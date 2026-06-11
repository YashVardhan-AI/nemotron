# ruff: noqa
# =====================================================================
# Kaggle cell: download the trained Tinker adapter + convert it to a
# vLLM-loadable rank-32 LoRA, entirely on Kaggle (no Modal, no upload).
#
# Requirements:
#   - Notebook internet ON (the download fetches from Tinker's servers).
#   - The base model attached (set BASE_MODEL to its path) OR downloadable.
#   - Your Tinker API key (Kaggle Secret "TINKER_API_KEY" or paste below).
#
# Output: /kaggle/working/nemotron-adapter-ready-to-submit
#   -> set the eval's ADAPTER_PATH to this dir; submission.zip-able.
#
# Run this BEFORE the eval cells.
# =====================================================================

# ---------------- config ----------------
# Sampler weights of your run (from training/sft/<ts>/checkpoints.jsonl ->
# "sampler_path"). NOT weights/final (that is training state for resuming).
TINKER_MODEL_PATH = (
    "tinker://ab25e405-af9e-5d12-a7e5-2e26c242ec2f:train:0/sampler_weights/final"
)
# build_lora_adapter needs the base model's config + state keys/shapes.
# On Kaggle, set this to your ATTACHED base-model dir to avoid a ~60GB download,
# e.g. "/kaggle/input/.../NVIDIA-Nemotron-3-Nano-30B-A3B-BF16".
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
RAW_DIR = "/kaggle/working/raw-adapter"
READY_DIR = "/kaggle/working/nemotron-adapter-ready-to-submit"

# Tinker API key: prefer a Kaggle Secret named TINKER_API_KEY, else paste here.
TINKER_API_KEY = ""

import os

if not TINKER_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient

        TINKER_API_KEY = UserSecretsClient().get_secret("TINKER_API_KEY")
    except ImportError:
        pass  # not running on Kaggle / no secrets client -> paste the key above
assert TINKER_API_KEY.startswith("tml-"), (
    "Set TINKER_API_KEY (Kaggle Secret 'TINKER_API_KEY' or paste in the cell)"
)
os.environ["TINKER_API_KEY"] = TINKER_API_KEY

# ---------------- 1. install matching SDKs (old tinker is server-rejected) ----
import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "tinker>=0.22", "tinker-cookbook>=0.4"],
    check=True,
)

# ---------------- 2. download + extract the sampler adapter from Tinker -------
import re, shutil, tarfile, urllib.request
import tinker

shutil.rmtree(RAW_DIR, ignore_errors=True)
os.makedirs(RAW_DIR, exist_ok=True)
sc = tinker.ServiceClient()
url = (
    sc.create_rest_client()
    .get_checkpoint_archive_url_from_tinker_path(TINKER_MODEL_PATH)
    .result()
    .url
)
_tar = "/kaggle/working/_adapter.tar"
print("downloading adapter ...")
urllib.request.urlretrieve(url, _tar)
print(f"  {os.path.getsize(_tar) / 1e6:.1f} MB")
with tarfile.open(_tar) as t:
    t.extractall(RAW_DIR)
os.remove(_tar)
# flatten: bring adapter_config.json + adapter_model.safetensors to RAW_DIR top
_cfg = _wts = None
for root, _dirs, files in os.walk(RAW_DIR):
    for f in files:
        if f == "adapter_config.json":
            _cfg = os.path.join(root, f)
        elif f == "adapter_model.safetensors":
            _wts = os.path.join(root, f)
assert _cfg and _wts, f"adapter files not found under {RAW_DIR}"
for p in (_cfg, _wts):
    d = os.path.join(RAW_DIR, os.path.basename(p))
    if os.path.abspath(p) != os.path.abspath(d):
        shutil.move(p, d)
print("raw adapter:", sorted(os.listdir(RAW_DIR)))

# ---------------- 3. force fused Mamba in_proj back to rank 32 (vLLM cap) -----
# Tinker trains in_proj as separate rank-32 gate_proj/x_proj; the official merge
# yields a rank-64 in_proj that vLLM (max_lora_rank=32) CANNOT load. Patch the
# merge to SVD-truncate the fused delta to rank 32 (plain Eckart-Young; the
# activation-aware variant was an A/B-NULL, so plain is the proven path).
import torch
import tinker_cookbook.weights._adapter as _A

FORCED_FUSED_RANK = 32


def _compress(B, Amat, rank):
    B, Amat = B.float(), Amat.float()
    U, S, Vh = torch.linalg.svd(B @ Amat, full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
    sr = torch.sqrt(S.clamp_min(0))
    return (U * sr).contiguous(), (sr.unsqueeze(1) * Vh).contiguous()


def _patched_merge(
    fused_model_key,
    adapter_layer_prefix,
    components,
    model_state_shapes,
    peft_weights,
    target_modules,
    profile,
):
    fused_out_dim = model_state_shapes[fused_model_key][0]
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]
    order = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            order = comps
            break
    assert order is not None
    comp = {n: (a, b) for n, a, b in components}
    A_parts, slices, merged_rank, off = [], [], 0, 0
    for name in order:
        if name not in comp:
            raise RuntimeError(f"missing component {name!r} for {fused_model_key!r}")
        a, b = comp[name]
        A_parts.append(a)
        slices.append((off, off + b.shape[0], a.shape[0]))
        off += b.shape[0]
        merged_rank += a.shape[0]
    mA = torch.cat(A_parts, dim=0)
    mB = torch.zeros(fused_out_dim, merged_rank, dtype=mA.dtype, device=mA.device)
    roff = 0
    for i, (r_start, r_end, r) in enumerate(slices):
        _, b = comp[order[i]]
        mB[r_start:r_end, roff : roff + r] = b
        roff += r
    final = merged_rank
    if merged_rank > FORCED_FUSED_RANK:
        mB, mA = _compress(mB, mA, FORCED_FUSED_RANK)
        final = FORCED_FUSED_RANK
        print(f"  {fused_model_key}: {merged_rank}->{FORCED_FUSED_RANK}")
    _A._add_peft_weight(
        f"{adapter_layer_prefix}.{fused_target_name}.weight",
        mA,
        mB,
        peft_weights,
        target_modules,
    )
    return final


_A._merge_fused_projections = _patched_merge
print("patched merge:", _A._merge_fused_projections.__name__)

# ---------------- 4. convert -> vLLM-ready PEFT adapter -----------------------
from tinker_cookbook import weights

shutil.rmtree(READY_DIR, ignore_errors=True)  # build_lora_adapter requires fresh dir
weights.build_lora_adapter(
    base_model=BASE_MODEL, adapter_path=RAW_DIR, output_path=READY_DIR
)
print("\nREADY adapter:", sorted(os.listdir(READY_DIR)))
print(f"Set ADAPTER_PATH = {READY_DIR!r} in the eval cell.")
