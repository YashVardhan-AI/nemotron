from val.holdout_registry import HoldoutRegistry


def test_reserve_and_query(tmp_path):
    path = tmp_path / "holdout_rules.json"
    reg = HoldoutRegistry(path)

    assert reg.is_reserved("cipher", "SIG1") is False
    assert reg.reserve("cipher", "SIG1") is True  # newly reserved
    assert reg.is_reserved("cipher", "SIG1") is True
    assert reg.reserve("cipher", "SIG1") is False  # already reserved


def test_persists_across_instances(tmp_path):
    path = tmp_path / "holdout_rules.json"
    HoldoutRegistry(path).reserve("bit_manipulation", "SIGX")

    reg2 = HoldoutRegistry(path)
    assert reg2.is_reserved("bit_manipulation", "SIGX") is True
    assert reg2.is_reserved("cipher", "SIGX") is False  # category-scoped
