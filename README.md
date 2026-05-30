# NVIDIA Progress Prize submission

This is the Github repository to the Progress Prize winning submission for NVIDIA Nemotron Model Reasoning Challenge.

Resources on Kaggle

- [Writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)
- [Notebook](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)


## Executing training


```
uv run python3 reasoning.py
uv run python3 augmentation.py
uv run python3 corpus.py
uv run python3 train_sft.py
uv run modal run upload_adapter.py
```
