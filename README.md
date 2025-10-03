# Adaptive DPO (Minimal Demo)

Quick demo repo to train DPO with an adaptive beta controller (target-KL, EMA, clipping) on `Qwen/Qwen2.5-7B-Instruct` using Unsloth + TRL, and evaluate with an LLM-as-judge (gpt-4o-mini).

## Quick start (Runpod L40 48GB)

1. Create venv and install PyTorch per CUDA (commonly CUDA 12.1 on L40):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

2. Export env (fill your keys):
```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=adaptive-dpo
export OPENAI_API_KEY=...
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_DISABLE_MULTIPROCESSING=1
export PYTHONHASHSEED=42
```

3. Prepare a tiny dev set (100–200):
```bash
python scripts/prepare_dev_set.py --dataset HuggingFaceH4/ultrafeedback_binarized --size 200 --out data/dev.jsonl
```

4. Train (1 epoch, small sample):
```bash
python scripts/train.py --config configs/train/qwen25_7b_adaptive_beta.yaml
```

5. Evaluate with gpt-4o-mini judge:
```bash
python scripts/eval.py --config configs/eval/judge_gpt4o_mini.yaml --ckpt outputs/last --dev data/dev.jsonl
```

See `configs/*` for tunables.
