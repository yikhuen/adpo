# Adaptive DPO

Quick demo to train DPO with an adaptive beta controller (target-KL, EMA, clipping) on `Qwen/Qwen2.5-7B-Instruct` using Unsloth + TRL, and evaluate with an LLM-as-judge (`gpt-4o-mini`).

## 0) Deploy a GPU (A40 recommended)
- GPU: A40 48GB (great value) or L4 24GB. GPU count: 1
- Template: PyTorch 2.8 (Ubuntu 24.04 + CUDA 12.8.x) – PyTorch is preinstalled
- Storage: 80–100 GB ephemeral. Do NOT attach a persistent volume unless you want to pay for storage after shutdown
- Check “SSH Terminal Access”. Jupyter optional

## 1) SSH access
- Add your public key to Runpod → Settings → SSH Public Keys
- Use the Exposed TCP command from the pod Connect tab (replace IP/PORT):
```bash
ssh -i ~/.ssh/id_rsa root@<PUBLIC_IP> -p <PORT>
```
Tip: If you see Permission denied, Stop → Start the pod to inject keys and ensure you’re using the matching private key path.

## 2) Keep jobs alive (tmux)
```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s run
# Detach with: Ctrl+b then d ; Reattach: tmux attach -t run
```

## 3) Get code and install deps
```bash
git clone https://github.com/yikhuen/adpo.git && cd adpo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# PyTorch is already in the image; just install the project deps
pip install -r requirements.txt --no-cache-dir
```

## 4) Environment variables (.env)
Create and load once per session:
```bash
cat > .env << 'EOF'
WANDB_API_KEY=YOUR_WANDB_KEY
WANDB_PROJECT=adaptive-dpo
OPENAI_API_KEY=YOUR_OPENAI_KEY
HF_HUB_ENABLE_HF_TRANSFER=1
HF_DATASETS_DISABLE_MULTIPROCESSING=1
PYTHONHASHSEED=42
EOF
set -a && source .env && set +a
```

## 5) Inspect and prepare datasets
- Preview a few formatted examples (ensures system/user prompts look right):
```bash
python scripts/inspect_dataset.py --alias ultrafeedback --split train --samples 3
python scripts/inspect_dataset.py --alias anthropic_hh --config configs/train/qwen25_7b_adaptive_beta.yaml --split train --samples 3
```
- Export a held-out prompt set for evaluation (reuses chat template formatting):
```bash
python scripts/prepare_dev_set.py \
  --dataset ultrafeedback \
  --size 200 \
  --split test \
  --out data/dev.jsonl
```

## 6) Train models
- Adaptive per-token KL controller:
```bash
python scripts/train.py --config configs/train/qwen25_7b_adaptive_beta.yaml
```
- Fixed-β baseline (set `fixed_beta` in config or override the file):
```bash
python scripts/train.py --config configs/train/qwen25_7b_fixed_beta.yaml
```
- Annealed-β schedule baseline:
```bash
python scripts/train.py --config configs/train/qwen25_7b_annealed_beta.yaml
```

Outputs are saved under `outputs/...` (per seed) with LoRA adapters, tokenizer, and `train_stats.json`.

## 7) Evaluate with LLM judges
Run multi-model comparisons with cached generations, metric exports, and judge agreement:
```bash
python scripts/eval.py \
  --config configs/eval/judge_gpt4o_mini.yaml \
  --force-judge    # optional; rerun judges even if cached decisions exist
```
Artifacts land in `research/results/eval/`:
- `responses/*.jsonl` – per-model generations stripped of prompts
- `decisions/*.jsonl` – per-judge pairwise choices
- `metrics/summary.json` & `summary.csv` – win rates + 95% CIs
- `metrics/judge_agreement.json` – agreement %, Cohen’s κ (when ≥2 judges)

Speed tips:
- Smaller dev set: `python scripts/prepare_dev_set.py --size 50 ...`
- Fast dry-run: `python scripts/eval.py --config ... --limit 50`
- Reduce decoding length via `generation.max_new_tokens` in the eval config.

## 8) Orchestrate complete phases (optional)
Automate the multi-run experiments described in `research/poster_plan.md`:
```bash
# Phase 1: fixed-β brittleness grid (β = 0.05, 0.10, 0.20)
python scripts/orchestrate.py phase1

# Phase 2: adaptive vs oracle vs annealed + evaluation
python scripts/orchestrate.py phase2

# Phase 3: ablation stress test (toggle EMA/deadband/clipping)
python scripts/orchestrate.py phase3

# Phase 4: generalization evaluation (create a matching eval config first)
python scripts/orchestrate.py phase4 --eval-config configs/eval/generalization.yaml
```
Results for each phase (training stats, evaluation metrics) are stored in `research/results/<phase>/`.

## 9) Save and copy results off the pod
```bash
tar -czf results.tgz outputs outputs_fixed
# from your PC
scp -P <PORT> -i ~/.ssh/id_rsa root@<PUBLIC_IP>:~/adpo/results.tgz \
  "/c/Users/<username>/Downloads/"
```

## Notes
- Training uses Unsloth QLoRA (LoRA adapters on a 4-bit base). Not full fine-tuning
- We log β, KL EMA, runtime stats, and throughput to Weights & Biases (`WANDB_*` envs required)
- If you see import issues, export once: `export PYTHONPATH=$PWD/src:$PYTHONPATH`
- For a detailed experiment playbook (phase breakdown, compute budget, risks) see `research/runbook.md`
