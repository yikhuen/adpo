# Adaptive DPO (Minimal Demo)

Quick demo to train DPO with an adaptive beta controller (target-KL, EMA, clipping) on `Qwen/Qwen2.5-7B-Instruct` using Unsloth + TRL, and evaluate with an LLM-as-judge (`gpt-4o-mini`).

## 0) Deploy a GPU on Runpod (A40 recommended)
- GPU: A40 48GB (great value) or L4 24GB. GPU count: 1
- Template: Runpod PyTorch 2.8 (Ubuntu 24.04 + CUDA 12.8.x) – PyTorch is preinstalled
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

## 5) Prepare a small held-out dev set
```bash
python scripts/prepare_dev_set.py \
  --dataset HuggingFaceH4/ultrafeedback_binarized \
  --size 200 \
  --out data/dev.jsonl
```

## 6) Train
- Adaptive-β DPO
```bash
python scripts/train.py --config configs/train/qwen25_7b_adaptive_beta.yaml
```
- Fixed-β DPO baseline
```bash
python scripts/train.py --config configs/train/qwen25_7b_fixed_beta.yaml
```
Outputs are saved to `outputs/` (adaptive) and `outputs_fixed/` (baseline) with LoRA adapters + tokenizer.

## 7) Evaluate (pairwise judge with gpt-4o-mini)
```bash
python scripts/eval.py \
  --config configs/eval/judge_gpt4o_mini.yaml \
  --ckpt-adaptive outputs \
  --ckpt-fixed outputs_fixed \
  --dev data/dev.jsonl
```
What you’ll see:
- Stage headers: generation and judging per pair, e.g., `=== adaptive_vs_base: Generating ... ===`
- Progress every 10 prompts: `[gen ...] 10/200`, `[judge ...] 10/200`
- Final JSON with win rate and 95% CI for each pair

Speed tips:
- Reduce dev set: `--size 50`
- In `scripts/eval.py`, lower `max_new_tokens` (default 512)

## 8) Save and copy results off the pod
```bash
tar -czf results.tgz outputs outputs_fixed
# from your PC
scp -P <PORT> -i ~/.ssh/id_rsa root@<PUBLIC_IP>:~/adpo/results.tgz \
  "/c/Users/chaiy/Downloads/"
```

## 9) Stop billing
- Delete the pod in the Runpod console (this kills tmux and frees ephemeral storage)
- If you created a persistent volume, delete it separately to stop storage charges

## Notes
- Training uses Unsloth QLoRA (LoRA adapters on a 4-bit base). Not full fine-tuning
- We log β, KL EMA, and related stats to Weights & Biases (`WANDB_*` envs required)
- If you see import issues, export once: `export PYTHONPATH=$PWD/src:$PYTHONPATH`
