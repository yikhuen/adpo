# Legacy RunPod / Cloud GPU Guide

These notes capture the original workflow for launching Adaptive DPO runs on RunPod. They are intentionally verbose and cover everything from provisioning to copying results off the pod. If you're running locally or on another cloud provider, use this as inspiration rather than a hard requirement.

---

## 0. Provision the Pod

- GPU: A40 48 GB (or L4 24 GB if budget constrained). GPU count: 1.
- Template: **PyTorch 2.8 (Ubuntu 24.04 + CUDA 12.8.x)** — PyTorch is preinstalled.
- Storage: 80–100 GB ephemeral. Avoid attaching persistent volumes unless you need them after shutdown.
- Enable **SSH Terminal Access** (Jupyter optional).
- Remember: RunPod only persists `/workspace` between restarts. Keep all code + outputs there.

## 1. SSH Access

1. Add your public key to RunPod → *Settings → SSH Public Keys*.
2. Use the exposed TCP endpoint from the pod *Connect* tab:

```bash
ssh -i ~/.ssh/id_rsa root@<PUBLIC_IP> -p <PORT>
```

If you hit `Permission denied`, stop → start the pod to re-inject keys and verify the private key path.

## 2. Keep Jobs Alive (tmux)

```bash
sudo apt-get update && sudo apt-get install -y tmux
tmux new -s run
# Detach: Ctrl+b → d     Reattach: tmux attach -t run
```

## 3. Clone & Install

Always operate inside `/workspace` so files persist:

```bash
cd /workspace
git clone https://github.com/yikhuen/adpo.git && cd adpo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Legacy approach (pre-package):
export PYTHONPATH=/workspace/adpo/src:$PYTHONPATH
```

(With the new packaging workflow you can instead run `pip install -e .`.)

## 4. Environment Variables via `.env`

Create locally:

```bash
WANDB_API_KEY=YOUR_WANDB_KEY
WANDB_PROJECT=adaptive-dpo
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
HF_HUB_ENABLE_HF_TRANSFER=1
HF_DATASETS_DISABLE_MULTIPROCESSING=1
PYTHONHASHSEED=42
```

Copy to the pod:

```bash
scp -P <PORT> -i ~/.ssh/<SSH_KEY> .env root@<PUBLIC_IP>:/workspace/adpo/.env
```

On the pod:

```bash
cd /workspace/adpo
cat -vet .env              # sanity check line endings
sed -i 's/\r$//' .env      # strip CRLF if needed
set -a && source .env && set +a
```

## 5. Dataset Prep

Preview formatted samples:

```bash
python scripts/inspect_dataset.py --config configs/train/qwen25_7b_adaptive_beta.yaml --split train --samples 3
python scripts/inspect_dataset.py --alias anthropic_hh --config configs/train/qwen25_7b_adaptive_beta.yaml --split train --samples 3
```

Export a held-out evaluation set:

```bash
python scripts/prepare_dev_set.py \
  --config configs/train/qwen25_7b_adaptive_beta.yaml \
  --size 200 \
  --split eval \
  --out data/dev.jsonl
```

## 6. Training Commands (Legacy)

```bash
python scripts/train.py --config configs/train/qwen25_7b_adaptive_beta.yaml
python scripts/train.py --config configs/train/qwen25_7b_fixed_beta.yaml
python scripts/train.py --config configs/train/qwen25_7b_annealed_beta.yaml
```

Outputs land under `outputs/...` with LoRA adapters + `train_stats.json`.

## 7. Evaluation Shortcuts

```bash
python scripts/eval.py openai-judge --force-judge
python scripts/eval.py openrouter-judge --force-judge
python scripts/eval.py all-judges --force-judge
```

Artifacts go to `research/results/eval*/` (responses, decisions, metrics, reward-hacking JSON, etc.).

## 8. Orchestrating Phases

```bash
python scripts/orchestrate.py phase1
python scripts/orchestrate.py phase2 --oracle-beta 0.2
python scripts/orchestrate.py phase3
python scripts/orchestrate.py phase4 --eval-config configs/eval/generalization.yaml
```

Phase-specific outputs are stored under `research/results/<phase>/`.

## 9. Copy Results Off the Pod

```bash
cd /workspace/adpo
tar -czf results.tgz outputs outputs_fixed
scp -P <PORT> -i ~/.ssh/id_rsa root@<PUBLIC_IP>:/workspace/adpo/results.tgz ~/Downloads/
```

---

These instructions remain here as a reference for older environments. For modern usage, rely on the packaged CLI/ pipelines described in the top-level README.***

