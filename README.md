# Adaptive DPO

Quick demo to train DPO with an adaptive beta controller (target-KL, EMA, clipping) on `Qwen/Qwen2.5-7B-Instruct` using Unsloth + TRL, and evaluate with an LLM-as-judge (`gpt-4o-mini`).

## 0) Deploy a GPU (A40 recommended)
- GPU: A40 48GB (great value) or L4 24GB. GPU count: 1
- Template: PyTorch 2.8 (Ubuntu 24.04 + CUDA 12.8.x) – PyTorch is preinstalled
- Storage: 80–100 GB ephemeral. Do NOT attach a persistent volume unless you want to pay for storage after shutdown
- Check "SSH Terminal Access". Jupyter optional
- **Important:** RunPod only persists data in `/workspace` when the pod stops. All project files should be placed in `/workspace` to survive pod restarts.

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
**Work in `/workspace` to ensure data persists across pod restarts:**
```bash
cd /workspace
git clone https://github.com/yikhuen/adpo.git && cd adpo
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# PyTorch is already in the image; just install the project deps
pip install -r requirements.txt --no-cache-dir

# Set PYTHONPATH so scripts can import adaptive_dpo module
export PYTHONPATH=/workspace/adpo/src:$PYTHONPATH
# (Add this to ~/.bashrc or run it each time you start a new terminal session)
```

## 4) Environment variables (.env)
Create and load once per session:
```bash
# On your local machine, create a file named .env with the following contents:
# (edit to add your keys before uploading)
WANDB_API_KEY=YOUR_WANDB_KEY
WANDB_PROJECT=adaptive-dpo
OPENAI_API_KEY=YOUR_OPENAI_KEY
GEMINI_API_KEY=YOUR_GEMINI_KEY
HF_HUB_ENABLE_HF_TRANSFER=1
HF_DATASETS_DISABLE_MULTIPROCESSING=1
PYTHONHASHSEED=42

# Then copy (upload) it to the GPU with:
# Replace <PUBLIC_IP>, <PORT>, and <SSH_KEY> with your actual values
scp -P <PORT> -i ~/.ssh/<SSH_KEY> .env root@<PUBLIC_IP>:/workspace/adpo/.env

# Now, once connected to the GPU terminal and in /workspace/adpo:
# 1. Check for carriage returns or whitespace issues:
cat -vet .env

# 2. Remove carriage returns and trailing whitespace if needed:
sed -i 's/\r$//' .env

# 3. Load the environment variables:
set -a && source .env && set +a
```

## 5) Inspect and prepare datasets
- Preview a few formatted examples (ensures system/user prompts look right):
python scripts/inspect_dataset.py \
  --config configs/train/qwen25_7b_adaptive_beta.yaml \
  --split train \
  --samples 3
python scripts/inspect_dataset.py --alias anthropic_hh --config configs/train/qwen25_7b_adaptive_beta.yaml --split train --samples 3
```
- Export a held-out prompt set for evaluation (reuses chat template formatting):
python scripts/prepare_dev_set.py \
  --config configs/train/qwen25_7b_adaptive_beta.yaml \
  --size 200 \
  --split eval \
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
Run multi-model comparisons with cached generations, metric exports, and judge agreement. Choose the judge mix that fits your keys:
```bash
# GPT-4o-mini only
python scripts/eval.py openai-judge --force-judge

# Gemini 2.0 Flash only
python scripts/eval.py gemini-judge --force-judge

# Both judges (default Phase 2 setting)
python scripts/eval.py all-judges --force-judge
```
Each shortcut loads a default config (`configs/eval/judge_openai_only.yaml`, `configs/eval/judge_gemini_only.yaml`, or `configs/eval/judge_gpt4o_mini.yaml`), which you can override with `--config path/to/config.yaml`. All three configs enable W&B logging by default, so summary tables and response-length bar charts are pushed automatically (set `WANDB_API_KEY` first).

Artifacts land in the directory specified by the chosen config (defaults: `research/results/eval/`, `research/results/eval_openai/`, `research/results/eval_gemini/`):
- `responses/*.jsonl` – per-model generations stripped of prompts
- `decisions/*.jsonl` – per-judge pairwise choices
- `metrics/summary.json` & `summary.csv` – win rates + 95% CIs
- `metrics/judge_agreement.json` – agreement %, Cohen’s κ (when ≥2 judges)
- `metrics/model_stats.json` – reward-hacking sanity check: Avg response length, refusal/safety rates per model
- `metrics/reward_hacking.json` – heuristic flags comparing those stats against configurable thresholds

Advanced WandB logging (optional) can be toggled directly in the `wandb:` block of any eval config. Besides the built-in summary/model tables, you can enable:

- **Decision table**  
  ```yaml
  wandb:
    decision_table:
      enabled: true
      max_rows: 200          # optional cap (default 200)
      prompt_chars: 512      # truncate long prompts/responses
      paths: []              # optional extra decision JSONL paths
  ```
  This logs `eval/decision_table`, sampling rows from `decisions/*.jsonl`.

- **Attachments** – bundle raw artefacts into the eval run:
  ```yaml
  wandb:
    attachments:
      include_prompt_file: true          # attach data/dev.jsonl (or override below)
      prompt_file: data/dev.jsonl        # optional path override
      prompt_file_name: prompts.jsonl    # rename inside the artifact
      include_decisions_dir: true        # attach decisions/
      include_responses_dir: true        # attach responses/
      decisions_name: decisions         # optional rename
      responses_name: responses
      extra_files:                       # arbitrary files or directories
        - path: results/entropy_bucket_summary.json
          name: entropy_bucket_summary.json
        - path: results/controller_plots/qwen25_adaptive_phase_portrait.png
          log_image: true                # also log as wandb.Image
  ```

- **Controller diagnostics**  
  ```yaml
  wandb:
    phase_trace:
      json: outputs/adaptive_beta/phase_trace.json
      plots:
        - results/controller_plots/qwen25_adaptive_phase_portrait.png
        - results/controller_plots/qwen25_adaptive_beta_kl_time.png
  ```
  The JSON gets hashed into `run.summary["phase_trace_stats"]` and each plot is logged as `wandb.Image`.

- **Entropy buckets / flip-rate summaries**  
  ```yaml
  wandb:
    entropy_buckets:
      summary: results/entropy_bucket_summary.json
      plot: results/entropy_bucket_plot.png
    fliprate:
      summary: results/fliprate_summary.json
      plot: results/fliprate_plot.png
  ```
  Each JSON becomes a table (`eval/entropy_buckets`, `eval/fliprate`) and associated plots are uploaded if present.

- **Aggregated summary from `summarize_eval_runs.py`**
  ```yaml
  wandb:
    eval_summary:
      path: results/eval_summary.csv
      name: eval_summary.csv
  ```
  A table `eval/aggregated_summary` is logged and the CSV is bundled into the artifact.

With these toggles in place, simply run `python scripts/eval.py …` and the pipeline handles the uploads automatically.

### Evaluate & Run Diagnostics

1. **Prepare the dev set (if needed):**
   ```bash
   python scripts/prepare_dev_set.py \
       --config configs/train/qwen25_7b_adaptive_beta.yaml \
       --size 200 \
       --split eval \
       --out data/dev.jsonl
   ```

2. **Run evaluation** (choose your judge mix):
   ```bash
   python scripts/eval.py openai-judge --force-judge
   # or: python scripts/eval.py gemini-judge --force-judge
   # or: python scripts/eval.py all-judges --force-judge
   ```

3. **Run diagnostics / attachments** (works immediately; no WandB export needed):
   ```bash
   python scripts/run_eval_with_diagnostics.py --eval-subcommand openai-judge
   ```
   - Use `--skip-eval` if you only want aggregation.
   - The wrapper auto-generates `research/results/<eval_dir>/metrics/wandb_export_local.csv` from cached `decisions/*.jsonl`. Override with `--summary-csv`, `--entropy-csv`, or `--fliprate-csv` if you have custom data.
   - Skip components via `--skip-phase-plots`, `--skip-entropy`, `--skip-fliprate`, `--skip-summary` as needed.

4. **Inspect WandB** – the run logs summary tables, controller plots, entropy buckets, flip rates, and bundles local artefacts automatically, provided referenced files exist.

## 8) Orchestrate complete phases (optional)
 Automate the multi-run experiments for the four study phases ([Phase 1](#phase-1), [Phase 2](#phase-2), [Phase 3](#phase-3), [Phase 4](#phase-4)):
```bash
# Phase 1: fixed-β brittleness grid (β = 0.05, 0.10, 0.20)
python scripts/orchestrate.py phase1

# Phase 2: adaptive vs oracle vs annealed + evaluation (single command, auto-logs to W&B)
python scripts/orchestrate.py phase2 --oracle-beta 0.2

# Phase 3: ablation stress test (toggle EMA/deadband/clipping)
python scripts/orchestrate.py phase3

# Phase 4: generalization evaluation (create a matching eval config first)
python scripts/orchestrate.py phase4 --eval-config configs/eval/generalization.yaml
```
Results for each phase (training stats, evaluation metrics) are stored in `research/results/<phase>/`.

- The `--oracle-beta` flag lets you inject the “best” fixed β identified in Phase 1 (or a dataset-specific value) so the Phase 2 oracle baseline trains with that exact setting and logs under a beta-specific subfolder.
- Phase 2 training now saves `phase_trace.json` plus `phase_plot.png` under each run output and logs the KL–β phase portrait to W&B (`phase/<run>_plot`), showing the controller trajectory and poison spike.
- Phase 2 evaluation runs at the end of the command and pushes win rates, judge agreement, response-length sanity checks, and model stats directly to W&B (artifacts + panels). Local files remain as cache only.
- The poison audit is executed automatically (default batch index 15) and logs per-sample DPO losses to W&B when `--audit-wandb-project` is provided. Re-run manually to inspect other batches:
  ```bash
  # Inspect shuffled batch (forensic checklist)
  python scripts/inspect_poison_batch.py \
    --config configs/train/qwen25_7b_adaptive_beta.yaml \
    --batch-index 15 \
    --batch-size 8 \
    --seed 42

  # Compute per-sample losses and log to W&B
  python scripts/poison_audit.py \
    --config configs/train/qwen25_7b_adaptive_beta.yaml \
    --model-dir outputs/adaptive_beta \
    --batch-index 15 \
    --batch-size 8 \
    --seed 42 \
    --phase-trace outputs/adaptive_beta/phase_trace.json \
    --wandb-project adaptive-dpo \
    --wandb-name phase2_poison_audit
  ```

## 9) Save and copy results off the pod
```bash
cd /workspace/adpo
tar -czf results.tgz outputs outputs_fixed
# from your PC
scp -P <PORT> -i ~/.ssh/id_rsa root@<PUBLIC_IP>:/workspace/adpo/results.tgz \
  "/c/Users/<username>/Downloads/"
```

## Notes
- Training uses Unsloth QLoRA (LoRA adapters on a 4-bit base). Not full fine-tuning
- We log β, KL EMA, runtime stats, and throughput to Weights & Biases (`WANDB_*` envs required)
- If you see import issues, export once: `export PYTHONPATH=/workspace/adpo/src:$PYTHONPATH`
- **All work should be done in `/workspace`** - this is the only directory that persists when RunPod pods stop
- For a detailed experiment playbook (phase breakdown, compute budget, risks) see `research/runbook.md`

 **Phase Plan**

 <a id="phase-1"></a>

 ## Phase 1 – Fixed-β Brittleness Study
- **Objective:** Demonstrate how static β choices create a brittle trade-off between policy drift and underfitting on UltraFeedback.
- **Configuration:** Qwen2.5-7B-Instruct, Unsloth QLoRA (4-bit), batch 2×4 tokens with gradient accumulation 8, learning rate 1e-5, one epoch.
- **Experiments:** Train SFT baseline; DPO with β ∈ {0.05, 0.10, 0.20}. Log per-step KLtoken, β (static), judge win rates (vs. SFT), response length, refusal rate, safety flags.
- **Analytics:** Wilson 95% CIs over 200-prompt evaluation set, GPT-4o-mini judge with deterministic decoding (T=0), secondary check using open-source judge (e.g., Llama Guard).
- **Deliverables:** Oracle bar chart (win rate vs. SFT), KL trajectory plot highlighting drift (low β) vs. underfitting (high β), failure-mode table (length, refusals, safety).

 <a id="phase-2"></a>

## Phase 2 – Adaptive Controller vs. Baselines
- **Objective:** Show the proposed controller matches/exceeds oracle performance in one run while retaining stability.
- **Baselines:** Oracle fixed β (best from Phase 1); annealed β schedule (e.g., β0=0.20 decaying to 0.05 with cosine anneal).
- **Controller Setup:** EMA α=0.10, deadband ±10% around KL* = 0.04 nats/token, η=0.01, βmin=0.05, βmax=2.0. Entropy spike enables after 10 warm-up steps with λ=4.0.
- **Training Schedule:** Use gradient accumulation 4 with `max_steps: 120` so each model receives ≥100 optimiser updates while keeping wall-clock reasonable.
- **Experiments:** Train each model with two random seeds; reuse Phase 1 logging; capture β trajectory for adaptive and annealed runs.
- **Analytics:** Report mean ± 95% CI across seeds; run paired bootstrap on win rates (adaptive vs. oracle, adaptive vs. annealed). Include stability plot overlaying KLema vs. KL* and β trajectory.
- **Deliverables:** “Money” bar chart with win rates and error bars; dual-axis stability plot; summary table for length/safety.
- **Evaluation commands:**
  ```bash
  python scripts/eval.py openai-judge --force-judge
  python scripts/eval.py gemini-judge --force-judge
  python scripts/eval.py all-judges --force-judge
  ```
  These configs log Phase 2 metrics, reward-hacking diagnostics, and plots to W&B automatically; set `WANDB_API_KEY` before running.
- **Robust hybrid controller:** Phase 2 now defaults to the robust hybrid PID controller. It keeps the EMA trend sensor but injects an entropy spike with a moderate sensitivity range ($\lambda=4$) after a brief warm-up. Baseline update:
  $\beta_{base}^{t+1} = \beta_{base}^{t} \cdot \exp(\eta \cdot \frac{KL_{sensor}-KL_{target}}{KL_{target}})$ whenever the KL deviates outside a ±deadband. The proportional term scales by
  $1 + \lambda \cdot \frac{\text{Entropy}_{batch}}{\log |V|}$. Configure it via:
  ```yaml
  beta_controller:
    kind: robust_hybrid
    target_kl: 0.04
    eta: 0.01
    lambda_entropy: 4.0
    vocab_size: 32000
    entropy_warmup_steps: 10
    beta_init: 0.10
    beta_min: 0.05
    beta_max: 2.0
trainer:
  gradient_accumulation_steps: 4
  max_steps: 120  # ensures ≥100 global updates
  ```
  Set `entropy_warmup_steps` > 0 if you want to disable the spike until the model stabilises.

 <a id="phase-3"></a>

 ## Phase 3 – Controller Ablation Justification
- **Objective:** Validate necessity of EMA, deadband, and clipping components.
- **Setup:** Same training recipe, 25% stratified subset of UltraFeedback, single seed.
- **Variants:** Full controller, No Deadband, No EMA, No Clipping.
- **Analytics:** Plot β trajectory and KLema for each variant; compute win rate vs. SFT on 100-prompt eval (deterministic); note qualitative instability (oscillation, divergence).
- **Theoretical Touch:** Add text panel outlining control-theory intuition (EMA for noise suppression, deadband to avoid chatter, clipping to bound gains).
- **Deliverables:** Bar chart of win rates per ablation; 2×2 β trajectory grid; succinct theoretical summary box.

 <a id="phase-4"></a>

 ## Phase 4 – Generalization Stress Test
- **Objective:** Prove the controller generalizes without retuning.
- **Models:** Best adaptive model and oracle fixed-β from Phase 2.
- **Datasets:** UltraFeedback (helpfulness), Anthropic HH-RLHF (harmlessness), standard sycophancy dataset (e.g., open-source SAA pairs). Convert to DPO pairs with consistent formatting.
  - **Preprocessing:** For HH, use harmless/helpful preference pairs, filter by quality label, and format as (prompt, preferred, dispreferred) strings with consistent system prompts. For sycophancy, align question-answer pairs, mark anti-sycophancy responses as preferred, and ensure tokenization matches UltraFeedback pipeline.
- **Protocol:** Evaluate zero-shot on new datasets using deterministic decoding; log win rate, refusal/safety metrics, and response length.
- **Analytics:** 95% CIs via Wilson interval; mention absence of retraining; run single seed due to budget, document limitation.
- **Deliverables:** Generalization matrix table covering all metrics; brief narrative of observed robustness vs. failure.

## Phase 5 – Synthesis and Future Work
- **Objective:** Craft the closing story that links brittleness discovery → adaptive solution → justified design → generalizable recipe.
- **Outputs:** Poster conclusion panel, planned extension bullets (PID, multi-objective, ORPO/SimPO transfer), highlight compute savings from reduced sweeps.

## Evaluation & Statistical Protocol
- **Judges:** Primary GPT-4o-mini with rubric, secondary open-source model (e.g., Prometheus 2 or Arena hardwired reward) for robustness.
- **Judge Consistency:** Compute agreement metrics (percentage agreement + Cohen’s κ) between judges on a 50-sample subset; manually inspect disagreements to adjust prompts or scoring rubrics if needed.
- **Sample Sizes:** 200 prompts for headline phases 1–2, 100 for ablations, 200 combined for generalization (≈70 per dataset).
- **Seeding:** Two seeds for SFT/oracle/adaptive/annealed; disclose when single-seed results are reported.
- **Intervals & Tests:** Wilson 95% CI for binary win rates; paired bootstrap (1k samples) for adaptive vs. baselines; report p-values where meaningful.
- **Logging:** Weights & Biases dashboards capturing KLtoken, KLema, β, loss, throughput; save checkpoints at mid-epoch and final.

## Compute & Resource Feasibility
- **Hardware Assumption:** Single 24–48 GB GPU (local 4090 or cloud L40S/A40). Per-run VRAM ≈20 GB with 4-bit QLoRA rank-32 adapters.
- **Runtime Estimates:** Full UltraFeedback epoch ≈3–3.5 GPU-hours; Phase 1 grid (3 runs) ≈10 GPU-hours; Phase 2 (3 configs ×2 seeds) ≈18 GPU-hours; Phase 3 ablations 4 runs ×1.5 GPU-hours ≈6 GPU-hours; Phase 4 evaluations ≈4 GPU-hours. Total ≈38 GPU-hours with serialization.
- **Cost Controls:** Use dataset subsets for ablations; cache LoRA adapters to resume; run generalization evaluations using existing checkpoints; schedule cloud bursts only for multi-seed phases.
- **Fallbacks:** Drop to Qwen2.5-3B or 1.8B if hardware constrained; reduce Phase 2 seeds to one and note limitation; cut Phase 4 to one additional dataset if time-compressed.
- **Risk Notes:** Monitor controller hyperparameters on small pilot before full runs; prepare emergency β clipping bounds; verify dataset preprocessing scripts for HH and sycophancy prior to large runs; pre-test judge prompts to ensure consistent scoring before committing to full evaluations.