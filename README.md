# Adaptive DPO

Adaptive DPO is a research-grade framework for Direct Preference Optimisation with **adaptive KL control**, modular evaluation, and end-to-end experiment orchestration. It targets runs on `Qwen/Qwen2.5-7B-Instruct` (via Unsloth + TRL) but the pipelines are controller/dataset agnostic.

---

## Quick Start

```bash
cd /workspace                   # standard project root on GPU clouds (RunPod, VastAI, etc.)
git clone https://github.com/yikhuen/adpo.git
cd adpo
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .            # exposes adaptive_dpo.* modules
pip install -r requirements.txt
```

Environment variables (set in your shell or `.env`):

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=adaptive-dpo
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
```

See [`docs/config.md`](docs/config.md) for training/eval/orchestration config schemas.

---

## Training

```bash
python scripts/train.py --config configs/train/qwen25_7b_adaptive_beta.yaml
```

This calls `adaptive_dpo.cli.train`, which delegates to `adaptive_dpo.pipelines.train`. Multiple seeds are handled automatically and summaries land under the configured `trainer.output_dir`.

---

## Evaluation

```bash
# Combined GPT-4o-mini + OpenRouter Gemini judges
python scripts/eval.py all-judges --config configs/eval/judge_gpt4o_mini.yaml --force-judge

# Shortcuts
python scripts/eval.py openai-judge  --config configs/eval/judge_openai_only.yaml
python scripts/eval.py openrouter-judge --config configs/eval/judge_openrouter_only.yaml
# For a lightweight sanity check (OpenAI only) you can reuse configs/eval/judge_phase1.yaml
python scripts/eval.py all-judges    --config configs/eval/judge_phase1.yaml  # Phase 1 template (candidate vs base)
```

The evaluation stack (`adaptive_dpo.eval.*`) splits responsibilities:

| Module | Responsibility |
| --- | --- |
| `prompts.py` | Prompt loading, shuffling, overrides |
| `generation.py` | Model loading + cache-aware generation |
| `judging.py` | Pairwise judge drivers (OpenAI, OpenRouter, HF causal) |
| `metrics.py` | Win-rate computation, agreement stats |
| `logging.py` | WandB logging, artifact bundling |
| `runner.py` | High-level orchestration (`run_evaluation`) |

Generations are cached per prompt ID (with resume support) so interrupted runs restart instantly.

---

## Orchestration (Phases 1–4)

```bash
# Phase 1: fixed β sweep + 50-prompt judge eval (writes phase1_results.json)
python scripts/orchestrate.py phase1 \
  --eval-config configs/eval/judge_phase1.yaml \
  --eval-prompt-size 50

# Phase 2: adaptive vs annealed vs fixed (+ diagnostics + Markdown report)
python scripts/orchestrate.py phase2 --oracle-beta 0.2

# Phase 3: ablation suite with automatic eval + phase portraits
python scripts/orchestrate.py phase3 --ablations full no_deadband no_ema no_clipping

# Phase 4: generalisation matrix (auto-prepares prompts when using alias:/config:)
python scripts/orchestrate.py phase4 \
  --eval-config configs/eval/judge_gpt4o_mini.yaml \
  --dataset uf=alias:ultrafeedback \
  --dataset hh=alias:anthropic_hh \
  --model adaptive=kind:lora,checkpoint:outputs/adaptive_beta/checkpoint-120
```

Highlights:

- Phase 1 now evaluates every β candidate (OpenAI + OpenRouter Gemini) and records the best pick inside `research/results/phase1_fixed_beta_grid/phase1_results.json`.
- Phase 2 automatically runs `scripts/run_eval_with_diagnostics.py` (controller plots, entropy buckets, flip-rate checks) and generates `results/phase2_report_latest.md` via `scripts/report_phase2.py`.
- Phase 3 triggers evaluations for each ablation, stores per-variant metrics, and emits `phase3/ablation_win_rates.png` using `scripts/plot_ablation_bar.py`.
- Phase 4 accepts dataset specs of the form `name=alias:<formatter>` or `name=config:<yaml>`; prompts are materialised with `scripts/prepare_dev_set.py` before running the sweep.

### Typical Repository Workflow

1. **Bootstrap the workspace.** Follow the _Quick Start_ section, export API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `WANDB_API_KEY`), and run `pytest` to ensure the environment is healthy.
2. **Phase 1 – fixed β sweep.**  
   ```bash
   python scripts/orchestrate.py phase1 \
     --eval-config configs/eval/judge_phase1.yaml \
     --eval-prompt-size 50
   ```
   Review `research/results/phase1_fixed_beta_grid/phase1_results.json` for the winning β value (under `best_beta`).
3. **Phase 2 – adaptive vs baselines.** Supply the β from Phase 1 via `--oracle-beta`:
   ```bash
   python scripts/orchestrate.py phase2 --oracle-beta <PHASE1_BEST>
   ```
   This run:
   - Trains adaptive/annealed/fixed controllers.
   - Reuses `configs/eval/judge_gpt4o_mini.yaml` with GPT‑4o-mini + OpenRouter Gemini judges.
   - Launches diagnostics (phase traces, entropy buckets, flip-rate) under `research/results/phase2_adaptive_vs_baselines/diagnostics`.
   - Writes a Markdown summary to `results/phase2_report_latest.md`.
4. **Phase 3 – controller ablations.**
   ```bash
   python scripts/orchestrate.py phase3 \
     --ablations full no_deadband no_ema no_clipping no_fast_loop
   ```
   Each variant is evaluated automatically; plots land under `research/results/phase3_ablation/phase_traces` and aggregated win-rates are charted in `research/results/phase3_ablation/ablation_win_rates.png`.
5. **Phase 4 – generalisation matrix.** Use `alias:` or `config:` dataset specs to auto-generate prompt files:
   ```bash
   python scripts/orchestrate.py phase4 \
     --eval-config configs/eval/judge_gpt4o_mini.yaml \
     --dataset uf=alias:ultrafeedback \
     --dataset hh=alias:anthropic_hh \
     --model adaptive=kind:lora,checkpoint:outputs/adaptive_beta/checkpoint-120
   ```
   Metrics per dataset are stored in `research/results/phase4_generalization/<dataset>/metrics/` and summarised in `research/results/phase4_generalization/summary.json`.
6. **Ad-hoc training/eval.** `scripts/train.py` and `scripts/eval.py` remain available for standalone experiments (e.g., debugging a new dataset formatter).
7. **Reporting.** Beside Phase 2’s Markdown, you can re-run `scripts/report_phase2.py --phase-dir ... --eval-output ...` or extend it for other phases to keep artifacts reproducible.

All heavy lifting lives in `adaptive_dpo.pipelines.orchestration`; the CLI layer simply parses options. Shared helpers cover config templating, artifact copying, dataset/model spec parsing, and eval triggering.

---

## Repository Layout

```
├─ configs/                # Train/eval YAML configs
├─ docs/
│  ├─ config.md            # Config schema & examples
│  └─ runpod.md            # Legacy GPU provisioning (RunPod/tmux/SCP)
├─ scripts/                # Thin entrypoints -> adaptive_dpo.cli.*
├─ src/adaptive_dpo/
│  ├─ cli/                 # Typer apps (train/eval/orchestrate)
│  ├─ controllers/         # Adaptive, hybrid, robust controllers + configs
│  ├─ data/                # Dataset loaders + formatters
│  ├─ eval/                # Prompt/generation/judging/logging modules
│  ├─ pipelines/           # Training/orchestration pipelines
│  └─ utils/               # Generation helpers, reproducibility tools
└─ tests/                  # Pytest suite (controllers, data, caching, orchestration)
```

Large PDFs, posters, and playbooks now live under `docs/` instead of the project root to keep checkouts lean.

---

## Testing & CI

```bash
pytest            # runs unit tests
ruff check .      # lint (optional but recommended)
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint + pytest on every push and PR.

---

## Contributing

1. Install dependencies (`pip install -e .`).
2. Make changes inside `src/adaptive_dpo` (pipelines/cli) rather than scripts.
3. Add or update tests under `tests/`.
4. Update docs (README or `docs/*.md`) when behaviour or configs change.
5. Submit a PR once lint/tests pass locally.

---

## Legacy Cloud Instructions

The previous README sections on RunPod provisioning, tmux usage, environment copying, and SCP backups were moved to [`docs/runpod.md`](docs/runpod.md). Refer there if you need the detailed GPU setup guide.

### Environment Sync Cheatsheet

- **Copy .env to a remote GPU:**
  ```bash
  scp .env user@remote-host:~/runs/adaptive_dpo/.env
  ```
- **Load env vars inside tmux/shell on the GPU:**
  ```bash
  cd ~/runs/adaptive_dpo
  python -m venv .venv && source .venv/bin/activate
  export $(grep -v '^#' .env | xargs)  # or use direnv
  ```
- **Persist env vars across sessions:** append the exports to `~/.bashrc` or use a secret manager (AWS/GCP/RunPod env panel) so that orchestrated runs inherit the keys automatically.

---

## License

MIT License. See `LICENSE`.

