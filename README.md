# Adaptive DPO

Adaptive DPO is a research-grade framework for Direct Preference Optimisation with **adaptive KL control**, modular evaluation, and end-to-end experiment orchestration. It targets runs on `Qwen/Qwen2.5-7B-Instruct` (via Unsloth + TRL) but the pipelines are controller/dataset agnostic.

---

## Quick Start

```bash
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
export GEMINI_API_KEY=...
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
# Combined GPT-4o-mini + Gemini judges
python scripts/eval.py all-judges --config configs/eval/judge_gpt4o_mini.yaml --force-judge

# Shortcuts
python scripts/eval.py openai-judge  --config configs/eval/judge_openai_only.yaml
python scripts/eval.py gemini-judge  --config configs/eval/judge_gemini_only.yaml
```

The evaluation stack (`adaptive_dpo.eval.*`) splits responsibilities:

| Module | Responsibility |
| --- | --- |
| `prompts.py` | Prompt loading, shuffling, overrides |
| `generation.py` | Model loading + cache-aware generation |
| `judging.py` | Pairwise judge drivers (OpenAI, Gemini, HF causal) |
| `metrics.py` | Win-rate computation, agreement stats |
| `logging.py` | WandB logging, artifact bundling |
| `runner.py` | High-level orchestration (`run_evaluation`) |

Generations are cached per prompt ID (with resume support) so interrupted runs restart instantly.

---

## Orchestration (Phases 1–4)

```bash
# Phase 1: fixed β sweep
python scripts/orchestrate.py phase1

# Phase 2: adaptive vs annealed vs fixed
python scripts/orchestrate.py phase2 --oracle-beta 0.2

# Phase 3: ablation suite
python scripts/orchestrate.py phase3 --ablations full no_deadband no_ema no_clipping

# Phase 4: generalisation matrix (datasets + models)
python scripts/orchestrate.py phase4 \
  --eval-config configs/eval/judge_gpt4o_mini.yaml \
  --dataset uf=data/dev.jsonl \
  --dataset hh=research/data/hh_eval.jsonl \
  --model adaptive=kind:lora,checkpoint:outputs/adaptive_beta/checkpoint-120
```

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

---

## License

MIT License. See `LICENSE`.

