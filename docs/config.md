# Config Reference

Adaptive DPO uses YAML configs for training, evaluation, and orchestration. This document highlights the required keys and useful overrides.

---

## Training (`configs/train/*.yaml`)

```yaml
project: adaptive-dpo
seed: 42
seeds: [42]

model:
  name: Qwen/Qwen2.5-7B-Instruct
  max_seq_length: 4096
  load_in_4bit: true

dataset:
  alias: ultrafeedback
  splits:
    train: train_prefs
    eval: test_prefs
  sample_frac: 0.005

beta_controller:
  kind: robust_hybrid             # or hybrid_entropy | pid
  target_kl: 0.04
  beta_init: 0.10
  beta_min: 0.05
  beta_max: 2.0
  eta: 0.01
  ema_alpha: 0.1
  deadband: 0.10
  lambda_entropy: 4.0
  entropy_warmup_steps: 10

trainer:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-6
  max_steps: 120
  logging_steps: 1
  output_dir: outputs/adaptive_beta
  report_to: wandb
```

Key fields:

- `model.*`: base checkpoint + precision settings.
- `dataset.alias`: mapped via `adaptive_dpo/data/formatters`.
- `beta_controller.kind`: `robust_hybrid`, `hybrid_entropy`, or `pid` (legacy).
- `seed` vs `seeds`: `seeds` overrides for multi-run sweeps.

---

## Evaluation (`configs/eval/*.yaml`)

```yaml
prompts:
  path: data/dev.jsonl
  limit: 200
  shuffle: false
  seed: 42

generation:
  batch_size: 8
  max_new_tokens: 1024
  cache: true

models:
  adaptive:
    kind: lora
    checkpoint: outputs/adaptive_beta/checkpoint-120
  base:
    kind: base

comparisons:
  - name: adaptive_vs_base
    a: adaptive
    b: base

judges:
  - name: primary
    provider: openai
    model: gpt-4o-mini
    temperature: 0.0
    max_tokens: 64
  - name: secondary
    provider: gemini
    model: gemini-2.0-flash-001

output:
  dir: research/results/eval

wandb:
  enabled: true
  project: adaptive-dpo
  name: phase2_eval
```

Highlights:

- `models.*`: `kind` ∈ {`lora`, `base`, `hf`}. HF models require `model: <hf_id>`.
- `comparisons`: `a`/`b` must reference names from `models`.
- `judges.provider`: `openai`, `gemini`, or `hf_causal`.
- `generation.cache`: disable to force regeneration per run.

---

## Orchestration CLI

The Typer commands accept structured options; examples:

```bash
# Phase 4 dataset + model specs
python scripts/orchestrate.py phase4 \
  --eval-config configs/eval/judge_gpt4o_mini.yaml \
  --dataset uf=data/dev.jsonl \
  --dataset hh=docs/data/hh_eval.jsonl \
  --model adaptive=kind:lora,checkpoint:outputs/adaptive_beta/checkpoint-120 \
  --model base=kind:base
```

- Dataset spec: `alias=path/to/prompts.jsonl`
- Model spec: `name=kind:lora,checkpoint:path` (additional `key:value` pairs allowed)

Validation happens before launching evaluations; missing models/datasets raise `typer.BadParameter`.

---

For additional legacy instructions (RunPod setup, tmux usage, SCP transfers) refer to [`docs/runpod.md`](docs/runpod.md).***

