# Adaptive Beta for DPO: A Practical Controller for Stable Preference Optimization

## 1. Problem Statement
Direct Preference Optimization (DPO) is a strong baseline for aligning LLMs using pairwise human preferences. However, practitioners often fix the inverse‑temperature parameter β, which implicitly controls the KL between the policy and a reference. Fixed β can be unstable across datasets/models and tends to either under‑regularize (quality drift, verbosity) or over‑regularize (underfitting). We target a practical question: can adaptive β control, driven by a per‑token KL target, consistently outperform fixed β while stabilizing training?

## 2. Prior Work and Related Literature (brief)
- DPO and variants: Rafailov et al., 2023; follow‑ups (ORPO, IPO, SimPO) explore alternative objectives and reference handling.
- KL targeting in RL: PPO/TRPO use adaptive KL penalties or trust regions; KL controllers are widely used to stabilize policy updates.
- Controller ideas: EMA smoothing, deadbands, and clipping are standard control heuristics in optimization and RL to avoid oscillations and runaway updates.
- Practical alignment reports: open‑source repos and blogs often note sensitivity to β and advocate manual tuning; systematic adaptive control for DPO is less documented compared to PPO.

## 3. Our Contribution (PoC)
- A simple, compute‑efficient adaptive β controller for DPO:
  - Per‑token KL proxy against a frozen reference on prompts.
  - β updated by an exponential controller toward a target KL with EMA smoothing, deadband, and clipping.
  - Implemented with Unsloth QLoRA on 7B models for low VRAM.
- Open, reproducible code and a minimal evaluation harness (LLM‑as‑judge + sanity metrics).

## 4. Method
### 4.1 Adaptive β Controller
- Target: per‑token KL setpoint (e.g., 0.03 nats/token).
- Signal: batch KL proxy E[log p_pol(x) − log p_ref(x)] on prompt tokens.
- Smoothing: KL_ema ← (1 − α) KL_ema + α KL_batch.
- Update: β ← clip(β · exp(η( KL_ema / KL_target − 1 )), β_min, β_max).
- Deadband: no update if KL_ema ∈ [0.8 KL_target, 1.2 KL_target].

### 4.2 Training Setup
- Model: Qwen/Qwen2.5‑7B‑Instruct, Unsloth QLoRA 4‑bit.
- Data: `HuggingFaceH4/ultrafeedback_binarized`, small subsample for PoC.
- Trainer: TRL DPO with our controller; seeds fixed for repeatability.
- Logging: β, KL_ema, losses, sequence lengths (W&B).

### 4.3 Evaluation
- Judge: gpt‑4o‑mini, pairwise comparisons with a fixed rubric.
- Systems compared: (i) Adaptive‑β DPO, (ii) Fixed‑β DPO, (iii) Base SFT.
- Metrics: judge win‑rate with 95% Wilson CIs; sequence length; refusal rate.

## 5. Experiments (current PoC)
- Hardware: Runpod A40 48GB; training with QLoRA; evaluation with API judge.
- Decoding: deterministic (temp=0), max_new_tokens=256–512.
- Dev set: 50 prompts (held‑out, not in training sample).
- Results (n=50):
  - Adaptive vs Base: 72% win (95% CI: 58%–83%) — clear gain.
  - Adaptive vs Fixed‑β: 60% win (95% CI: 46%–72%) — positive, but inconclusive.

## 6. Interpretation
- The PoC indicates adaptive control can improve over the base model and may outperform a single fixed‑β choice.
- With 50 prompts the uncertainty vs fixed‑β remains large; more data needed to claim a consistent win.
- KL control behaved stably (qualitative); we will quantify stability in follow‑ups.

## 7. Limitations
- Small evaluation set (n=50) and a single judge; risk of judge/model bias.
- One model family and one dataset subsample; limited generality.
- Per‑token KL proxy is approximate; need to measure true KL proxies over prompts/responses.

## 8. Planned Extensions (toward full poster)
1. Stronger evaluation
   - n=100–200 prompts; two judges (API + open model) and two seeds.
   - Baselines: fixed‑β grid (e.g., 0.05, 0.10, 0.20); best static β vs adaptive.
   - Report: win‑rate + 95% CIs; response length and refusal rate; KL and β trajectories.
2. Ablations
   - Remove EMA, deadband, clipping individually; study stability and performance.
   - Vary KL targets (per‑token and per‑sequence) and controller gains (η, α).
3. Generalization
   - Repeat on Llama‑3‑8B/Mistral‑7B and an additional preference dataset (e.g., HH‑RLHF or HelpSteer2).
4. Practicality
   - Throughput/latency tradeoffs, cost analysis (GPU + API), and guidance for default settings.
5. Potential novelty extensions
   - Adaptive target scheduling (setpoint evolves with length/difficulty).
   - Difficulty‑aware β scaling from reward/entropy proxies.
   - Early warning detector for β/ KL runaway with automatic dampening.

## 9. Experimental Protocol (for the poster)
- Training: identical configs across runs; log seeds; fixed decoding.
- Evaluation: same prompts across systems; randomize pair order; blind judge rubric.
- Statistics: report mean ± 95% CI; perform two‑sided binomial tests for win rates.

## 10. Expected Outcomes
- Demonstrate that adaptive β meets KL targets and improves instruction following versus base and typical fixed‑β settings at 7B scale.
- Provide a small set of practical defaults and a guide for controller tuning.

## 11. Timeline (2–3 weeks)
- Week 1: expand eval to 100–200; fixed‑β grid; logging of length/refusals/KL; ablations.
- Week 2: cross‑model or cross‑dataset; second judge; compile plots and examples.
- Week 3: write poster; artifact release (code + configs); final sanity pass.

## 12. Risks and Mitigations
- Judge bias → use two judges (API + open model), shuffle pair order.
- Variance → two seeds; report CIs, not point estimates.
- Compute cost → 7B QLoRA; batched eval; shorter max_new_tokens where safe.

## 13. References (selected)
- Rafailov et al., 2023. Direct Preference Optimization.
- Schulman et al., 2017. Proximal Policy Optimization (for KL penalties and targeting).
- Recent DPO variants: IPO/ORPO/SimPO and alignment handbooks for practical baselines.

---
Contact: repo README for exact run commands (Runpod setup, training, batched evaluation, result export).
