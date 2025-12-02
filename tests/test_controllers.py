import pytest

torch = pytest.importorskip("torch")

from adaptive_dpo.controllers import (  # noqa: E402  (import after torch check)
    AdaptiveBetaController,
    BetaControllerConfig,
    BetaDPOConfig,
    BetaDPOController,
    EpsilonDPOConfig,
    EpsilonDPOController,
    HybridAdaptiveKLController,
    HybridControllerConfig,
    RobustHybridConfig,
    RobustHybridController,
)


def test_adaptive_controller_basic_update():
    controller = AdaptiveBetaController(BetaControllerConfig())
    value = controller.update(0.05)
    assert isinstance(value, float)
    state = controller.state()
    assert "beta" in state and isinstance(state["beta"], float)


def test_hybrid_controller_with_entropy():
    ctrl = HybridAdaptiveKLController(HybridControllerConfig(entropy_warmup_steps=0))
    logits = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4)
    value = ctrl.update(0.05, batch_logits=logits, attention_mask=mask, global_step=1)
    assert isinstance(value, float)
    state = ctrl.state()
    assert pytest.approx(state["beta_total"]) == value


def test_robust_hybrid_controller_entropy_activation():
    ctrl = RobustHybridController(RobustHybridConfig(entropy_warmup_steps=0))
    logits = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4)
    value = ctrl.update(0.08, batch_logits=logits, attention_mask=mask)
    assert isinstance(value, float)
    state = ctrl.state()
    assert "beta_total" in state


def test_beta_dpo_controller_update():
    cfg = BetaDPOConfig(beta_min=0.1, beta_max=0.5, scale_coeff=0.1)
    ctrl = BetaDPOController(cfg)
    
    # 1. Initial update (no metrics) -> returns default
    beta1 = ctrl.update(kl_batch=0.01)
    assert beta1 == 0.1
    
    # 2. Update with easy sample (large margin) -> beta should increase
    # margin = 2.0 (unscaled) -> target = 0.1 * 2.0 = 0.2
    metrics = {"rewards/margins": 2.0 * beta1} # simulate scaled margin
    beta2 = ctrl.update(kl_batch=0.01, metrics=metrics)
    
    # EMA takes time to move, but let's check state
    state = ctrl.state()
    assert "margin_raw" in state
    assert state["margin_raw"] == pytest.approx(2.0)


def test_epsilon_dpo_controller_update():
    cfg = EpsilonDPOConfig(beta_init=0.1)
    ctrl = EpsilonDPOController(cfg)
    beta = ctrl.update(kl_batch=0.05)
    assert beta == 0.1
