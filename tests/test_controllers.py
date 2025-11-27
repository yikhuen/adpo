import pytest

torch = pytest.importorskip("torch")

from adaptive_dpo.controllers import (  # noqa: E402  (import after torch check)
    AdaptiveBetaController,
    BetaControllerConfig,
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

