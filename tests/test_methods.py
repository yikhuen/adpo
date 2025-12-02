import pytest

pytest.importorskip("torch")

from adaptive_dpo.controllers.methods import resolve_method_config


def test_method_defaults_to_adaptive_with_controller():
    cfg = {
        "model": {},
        "trainer": {},
        "dataset": {},
        "beta_controller": {"kind": "pid"},
    }
    spec = resolve_method_config(cfg)
    assert spec.name == "adaptive"
    assert spec.trainer_kind == "dpo"


def test_method_parses_simpo_params():
    cfg = {
        "model": {},
        "trainer": {},
        "dataset": {},
        "method": {"name": "simpo", "params": {"gamma": 0.75}},
    }
    spec = resolve_method_config(cfg)
    assert spec.name == "simpo"
    assert spec.reference_free is True
    assert spec.loss_override == "simpo"
    assert spec.simpo_gamma == 0.75


def test_method_parses_kto_weights():
    cfg = {
        "model": {},
        "trainer": {},
        "dataset": {},
        "method": {"name": "kto", "params": {"desirable_weight": 1.5, "undesirable_weight": 0.8}},
    }
    spec = resolve_method_config(cfg)
    assert spec.name == "kto"
    assert spec.trainer_kind == "kto"
    assert spec.desirable_weight == 1.5
    assert spec.undesirable_weight == 0.8


def test_method_parses_beta_dpo():
    cfg = {
        "model": {},
        "trainer": {},
        "dataset": {},
        "method": {"name": "beta_dpo", "params": {"scale_coeff": 0.2}},
    }
    spec = resolve_method_config(cfg)
    assert spec.name == "beta_dpo"
    assert spec.beta_dpo_config.scale_coeff == 0.2


def test_method_parses_epsilon_dpo():
    cfg = {
        "model": {},
        "trainer": {},
        "dataset": {},
        "method": {"name": "epsilon_dpo", "params": {"epsilon": 0.02}},
    }
    spec = resolve_method_config(cfg)
    assert spec.name == "epsilon_dpo"
    assert spec.epsilon_dpo_config.epsilon == 0.02
