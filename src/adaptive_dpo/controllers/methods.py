from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .beta_dpo import BetaDPOConfig
from .epsilon_dpo import EpsilonDPOConfig
from .ipo import IPOMethodConfig
from .kto import KTOAlgorithmConfig
from .simpo import SimPOConfig


@dataclass
class MethodSpec:
    """Normalized representation of a training method selection."""

    name: str
    label: str
    trainer_kind: str = "dpo"
    loss_type_arg: str = "sigmoid"
    loss_override: Optional[str] = None
    reference_free: bool = False
    simpo_gamma: float = 0.5
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    beta_dpo_config: Optional[BetaDPOConfig] = None
    epsilon_dpo_config: Optional[EpsilonDPOConfig] = None


def _infer_default_method(cfg: Dict[str, Any]) -> str:
    if cfg.get("beta_controller"):
        return "adaptive"
    if cfg.get("beta_schedule"):
        return "annealed"
    return "fixed"


def resolve_method_config(cfg: Dict[str, Any]) -> MethodSpec:
    """Resolve the requested training method and its hyperparameters."""

    method_block = cfg.get("method") or {}
    params = method_block.get("params") or {}
    name = (method_block.get("name") or _infer_default_method(cfg)).lower()
    label = method_block.get("label") or name

    if name == "adaptive":
        return MethodSpec(name="adaptive", label=label, trainer_kind="dpo")
    if name == "fixed":
        return MethodSpec(name="fixed", label=label, trainer_kind="dpo")
    if name == "annealed":
        return MethodSpec(name="annealed", label=label, trainer_kind="dpo")
    if name == "ipo":
        IPOMethodConfig(**params)
        return MethodSpec(
            name="ipo",
            label=label,
            trainer_kind="dpo",
            loss_type_arg="ipo",
        )
    if name == "simpo":
        simpo_cfg = SimPOConfig(**params)
        return MethodSpec(
            name="simpo",
            label=label,
            trainer_kind="dpo",
            loss_type_arg="sigmoid",
            loss_override="simpo",
            reference_free=True,
            simpo_gamma=float(simpo_cfg.gamma),
        )
    if name == "kto":
        kto_cfg = KTOAlgorithmConfig(**params)
        return MethodSpec(
            name="kto",
            label=label,
            trainer_kind="kto",
            desirable_weight=float(kto_cfg.desirable_weight),
            undesirable_weight=float(kto_cfg.undesirable_weight),
        )
    if name == "beta_dpo":
        beta_cfg = BetaDPOConfig(**params)
        return MethodSpec(
            name="beta_dpo",
            label=label,
            trainer_kind="dpo",
            loss_type_arg="sigmoid",
            beta_dpo_config=beta_cfg,
        )
    if name == "epsilon_dpo":
        eps_cfg = EpsilonDPOConfig(**params)
        return MethodSpec(
            name="epsilon_dpo",
            label=label,
            trainer_kind="dpo",
            loss_type_arg="sigmoid",
            epsilon_dpo_config=eps_cfg,
        )

    raise ValueError(f"Unsupported training method '{name}'.")
