"""
Adaptive DPO package bootstrap.

We attempt to import `unsloth` as early as possible so that its performance
patches apply before any downstream module pulls in `transformers`. If the
dependency is missing (e.g., on a CPU-only environment), we fail silently and
allow the lazy import guards elsewhere to raise more helpful errors.
"""

from __future__ import annotations

__all__ = []

try:  # pragma: no cover - optional GPU dependency
    import unsloth  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - best effort
    # Unsloth is optional in many CI/test environments; skip if unavailable.
    pass
