from __future__ import annotations

from typing import Iterable, Optional

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


def progress(iterable: Iterable, *, total: Optional[int] = None, desc: Optional[str] = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)

