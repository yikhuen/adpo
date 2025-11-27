import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()


