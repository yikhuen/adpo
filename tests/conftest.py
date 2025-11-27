import os
import sys
import types
import warnings
from pathlib import Path


_GPU_MOCK_WARNING = (
    "Using mocked GPU dependencies in tests. Install the real packages to enable training/eval."
)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _install_gpu_dependency_mocks() -> None:
    """Provide lightweight stand-ins for GPU-only packages during CI tests."""
    if os.getenv("ADPO_DISABLE_GPU_MOCKS") == "1":
        return
    _mock_unsloth()
    _mock_trl()


def _mock_unsloth() -> None:
    if "unsloth" in sys.modules:
        return
    try:  # pragma: no cover - real import when available
        import unsloth  # type: ignore  # noqa: F401
    except Exception:
        module = types.ModuleType("unsloth")

        class _MockModel:
            device = "cpu"

            def eval(self):
                return self

            def generate(self, **kwargs):
                inputs = kwargs.get("input_ids") or []
                try:
                    batch_size = len(inputs)
                except TypeError:
                    batch_size = 1
                return [f"<mock-response-{i}>" for i in range(batch_size or 1)]

        class _MockBatch(dict):
            def __init__(self, batch_size: int):
                super().__init__(input_ids=[f"<mock-input-{i}>" for i in range(max(1, batch_size))])

            def to(self, *_args, **_kwargs):
                return self

        class _MockTokenizer:
            def __call__(self, prompts, **_kwargs):
                batch_size = len(prompts) if prompts is not None else 1
                return _MockBatch(batch_size)

            def batch_decode(self, outputs, **_kwargs):
                return list(outputs)

        class _MockFastLanguageModel:
            _MSG = (
                "FastLanguageModel is mocked during tests. Install unsloth with GPU support for real usage."
            )

            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                warnings.warn(_GPU_MOCK_WARNING, RuntimeWarning)
                return _MockModel(), _MockTokenizer()

            @staticmethod
            def get_peft_model(model, **_kwargs):
                return model

            @staticmethod
            def for_inference(model):
                return model

        module.FastLanguageModel = _MockFastLanguageModel  # type: ignore[attr-defined]
        sys.modules["unsloth"] = module


def _mock_trl() -> None:
    if "trl" in sys.modules:
        return
    module = types.ModuleType("trl")

    class _DummyDPOConfig:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class _DummyDPOTrainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    module.DPOConfig = _DummyDPOConfig  # type: ignore[attr-defined]
    module.DPOTrainer = _DummyDPOTrainer  # type: ignore[attr-defined]
    sys.modules["trl"] = module


_install_gpu_dependency_mocks()
_ensure_src_on_path()


