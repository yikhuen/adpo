from .anthropic_hh import DEFAULT_PATH as ANTHROPIC_PATH, format_example as format_anthropic_hh
from .helpsteer2 import DEFAULT_PATH as HELPSTEER2_PATH, format_example as format_helpsteer2
from .sycophancy import DEFAULT_PATH as SYCOPHANCY_PATH, format_example as format_sycophancy
from .ultrafeedback import DEFAULT_PATH as ULTRAFEEDBACK_PATH, format_example as format_ultrafeedback

FORMATTERS = {
    "ultrafeedback": {
        "default_path": ULTRAFEEDBACK_PATH,
        "formatter": format_ultrafeedback,
    },
    "anthropic_hh": {
        "default_path": ANTHROPIC_PATH,
        "formatter": format_anthropic_hh,
    },
    "sycophancy": {
        "default_path": SYCOPHANCY_PATH,
        "formatter": format_sycophancy,
    },
    "helpsteer2": {
        "default_path": HELPSTEER2_PATH,
        "formatter": format_helpsteer2,
    },
}

__all__ = ["FORMATTERS"]

