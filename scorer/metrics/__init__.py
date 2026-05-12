from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    name: str
    global_score: float        # 0-100
    heatmap: np.ndarray        # (H, W) float32, pixel-level quality
    region_scores: dict[str, float]
    diagnosis: str
    metadata: dict


# Registry of all metric functions, filled by imports below
_METRIC_REGISTRY: list = []


def register(func):
    _METRIC_REGISTRY.append(func)
    return func


# Import all metric modules to populate the registry
from . import exposure  # noqa: E402
from . import brightness  # noqa: E402
from . import contrast  # noqa: E402
from . import color_accuracy  # noqa: E402
from . import white_balance  # noqa: E402
from . import sharpness  # noqa: E402
from . import noise  # noqa: E402
from . import dynamic_range  # noqa: E402
from . import texture  # noqa: E402
from . import uniformity  # noqa: E402
from . import fringing  # noqa: E402


def get_all_metrics() -> list:
    return list(_METRIC_REGISTRY)
