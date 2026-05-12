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


def get_all_metrics() -> list:
    return list(_METRIC_REGISTRY)
