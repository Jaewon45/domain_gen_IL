"""Canonical CMNIST protocol definitions and post-training utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

CONDITION_ALIASES = {
    "balanced": "balanced",
    "balanced_visible": "balanced",
    "long_tail": "long_tail",
    "long_tail_visible": "long_tail",
    "scarce_tail": "scarce_tail",
    "near_missing_tail": "scarce_tail",
    "missing_tail": "missing_tail",
}

ANCHORS = (0.1, 0.2, 0.5, 0.9)
EVAL_GRID = tuple(round(0.1 * index, 1) for index in range(11))
LAMBDA_EVAL = 0.9
RESULTS_ROOT = "/home/ra95tig/results_final"
SOURCE_COUNTS = {
    "balanced": (2000, 2000, 2000, 2000),
    "long_tail": (5000, 2000, 800, 200),
    "scarce_tail": (5800, 1800, 350, 50),
    "missing_tail": (6000, 1500, 500, 0),
}


def canonical_condition(value: str) -> str:
    try:
        return CONDITION_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown CMNIST support condition: {value}") from exc


def source_counts(condition: str):
    return SOURCE_COUNTS[canonical_condition(condition)]


def deployment_epsilon(condition: str) -> float:
    counts = source_counts(condition)
    return sum(0.25 for count in counts if count == 0)


def discrete_cvar(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    """Compute exact upper discrete CVaR with fractional boundary mass."""
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and equally sized")
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    numeric_weights = [float(weight) for weight in weights]
    if any(weight < 0.0 for weight in numeric_weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(numeric_weights) - 1.0) > 1e-8:
        raise ValueError("weights must sum to one")
    tail_mass = 1.0 - float(alpha)
    used = 0.0
    total = 0.0
    for value, weight in sorted(zip(values, numeric_weights), key=lambda item: float(item[0]), reverse=True):
        take = min(weight, tail_mass - used)
        total += float(value) * take
        used += take
        if used >= tail_mass - 1e-12:
            break
    return total / tail_mass
