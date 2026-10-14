"""Turn telemetry readings into the feature vectors the detector consumes.

The feature set is the five sensor fields from ``docs/05_DATA_SCHEMA.md``, in a
fixed order. Isolation Forest is tree-based and scale-invariant, so values are
used as-is (no normalisation).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from src.database import TELEMETRY_FIELDS

FEATURE_NAMES: tuple[str, ...] = TELEMETRY_FIELDS


def extract_features(reading: Mapping[str, Any]) -> list[float]:
    """Return the 5-feature vector for one telemetry reading.

    Order follows ``FEATURE_NAMES``. Raises ``ValueError`` if a feature is
    missing or not a finite number (the model cannot score NaN/inf).
    """
    missing = [name for name in FEATURE_NAMES if name not in reading]
    if missing:
        raise ValueError(f"missing features: {', '.join(missing)}")
    try:
        vector = [float(reading[name]) for name in FEATURE_NAMES]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"features must be numeric: {exc}") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("features must be finite numbers")
    return vector


def build_feature_matrix(readings: Iterable[Mapping[str, Any]]) -> np.ndarray:
    """Stack readings into an ``(n_readings, 5)`` float array."""
    rows = [extract_features(reading) for reading in readings]
    return np.array(rows, dtype=float).reshape(-1, len(FEATURE_NAMES))
