"""Isolation Forest training, persistence and raw prediction.

This module speaks scikit-learn's vocabulary (``-1`` = anomaly, ``1`` =
normal). The application-level ``NORMAL``/``ANOMALY`` strings live in
``src/prediction_service.py``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import get_config, resolve_path


def default_model_path() -> Path:
    """Where the model is serialized, from ``ml.model_path`` in the config."""
    return resolve_path(get_config().data["ml"]["model_path"])


def train(
    features: np.ndarray,
    contamination: float | None = None,
    random_state: int | None = None,
    n_estimators: int = 100,
) -> IsolationForest:
    """Fit an Isolation Forest on an ``(n, 5)`` feature matrix.

    ``contamination`` and ``random_state`` default to the ``ml`` section of
    ``config/project.yaml``.
    """
    ml = get_config().data["ml"]
    if contamination is None:
        contamination = ml["contamination"]
    if random_state is None:
        random_state = ml["random_state"]
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    return model.fit(features)


def predict(
    model: IsolationForest, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score a feature matrix; returns ``(labels, scores)``.

    ``labels`` are sklearn's raw ``-1`` (anomaly) / ``1`` (normal). ``scores``
    are the raw ``score_samples`` values: lower means more anomalous.
    """
    return model.predict(features), model.score_samples(features)


def save_model(model: IsolationForest, path: str | Path | None = None) -> Path:
    """Serialize the model with joblib, creating parent dirs; returns the path."""
    target = resolve_path(path) if path else default_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: str | Path | None = None) -> IsolationForest:
    """Load a model saved by :func:`save_model`.

    Raises ``FileNotFoundError`` (naming the train script) if it doesn't exist.
    """
    source = resolve_path(path) if path else default_model_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"No trained model at {source}; run `python scripts/train_model.py`"
        )
    return joblib.load(source)
