"""Application-level anomaly prediction: telemetry reading -> NORMAL/ANOMALY."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.ensemble import IsolationForest

from src.anomaly_detector import load_model, predict
from src.feature_engineering import build_feature_matrix

NORMAL = "NORMAL"
ANOMALY = "ANOMALY"


@dataclass(frozen=True)
class Prediction:
    """Result for one reading: the label and the raw Isolation Forest score."""

    label: str
    score: float

    @property
    def is_anomaly(self) -> bool:
        return self.label == ANOMALY


class PredictionService:
    """Wraps a trained model behind the strings the rest of the app uses."""

    def __init__(self, model: IsolationForest) -> None:
        self._model = model

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> PredictionService:
        """Build a service from the serialized model (config path by default)."""
        return cls(load_model(model_path))

    def predict_many(self, readings: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        """Classify several readings at once (one model call)."""
        if not readings:
            return []
        labels, scores = predict(self._model, build_feature_matrix(readings))
        return [
            Prediction(ANOMALY if label == -1 else NORMAL, float(score))
            for label, score in zip(labels, scores, strict=True)
        ]

    def predict(self, reading: Mapping[str, Any]) -> Prediction:
        """Classify one telemetry reading."""
        return self.predict_many([reading])[0]
