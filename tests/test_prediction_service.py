import random

import pytest

from src.anomaly_detector import save_model, train
from src.feature_engineering import build_feature_matrix
from src.mqtt_publisher import generate_reading
from src.prediction_service import ANOMALY, NORMAL, Prediction, PredictionService


@pytest.fixture(scope="module")
def model():
    rng = random.Random(0)
    readings = [generate_reading("MOTOR-001", rng) for _ in range(800)]
    return train(build_feature_matrix(readings), contamination=0.05, random_state=0)


@pytest.fixture
def service(model):
    return PredictionService(model)


HEALTHY = {
    "device_id": "MOTOR-001",
    "timestamp": "2026-01-01T00:00:00Z",
    "voltage": 400.0,
    "current": 12.0,
    "power_kw": 7.73,
    "power_factor": 0.93,
    "temperature": 52.0,
}
FAULTY = {**HEALTHY, "temperature": 150.0, "current": 30.0, "voltage": 520.0}


def test_healthy_reading_is_normal(service):
    prediction = service.predict(HEALTHY)

    assert prediction.label == NORMAL
    assert not prediction.is_anomaly


def test_faulty_reading_is_anomaly_with_lower_score(service):
    prediction = service.predict(FAULTY)

    assert prediction.label == ANOMALY
    assert prediction.is_anomaly
    assert prediction.score < service.predict(HEALTHY).score


def test_score_is_a_plain_float(service):
    assert type(service.predict(HEALTHY).score) is float


def test_predict_many_preserves_order(service):
    predictions = service.predict_many([FAULTY, HEALTHY, FAULTY])

    assert [p.label for p in predictions] == [ANOMALY, NORMAL, ANOMALY]


def test_predict_many_empty_returns_empty_list(service):
    assert service.predict_many([]) == []


def test_predict_rejects_reading_missing_a_feature(service):
    incomplete = {k: v for k, v in HEALTHY.items() if k != "voltage"}

    with pytest.raises(ValueError, match="voltage"):
        service.predict(incomplete)


def test_load_reads_a_saved_model(model, tmp_path):
    path = save_model(model, tmp_path / "model.joblib")

    assert PredictionService.load(path).predict(FAULTY).is_anomaly


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PredictionService.load(tmp_path / "missing.joblib")


def test_prediction_is_immutable():
    with pytest.raises(AttributeError):
        Prediction(NORMAL, 0.0).label = ANOMALY  # type: ignore[misc]
