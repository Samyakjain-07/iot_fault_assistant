import random

import numpy as np
import pytest

from src.anomaly_detector import load_model, predict, save_model, train
from src.feature_engineering import build_feature_matrix
from src.mqtt_publisher import generate_reading


@pytest.fixture(scope="module")
def healthy_features():
    rng = random.Random(0)
    return build_feature_matrix(generate_reading("MOTOR-001", rng) for _ in range(800))


@pytest.fixture(scope="module")
def model(healthy_features):
    return train(healthy_features, contamination=0.05, random_state=0)


def _features(**overrides):
    reading = {
        "voltage": 400.0,
        "current": 12.0,
        "power_kw": 7.73,
        "power_factor": 0.93,
        "temperature": 52.0,
    }
    reading.update(overrides)
    return build_feature_matrix([reading])


def test_typical_healthy_reading_is_normal(model):
    labels, _ = predict(model, _features())

    assert labels.tolist() == [1]


def test_grossly_abnormal_reading_is_anomaly(model):
    extreme = _features(temperature=150.0, current=30.0, voltage=520.0)
    labels, _ = predict(model, extreme)

    assert labels.tolist() == [-1]


def test_anomalous_reading_scores_lower_than_healthy(model):
    _, healthy = predict(model, _features())
    _, faulty = predict(model, _features(temperature=150.0, current=30.0))

    assert faulty[0] < healthy[0]


def test_predict_returns_one_label_and_score_per_row(model, healthy_features):
    labels, scores = predict(model, healthy_features[:10])

    assert labels.shape == scores.shape == (10,)
    assert set(labels.tolist()) <= {-1, 1}


def test_contamination_controls_share_flagged_on_training_data(healthy_features):
    model = train(healthy_features, contamination=0.1, random_state=0)
    labels, _ = predict(model, healthy_features)

    assert (labels == -1).mean() == pytest.approx(0.1, abs=0.02)


def test_training_is_deterministic_for_a_seed(healthy_features):
    a = train(healthy_features, random_state=7)
    b = train(healthy_features, random_state=7)

    assert np.array_equal(
        predict(a, healthy_features)[1], predict(b, healthy_features)[1]
    )


def test_save_and_load_round_trip(model, healthy_features, tmp_path):
    path = save_model(model, tmp_path / "nested" / "model.joblib")

    assert path.is_file()
    restored = load_model(path)
    assert np.array_equal(
        predict(model, healthy_features)[1], predict(restored, healthy_features)[1]
    )


def test_load_missing_model_names_the_train_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        load_model(tmp_path / "missing.joblib")
