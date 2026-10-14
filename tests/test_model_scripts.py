import pytest

from scripts import evaluate_model, train_model
from src.anomaly_detector import train
from src.feature_engineering import build_feature_matrix
from src.mqtt_publisher import FAULT_SCENARIOS
from src.prediction_service import PredictionService

# --- compute_metrics ------------------------------------------------------


def test_compute_metrics_known_values():
    actual = [True, True, True, True, False, False, False, False]
    predicted = [True, True, True, False, True, False, False, False]

    metrics = evaluate_model.compute_metrics(actual, predicted)

    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (3, 1, 1, 3)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.75)


def test_compute_metrics_perfect_and_worst_cases():
    perfect = evaluate_model.compute_metrics([True, False], [True, False])
    nothing_flagged = evaluate_model.compute_metrics([True, True], [False, False])

    assert (perfect["precision"], perfect["recall"], perfect["f1"]) == (1.0, 1.0, 1.0)
    assert (nothing_flagged["precision"], nothing_flagged["recall"]) == (0.0, 0.0)
    assert nothing_flagged["f1"] == 0.0


def test_compute_metrics_all_healthy_has_no_division_error():
    metrics = evaluate_model.compute_metrics([False, False], [False, False])

    assert metrics["f1"] == 0.0
    assert metrics["tn"] == 2


# --- eval set / training data ---------------------------------------------


def test_eval_set_composition_and_labels():
    cases = evaluate_model.build_eval_set(healthy=20, per_fault=4, seed=1)

    assert len(cases) == 20 + 4 * len(FAULT_SCENARIOS)
    assert sum(fault is None for _, fault in cases) == 20
    for fault in FAULT_SCENARIOS:
        assert sum(f == fault for _, f in cases) == 4


def test_eval_set_is_reproducible_and_differs_by_seed():
    first = evaluate_model.build_eval_set(5, 2, seed=1)
    again = evaluate_model.build_eval_set(5, 2, seed=1)
    other = evaluate_model.build_eval_set(5, 2, seed=2)

    strip = lambda cases: [(r["voltage"], r["current"], f) for r, f in cases]  # noqa: E731
    assert strip(first) == strip(again)
    assert strip(first) != strip(other)


def test_training_readings_are_healthy_and_seeded():
    readings = train_model.generate_training_readings(50, seed=3)

    again = train_model.generate_training_readings(50, seed=3)
    sensors = ("voltage", "current", "power_kw", "power_factor", "temperature")
    assert len(readings) == 50
    # Timestamps use the wall clock, so compare the sensor values only.
    assert [[r[k] for k in sensors] for r in readings] == [
        [r[k] for k in sensors] for r in again
    ]
    # Healthy only: no reading reaches any fault's range.
    assert all(380 < r["voltage"] < 420 and r["temperature"] < 65 for r in readings)


# --- end to end: train on healthy data, evaluate on held-out faults -------


def test_trained_model_detects_most_injected_faults_with_few_false_alarms():
    train_readings = train_model.generate_training_readings(1500, seed=42)
    service = PredictionService(
        train(build_feature_matrix(train_readings), contamination=0.05, random_state=42)
    )

    metrics, per_fault = evaluate_model.evaluate(
        service, healthy=400, per_fault=40, seed=2024
    )

    assert metrics["recall"] > 0.7
    assert metrics["precision"] > 0.6
    assert metrics["fp"] / 400 < 0.15
    assert set(per_fault) == set(FAULT_SCENARIOS)
