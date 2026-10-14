import numpy as np
import pytest

from src.feature_engineering import (
    FEATURE_NAMES,
    build_feature_matrix,
    extract_features,
)


def _reading(**overrides):
    reading = {
        "device_id": "MOTOR-001",
        "timestamp": "2026-01-01T00:00:00Z",
        "voltage": 400.0,
        "current": 12.0,
        "power_kw": 7.7,
        "power_factor": 0.93,
        "temperature": 52.0,
    }
    reading.update(overrides)
    return reading


def test_feature_names_are_the_five_schema_features():
    assert FEATURE_NAMES == (
        "voltage",
        "current",
        "power_kw",
        "power_factor",
        "temperature",
    )


def test_extract_features_returns_five_values_in_fixed_order():
    assert extract_features(_reading()) == [400.0, 12.0, 7.7, 0.93, 52.0]


def test_extract_features_ignores_non_feature_keys():
    assert len(extract_features(_reading(id=5, note="x"))) == 5


def test_extract_features_coerces_ints_to_float():
    vector = extract_features(_reading(voltage=400))

    assert vector[0] == 400.0
    assert isinstance(vector[0], float)


def test_missing_feature_rejected():
    reading = _reading()
    del reading["temperature"]

    with pytest.raises(ValueError, match="missing features: temperature"):
        extract_features(reading)


@pytest.mark.parametrize("value", ["abc", None, [1]])
def test_non_numeric_feature_rejected(value):
    with pytest.raises(ValueError, match="numeric"):
        extract_features(_reading(current=value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_feature_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        extract_features(_reading(voltage=value))


def test_build_feature_matrix_shape_and_rows():
    matrix = build_feature_matrix([_reading(), _reading(voltage=410.0)])

    assert matrix.shape == (2, 5)
    assert matrix[1, 0] == 410.0
    assert matrix.dtype == float


def test_build_feature_matrix_empty_input_keeps_five_columns():
    assert build_feature_matrix([]).shape == (0, 5)


def test_build_feature_matrix_accepts_generator():
    matrix = build_feature_matrix(_reading(voltage=400 + i) for i in range(3))

    assert np.array_equal(matrix[:, 0], [400.0, 401.0, 402.0])
