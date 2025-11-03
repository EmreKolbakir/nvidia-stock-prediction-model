import numpy as np
import pytest

from src.evaluation.metrics import (
    directional_accuracy,
    mae,
    mape,
    regression_report,
    rmse,
    r_squared,
)


def test_regression_metrics_compute_expected_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    report = regression_report(y_true, y_pred)
    assert pytest.approx(report["rmse"], rel=1e-3) == rmse(y_true, y_pred)
    assert pytest.approx(report["mape"], rel=1e-3) == mape(y_true, y_pred)
    assert report["directional_accuracy"] == directional_accuracy(y_true, y_pred)
    assert pytest.approx(report["mae"], rel=1e-3) == mae(y_true, y_pred)
    assert pytest.approx(report["r2"], rel=1e-3) == r_squared(y_true, y_pred)
