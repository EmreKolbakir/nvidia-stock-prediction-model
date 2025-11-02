"""Evaluation metrics for regression-based stock prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error with zero protection."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def directional_accuracy(y_true, y_pred) -> float:
    """Share of predictions matching the sign of the true value."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return float("nan")
    sign_true = np.sign(y_true[mask])
    sign_pred = np.sign(y_pred[mask])
    return float(np.mean(sign_true == sign_pred))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred) -> float:
    """Coefficient of determination (R²)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    mean_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_true) ** 2)
    if np.isclose(ss_tot, 0):
        return float("nan")
    return float(1 - ss_res / ss_tot)


def regression_report(y_true, y_pred) -> Dict[str, float]:
    """Return a metrics dictionary."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
