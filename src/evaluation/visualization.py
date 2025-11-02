"""Visualization helpers for evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go


def plot_actual_vs_predicted(
    dates: Iterable,
    actual: Iterable,
    predicted: Iterable,
    title: str = "Actual vs Predicted",
    y_axis_label: str = "Value",
) -> go.Figure:
    """Build a Plotly line chart comparing actual and predicted values."""
    frame = pd.DataFrame(
        {"date": pd.to_datetime(list(dates)), "actual": actual, "predicted": predicted}
    ).sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["actual"],
            name="Actual",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["predicted"],
            name="Predicted",
            mode="lines",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_label,
        template="plotly_white",
    )
    return fig


def save_figure_html(fig: go.Figure, output_path: Path) -> Path:
    """Persist a Plotly figure as self-contained HTML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def save_figure_png(fig: go.Figure, output_path: Path, scale: float = 2.0) -> Path:
    """Persist a Plotly figure as a static PNG (requires kaleido)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(output_path), format="png", scale=scale)
    except ValueError:
        # Kaleido not installed or unavailable; skip PNG export silently.
        return output_path
    return output_path
