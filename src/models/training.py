"""Model training utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


def select_model(model_cfg: Dict):
    """Instantiate a model from configuration."""
    model_type = model_cfg.get("type", "random_forest").lower()
    params = model_cfg.get("params", {})
    if model_type == "random_forest":
        model = RandomForestRegressor(**params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(**params)
    elif model_type == "ridge":
        model = Ridge(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, model_type


def train_model(model_cfg: Dict, features: np.ndarray, target: np.ndarray) -> Tuple[object, Dict[str, object]]:
    """Train the selected model and return the fitted estimator and metadata."""
    model, model_type = select_model(model_cfg)
    metadata: Dict[str, object] = {"model_type": model_type}

    search_cfg: Optional[Dict[str, object]] = model_cfg.get("search")
    if search_cfg and search_cfg.get("type") == "grid":
        param_grid = search_cfg.get("param_grid", {})
        if not param_grid:
            raise ValueError("Grid search requires a non-empty 'param_grid'.")
        cv = int(search_cfg.get("cv", 3))
        n_jobs = search_cfg.get("n_jobs")
        n_samples = features.shape[0]
        if n_samples >= cv and n_samples > 0:
            gs = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs)
            gs.fit(features, target)
            metadata.update(
                {
                    "best_params": gs.best_params_,
                    "best_score": float(gs.best_score_),
                    "search_type": "grid",
                }
            )
            return gs.best_estimator_, metadata
        metadata.update(
            {
                "search_type": "grid",
                "search_skipped": f"n_samples={n_samples} < cv={cv}",
            }
        )

    model.fit(features, target)
    metadata.update(
        {
            "best_params": model.get_params(),
            "best_score": None,
            "search_type": None,
        }
    )
    return model, metadata
