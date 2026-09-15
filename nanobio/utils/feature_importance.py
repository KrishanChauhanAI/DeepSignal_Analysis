"""
Feature importance extraction for tree-based classical models (Section 43).

Writes per-model feature importance CSVs so trees are not silent about
which features drove their predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger("nanobio.utils.feature_importance")


def extract_and_save_feature_importance(
    model: Any,
    model_name: str,
    feature_names: List[str],
    output_dir: Path,
    timestamp: str,
) -> Path:
    """
    Extract feature importance if the model supports it, then persist to CSV.

    Supports:
        - sklearn tree models (RandomForest, ExtraTrees, GradientBoosting)
        - XGBoost
        - LogisticRegression (via absolute coefficient magnitudes)

    Returns:
        Path to the written CSV, or empty Path if unsupported.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importance = None
    importance_type = None

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        importance_type = "gini_or_gain"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            # Multiclass: aggregate as mean absolute value across classes
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        importance_type = "abs_coefficient"
    else:
        logger.debug("Model %s does not expose feature importance", model_name)
        return Path("")

    # Handle length mismatch (SelectKBest may have reduced features)
    if len(importance) != len(feature_names):
        logger.warning(
            "Feature-importance length %d does not match feature_names %d for %s; "
            "using generic indices",
            len(importance), len(feature_names), model_name,
        )
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "importance_type": importance_type,
    }).sort_values("importance", ascending=False)

    out_path = output_dir / f"{timestamp}_feature_importance_{model_name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Feature importance for %s → %s (top-3: %s)",
                model_name, out_path,
                df.head(3)["feature"].tolist())
    return out_path