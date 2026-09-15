"""
Clustering diagnostics on learned embeddings (exploratory only).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

logger = logging.getLogger("nanobio.utils.clustering")


def run_clustering(
    embeddings: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    n_clusters: int = 2,
    seed: int = 42,
) -> Dict[str, Any]:
    """KMeans clustering with optional evaluation against true labels."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    result = {
        "n_clusters": n_clusters,
        "inertia": float(km.inertia_),
    }

    if true_labels is not None:
        valid = labels >= 0
        if valid.sum() > 1:
            result["ari"] = float(adjusted_rand_score(
                true_labels[valid], labels[valid],
            ))
        try:
            if len(np.unique(labels)) >= 2:
                result["silhouette"] = float(silhouette_score(
                    embeddings, labels,
                ))
        except Exception:
            pass

    logger.info("Clustering: %d clusters, inertia=%.2f", n_clusters, km.inertia_)
    return result