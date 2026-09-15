"""
Standalone prototype-based classifier for few-shot and research use.

Decoupled from any encoder: operates on precomputed embeddings.
Supports Euclidean and cosine distance with calibrated probabilities.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger("nanobio.models.prototype")


class PrototypeClassifier:
    """Nearest-prototype classifier in embedding space."""

    def __init__(self, distance_metric: str = "euclidean",
                 normalize: bool = True, temperature: float = 1.0):
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.temperature = temperature
        self.prototypes: Optional[np.ndarray] = None
        self.class_labels: Optional[np.ndarray] = None

    def _norm(self, x: np.ndarray) -> np.ndarray:
        if self.normalize:
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        embeddings = self._norm(embeddings)
        self.class_labels = np.unique(labels)
        self.prototypes = np.stack([
            embeddings[labels == c].mean(axis=0) for c in self.class_labels
        ])

    def predict(self, embeddings: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (predictions, probabilities, distances)."""
        if self.prototypes is None:
            raise RuntimeError("Not fitted.")
        embeddings = self._norm(embeddings)

        if self.distance_metric == "cosine":
            pn = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8)
            distances = 1.0 - embeddings @ pn.T
        else:
            distances = cdist(embeddings, self.prototypes, metric="euclidean")

        logits = -distances / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        preds = self.class_labels[distances.argmin(axis=1)]
        return preds, probs, distances

    def predict_with_rejection(self, embeddings: np.ndarray, threshold: float
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        preds, probs, dists = self.predict(embeddings)
        is_known = dists.min(axis=1) <= threshold
        preds_unk = preds.copy()
        preds_unk[~is_known] = -1
        return preds_unk, probs, dists, is_known

    def get_support_distances(self, embeddings: np.ndarray,
                               labels: np.ndarray) -> Dict[int, np.ndarray]:
        embeddings = self._norm(embeddings)
        out = {}
        for i, c in enumerate(self.class_labels):
            mask = labels == c
            if mask.sum() == 0:
                continue
            d = cdist(embeddings[mask], self.prototypes[i:i+1],
                       metric="euclidean").ravel()
            out[int(c)] = d
        return out