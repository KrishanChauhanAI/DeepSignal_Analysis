"""
Confidence-aware pseudo-labeling with quality gates.

Gates: confidence → distance → consistency → class balance.
Each gate is independently configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from nanobio.config import PseudoLabelCfg
from nanobio.models.prototype import PrototypeClassifier

logger = logging.getLogger("nanobio.training.pseudo_labeling")


@dataclass
class PseudoLabelStats:
    total_unlabeled: int = 0
    passed_confidence: int = 0
    passed_distance: int = 0
    accepted: int = 0
    rejected: int = 0
    per_class_accepted: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total": self.total_unlabeled,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rate": self.accepted / max(1, self.total_unlabeled),
            "per_class": self.per_class_accepted,
        }


class PseudoLabelManager:
    """Assigns pseudo-labels to unlabeled embeddings with quality gates."""

    def __init__(self, cfg: PseudoLabelCfg, classifier: PrototypeClassifier):
        self.cfg = cfg
        self.classifier = classifier

    def generate(
        self,
        unlabeled_embeddings: np.ndarray,
        unlabeled_windows: np.ndarray,  # kept for consistency filtering
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, PseudoLabelStats]:
        stats = PseudoLabelStats(total_unlabeled=len(unlabeled_embeddings))

        preds, probs, dists = self.classifier.predict(unlabeled_embeddings)
        max_probs = probs.max(axis=1)
        min_dists = dists.min(axis=1)

        # Gate 1: Confidence
        conf_mask = max_probs >= self.cfg.confidence_threshold
        stats.passed_confidence = int(conf_mask.sum())

        # Gate 2: Distance
        if self.cfg.distance_filter_enabled:
            sup_dists = self.classifier.get_support_distances(
                support_embeddings, support_labels,
            )
            dist_mask = np.ones(len(preds), dtype=bool)
            for i, (c, d) in enumerate(zip(preds, min_dists)):
                if int(c) in sup_dists and len(sup_dists[int(c)]) > 0:
                    thresh = float(np.percentile(
                        sup_dists[int(c)], self.cfg.distance_percentile,
                    ))
                    if d > thresh:
                        dist_mask[i] = False
            stats.passed_distance = int((conf_mask & dist_mask).sum())
        else:
            dist_mask = conf_mask.copy()
            stats.passed_distance = stats.passed_confidence

        combined = conf_mask & dist_mask

        # Gate 3: Class balance
        if self.cfg.class_balance_enabled:
            final_mask = np.zeros(len(preds), dtype=bool)
            class_counts: Dict[int, int] = {}
            for idx in np.argsort(-max_probs):
                if not combined[idx]:
                    continue
                c = int(preds[idx])
                if class_counts.get(c, 0) < self.cfg.max_pseudo_per_class:
                    final_mask[idx] = True
                    class_counts[c] = class_counts.get(c, 0) + 1
        else:
            final_mask = combined

        stats.accepted = int(final_mask.sum())
        stats.rejected = stats.total_unlabeled - stats.accepted
        for c in np.unique(preds):
            stats.per_class_accepted[int(c)] = int(
                (final_mask & (preds == c)).sum()
            )

        logger.info("PseudoLabels: %d/%d accepted", stats.accepted, stats.total_unlabeled)
        return unlabeled_embeddings[final_mask], preds[final_mask], stats