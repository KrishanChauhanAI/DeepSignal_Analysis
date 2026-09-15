"""
Unified metric engine with applicability flags.

Closes Section 31 gaps: adds AUPRC, F1-micro, Specificity, Gini,
Hamming Loss, Top-k Accuracy alongside the existing metrics.

Backwards compatible: `compute_all()` still returns the same dict format
with existing keys unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize


class MetricEngine:
    """Classification metrics with applicability flags."""

    @staticmethod
    def _normalize_probabilities(
        y_prob: Optional[np.ndarray], num_classes: int, n_samples: int,
    ) -> Optional[np.ndarray]:
        """Force y_prob into shape (n_samples, num_classes)."""
        if y_prob is None:
            return None
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim == 1:
            if y_prob.shape[0] != n_samples:
                return None
            return np.column_stack([1.0 - y_prob, y_prob])
        if y_prob.ndim == 2:
            if y_prob.shape[0] != n_samples:
                return None
            if y_prob.shape[1] == 1:
                return np.column_stack([1.0 - y_prob[:, 0], y_prob[:, 0]])
            if y_prob.shape[1] == num_classes:
                return y_prob
        return None

    @staticmethod
    def _specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                                num_classes: int) -> float:
        """
        Macro-averaged specificity (true-negative rate).

        For each class c: TN / (TN + FP), then average across classes.
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        specs = []
        total = cm.sum()
        for c in range(num_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = total - tp - fn - fp
            denom = tn + fp
            specs.append(tn / denom if denom > 0 else 0.0)
        return float(np.mean(specs))

    @staticmethod
    def compute_all(
        y_true: np.ndarray, y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None, num_classes: int = 2,
    ) -> Dict[str, Any]:
        m = OrderedDict()

        def add(name, value, applicable=True, definition="", averaging=""):
            m[name] = {
                "value": value if applicable else "N/A",
                "applicable": applicable,
                "definition": definition,
                "averaging": averaging,
            }

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        n_samples = len(y_true)

        # ── Hard classification metrics ──
        add("accuracy", float(accuracy_score(y_true, y_pred)))
        add("balanced_accuracy", float(balanced_accuracy_score(y_true, y_pred)))
        add("precision_weighted",
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            averaging="weighted")
        add("recall_weighted",
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            averaging="weighted")
        add("f1_weighted",
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            averaging="weighted")
        add("f1_macro",
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            averaging="macro")
        # NEW: F1 micro
        add("f1_micro",
            float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            averaging="micro")
        add("mcc", float(matthews_corrcoef(y_true, y_pred)))
        add("cohen_kappa", float(cohen_kappa_score(y_true, y_pred)))
        # NEW: specificity (macro-averaged TNR)
        add("specificity_macro",
            MetricEngine._specificity_per_class(y_true, y_pred, num_classes),
            definition="Mean per-class true-negative rate",
            averaging="macro")
        # NEW: Hamming loss (fraction of wrong labels; = 1 - accuracy for single-label)
        add("hamming_loss", float(hamming_loss(y_true, y_pred)),
            definition="Fraction of labels wrong (single-label: 1 - accuracy)")

        # ── Probability-dependent metrics ──
        y_prob_norm = MetricEngine._normalize_probabilities(
            y_prob, num_classes, n_samples,
        )

        if y_prob_norm is not None and num_classes >= 2:
            # ROC-AUC
            try:
                if num_classes == 2:
                    auc_val = float(roc_auc_score(y_true, y_prob_norm[:, 1]))
                else:
                    auc_val = float(roc_auc_score(
                        y_true, y_prob_norm, multi_class="ovr", average="weighted",
                    ))
                add("roc_auc", auc_val, definition="Area under ROC curve")
                # NEW: Gini = 2*AUC - 1
                add("gini", 2.0 * auc_val - 1.0,
                    definition="Gini coefficient = 2·AUC − 1")
            except Exception as e:
                add("roc_auc", None, applicable=False,
                    definition=f"Not computable: {e}")
                add("gini", None, applicable=False)

            # Log loss
            try:
                add("log_loss",
                    float(log_loss(y_true, y_prob_norm,
                                    labels=list(range(num_classes)))),
                    definition="Cross-entropy loss")
            except Exception as e:
                add("log_loss", None, applicable=False,
                    definition=f"Not computable: {e}")

            # NEW: AUPRC (weighted one-vs-rest)
            try:
                if num_classes == 2:
                    auprc = float(average_precision_score(y_true, y_prob_norm[:, 1]))
                else:
                    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
                    auprc = float(average_precision_score(
                        y_bin, y_prob_norm, average="weighted",
                    ))
                add("auprc", auprc,
                    definition="Area under Precision-Recall curve",
                    averaging="weighted")
            except Exception as e:
                add("auprc", None, applicable=False,
                    definition=f"Not computable: {e}")

            # NEW: Top-k accuracy (k=2 if enough classes)
            if num_classes >= 3:
                try:
                    top2 = float(top_k_accuracy_score(
                        y_true, y_prob_norm, k=2,
                        labels=list(range(num_classes)),
                    ))
                    add("top2_accuracy", top2, definition="Top-2 accuracy")
                except Exception:
                    add("top2_accuracy", None, applicable=False)
            else:
                add("top2_accuracy", None, applicable=False,
                    definition="Only meaningful for ≥3 classes")
        else:
            reason = ("Requires probabilities" if y_prob is None
                      else "Probability shape unusable")
            for k in ("roc_auc", "gini", "log_loss", "auprc", "top2_accuracy"):
                add(k, None, applicable=False, definition=reason)

        # ── Regression metrics: N/A ──
        for name in ("mse", "rmse", "mae", "r_squared", "mape"):
            add(name, None, applicable=False,
                definition="Regression metric — not meaningful for classification")

        return m