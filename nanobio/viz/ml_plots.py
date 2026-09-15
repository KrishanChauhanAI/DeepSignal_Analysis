"""
Machine-learning visualizations (Section 24) and training curves (Section 32).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("nanobio.viz.ml_plots")


class MLPlotter:
    """Section 24 + 32 plot generator."""

    def __init__(self, output_dir: Path, timestamp: str, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.dpi = dpi
        self._c = 0

    def _path(self, prefix: str) -> Path:
        self._c += 1
        return self.output_dir / f"{self.timestamp}_{prefix}_{self._c}.png"

    # ── Section 24: Confusion Matrix ──
    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        categories: List[str], model_name: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=categories,
            cmap="Blues", ax=ax, colorbar=False, xticks_rotation=30,
        )
        ax.set_title(f"Confusion Matrix — {model_name}")
        fig.tight_layout()
        p = self._path(f"CM_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p

    # ── Section 24: ROC (binary + multiclass OvR) ──
    def plot_roc(
        self, y_true: np.ndarray, y_prob: np.ndarray,
        categories: List[str], model_name: str,
    ) -> Path:
        num_classes = len(categories)
        fig, ax = plt.subplots(figsize=(7, 6))

        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}", lw=2)
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            for i, cat in enumerate(categories):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(fpr, tpr, lw=1.5,
                        label=f"{cat} (AUC={auc(fpr, tpr):.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {model_name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = self._path(f"ROC_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p

    # ── Section 24: PR curve + AUPRC ──
    def plot_pr(
        self, y_true: np.ndarray, y_prob: np.ndarray,
        categories: List[str], model_name: str,
    ) -> Path:
        num_classes = len(categories)
        fig, ax = plt.subplots(figsize=(7, 6))

        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            ax.plot(recall, precision, lw=2,
                    label=f"AUPRC = {auc(recall, precision):.3f}")
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            for i, cat in enumerate(categories):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                ax.plot(recall, precision, lw=1.5,
                        label=f"{cat} (AUPRC={auc(recall, precision):.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall — {model_name}")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = self._path(f"PR_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p

    # ── Section 24: Calibration plot ──
    def plot_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray,
        categories: List[str], model_name: str, n_bins: int = 10,
    ) -> Path:
        num_classes = len(categories)
        fig, ax = plt.subplots(figsize=(7, 6))

        if num_classes == 2:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_prob[:, 1], n_bins=n_bins, strategy="uniform",
            )
            ax.plot(mean_pred, frac_pos, "o-", label="Model")
        else:
            for i, cat in enumerate(categories):
                y_bin = (y_true == i).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                frac_pos, mean_pred = calibration_curve(
                    y_bin, y_prob[:, i], n_bins=n_bins, strategy="uniform",
                )
                ax.plot(mean_pred, frac_pos, "o-", label=cat, lw=1.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration — {model_name}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = self._path(f"Calibration_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p

    # ── Section 24: Probability distribution ──
    def plot_probability_distribution(
        self, y_true: np.ndarray, y_prob: np.ndarray,
        categories: List[str], model_name: str,
    ) -> Path:
        num_classes = len(categories)
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, cat in enumerate(categories):
            mask = y_true == i
            if mask.sum() == 0:
                continue
            probs_of_true = y_prob[mask, i]
            sns.kdeplot(probs_of_true, ax=ax, label=f"True={cat}",
                        fill=True, alpha=0.3, clip=(0, 1))

        ax.set_xlabel("Predicted probability of the TRUE class")
        ax.set_ylabel("Density")
        ax.set_title(f"Predicted-Probability Distribution — {model_name}")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = self._path(f"ProbDist_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p

    # ── Section 24: Class-wise metrics table (saved as CSV, not plot) ──
    def save_classwise_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        categories: List[str], model_name: str, out_dir: Path,
    ) -> Path:
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
        )
        rows = []
        for i, cat in enumerate(categories):
            mask_t = (y_true == i)
            mask_p = (y_pred == i)
            support = int(mask_t.sum())
            if support == 0:
                rows.append({
                    "class": cat, "support": 0,
                    "precision": None, "recall": None, "f1": None,
                    "accuracy_within_class": None,
                })
                continue
            p = float(precision_score(y_true == i, y_pred == i, zero_division=0))
            r = float(recall_score(y_true == i, y_pred == i, zero_division=0))
            f = float(f1_score(y_true == i, y_pred == i, zero_division=0))
            acc_c = float((mask_t & mask_p).sum() / support)
            rows.append({
                "class": cat, "support": support,
                "precision": p, "recall": r, "f1": f,
                "accuracy_within_class": acc_c,
            })
        df = pd.DataFrame(rows)
        out = Path(out_dir) / f"{self.timestamp}_classwise_{model_name}.csv"
        df.to_csv(out, index=False)
        return out

    # ── Section 32: Training curves (loss / acc / lr / epoch_time) ──
    def plot_training_curves(
        self, history: Dict[str, List[float]], model_name: str,
    ) -> Path:
        # Which curves are present?
        has_lr = "learning_rate" in history and len(history["learning_rate"]) > 0
        has_time = "epoch_time_s" in history and len(history["epoch_time_s"]) > 0

        n_panels = 2 + int(has_lr) + int(has_time)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        epochs = history.get("epoch", list(range(1, len(history.get("loss", [])) + 1)))
        idx = 0

        # Loss
        if "loss" in history:
            axes[idx].plot(epochs, history["loss"], label="Train", marker="o", ms=3)
            if "val_loss" in history:
                axes[idx].plot(epochs, history["val_loss"], label="Val", marker="s", ms=3)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Loss")
            axes[idx].set_title(f"{model_name} — Loss")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            idx += 1

        # Accuracy
        if "accuracy" in history:
            axes[idx].plot(epochs, history["accuracy"], label="Train", marker="o", ms=3)
            if "val_accuracy" in history:
                axes[idx].plot(epochs, history["val_accuracy"], label="Val", marker="s", ms=3)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Accuracy")
            axes[idx].set_title(f"{model_name} — Accuracy")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            idx += 1

        # Learning rate
        if has_lr:
            axes[idx].plot(epochs, history["learning_rate"], "g-", marker="d", ms=3)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Learning rate")
            axes[idx].set_yscale("log")
            axes[idx].set_title(f"{model_name} — LR schedule")
            axes[idx].grid(alpha=0.3, which="both")
            idx += 1

        # Epoch time (cumulative)
        if has_time:
            cum_time = np.cumsum(history["epoch_time_s"])
            ax2 = axes[idx]
            ax2.plot(epochs, history["epoch_time_s"], "b-", label="Per epoch", marker="o", ms=3)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Seconds per epoch", color="b")
            ax2.tick_params(axis="y", labelcolor="b")
            ax2t = ax2.twinx()
            ax2t.plot(epochs, cum_time, "r--", label="Cumulative", marker="x", ms=3)
            ax2t.set_ylabel("Cumulative seconds", color="r")
            ax2t.tick_params(axis="y", labelcolor="r")
            ax2.set_title(f"{model_name} — Epoch time")
            idx += 1

        fig.tight_layout()
        p = self._path(f"Training_{model_name}")
        fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return p