"""
Loss functions for class imbalance handling (Section 29).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for class imbalance.

    L = -α_c · (1 - p_t)^γ · log(p_t)

    γ=0 reduces to weighted cross-entropy. Higher γ down-weights easy examples.

    Args:
        alpha: Per-class weight tensor of shape (num_classes,). If None, uniform.
        gamma: Focusing parameter (default 2.0 as in the paper).
        label_smoothing: Optional label smoothing factor.
    """

    def __init__(
        self, alpha: torch.Tensor = None, gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with per-sample reduction
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        # p_t = probability assigned to the true class
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()