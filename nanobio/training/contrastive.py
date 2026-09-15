"""
Contrastive loss functions for self-supervised learning.

Separated from trainer.py for modularity. The ContrastiveTrainer
class lives in trainer.py alongside other trainers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent loss for contrastive learning.

    z1, z2: (B, D) L2-normalized embeddings.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)