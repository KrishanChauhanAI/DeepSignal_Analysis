"""
TimesNet (Wu et al., ICLR 2023) — compact from-scratch re-implementation
adapted for univariate high-frequency signal classification.

BUG-002 FIX:
  InceptionBlock2D now includes a 1×1 projection convolution that maps
  the concatenated multi-kernel channels (out_channels × num_kernels)
  back to out_channels. This prevents the batch-dimension corruption
  that occurred during the reshape in TimesBlock.forward().
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock2D(nn.Module):
    """
    Multi-scale 2-D convolution block (parallel kernels, concatenated,
    then projected back to out_channels via 1×1 conv).

    BUG-002 FIX: Added self.projection to resolve channel dimension.
    Previously, the concatenated output had out_channels * num_kernels
    channels, which silently corrupted the batch dimension during the
    reshape in TimesBlock.forward().
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        # BUG-002 FIX: 1×1 projection to map concatenated channels back
        self.projection = nn.Conv2d(
            out_channels * num_kernels, out_channels,
            kernel_size=1, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([k(x) for k in self.kernels], dim=1)
        return self.projection(concat)


class TimesBlock(nn.Module):
    """
    One TimesNet block: FFT period detection -> 2D reshape -> Inception2D.

    Periods are derived per-sample from FFT amplitude peaks; samples are
    GROUPED BY PERIOD so convolutions run batched per unique period.
    """

    def __init__(
        self, d_model: int, d_ff: int, top_k: int,
        num_kernels: int = 6, dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock2D(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock2D(d_ff, d_model, num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape

        # ── Period detection ──
        fft = torch.fft.rfft(x, dim=1)
        amp = fft.abs().mean(dim=-1)
        amp[:, 0] = 0.0
        if T % 2 == 0:
            amp[:, -1] = 0.0
        k = min(self.top_k, amp.size(1) - 1)
        top_amp, top_idx = torch.topk(amp, k, dim=1)
        weights = F.softmax(top_amp, dim=1)
        top_idx_np = top_idx.detach().cpu().numpy()

        out = torch.zeros_like(x)

        # ── Group samples by period for batched convolutions ──
        by_period: dict = {}
        for b in range(B):
            for j in range(k):
                idx = max(1, int(top_idx_np[b, j]))
                p = min(T, max(2, T // idx))
                by_period.setdefault(p, []).append((b, j))

        for p, items in by_period.items():
            t_pad = math.ceil(T / p) * p
            k_len = t_pad // p
            xb = torch.stack([x[b] for b, _ in items], dim=0)
            if t_pad > T:
                xb = F.pad(xb, (0, 0, 0, t_pad - T))
            x2 = xb.reshape(-1, k_len, p, D).permute(0, 3, 1, 2)
            y2 = self.conv(x2)
            # BUG-002 FIX: y2 now has shape (n, D, k_len, p) because
            # InceptionBlock2D projects back to D channels.
            # The reshape correctly maps to (n, t_pad, D).
            y = y2.permute(0, 2, 3, 1).reshape(-1, t_pad, D)[:, :T]
            for (b, j), row in zip(items, y):
                out[b] = out[b] + weights[b, j] * row

        return self.norm(x + self.dropout(out))


class TimesNetClassifier(nn.Module):
    """
    TimesNet for classification on raw windows.

    Defaults are deliberately SMALL (d_model=32, e_layers=2): our dataset is
    file-limited and windows noisy; a full-size TimesNet would overfit.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 3,
        e_layers: int = 2,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, top_k, num_kernels, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.fc(self.dropout(x.mean(dim=1)))