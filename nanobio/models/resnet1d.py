"""
Baseline deep-learning architectures operating on RAW signal windows.

Models included:
    - ResNet1D    (Wang et al. 2017)
    - TCN         (Bai, Kolter & Koltun 2018)
    - InceptionTime (Fawaz et al. 2020)
    - CNNBiLSTM   (hybrid convolutional + recurrent)
    - Transformer (patch-based)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Basic residual convolutional block for 1-D signals."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet1D(nn.Module):
    """1-D ResNet baseline (Wang et al. 2017)."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128),
            ResidualBlock1D(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


class TCN(nn.Module):
    """Temporal Convolutional Network."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        channels = [64, 64, 128, 128]
        layers = []
        prev = in_channels
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers += [
                nn.Conv1d(prev, ch, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(0.2),
            ]
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))

'''
class InceptionTime(nn.Module):
    """InceptionTime multi-kernel CNN (Fawaz et al. 2020)."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, 32, 1, bias=False)
        self.conv_small = nn.Conv1d(32, 32, 10, padding=5, bias=False)
        self.conv_med = nn.Conv1d(32, 32, 20, padding=10, bias=False)
        self.conv_large = nn.Conv1d(32, 32, 40, padding=20, bias=False)
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            nn.Conv1d(in_channels, 32, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        c1, c2, c3 = self.conv_small(b), self.conv_med(b), self.conv_large(b)
        c4 = self.pool_conv(x)
        min_len = min(c.size(-1) for c in (c1, c2, c3, c4))
        merged = torch.cat([c[..., :min_len] for c in (c1, c2, c3, c4)], dim=1)
        x = F.relu(self.bn(merged))
        x = self.gap(x).squeeze(-1)
        return self.fc(self.dropout(x))
'''
class InceptionBlock(nn.Module):
    """Single InceptionTime block with residual shortcut."""
    def __init__(self, in_channels: int, out_channels: int, bottleneck_ch: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_ch, 1, bias=False)
        self.conv_small = nn.Conv1d(bottleneck_ch, bottleneck_ch, 10, padding=5, bias=False)
        self.conv_med = nn.Conv1d(bottleneck_ch, bottleneck_ch, 20, padding=10, bias=False)
        self.conv_large = nn.Conv1d(bottleneck_ch, bottleneck_ch, 40, padding=20, bias=False)
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            nn.Conv1d(in_channels, bottleneck_ch, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(bottleneck_ch * 4)
        self.shortcut = (
            nn.Conv1d(in_channels, bottleneck_ch * 4, 1, bias=False)
            if in_channels != bottleneck_ch * 4 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        c1 = self.conv_small(b)
        c2 = self.conv_med(b)
        c3 = self.conv_large(b)
        c4 = self.pool_conv(x)
        min_len = min(c.size(-1) for c in (c1, c2, c3, c4))
        merged = torch.cat([c[..., :min_len] for c in (c1, c2, c3, c4)], dim=1)
        out = F.relu(self.bn(merged))
        return out + self.shortcut(x[..., :min_len])


class InceptionTime(nn.Module):
    """
    InceptionTime multi-kernel CNN (Fawaz et al. 2020).

    BUG-015 FIX: Expanded from 1 block to 3 blocks with residual
    connections. The original paper uses 6 blocks, but 3 is sufficient
    for our 1024-sample windows and small dataset.
    """
    def __init__(self, num_classes: int, in_channels: int = 1, n_blocks: int = 3):
        super().__init__()
        blocks = []
        ch_in = in_channels
        ch_out = 128  # bottleneck_ch=32 × 4 branches
        for _ in range(n_blocks):
            blocks.append(InceptionBlock(ch_in, ch_out))
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(ch_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.dropout(x))


class CNNBiLSTM(nn.Module):
    """Hybrid CNN + BiLSTM for raw signals."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            128, 64, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


class TransformerClassifier(nn.Module):
    """Patch-based Transformer encoder."""
    def __init__(
        self, num_classes: int, in_channels: int = 1,
        patch_size: int = 16, d_model: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(in_channels * patch_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=256,
            dropout=0.2, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=3)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        np_ = T // self.patch_size
        x = x[:, :, : np_ * self.patch_size]
        x = x.reshape(B, C, np_, self.patch_size).permute(0, 2, 1, 3).reshape(B, np_, -1)
        x = self.embed(x) + self.pos[:, :np_, :]
        x = self.encoder(x)
        return self.fc(self.dropout(self.norm(x.mean(1))))

# ═══════════════════════════════════════════════════════════════════
# RESEARCH ENCODERS (self-supervised representation learning)
# Added to existing models/resnet1d.py to reuse the same module.
# ═══════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """Dilated causal conv block with residual connection."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.shortcut = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)[..., :x.size(-1)]))
        out = self.drop(out)
        out = self.bn2(self.conv2(out)[..., :x.size(-1)])
        out = self.drop(out)
        return F.gelu(out + self.shortcut(x))


class TCNEncoder(nn.Module):
    """TCN encoder for self-supervised learning.
    Maps (B, 1, T) → (B, embedding_dim)."""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64,
                 num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        channels = [1] + [hidden_dim] * (num_layers - 1) + [embedding_dim]
        for i in range(len(channels) - 1):
            layers.append(_TCNBlock(
                channels[i], channels[i + 1],
                kernel_size, 2 ** i, dropout,
            ))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(x)).squeeze(-1)


class CNNEncoder(nn.Module):
    """Simple CNN encoder for ablation baselines."""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64,
                 num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = 1
        for i in range(num_layers):
            out_ch = min(hidden_dim * (2 ** min(i, 3)), 512)
            layers.extend([
                nn.Conv1d(ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.MaxPool1d(2),
            ])
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.features(x)).squeeze(-1)
        return self.proj(self.drop(h))


class ResNetEncoder(nn.Module):
    """ResNet encoder variant for SSL (reuses ResidualBlock1D above)."""
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64,
                 num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        blocks = []
        ch = hidden_dim
        for i in range(num_layers):
            out_ch = min(ch * 2, embedding_dim) if i > 0 else ch
            blocks.append(ResidualBlock1D(ch, out_ch, kernel_size))
            if i < num_layers - 1:
                blocks.append(nn.MaxPool1d(2))
            ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.blocks(self.stem(x))).squeeze(-1)
        return self.proj(h)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive pretraining (discarded after)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Factory function for research encoders
_ENCODER_REGISTRY = {
    "tcn": TCNEncoder,
    "resnet": ResNetEncoder,
    "cnn": CNNEncoder,
}

def build_ssl_encoder(encoder_type: str = "tcn", **kwargs) -> nn.Module:
    """Build an encoder for self-supervised learning."""
    if encoder_type not in _ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder '{encoder_type}'. "
                       f"Available: {list(_ENCODER_REGISTRY.keys())}")
    return _ENCODER_REGISTRY[encoder_type](**kwargs)