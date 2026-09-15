"""
Hierarchical Attention-Based Hybrid Prototypical Network.

Architecture:
    Raw file (detrended)
        │
        ▼ Segment into M windows
        ▼ Per-segment z-score
    Segment encoder f_φ (1-D ResNet)
        │
        ▼ M segment embeddings z_m
    Attention aggregator
        α_m = softmax(w^T tanh(W z_m))
        Z_learned = Σ α_m z_m
        │
        ▼ Hybrid fusion (concat with mean physical features)
    Metric embedding
        │
        ▼ Prototypical distance classifier
        C_k = mean of support Z for class k
        p(y=k|q) = softmax(-‖Z_q - C_k‖²)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanobio.models.resnet1d import ResidualBlock1D


class SegmentEncoder(nn.Module):
    """Maps a segment (1, T) → embedding (embed_dim,)."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.block1 = ResidualBlock1D(32, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = ResidualBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = ResidualBlock1D(128, embed_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return self.gap(x).squeeze(-1)


class AttentionAggregator(nn.Module):
    """Additive attention across segments within one file."""
    def __init__(self, embed_dim: int, attention_hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

    def forward(
        self, segment_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.attn(segment_embeddings).squeeze(-1)
        weights = F.softmax(logits, dim=0)
        aggregated = (weights.unsqueeze(-1) * segment_embeddings).sum(dim=0)
        return aggregated, weights


class HybridFusion(nn.Module):
    """Fuse learned file embedding with mean physical features."""
    def __init__(self, learned_dim: int, phys_dim: int, metric_dim: int):
        super().__init__()
        self.norm_phys = nn.LayerNorm(phys_dim)
        self.mlp = nn.Sequential(
            nn.Linear(learned_dim + phys_dim, learned_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(learned_dim, metric_dim),
        )

    def forward(
        self, learned_embed: torch.Tensor, phys_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([learned_embed, self.norm_phys(phys_features)], dim=-1))


@dataclass
class ProtoNetOutput:
    """Structured episode output."""
    logits: torch.Tensor
    query_embeddings: torch.Tensor
    prototypes: torch.Tensor
    attention_weights: Optional[dict] = None


class HierarchicalProtoNet(nn.Module):
    """
    Full Hierarchical ProtoNet.

    Produces a single embedding per FILE (not per window).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        metric_dim: int = 64,
        attention_hidden: int = 64,
        phys_dim: int = 13,
        hybrid_enabled: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.metric_dim = metric_dim
        self.hybrid_enabled = hybrid_enabled
        self.segment_encoder = SegmentEncoder(embed_dim=embed_dim)
        self.aggregator = AttentionAggregator(embed_dim, attention_hidden)
        self.fusion = (
            HybridFusion(embed_dim, phys_dim, metric_dim)
            if hybrid_enabled else nn.Linear(embed_dim, metric_dim)
        )

    def encode_file(
        self,
        segments: torch.Tensor,
        phys_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode ONE file (M segments) → metric embedding."""
        seg_embeds = self.segment_encoder(segments)
        file_learned, attn = self.aggregator(seg_embeds)
        if self.hybrid_enabled and phys_features is not None:
            phys_mean = phys_features.mean(dim=0)
            file_embed = self.fusion(file_learned, phys_mean)
        else:
            file_embed = self.fusion(file_learned)
        return file_embed, (attn if return_attention else None)

    def forward(
        self,
        support_segments: List[torch.Tensor],
        support_phys: List[torch.Tensor],
        support_labels: torch.Tensor,
        query_segments: List[torch.Tensor],
        query_phys: List[torch.Tensor],
        n_way: int,
        return_attention: bool = False,
    ) -> ProtoNetOutput:
        """Run one episodic forward pass."""
        support_embeds, support_attns = [], [] if return_attention else None
        for segs, phys in zip(support_segments, support_phys):
            e, a = self.encode_file(segs, phys, return_attention=return_attention)
            support_embeds.append(e)
            if support_attns is not None:
                support_attns.append(a)
        support_embeds = torch.stack(support_embeds, dim=0)

        prototypes = torch.zeros(
            n_way, support_embeds.size(-1),
            device=support_embeds.device, dtype=support_embeds.dtype,
        )
        for k in range(n_way):
            mask = (support_labels == k)
            if mask.sum() > 0:
                prototypes[k] = support_embeds[mask].mean(dim=0)

        query_embeds, query_attns = [], [] if return_attention else None
        for segs, phys in zip(query_segments, query_phys):
            e, a = self.encode_file(segs, phys, return_attention=return_attention)
            query_embeds.append(e)
            if query_attns is not None:
                query_attns.append(a)
        query_embeds = torch.stack(query_embeds, dim=0)

        dists = torch.cdist(query_embeds, prototypes, p=2) ** 2
        logits = -dists

        all_attns = None
        if return_attention:
            all_attns = {"support": support_attns, "query": query_attns}

        return ProtoNetOutput(
            logits=logits, query_embeddings=query_embeds,
            prototypes=prototypes, attention_weights=all_attns,
        )