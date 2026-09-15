"""
Attention overlay visualization for ProtoNet.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_overlay(
    segments: np.ndarray,
    attention_weights: np.ndarray,
    segment_size: int,
    stride: int,
    file_name: str,
    category: str,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """
    Visualize attention weights across segments of a file.

    Args:
        segments: (M, segment_size) segments.
        attention_weights: (M,) attention weights.
        segment_size: Samples per segment.
        stride: Stride between segment starts.
        file_name, category: Metadata for the title.
        output_path: PNG path.

    Returns:
        Path to saved figure.
    """
    M = len(segments)
    signal = np.zeros((M - 1) * stride + segment_size, dtype=np.float32)
    for i in range(M):
        signal[i * stride: i * stride + segment_size] = segments[i]

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

    # Top: signal
    axes[0].plot(signal, linewidth=0.5, color="#333")
    axes[0].set_title(
        f"Signal + attention | Class: {category} | File: {file_name}",
        fontsize=11,
    )
    axes[0].set_ylabel("Signal")

    # Bottom: attention heatmap projected onto time axis
    attn_curve = np.zeros_like(signal, dtype=np.float32)
    for i in range(M):
        attn_curve[i * stride: i * stride + segment_size] = attention_weights[i]
    axes[1].fill_between(
        np.arange(len(attn_curve)), 0, attn_curve,
        color="#c92020", alpha=0.7,
    )
    axes[1].set_ylabel("Attention")
    axes[1].set_xlabel("Sample")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path