"""
Filter comparison plots (raw vs downsampled at various rates).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from nanobio.signal.filters import FilterResult
from nanobio.viz.style import FREQ_COLOR_MAP

logger = logging.getLogger("nanobio.viz.filter_plots")


class FilterComparisonPlotter:
    """Plot filter comparisons (sequential and combined)."""

    def __init__(
        self, output_dir: Path, timestamp: str,
        dpi: int = 150, fig_format: str = "png",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.dpi = dpi
        self.fig_format = fig_format
        self._counter = 0

    def _next(self, tag: str) -> Path:
        self._counter += 1
        return self.output_dir / f"{self.timestamp}_filter_{tag}_{self._counter}.{self.fig_format}"

    def plot_sequential(
        self, raw_signal: np.ndarray, filtered_result: FilterResult,
        raw_fs: int, category: str, filename: str,
    ) -> Path:
        """Raw vs one filtered signal in a two-row figure."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        n_show = min(10_000, len(raw_signal))
        t_raw = np.arange(n_show) / raw_fs * 1000
        axes[0].plot(
            t_raw, raw_signal[:n_show], linewidth=0.5,
            color=FREQ_COLOR_MAP.get(raw_fs, "#1a1a1a"), alpha=0.8,
        )
        axes[0].set_title(f"Raw {raw_fs/1e6:.0f} MHz | {category} | {filename}")
        axes[0].set_xlabel("Time (ms)"); axes[0].set_ylabel("ADP (mV)")

        n_f = min(int(n_show * filtered_result.target_rate_hz / raw_fs), len(filtered_result.signal))
        t_f = np.arange(n_f) / filtered_result.target_rate_hz * 1000
        axes[1].plot(
            t_f, filtered_result.signal[:n_f], linewidth=0.8,
            color=FREQ_COLOR_MAP.get(filtered_result.target_rate_hz, "#333"),
        )
        label = f"Filter: {filtered_result.filter_type}"
        if filtered_result.filter_impl == "butterworth":
            label += f" (order={filtered_result.order})"
        elif filtered_result.filter_impl == "gaussian":
            label += f" (σ={filtered_result.sigma_samples:.1f})"
        label += f" | {filtered_result.phase_mode}"
        axes[1].set_title(
            f"Downsampled to {filtered_result.target_rate_hz} Hz | {label}"
        )
        axes[1].set_xlabel("Time (ms)"); axes[1].set_ylabel("ADP (mV)")
        fig.tight_layout()
        tag = f"seq_{filtered_result.filter_type}_{filtered_result.target_rate_hz}Hz"
        path = self._next(tag)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_combined(
        self, results: Dict[str, FilterResult], filter_type: str,
        raw_signal: np.ndarray, raw_fs: int, category: str, filename: str,
    ) -> Path:
        """All frequencies overlaid on one axis."""
        fig, ax = plt.subplots(figsize=(16, 8))
        n_show = min(10_000, len(raw_signal))
        t_raw = np.arange(n_show) / raw_fs * 1000
        ax.plot(
            t_raw, raw_signal[:n_show], linewidth=0.3,
            color=FREQ_COLOR_MAP[raw_fs], alpha=0.5,
            label=f"Raw {raw_fs/1e6:.0f} MHz",
        )
        for key, result in sorted(
            results.items(), key=lambda x: x[1].target_rate_hz, reverse=True,
        ):
            if result.filter_type != filter_type:
                continue
            rate = result.target_rate_hz
            n_f = min(int(n_show * rate / raw_fs), len(result.signal))
            t_f = np.arange(n_f) / rate * 1000
            lw = 0.5 if rate > 1000 else 1.0 if rate > 10 else 1.5
            ax.plot(
                t_f, result.signal[:n_f], linewidth=lw,
                color=FREQ_COLOR_MAP.get(rate, "#666"), alpha=0.7,
                label=f"{rate} Hz",
            )
        ax.set_title(f"All Frequencies — Filter: {filter_type} | {category} | {filename}")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("ADP (mV)")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        path = self._next(f"combined_{filter_type}")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path