"""
Baseline overlay visualization.

Produces the plot style shown in the user's reference images:
raw noisy signal (red) with the per-file baseline overlaid (dark black line).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from nanobio.signal.baseline import BaselineResult

logger = logging.getLogger("nanobio.viz.baseline_plots")


class BaselineOverlayPlotter:
    """Generate raw-signal + baseline overlay plots per file."""

    def __init__(
        self,
        output_dir: Path,
        timestamp: str,
        dpi: int = 150,
        fig_format: str = "png",
        signal_color: str = "#c92020",
        baseline_color: str = "#000000",
        signal_units: str = "mV",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.dpi = dpi
        self.fig_format = fig_format
        self.signal_color = signal_color
        self.baseline_color = baseline_color
        self.signal_units = signal_units
        self._counter = 0

    def _next_path(self, tag: str) -> Path:
        self._counter += 1
        return self.output_dir / f"{self.timestamp}_baseline_{tag}_{self._counter}.{self.fig_format}"

    '''
    def plot_file(
        self,
        signal: np.ndarray,
        baseline_result: BaselineResult,
        sampling_rate_hz: int,
        file_name: str,
        category: str,
        max_display_seconds: Optional[float] = None,
    ) -> Path:
        """Plot one file's raw signal with baseline overlay."""
        if max_display_seconds is not None:
            n = min(len(signal), int(max_display_seconds * sampling_rate_hz))
            signal = signal[:n]
            baseline = baseline_result.baseline[:n]
        else:
            baseline = baseline_result.baseline

        t = np.arange(len(signal)) / sampling_rate_hz

        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(
            t, signal, color=self.signal_color, linewidth=0.4, alpha=0.85,
            label="Raw signal", zorder=1,
        )
        ax.plot(
            t, baseline, color=self.baseline_color, linewidth=1.2,
            label=f"Per-file baseline ({baseline_result.method})", zorder=2,
        )
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(f"ADP ({self.signal_units})", fontsize=11)
        ax.set_title(
            f"Raw + per-file baseline | Class: {category} | "
            f"File: {file_name} | fs = {sampling_rate_hz:,} Hz",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=9)
        ax.text(
            0.01, 0.98,
            f"Baseline range: {baseline_result.baseline_range:.2f} {self.signal_units}\n"
            f"Residual std:   {baseline_result.residual_std:.2f} {self.signal_units}",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        fig.tight_layout()

        tag = f"{category}_{Path(file_name).stem[:30]}"
        path = self._next_path(tag)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path
        '''
    def plot_file(
        self,
        signal: np.ndarray,
        baseline_result: BaselineResult,
        sampling_rate_hz: int,
        file_name: str,
        category: str,
        max_display_seconds: Optional[float] = None,
    ) -> Path:
        """Plot one file's raw signal with baseline overlay."""
        if max_display_seconds is not None:
            n = min(len(signal), int(max_display_seconds * sampling_rate_hz))
            signal = signal[:n]
            baseline = baseline_result.baseline[:n]
        else:
            baseline = baseline_result.baseline

        t = np.arange(len(signal)) / sampling_rate_hz

        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(
            t, signal, color=self.signal_color, linewidth=0.4, alpha=0.85,
            label="Raw signal", zorder=1,
        )
        ax.plot(
            t, baseline, color=self.baseline_color, linewidth=1.2,
            label=f"Per-file baseline ({baseline_result.method})", zorder=2,
        )
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(f"ADP ({self.signal_units})", fontsize=11)
        ax.set_title(
            f"Raw + per-file baseline | Class: {category} | "
            f"File: {file_name} | fs = {sampling_rate_hz:,} Hz",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=9)

        # Annotation with baseline quality metrics (flag-controlled)
        lines = []
        if baseline_result.baseline_range is not None:
            lines.append(
                f"Baseline range: {baseline_result.baseline_range:.2f} {self.signal_units}"
            )
        if baseline_result.residual_std is not None:
            lines.append(
                f"Residual std:   {baseline_result.residual_std:.2f} {self.signal_units}"
            )
        lines.append(f"Method: {baseline_result.method}")
        if baseline_result.score is not None:
            lines.append(f"Composite score: {baseline_result.score.composite:.4f}")
            lines.append(
                f"evt_pres_loss={baseline_result.score.event_preservation_loss:.3f}  "
                f"acf1={baseline_result.score.residual_autocorr_lag1:.3f}"
            )
        if lines:
            ax.text(
                0.01, 0.98, "\n".join(lines),
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                fontfamily="monospace",
            )

        fig.tight_layout()

        tag = f"{category}_{Path(file_name).stem[:30]}"
        path = self._next_path(tag)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_batch(self, files_data: List[dict], max_files: int = 20) -> List[Path]:
        paths = []
        for item in files_data[:max_files]:
            try:
                p = self.plot_file(
                    signal=item["signal"],
                    baseline_result=item["baseline_result"],
                    sampling_rate_hz=item["sampling_rate_hz"],
                    file_name=item["file_name"],
                    category=item["category"],
                    max_display_seconds=item.get("max_display_seconds"),
                )
                paths.append(p)
            except Exception as e:
                logger.error("Baseline plot failed: %s", e)
        logger.info("Saved %d baseline overlay figures", len(paths))
        return paths