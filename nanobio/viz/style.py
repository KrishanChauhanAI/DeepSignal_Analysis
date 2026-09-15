"""
Consistent visualization styling.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from nanobio.config import VisualizationCfg

# Frequency-to-color mapping (darker for lower frequencies)
FREQ_COLOR_MAP = {
    1_000_000: "#1a1a1a",
    50_000: "#2d2d8e",
    10_000: "#3838b8",
    1_000: "#4444e0",
    100: "#7777f0",
    10: "#aaaafc",
    2: "#ccccff",
}


def apply_default_style(cfg: VisualizationCfg) -> None:
    """Apply consistent styling to matplotlib and seaborn."""
    sns.set_context("paper", font_scale=cfg.font_scale)
    sns.set_style(cfg.style, {"grid.linestyle": "--", "grid.alpha": 0.3})
    plt.rcParams["font.family"] = cfg.font_family
    plt.rcParams["figure.dpi"] = cfg.dpi