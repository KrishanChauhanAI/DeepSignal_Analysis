"""
Detrending for MODEL INPUT only.

The detrended signal is what the model actually consumes. It is
zero-mean after subtracting the per-file baseline.
"""

from __future__ import annotations

import logging

import numpy as np

from nanobio.signal.baseline import BaselineResult

logger = logging.getLogger("nanobio.signal.detrend")


class Detrender:
    """Apply detrending to a signal using a pre-computed baseline."""

    @staticmethod
    def subtract_baseline(
        signal: np.ndarray, baseline_result: BaselineResult,
    ) -> np.ndarray:
        """Standard: subtract the per-file baseline."""
        return (signal - baseline_result.baseline).astype(signal.dtype)

    @staticmethod
    def mean_center(signal: np.ndarray) -> np.ndarray:
        """Minimal fallback: just remove the global mean."""
        return (signal - float(np.mean(signal))).astype(signal.dtype)

    @staticmethod
    def linear(signal: np.ndarray) -> np.ndarray:
        """Remove a global linear trend."""
        t = np.arange(len(signal))
        c = np.polyfit(t, signal, 1)
        trend = np.polyval(c, t)
        return (signal - trend).astype(signal.dtype)

    @classmethod
    def apply(
        cls, signal: np.ndarray, method: str,
        baseline_result: BaselineResult = None,
    ) -> np.ndarray:
        """Dispatcher for the configured detrending method."""
        if method == "subtract_baseline" and baseline_result is not None:
            return cls.subtract_baseline(signal, baseline_result)
        if method == "mean_center":
            return cls.mean_center(signal)
        if method == "linear":
            return cls.linear(signal)
        if method == "none":
            return signal.copy()
        raise ValueError(f"Unknown detrending method: {method!r}")