"""
Per-file baseline estimation with objective auto-selection.

CRITICAL DISTINCTION:
    - Baseline is for VISUALIZATION and as a detrending reference.
    - It is NOT fed to the model.
    - Each file has its own baseline (no universal baseline).

Auto-selection (when method="auto"):
    Runs constant, linear, rolling_median, and savgol candidates,
    scores each with residual variance, baseline smoothness,
    residual autocorrelation, event preservation, and reconstruction
    error, then picks the winner by composite score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger("nanobio.signal.baseline")


@dataclass
class BaselineScore:
    """Per-candidate objective metrics (lower is better for all)."""
    residual_variance: float = 0.0
    baseline_smoothness: float = 0.0
    residual_autocorr_lag1: float = 0.0
    event_preservation_loss: float = 0.0   # 0 = events fully preserved
    reconstruction_error: float = 0.0
    composite: float = 0.0                 # weighted sum of the five

    def to_dict(self) -> Dict[str, float]:
        return {
            "residual_variance": self.residual_variance,
            "baseline_smoothness": self.baseline_smoothness,
            "residual_autocorr_lag1": self.residual_autocorr_lag1,
            "event_preservation_loss": self.event_preservation_loss,
            "reconstruction_error": self.reconstruction_error,
            "composite": self.composite,
        }


@dataclass
class BaselineResult:
    """Per-file baseline with quality metrics and optional scorecard."""
    baseline: np.ndarray
    method: str = "rolling_median"
    parameters: Dict[str, Any] = field(default_factory=dict)
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    baseline_range: Optional[float] = 0.0
    residual_std: Optional[float] = 0.0
    score: Optional[BaselineScore] = None
    all_candidates: Optional[Dict[str, BaselineScore]] = None  # method → score


class BaselineEstimator:
    """Estimate slow-varying baseline per file, with optional auto-selection."""

    # Weights for composite score (must sum to 1.0)
    WEIGHTS = {
        "residual_variance": 0.20,
        "baseline_smoothness": 0.25,
        "residual_autocorr_lag1": 0.20,
        "event_preservation_loss": 0.25,
        "reconstruction_error": 0.10,
    }

    def __init__(self, sampling_rate_hz: int):
        self.fs = sampling_rate_hz

    # ── Candidate generators ───────────────────────────────────────────────

    def constant(self, signal: np.ndarray) -> BaselineResult:
        """Subtract global mean (constant baseline)."""
        mean_val = float(np.mean(signal))
        baseline = np.full_like(signal, mean_val)
        residual = signal - baseline
        return BaselineResult(
            baseline=baseline,
            method="constant",
            parameters={"mean": mean_val},
            baseline_mean=mean_val,
            baseline_std=0.0,
            baseline_range=0.0,
            residual_std=float(np.std(residual)),
        )

    def linear(self, signal: np.ndarray) -> BaselineResult:
        """Global linear trend fit."""
        t = np.arange(len(signal), dtype=np.float64)
        coeffs = np.polyfit(t, signal.astype(np.float64), 1)
        baseline = np.polyval(coeffs, t).astype(signal.dtype)
        residual = signal - baseline
        return BaselineResult(
            baseline=baseline,
            method="linear",
            parameters={"slope": float(coeffs[0]), "intercept": float(coeffs[1])},
            baseline_mean=float(np.mean(baseline)),
            baseline_std=float(np.std(baseline)),
            baseline_range=float(np.ptp(baseline)),
            residual_std=float(np.std(residual)),
        )

    def rolling_median(
        self, signal: np.ndarray, window_seconds: float = 1.0,
    ) -> BaselineResult:
        """
        Rolling-median baseline.

        Median is robust to transient blockade events; those events are
        NOT absorbed into the baseline (when window ≫ event duration).
        """
        w = max(3, int(window_seconds * self.fs))
        if w % 2 == 0:
            w += 1
        # Cap window at signal length
        w = min(w, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        w = max(3, w)
        baseline = (
            pd.Series(signal.astype(np.float64))
            .rolling(window=w, center=True, min_periods=1)
            .median()
            .values
        ).astype(signal.dtype)
        residual = signal - baseline
        return BaselineResult(
            baseline=baseline,
            method="rolling_median",
            parameters={"window_seconds": window_seconds, "window_samples": w},
            baseline_mean=float(np.mean(baseline)),
            baseline_std=float(np.std(baseline)),
            baseline_range=float(np.ptp(baseline)),
            residual_std=float(np.std(residual)),
        )

    def savgol(
        self, signal: np.ndarray, window: int = 501, poly_order: int = 3,
    ) -> BaselineResult:
        """Savitzky-Golay smoothing (smoother; less event-robust)."""
        if window % 2 == 0:
            window += 1
        window = min(window, len(signal) - 1 if len(signal) % 2 == 0 else len(signal))
        if window < poly_order + 2:
            window = poly_order + 3 + (0 if (poly_order + 3) % 2 else 1)
        window = max(poly_order + 2 | 1, window)  # ensure odd and large enough
        if window % 2 == 0:
            window += 1
        if window >= len(signal):
            window = len(signal) - 1 if len(signal) % 2 == 0 else len(signal) - 2
            window = max(poly_order + 2 | 1, window)
            if window % 2 == 0:
                window -= 1
        baseline = savgol_filter(
            signal.astype(np.float64), window_length=window, polyorder=poly_order,
        ).astype(signal.dtype)
        residual = signal - baseline
        return BaselineResult(
            baseline=baseline,
            method="savgol",
            parameters={"window": window, "poly_order": poly_order},
            baseline_mean=float(np.mean(baseline)),
            baseline_std=float(np.std(baseline)),
            baseline_range=float(np.ptp(baseline)),
            residual_std=float(np.std(residual)),
        )

    # ── Objective scoring ──────────────────────────────────────────────────

    def _score(
        self, signal: np.ndarray, result: BaselineResult,
        event_sigma: float = 5.0,
    ) -> BaselineScore:
        """
        Score one baseline candidate. Lower is better for every metric.

        Metrics:
            residual_variance:
                Var(signal − baseline). Too low → baseline ate the signal.
                Too high → baseline missed slow drift. We use absolute variance
                (normalized later across candidates).

            baseline_smoothness:
                Mean squared second difference of the baseline.
                High = jittery/noisy baseline (bad). Low = smooth (good).

            residual_autocorr_lag1:
                |ACF(residual, lag=1)|. High residual autocorrelation means
                the residual still contains slow structure the baseline missed.

            event_preservation_loss:
                1 − (event energy in residual / event energy in original).
                Events are defined as |signal − median| > event_sigma × MAD.
                0 = events fully preserved in residual; 1 = events destroyed.

            reconstruction_error:
                RMSE of (baseline + residual) vs original signal.
                Should be ~0 for any additive method; catches numerical issues.
        """
        residual = signal.astype(np.float64) - result.baseline.astype(np.float64)
        baseline = result.baseline.astype(np.float64)
        sig = signal.astype(np.float64)

        # 1. Residual variance
        res_var = float(np.var(residual))

        # 2. Baseline smoothness (mean sq 2nd difference)
        if len(baseline) > 2:
            d2 = np.diff(baseline, n=2)
            smoothness = float(np.mean(d2 ** 2))
        else:
            smoothness = 0.0

        # 3. Residual lag-1 autocorrelation (absolute value)
        if len(residual) > 10 and np.std(residual) > 1e-15:
            r = residual - residual.mean()
            c0 = float(np.dot(r, r))
            c1 = float(np.dot(r[:-1], r[1:]))
            acf1 = abs(c1 / (c0 + 1e-15))
        else:
            acf1 = 0.0

        # 4. Event preservation
        #    Define "events" on the ORIGINAL signal via robust threshold,
        #    then check how much of that event energy survives in residual.
        med = float(np.median(sig))
        mad = float(np.median(np.abs(sig - med))) + 1e-15
        event_mask = np.abs(sig - med) > (event_sigma * mad * 1.4826)
        if event_mask.any():
            orig_event_energy = float(np.sum((sig[event_mask] - med) ** 2))
            res_event_energy = float(np.sum(residual[event_mask] ** 2))
            # Loss = fraction of event energy absorbed by baseline
            preservation_loss = 1.0 - min(1.0, res_event_energy / (orig_event_energy + 1e-15))
            preservation_loss = float(max(0.0, preservation_loss))
        else:
            # No events detected — preservation is moot; don't penalize
            preservation_loss = 0.0

        # 5. Reconstruction error
        recon = baseline + residual
        recon_err = float(np.sqrt(np.mean((recon - sig) ** 2)))

        score = BaselineScore(
            residual_variance=res_var,
            baseline_smoothness=smoothness,
            residual_autocorr_lag1=acf1,
            event_preservation_loss=preservation_loss,
            reconstruction_error=recon_err,
        )
        return score

    def _normalize_and_composite(
        self, scores: Dict[str, BaselineScore],
    ) -> Dict[str, BaselineScore]:
        """
        Min-max normalize each metric across candidates, then compute
        weighted composite. Lower composite = better.
        """
        metric_names = [
            "residual_variance", "baseline_smoothness",
            "residual_autocorr_lag1", "event_preservation_loss",
            "reconstruction_error",
        ]
        # Collect raw values
        raw = {m: [] for m in metric_names}
        for sc in scores.values():
            for m in metric_names:
                raw[m].append(getattr(sc, m))

        # Min-max normalize (constant metrics → 0)
        norms = {}
        for m in metric_names:
            vals = np.array(raw[m], dtype=np.float64)
            lo, hi = vals.min(), vals.max()
            if hi - lo < 1e-15:
                norms[m] = np.zeros_like(vals)
            else:
                norms[m] = (vals - lo) / (hi - lo)

        # Assign composites
        methods = list(scores.keys())
        for i, method in enumerate(methods):
            composite = 0.0
            for m in metric_names:
                composite += self.WEIGHTS[m] * float(norms[m][i])
            scores[method].composite = float(composite)

        return scores

    # ── Auto-selection ─────────────────────────────────────────────────────

    def auto_select(
        self,
        signal: np.ndarray,
        window_seconds: float = 1.0,
        savgol_window: int = 501,
        poly_order: int = 3,
        event_sigma: float = 5.0,
        candidates: Optional[List[str]] = None,
    ) -> BaselineResult:
        """
        Run all candidate methods, score them, return the winner.

        Args:
            signal: 1-D array at self.fs.
            window_seconds: Rolling-median window.
            savgol_window / poly_order: Savitzky-Golay params.
            event_sigma: Threshold for event-preservation metric.
            candidates: Subset of {"constant","linear","rolling_median","savgol"}.

        Returns:
            BaselineResult of the winning method, with .score and
            .all_candidates filled for logging/metadata.
        """
        if candidates is None:
            candidates = ["constant", "linear", "rolling_median", "savgol"]

        generators = {
            "constant": lambda: self.constant(signal),
            "linear": lambda: self.linear(signal),
            "rolling_median": lambda: self.rolling_median(signal, window_seconds),
            "savgol": lambda: self.savgol(signal, savgol_window, poly_order),
        }

        results: Dict[str, BaselineResult] = {}
        scores: Dict[str, BaselineScore] = {}

        for name in candidates:
            if name not in generators:
                logger.warning("Unknown baseline candidate '%s' — skipping", name)
                continue
            try:
                res = generators[name]()
                sc = self._score(signal, res, event_sigma=event_sigma)
                res.score = sc
                results[name] = res
                scores[name] = sc
            except Exception as e:
                logger.warning("Baseline candidate '%s' failed: %s", name, e)

        if not scores:
            logger.error("All baseline candidates failed — falling back to constant")
            return self.constant(signal)

        scores = self._normalize_and_composite(scores)

        # Pick lowest composite
        winner_name = min(scores, key=lambda k: scores[k].composite)
        winner = results[winner_name]
        winner.all_candidates = scores

        logger.info(
            "Baseline auto-select → %s (composite=%.4f) | scores: %s",
            winner_name,
            scores[winner_name].composite,
            {k: f"{v.composite:.3f}" for k, v in scores.items()},
        )
        # Detailed per-metric log at DEBUG
        for name, sc in scores.items():
            logger.debug(
                "  %-16s var=%.4g smooth=%.4g acf1=%.4f evt_loss=%.4f recon=%.4g → comp=%.4f%s",
                name, sc.residual_variance, sc.baseline_smoothness,
                sc.residual_autocorr_lag1, sc.event_preservation_loss,
                sc.reconstruction_error, sc.composite,
                " ← WINNER" if name == winner_name else "",
            )

        return winner

    # ── Public dispatcher ──────────────────────────────────────────────────

    def estimate(
        self,
        signal: np.ndarray,
        method: str = "rolling_median",
        window_seconds: float = 1.0,
        savgol_window: int = 501,
        poly_order: int = 3,
        compute_range: bool = True,
        compute_residual: bool = True,
        event_sigma: float = 5.0,
        auto_candidates: Optional[List[str]] = None,
        **kwargs,
    ) -> BaselineResult:
        """
        Estimate baseline for one file.

        Args:
            method:
                "auto"            → objective auto-selection among candidates
                "rolling_median"  → fixed rolling median
                "constant"        → global mean
                "linear"          → linear trend
                "savgol"          → Savitzky-Golay
            window_seconds: Rolling-median window length.
            savgol_window / poly_order: Savitzky-Golay params.
            compute_range: If False, baseline_range is set to None.
            compute_residual: If False, residual_std is set to None.
            event_sigma: Sigma threshold for event-preservation scoring.
            auto_candidates: Methods to try when method="auto".

        Returns:
            BaselineResult (with .score and .all_candidates if method="auto").
        """
        if method == "auto":
            result = self.auto_select(
                signal,
                window_seconds=window_seconds,
                savgol_window=savgol_window,
                poly_order=poly_order,
                event_sigma=event_sigma,
                candidates=auto_candidates,
            )
        elif method == "constant":
            result = self.constant(signal)
        elif method == "linear":
            result = self.linear(signal)
        elif method == "savgol":
            result = self.savgol(signal, savgol_window, poly_order)
        elif method == "rolling_median":
            result = self.rolling_median(signal, window_seconds)
        else:
            raise ValueError(
                f"Unknown baseline method: {method!r}. "
                f"Use auto|constant|linear|rolling_median|savgol."
            )

        # Honour user flags for optional diagnostics
        if not compute_range:
            result.baseline_range = None
        if not compute_residual:
            result.residual_std = None

        return result