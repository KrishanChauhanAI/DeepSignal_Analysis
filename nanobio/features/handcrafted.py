"""
Handcrafted feature extraction.

Two extractors:
    - FeatureExtractor: full 36-feature vector for CLASSICAL models.
    - HybridFeatureExtractor: compact 13-feature vector for ProtoNet hybrid branch.

Deep learning models (ResNet1D, TCN, InceptionTime) receive RAW windows
directly and do not use these features. The Prototypical Network uses
the hybrid vector concatenated to its learned embedding.

Cleanup vs. legacy 43-feature list:
    - Removed 'variance' (duplicate of std²)
    - Removed 'manhattan_norm' (near-duplicate of manhattan_mean)
    - Removed 'peak_prominence' placeholder (was always 0.0)
    - Removed 'var_coeff' (unstable when mean ≈ 0 after detrending)
    - Fixed 'mean_prominence' by computing it properly
    - Added crest / shape / impulse factors (event-sensitive)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import entropy, kurtosis, skew


class FeatureExtractor:
    """Full physical feature vector for classical models."""

    FEATURE_NAMES: List[str] = [
        # Statistical (14)
        "mean", "median", "std", "skew", "kurtosis",
        "min", "max", "ptp", "q25", "q75", "rms", "iqr",
        "zero_cross", "abs_mean",
        # Spectral (8)
        "total_power", "max_psd", "dominant_freq",
        "low_band", "mid_band", "high_band",
        "spectral_entropy", "spectral_centroid",
        # Peaks (3)
        "num_peaks", "mean_peak", "mean_prominence",
        # Nonlinear (4)
        "entropy", "hurst", "hjorth_mobility", "hjorth_complexity",
        # Shape factors (3)
        "crest_factor", "shape_factor", "impulse_factor",
        # Autocorrelation (2)
        "autocorr_lag1", "autocorr_lag2",
        # Dynamics (4)
        "mean_diff", "std_diff", "tunnel_proxy", "trend",
    ]

    @staticmethod
    def _hjorth(x: np.ndarray) -> Tuple[float, float]:
        d1, d2 = np.diff(x), np.diff(np.diff(x))
        v0, v1, v2 = np.var(x), np.var(d1), np.var(d2)
        mob = float(np.sqrt(v1 / (v0 + 1e-10)))
        comp = float(np.sqrt(v2 / (v1 + 1e-10)) / (mob + 1e-10))
        return mob, comp

    @staticmethod
    def _hurst(x: np.ndarray) -> float:
        n = len(x)
        if n < 20:
            return 0.5
        cd = np.cumsum(x - np.mean(x))
        s = np.std(cd)
        return 0.5 if s == 0 else float(np.log(s) / np.log(n))

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract full feature vector."""
        x = np.asarray(window, dtype=np.float64).ravel()
        mean_val = float(np.mean(x))
        median_val = float(np.median(x))
        std_val = float(np.std(x))
        q25, q75 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
        abs_mean = float(np.mean(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak_abs = float(np.max(np.abs(x)))

        # Statistical (14)
        stats = [
            mean_val, median_val, std_val,
            float(skew(x)), float(kurtosis(x)),
            float(np.min(x)), float(np.max(x)), float(np.ptp(x)),
            q25, q75, rms, q75 - q25,
            float(np.sum(np.diff(np.sign(x)) != 0)), abs_mean,
        ]

        # Spectral (8)
        # BUG-003 FIX: rfftfreq returns values in [0, 0.5].
        # The old code used `freqs > 0.5` for high_band, which was
        # always empty (0.0). Now we partition [0, 0.5] into three
        # physically meaningful bands:
        #   low_band:  [0, 0.1]   → 0–10% of Nyquist
        #   mid_band:  (0.1, 0.35] → 10–35% of Nyquist
        #   high_band: (0.35, 0.5] → 35–50% of Nyquist (was always 0)
        psd = np.abs(rfft(x)) ** 2
        freqs = rfftfreq(len(x))
        tp = float(np.sum(psd)) + 1e-12
        dom = float(freqs[np.argmax(psd[1:]) + 1]) if len(psd) > 1 else 0.0
        pn = psd / tp
        se = float(-np.sum(pn * np.log(pn + 1e-12)))
        sc = float(np.sum(freqs * psd) / tp)
        spec = [
            float(tp), float(np.max(psd)), dom,
            float(np.sum(psd[freqs <= 0.1]) / tp),
            float(np.sum(psd[(freqs > 0.1) & (freqs <= 0.35)]) / tp),
            float(np.sum(psd[freqs > 0.35]) / tp),  # BUG-003 FIX: was > 0.5
            se, sc,
        ]

        # Peaks (3)
        peaks, _ = find_peaks(x)
        if len(peaks) > 0:
            prominences = peak_prominences(x, peaks)[0]
            peak_feats = [
                float(len(peaks)), float(np.mean(x[peaks])),
                float(np.mean(prominences)),
            ]
        else:
            peak_feats = [0.0, 0.0, 0.0]

        # Nonlinear (4)
        mob, comp = self._hjorth(x)
        hist = np.histogram(x, bins=20)[0] + 1e-10
        nonlin = [float(entropy(hist)), self._hurst(x), mob, comp]

        # Shape (3)
        shape = [
            peak_abs / (rms + 1e-10),
            rms / (abs_mean + 1e-10),
            peak_abs / (abs_mean + 1e-10),
        ]

        # Autocorrelation (2)
        if len(x) > 5:
            l1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            l2 = float(np.corrcoef(x[:-2], x[2:])[0, 1])
            l1 = 0.0 if np.isnan(l1) else l1
            l2 = 0.0 if np.isnan(l2) else l2
        else:
            l1 = l2 = 0.0

        # Dynamics (4)
        d1 = np.diff(x)
        trend = float(np.polyfit(np.arange(len(x)), x, 1)[0])
        dyn = [
            float(np.mean(np.abs(d1))), float(np.std(d1)),
            float(np.sum(np.abs(d1) > std_val)), trend,
        ]

        feats = stats + spec + peak_feats + nonlin + shape + [l1, l2] + dyn
        assert len(feats) == len(self.FEATURE_NAMES), (
            f"Feature count mismatch: {len(feats)} vs {len(self.FEATURE_NAMES)}"
        )
        return np.nan_to_num(np.array(feats, dtype=np.float32))


class HybridFeatureExtractor:
    """
    Compact 13-feature vector for the ProtoNet hybrid branch.

    Anchors the learned embedding to physically meaningful quantities.
    """

    NAMES: List[str] = [
        "std", "rms", "ptp", "skew", "iqr",
        "zero_cross_rate", "delta_i_proxy", "psd_high_frac",
        "crest_factor", "autocorr_lag1", "mean_diff",
        "tunnel_proxy", "spectral_entropy",
    ]

    def extract(self, window: np.ndarray) -> np.ndarray:
        """Extract compact hybrid vector."""
        x = np.asarray(window, dtype=np.float64).ravel()
        mean_val = float(np.mean(x))
        std_val = float(np.std(x)) + 1e-10
        abs_mean = float(np.mean(np.abs(x))) + 1e-10
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak_abs = float(np.max(np.abs(x)))
        q25, q75 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
        delta_i = float(np.max(np.abs(x - mean_val)) / std_val)
        zc = float(np.sum(np.diff(np.sign(x)) != 0) / max(1, len(x) - 1))

        psd = np.abs(rfft(x)) ** 2
        freqs = rfftfreq(len(x))
        tp = float(np.sum(psd)) + 1e-12
        pn = psd / tp
        se = float(-np.sum(pn * np.log(pn + 1e-12)))
        high = float(np.sum(psd[freqs > 0.3]) / tp)

        if len(x) > 5:
            l1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
            l1 = 0.0 if np.isnan(l1) else l1
        else:
            l1 = 0.0

        d1 = np.diff(x)
        crest = peak_abs / (rms + 1e-10)

        feats = [
            std_val, rms, float(np.ptp(x)), float(skew(x)), q75 - q25,
            zc, delta_i, high, crest, l1,
            float(np.mean(np.abs(d1))),
            float(np.sum(np.abs(d1) > std_val)),
            se,
        ]
        assert len(feats) == len(self.NAMES)
        return np.nan_to_num(np.array(feats, dtype=np.float32))