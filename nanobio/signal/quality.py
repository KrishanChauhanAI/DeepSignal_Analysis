"""
Automated quality control for raw signal files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple  # BUG-010 FIX: Added Tuple

import numpy as np

logger = logging.getLogger("nanobio.signal.quality")


@dataclass
class QualityReport:
    """Quality-check results for a single file."""
    file_path: str
    passed: bool = True
    num_samples: int = 0
    duration_seconds: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    num_nan: int = 0
    num_inf: int = 0
    is_clipped: bool = False
    clip_fraction: float = 0.0
    signal_variance: float = 0.0
    is_low_variance: bool = False
    is_saturated: bool = False
    baseline_drift_fraction: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # NEW — Section 41 additions
    sampling_rate_verified: bool = True
    impossible_value_count: int = 0
    missing_sample_gaps: int = 0
    excessive_noise: bool = False
    noise_std: float = 0.0
    corrupt_reason: str = ""
    questionable: bool = False       # true if warnings but not errors


class DataQualityChecker:
    """Runs quality checks on a raw signal."""

    def __init__(
        self,
        sampling_rate_hz: int = 1_000_000,
        max_clip_fraction: float = 0.05,
        min_variance: float = 1e-12,
        max_drift_fraction: float = 0.5,
        # NEW
        impossible_value_range: Tuple[float, float] = (-1e9, 1e9),
        max_noise_std_ratio: float = 0.5,   # noise_std / signal_range
        verify_sampling_rate: bool = True,
    ):
        self.sampling_rate = sampling_rate_hz
        self.max_clip = max_clip_fraction
        self.min_variance = min_variance
        self.max_drift = max_drift_fraction
        self.impossible_range = impossible_value_range
        self.max_noise_ratio = max_noise_std_ratio
        self.verify_rate = verify_sampling_rate

    def check(self, signal: np.ndarray, file_path: str) -> QualityReport:
        """Run all QC checks."""
        report = QualityReport(file_path=file_path)
        sig_1d = signal[:, 0] if signal.ndim > 1 else signal
        report.num_samples = len(sig_1d)
        report.duration_seconds = len(sig_1d) / self.sampling_rate

        nan_mask = np.isnan(sig_1d)
        inf_mask = np.isinf(sig_1d)
        report.num_nan = int(nan_mask.sum())
        report.num_inf = int(inf_mask.sum())
        report.has_nan = report.num_nan > 0
        report.has_inf = report.num_inf > 0
        if report.has_nan or report.has_inf:
            report.errors.append(
                f"NaN={report.num_nan} Inf={report.num_inf}"
            )

        clean = sig_1d[~nan_mask & ~inf_mask]
        if len(clean) == 0:
            report.passed = False
            report.errors.append("All samples invalid")
            return report

        # Variance
        report.signal_variance = float(np.var(clean))
        if report.signal_variance < self.min_variance:
            report.is_low_variance = True
            report.warnings.append("Very low variance")

        # Clipping via z-score
        std_val = float(np.std(clean))
        if std_val > 0:
            z = np.abs((clean - np.mean(clean)) / std_val)
            report.clip_fraction = float(np.mean(z > 4.0))
            if report.clip_fraction > self.max_clip:
                report.is_clipped = True
                report.errors.append(
                    f"Excessive clipping: {report.clip_fraction*100:.2f}%"
                )

        # Baseline drift
        if len(clean) > 1000:
            n_seg = 10
            seg_means = [
                float(np.mean(clean[i * len(clean) // n_seg:
                                     (i + 1) * len(clean) // n_seg]))
                for i in range(n_seg)
            ]
            drift_range = max(seg_means) - min(seg_means)
            full_range = np.ptp(clean)
            if full_range > 0:
                report.baseline_drift_fraction = float(drift_range / full_range)
                if report.baseline_drift_fraction > self.max_drift:
                    report.warnings.append(
                        f"Baseline drift {report.baseline_drift_fraction*100:.1f}%"
                    )

        # Saturation (constant runs)
        if len(clean) > 100:
            zero_diff_frac = float(np.sum(np.abs(np.diff(clean)) < 1e-15) / (len(clean) - 1))
            if zero_diff_frac > 0.1:
                report.is_saturated = True
                report.warnings.append(f"Saturation {zero_diff_frac*100:.1f}%")

        # NEW: Section 41 additional checks
        self._check_impossible_values(clean, report)
        self._check_excessive_noise(clean, report)
        self._check_missing_samples(sig_1d, report)

        # NEW: mark questionable if warnings exist but no errors
        report.passed = len(report.errors) == 0
        report.questionable = report.passed and len(report.warnings) > 0

        #report.passed = len(report.errors) == 0
        return report

    def _check_impossible_values(
        self, signal: np.ndarray, report: QualityReport,
    ) -> None:
        """Flag values outside the physically plausible range."""
        low, high = self.impossible_range
        mask = (signal < low) | (signal > high)
        n_bad = int(np.sum(mask))
        report.impossible_value_count = n_bad
        if n_bad > 0:
            frac = n_bad / len(signal)
            msg = (
                f"{n_bad} impossible values outside [{low:.2g}, {high:.2g}] "
                f"({frac * 100:.4f}%)"
            )
            if frac > 0.01:
                report.errors.append(msg)
            else:
                report.warnings.append(msg)

    def _check_excessive_noise(
        self, clean: np.ndarray, report: QualityReport,
    ) -> None:
        """Flag files where noise_std / signal_range exceeds the threshold."""
        if len(clean) < 100:
            return
        # Rough noise estimate: std of first-differences / sqrt(2)
        diff = np.diff(clean)
        noise_std = float(np.std(diff) / np.sqrt(2.0))
        report.noise_std = noise_std
        sig_range = float(np.ptp(clean))
        if sig_range > 0:
            ratio = noise_std / sig_range
            if ratio > self.max_noise_ratio:
                report.excessive_noise = True
                report.warnings.append(
                    f"Excessive noise: std/range={ratio:.3f} > {self.max_noise_ratio}"
                )

    def _check_missing_samples(
        self, signal: np.ndarray, report: QualityReport,
    ) -> None:
        """
        Detect large gaps that indicate dropped samples.

        Uses local variance: a stretch of identical consecutive values
        longer than a threshold suggests either saturation or a fill-in
        (dropped-sample replacement).
        """
        if len(signal) < 100:
            return
        # Count runs of identical adjacent samples
        diff_zero = np.diff(signal) == 0
        # Find runs
        gaps = 0
        run_len = 0
        max_run = int(self.sampling_rate * 0.001)  # 1 ms of identical values
        for is_zero in diff_zero:
            if is_zero:
                run_len += 1
                if run_len == max_run:
                    gaps += 1
            else:
                run_len = 0
        report.missing_sample_gaps = gaps
        if gaps > 0:
            report.warnings.append(
                f"{gaps} suspected missing-sample gap(s) "
                f"(runs > {max_run} identical values)"
            )