"""
Standalone data-quality report writer (Section 41).
Consolidates per-file QC results into a single CSV for auditing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from nanobio.signal.quality import QualityReport

logger = logging.getLogger("nanobio.utils.qc_report")


def write_qc_report(
    reports: List[QualityReport],
    output_path: Path,
) -> Path:
    """
    Serialize all QC reports into a flat CSV, one row per file.

    Columns include every field of QualityReport plus a computed
    `severity` column: 'error' | 'questionable' | 'ok'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in reports:
        severity = (
            "error" if not r.passed
            else "questionable" if r.questionable
            else "ok"
        )
        row = {
            "file_path": r.file_path,
            "severity": severity,
            "passed": r.passed,
            "num_samples": r.num_samples,
            "duration_seconds": r.duration_seconds,
            "has_nan": r.has_nan,
            "num_nan": r.num_nan,
            "has_inf": r.has_inf,
            "num_inf": r.num_inf,
            "is_clipped": r.is_clipped,
            "clip_fraction": r.clip_fraction,
            "signal_variance": r.signal_variance,
            "is_low_variance": r.is_low_variance,
            "is_saturated": r.is_saturated,
            "baseline_drift_fraction": r.baseline_drift_fraction,
            "impossible_value_count": r.impossible_value_count,
            "missing_sample_gaps": r.missing_sample_gaps,
            "excessive_noise": r.excessive_noise,
            "noise_std": r.noise_std,
            "sampling_rate_verified": r.sampling_rate_verified,
            "n_warnings": len(r.warnings),
            "n_errors": len(r.errors),
            "warnings": " | ".join(r.warnings) if r.warnings else "",
            "errors": " | ".join(r.errors) if r.errors else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    n_total = len(df)
    n_ok = (df["severity"] == "ok").sum()
    n_q = (df["severity"] == "questionable").sum()
    n_err = (df["severity"] == "error").sum()
    logger.info(
        "QC report → %s | %d ok, %d questionable, %d error out of %d files",
        output_path, n_ok, n_q, n_err, n_total,
    )
    return output_path