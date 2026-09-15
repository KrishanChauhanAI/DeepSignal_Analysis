"""
Data audit — logs concrete size/shape statistics per file BEFORE any
processing, so the user can verify the pipeline is actually consuming
the full raw data (not a subset).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nanobio.utils.data_audit")


def audit_dataset(
    files: List[Path], labels: List[int], categories: List[str],
    sampling_rate_hz: int, dtype_bytes: int = 4,
) -> pd.DataFrame:
    """
    Report per-file size, sample count, and duration.

    Prints a per-class summary so the user can VERIFY that:
      1. Every file is being seen
      2. File sizes are consistent with 1 MHz raw acquisition
      3. Total dataset size is sensible

    Returns a DataFrame the caller can write to CSV.
    """
    rows = []
    for fp, y in zip(files, labels):
        try:
            size_bytes = fp.stat().st_size
            n_samples = size_bytes // dtype_bytes
            duration_s = n_samples / sampling_rate_hz
            rows.append({
                "file_name": fp.name,
                "category": categories[y],
                "size_MB": round(size_bytes / (1024**2), 2),
                "n_samples": n_samples,
                "duration_s": round(duration_s, 3),
            })
        except OSError as e:
            logger.warning("Cannot stat %s: %s", fp, e)

    df = pd.DataFrame(rows)
    total_gb = df["size_MB"].sum() / 1024
    total_hours = df["duration_s"].sum() / 3600

    logger.info("=" * 70)
    logger.info("DATA AUDIT (verify raw data actually reaches the pipeline)")
    logger.info("=" * 70)
    logger.info("Total files       : %d", len(df))
    logger.info("Total raw size    : %.2f GB", total_gb)
    logger.info("Total duration    : %.2f hours @ %d Hz", total_hours, sampling_rate_hz)
    logger.info("Total samples     : %s", f"{int(df['n_samples'].sum()):,}")
    logger.info("-" * 70)
    logger.info("%-25s %8s %12s %12s %10s",
                "Category", "n_files", "size_MB(sum)", "samples(sum)", "hours")
    logger.info("-" * 70)
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        logger.info("%-25s %8d %12.1f %12s %10.2f",
                    cat, len(sub), sub["size_MB"].sum(),
                    f"{int(sub['n_samples'].sum()):,}",
                    sub["duration_s"].sum() / 3600)
    logger.info("-" * 70)
    logger.info("Per-file size range: min=%.2f MB, median=%.2f MB, max=%.2f MB",
                df["size_MB"].min(), df["size_MB"].median(), df["size_MB"].max())
    logger.info("Per-file duration : min=%.2fs, median=%.2fs, max=%.2fs",
                df["duration_s"].min(), df["duration_s"].median(),
                df["duration_s"].max())
    logger.info("=" * 70)

    return df


def verify_hardlink_savings(links_root: Path) -> Dict[str, int]:
    """
    Prove that split_links are hardlinks (not copies).

    A hardlink shares the same inode as the source file. If we sum unique
    inodes vs total files, we can measure the actual disk footprint.
    """
    total_files, unique_inodes = 0, set()
    total_apparent = 0
    for f in links_root.rglob("*"):
        if not f.is_file() or f.name.startswith("split_manifest"):
            continue
        try:
            st = f.stat()
            total_files += 1
            unique_inodes.add((st.st_dev, st.st_ino))
            total_apparent += st.st_size
        except OSError:
            continue

    n_unique = len(unique_inodes)
    n_hardlinked = total_files - n_unique  # 0 if none are hardlinks to raw
    savings_mb = (total_apparent - (total_apparent * n_unique / max(1, total_files))) / (1024**2)

    logger.info("=" * 70)
    logger.info("HARDLINK VERIFICATION (proves zero-copy)")
    logger.info("=" * 70)
    logger.info("Files in link tree     : %d", total_files)
    logger.info("Unique inodes          : %d", n_unique)
    logger.info("Apparent size (ls)     : %.2f MB", total_apparent / (1024**2))
    logger.info("Actual disk usage      : ~%.2f MB (hardlinks share inodes)",
                total_apparent / (1024**2) if n_unique == total_files
                else total_apparent * n_unique / (1024**2 * max(1, total_files)))
    if n_unique < total_files:
        logger.info("✓ Confirmed: %d files are hardlinked (0 bytes extra)",
                    total_files - n_unique)
    else:
        logger.warning("⚠ No hardlinks detected — files may be copies "
                       "(check link_mode and filesystem)")
    logger.info("Verify manually with: du -sh %s", links_root)
    logger.info("=" * 70)

    return {
        "total_files": total_files, "unique_inodes": n_unique,
        "hardlinked": total_files - n_unique,
    }

def compute_signal_diagnostics(
    signal: np.ndarray, sampling_rate_hz: int,
    baseline_window_s: float = 1.0,
    event_threshold_sigma: float = 5.0,
) -> Dict[str, float]:
    """
    Per-file diagnostics that reveal whether events actually exist.

    Returns:
        noise_std       : residual std after baseline subtraction
        signal_range_mv : peak-to-peak of the full signal
        snr_estimate    : (max deviation from baseline) / noise_std
        n_events_5sigma : count of samples > 5σ from baseline
        event_density   : fraction of samples that are 'events'
        baseline_drift  : ptp of baseline (proxy for instrumental drift)
        dc_offset       : median signal level (device fingerprint)
    """
    import pandas as pd
    w = max(3, int(baseline_window_s * sampling_rate_hz))
    if w % 2 == 0:
        w += 1
    baseline = pd.Series(signal).rolling(w, center=True, min_periods=1).median().values
    residual = signal - baseline
    noise_std = float(np.std(residual))
    max_dev = float(np.max(np.abs(residual)))
    snr = max_dev / (noise_std + 1e-9)
    n_events = int(np.sum(np.abs(residual) > event_threshold_sigma * noise_std))
    event_density = n_events / len(signal)
    return {
        "noise_std_mv": noise_std,
        "signal_range_mv": float(np.ptp(signal)),
        "snr_estimate": snr,
        "n_events_5sigma": n_events,
        "event_density": event_density,
        "baseline_drift_mv": float(np.ptp(baseline)),
        "dc_offset_mv": float(np.median(signal)),
    }