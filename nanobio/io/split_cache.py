"""
Split cache manager.

Materializes the logical train/val/test split as a browsable folder tree
containing one processed .npz file per source recording. Each .npz holds:

    signal_ds        — downsampled signal (model input rate)
    baseline         — per-file baseline (for visualization / reference)
    detrended        — signal − baseline (what the model consumes)
    segments         — (M, L) normalized segments for DL / ProtoNet
    phys_features    — (M, P) hybrid features for ProtoNet
    window_features  — (M, F) full handcrafted features for classical models
    label, category, original_path, md5, fs, fingerprint

Cache validity is tied to a fingerprint of the preprocessing config:
change model_input_rate / window / stride / baseline / detrend / feature
version and the cache is automatically rebuilt for every file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("nanobio.io.split_cache")

# Bump this whenever feature semantics change (e.g., the high_band bug fix).
FEATURE_VERSION = "v2.1-highband-fixed"


def config_fingerprint(
    model_input_rate_hz: int,
    segment_length: int,
    stride: int,
    normalize_per_segment: bool,
    baseline_method: str,
    baseline_window_s: float,
    detrend_method: str,
) -> str:
    """Stable hash of everything that affects cache contents."""
    raw = "|".join(map(str, [
        model_input_rate_hz, segment_length, stride, normalize_per_segment,
        baseline_method, baseline_window_s, detrend_method, FEATURE_VERSION,
    ]))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class SplitCacheManager:
    """
    Builds and loads the per-split .npz cache.

    Args:
        cache_root: e.g. data/split_cache.
        fingerprint: from config_fingerprint(); stored in every .npz.
    """

    def __init__(self, cache_root: Path, fingerprint: str):
        self.root = Path(cache_root)
        self.fingerprint = fingerprint
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ───────────────────────────────────────────────────────

    def _npz_path(self, split: str, category: str, file_path: Path) -> Path:
        d = self.root / split / category
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{file_path.stem}.npz"

    # ── Per-file processing (pure function → parallelizable) ───────────────

    @staticmethod
    def process_one(
        fp: Path,
        label: int,
        category: str,
        cfg_snapshot: dict,
        fingerprint: str,
    ) -> Optional[dict]:
        """
        Read ONE raw file and produce the full cache payload dict.
        Import-heavy dependencies are imported here so this function can
        run in worker processes.
        """
        try:
            import numpy as np

            from nanobio.features.handcrafted import (
                FeatureExtractor,
                HybridFeatureExtractor,
            )
            from nanobio.io.reader import read_signal_file
            from nanobio.signal.baseline import BaselineEstimator
            from nanobio.signal.detrend import Detrender
            from nanobio.signal.filters import SignalProcessor
            from nanobio.signal.quality import DataQualityChecker

            s = cfg_snapshot
            raw = read_signal_file(fp)

            qc = DataQualityChecker(
                sampling_rate_hz=s["original_rate_hz"],
                max_clip_fraction=s["max_clip_fraction"],
            )
            qr = qc.check(raw.reshape(-1, 1), str(fp))
            if not qr.passed:
                return {"status": "qc_fail", "file": fp.name, "errors": qr.errors}

            sp = SignalProcessor(s["original_rate_hz"], s["lpf_order"])
            fr = sp.downsample(raw, s["model_input_rate_hz"], "LPF")

            be = BaselineEstimator(s["model_input_rate_hz"])
            br = be.estimate(
                fr.signal, method=s["baseline_method"],
                window_seconds=s["baseline_window_s"],
            )
            det = Detrender.apply(
                fr.signal, s["detrend_method"], br,
            )

            L, S = s["segment_length"], s["stride"]
            fe_full = FeatureExtractor()
            fe_hyb = HybridFeatureExtractor()
            segs, phys, wins = [], [], []
            start = 0
            while start + L <= len(det):
                w = det[start:start + L].copy()
                if s["normalize_per_segment"]:
                    w = (w - w.mean()) / (w.std() + 1e-6)
                segs.append(w)
                phys.append(fe_hyb.extract(w))
                wins.append(fe_full.extract(w))
                start += S
            if not segs:
                return {"status": "no_windows", "file": fp.name}

            import hashlib as _h
            h = _h.md5()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)

            return {
                "status": "ok",
                "file": fp.name,
                "payload": dict(
                    signal_ds=fr.signal.astype(np.float32),
                    baseline=br.baseline.astype(np.float32),
                    detrended=det.astype(np.float32),
                    segments=np.asarray(segs, np.float32),
                    phys_features=np.asarray(phys, np.float32),
                    window_features=np.asarray(wins, np.float32),
                    label=np.int64(label),
                    category=np.str_(category),
                    original_path=np.str_(str(fp.resolve())),
                    md5=np.str_(h.hexdigest()),
                    fs=np.int64(s["model_input_rate_hz"]),
                    fingerprint=np.str_(fingerprint),
                ),
            }
        except Exception as e:
            logger.exception("Cache build failed for %s", fp)
            return {"status": "error", "file": fp.name, "errors": [str(e)]}

    # ── Build the whole tree ───────────────────────────────────────────────

    def build(
        self,
        splits: Dict[str, Tuple[List[Path], List[int]]],
        categories: List[str],
        cfg_snapshot: dict,
        n_workers: int = 1,
        rebuild: bool = False,
    ) -> Dict[str, dict]:
        """
        Process every source file and write <split>/<category>/<stem>.npz.

        Files whose .npz exists with the SAME fingerprint are skipped
        (instant on re-runs). With rebuild=True everything is reprocessed.
        """
        cfg_snapshot = dict(cfg_snapshot)
        cfg_snapshot["__fingerprint"] = self.fingerprint
        report = {"built": 0, "skipped": 0, "failed": 0}
        manifest_rows: List[dict] = []

        tasks, targets = [], []
        for split, (files, labels) in splits.items():
            for fp, y in zip(files, labels):
                out = self._npz_path(split, categories[y], fp)
                targets.append((split, out))
                if (not rebuild) and out.exists():
                    try:
                        with np.load(out, allow_pickle=False) as z:
                            if str(z["fingerprint"]) == self.fingerprint:
                                report["skipped"] += 1
                                manifest_rows.append({
                                    "split": split,
                                    "category": categories[y],
                                    "label": int(y),
                                    "file_name": fp.name,
                                    "file_path": str(fp.resolve()),
                                    "cache_path": str(out),
                                })
                                continue
                    except Exception:
                        pass  # corrupt cache → rebuild
                tasks.append((fp, y, categories[y], split, out))

        if tasks:
            if n_workers > 1:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_workers)(
                    delayed(self.process_one)(
                        fp, y, cat, cfg_snapshot, self.fingerprint,
                    ) for fp, y, cat, _, _ in tasks
                )
            else:
                results = [
                    self.process_one(fp, y, cat, cfg_snapshot, self.fingerprint)
                    for fp, y, cat, _, _ in tasks
                ]

            for (fp, y, cat, split, out), res in zip(tasks, results):
                if res is None or res["status"] != "ok":
                    reason = (res or {}).get("errors", ["unknown"])
                    logger.error("Cache FAIL %s (%s): %s", fp.name, (res or {}).get("status"), reason)
                    report["failed"] += 1
                    continue
                np.savez_compressed(out, **res["payload"])
                report["built"] += 1
                manifest_rows.append({
                    "split": split, "category": cat, "label": int(y),
                    "file_name": fp.name, "file_path": str(fp.resolve()),
                    "cache_path": str(out),
                })

        # Top-level manifest (visual + machine-readable)
        df = pd.DataFrame(manifest_rows).sort_values(["split", "category", "file_name"])
        df.to_csv(self.root / "split_manifest.csv", index=False)
        (self.root / "split_manifest.json").write_text(json.dumps({
            "created": datetime.now().isoformat(),
            "fingerprint": self.fingerprint,
            "counts": df["split"].value_counts().to_dict(),
            "files": manifest_rows,
        }, indent=2), encoding="utf-8")

        logger.info(
            "Split cache: built=%d skipped=%d failed=%d → %s",
            report["built"], report["skipped"], report["failed"], self.root,
        )
        return report

    # ── Load back for model training ───────────────────────────────────────

    def load_split(self, split: str) -> List[dict]:
        """Load every .npz of a split into the in-memory file cache format."""
        entries = []
        split_dir = self.root / split
        if not split_dir.exists():
            return entries
        for npz_path in sorted(split_dir.rglob("*.npz")):
            with np.load(npz_path, allow_pickle=False) as z:
                if str(z["fingerprint"]) != self.fingerprint:
                    logger.warning("Stale cache (fingerprint mismatch): %s", npz_path)
                    continue
                entries.append({
                    "path": str(z["original_path"]),
                    "file_name": npz_path.stem,
                    "label": int(z["label"]),
                    "segments": z["segments"],
                    "phys_features": z["phys_features"],
                    "window_features": z["window_features"],
                    "signal_ds": z["signal_ds"],
                    "baseline": z["baseline"],
                    "detrended": z["detrended"],
                })
        logger.info("Loaded %d cached files from split '%s'", len(entries), split)
        return entries

    # ── Optional: mirror raw files (symlink or copy) for browsing ──────────

    def link_raw_files(
        self,
        splits: Dict[str, Tuple[List[Path], List[int]]],
        categories: List[str],
        mode: str = "symlink",   # "symlink" | "copy"
    ) -> None:
        """
        Additionally place raw files (or links to them) under the split
        tree:  <split>/<category>/<original_name>.dat
        Useful for human browsing; NOT needed by the pipeline.
        """
        for split, (files, labels) in splits.items():
            for fp, y in zip(files, labels):
                d = self.root / split / categories[y]
                d.mkdir(parents=True, exist_ok=True)
                dst = d / fp.name
                if dst.exists():
                    continue
                if mode == "symlink":
                    try:
                        dst.symlink_to(fp.resolve())
                    except OSError:
                        logger.warning("Symlink failed (Windows?). Falling back to copy: %s", fp.name)
                        shutil.copy2(fp, dst)
                else:
                    shutil.copy2(fp, dst)