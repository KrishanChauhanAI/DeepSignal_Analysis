"""
Master pipeline orchestrator.

Wires all modules together and runs enabled phases.
"""

from __future__ import annotations

import gc
import json
import logging
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn  # NEW: needed for DL overfit gate
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,  # NEW: drop constant features before f_classif
    f_classif,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from nanobio.config import PipelineConfig
from nanobio.features import (
    FeatureExtractor,
    HybridFeatureExtractor,
)
from nanobio.io import (
    DataConverter,
    DatasetSplitter,
    read_signal_file,
)
from nanobio.models import (
    TCN,
    CNNBiLSTM,
    HierarchicalProtoNet,
    InceptionTime,
    ResNet1D,
    TransformerClassifier,
    build_classical_models,
)
from nanobio.signal import (
    BaselineEstimator,
    DataQualityChecker,
    Detrender,
    EventDetector,
    SignalProcessor,
)
from nanobio.training import (
    MetricEngine,
    ProtoNetTrainer,
    TorchClassifierTrainer,
)
from nanobio.viz import (
    BaselineOverlayPlotter,
    FilterComparisonPlotter,
    apply_default_style,
)

logger = logging.getLogger("nanobio.pipeline")

from nanobio.config import MLVizCfg  # NEW
from nanobio.features.event_features import EventFeatureExtractor
from nanobio.io.split_links import SplitLinkManager
from nanobio.utils.environment import (
    capture_environment,
    log_environment_summary,
    save_environment,
)
from nanobio.utils.experiment_registry import ExperimentRegistry
from nanobio.utils.feature_importance import extract_and_save_feature_importance
from nanobio.utils.logging_utils import OperationTimer
from nanobio.utils.qc_report import write_qc_report
from nanobio.viz.analysis_plots import SignalAnalysisPlotter
from nanobio.viz.comprehensive_analysis import ComprehensivePlotter
from nanobio.viz.ml_plots import MLPlotter

# NEW: SOTA imports (optional deps — graceful degradation inside adapters)
try:
    from nanobio.models.timesnet import TimesNetClassifier
    _TIMESNET_AVAILABLE = True
except ImportError:
    _TIMESNET_AVAILABLE = False
    logger.warning("TimesNet unavailable — nanobio/models/timesnet.py missing")

try:
    from nanobio.models.sota_adapters import (
        fit_predict_minirocket,
        fit_predict_tabpfn,
    )
    _SOTA_ADAPTERS_AVAILABLE = True
except ImportError:
    _SOTA_ADAPTERS_AVAILABLE = False
    logger.warning("SOTA adapters unavailable — nanobio/models/sota_adapters.py missing")


# ────────────────────────────────────────────────────────────────────────────
# Minimal PyTorch dataset for raw windows
# ────────────────────────────────────────────────────────────────────────────

class RawWindowDataset(Dataset):
    """Dataset of raw (windows, labels) for baseline DL classifiers."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T) → (N, 1, T)
        self.X = torch.from_numpy(X[:, None, :].astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.labels = y  # for sampler indexing

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ────────────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────────────

class NanoBioPipeline:
    """Master orchestrator."""

    def __init__(self, config: PipelineConfig):
        config.resolve_run_flags()
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"exp_{self.timestamp}"

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        apply_default_style(config.visualization)

        # Directories
        root = Path(config.project_root)
        self.dirs = {
            "figures": root / "figures" / self.experiment_id,
            "reports": root / "reports" / self.experiment_id,
            "models": root / "models" / self.experiment_id,
            "logs": root / "logs",
            "outputs": root / "outputs" / self.experiment_id,
            "converted_csv": root / "data" / "converted" / "csv",
            "converted_txt": root / "data" / "converted" / "txt",
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        try:
            config.to_yaml(root / "configs" / f"{self.experiment_id}.yaml")
        except Exception:
            pass

        # NEW: Section 37 — capture and save environment
        if getattr(config, "tracking", None) and config.tracking.save_environment:
            try:
                env = capture_environment(self.experiment_id, config.to_dict())
                env_path = self.dirs["outputs"] / f"{self.timestamp}_environment.json"
                save_environment(env, env_path)
                log_environment_summary(env)
            except Exception as e:
                logger.warning("Environment capture failed: %s", e)

        # NEW: Section 43 — register this experiment in the local registry
        self.registry = None
        if getattr(config, "tracking", None) and config.tracking.enabled:
            try:
                self.registry = ExperimentRegistry(Path(config.project_root))
                self.registry.register(
                    experiment_id=self.experiment_id,
                    config=config.to_dict(),
                    artifact_paths={
                        "figures": str(self.dirs["figures"]),
                        "reports": str(self.dirs["reports"]),
                        "models": str(self.dirs["models"]),
                        "outputs": str(self.dirs["outputs"]),
                    },
                )
            except Exception as e:
                logger.warning("Experiment registration failed: %s", e)

        # NEW: Section 41 — track QC reports for later CSV export
        self._qc_reports: List = []
        # NEW: Section 37 — track per-file preprocessing metadata
        self._per_file_preprocessing: List[Dict] = []

        # Components
        self.signal_processor = SignalProcessor(
            config.sampling.original_rate_hz, 
            config.filters.lpf_order,
        )
        self.baseline_estimator = BaselineEstimator(
            config.sampling.model_input_rate_hz,
        )
        self.detrender = Detrender()
        """self.qc = DataQualityChecker(
            sampling_rate_hz=config.sampling.original_rate_hz,
            max_clip_fraction=config.quality.max_clip_fraction,
            min_variance=config.quality.min_variance,
            max_drift_fraction=config.quality.max_baseline_drift_fraction,
        )"""
        self.qc = DataQualityChecker(
            sampling_rate_hz=config.sampling.original_rate_hz,
            max_clip_fraction=config.quality.max_clip_fraction,
            min_variance=config.quality.min_variance,
            max_drift_fraction=config.quality.max_baseline_drift_fraction,
            # NEW
            impossible_value_range=(
                getattr(config.quality, "impossible_value_min", -1e9),
                getattr(config.quality, "impossible_value_max", 1e9),
            ),
            max_noise_std_ratio=getattr(config.quality, "max_noise_std_ratio", 0.5),
            verify_sampling_rate=getattr(config.quality, "verify_sampling_rate", True),
        )
        self.event_detector = EventDetector(config.events, 
                                            config.sampling.model_input_rate_hz
        )

        # NEW: event-level feature extraction
        self.event_extractor = EventFeatureExtractor(self.event_detector)

        self.splitter = DatasetSplitter(config.split)
        self.feature_extractor = FeatureExtractor()
        self.hybrid_extractor = HybridFeatureExtractor()

        self.baseline_plotter = BaselineOverlayPlotter(
            output_dir=self.dirs["figures"],
            timestamp=self.timestamp,
            dpi=config.visualization.dpi,
            fig_format=config.visualization.fig_format,
            signal_color=config.visualization.signal_color,
            baseline_color=config.visualization.baseline_color,
        )
        self.filter_plotter = FilterComparisonPlotter(
            output_dir=self.dirs["figures"],
            timestamp=self.timestamp,
            dpi=config.visualization.dpi,
        )
        # NEW: Section 24 / 32 ML plotter
        self.ml_plotter = MLPlotter(
            output_dir=self.dirs["figures"],
            timestamp=self.timestamp,
            dpi=getattr(config.visualization, "dpi", 150),
        )

        # State
        self.categories: List[str] = []
        self.label_map: Dict[str, int] = {}
        self.splits = None
        self.classical_data = {}
        self.dl_data = {}
        self.proto_cache = {"train": [], "val": [], "test": []}
        self.results: Dict[str, dict] = {}

        # --- NEW: Initialize the in-memory memo cache ---
        self._prep_memo: Dict[str, tuple] = {}

        # NEW: split link manager (populated in phase1 when links_enabled)
        self.link_mgr = None
        self.split_link_root = None

        # NEW: DL overfit gate result — cached across phases so we don't retest
        # for both baseline DL and TimesNet in the same run
        self._dl_gate_passed: bool = None  # None = not tested yet

        # File handler
        fh = logging.FileHandler(self.dirs["logs"] / f"{self.experiment_id}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
        ))
        logger.addHandler(fh)
        logger.info("Init exp=%s device=%s mode=%s",
                    self.experiment_id, self.device, config.models.run_mode)

    # ────────────────────────────────────────────────────────────────────
    # Phase 1: discover files, run QC, split
    # ────────────────────────────────────────────────────────────────────
    """
    def phase1_data_integrity(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 1: DATA INTEGRITY + FILE-LEVEL SPLIT")
        logger.info("=" * 70)

        root = Path(self.config.data_root)
        self.categories = sorted([
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if len(self.categories) < 2:
            raise RuntimeError(f"Need ≥2 category folders in {root}")
        self.label_map = {c: i for i, c in enumerate(self.categories)}

        # Optional data conversion
        if self.config.data_conversion:
            converter = DataConverter(
                csv_output_dir=self.dirs["converted_csv"],
                txt_output_dir=self.dirs["converted_txt"],
                sampling_rate_hz=self.config.sampling.original_rate_hz,
            )
            converter.convert_all(root, extensions=self.config.supported_extensions)

        # Discover files
        files, labels = [], []
        for c in self.categories:
            for ext in self.config.supported_extensions:
                for fp in sorted((root / c).glob(f"*{ext}")):
                    files.append(fp)
                    labels.append(self.label_map[c])

        logger.info(
            "Discovered %d files | class distribution: %s",
            len(files),
            dict(Counter(self.categories[l] for l in labels)),
        )

        self.splits = self.splitter.split(files, labels, self.categories)

        # --- Zero-copy browsable split tree (Hardlinks / Symlinks) ---
        self.split_link_root = None

        # Safely get configs (in case config.py hasn't been fully updated yet)
        links_enabled = getattr(self.config.split, "links_enabled", True)
        if links_enabled:
            links_dir = getattr(self.config.split, "links_dir", "") or str(
                Path(self.config.data_root).parent / "split_links"
            )
            self.link_mgr = SplitLinkManager(
                Path(links_dir),
                mode=getattr(self.config.split, "link_mode", "auto"),
                verify_md5=getattr(self.config.split, "verify_md5", False),
            )
            rep = self.link_mgr.build(
                splits={
                    "train": (self.splits.train_files, self.splits.train_labels),
                    "val":   (self.splits.val_files,   self.splits.val_labels),
                    "test":  (self.splits.test_files,  self.splits.test_labels),
                },
                categories=self.categories,
                ratios=(self.config.split.train_ratio,
                        self.config.split.val_ratio,
                        self.config.split.test_ratio),
                seed=self.config.split.seed,
                purge_on_change=getattr(self.config.split, "purge_on_change", True),
            )
            vrep = self.link_mgr.verify()
            self.split_link_root = Path(links_dir)
            logger.info("Manifest: %s", self.split_link_root / "split_manifest.csv")
    """
    def phase1_data_integrity(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 1: DATA INTEGRITY + FILE-LEVEL SPLIT")
        logger.info("=" * 70)

        root = Path(self.config.data_root)
        self.categories = sorted([
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if len(self.categories) < 2:
            raise RuntimeError(f"Need ≥2 category folders in {root}")
        self.label_map = {c: i for i, c in enumerate(self.categories)}

        # Optional data conversion
        if self.config.data_conversion:
            converter = DataConverter(
                csv_output_dir=self.dirs["converted_csv"],
                txt_output_dir=self.dirs["converted_txt"],
                sampling_rate_hz=self.config.sampling.original_rate_hz,
            )
            converter.convert_all(root, extensions=self.config.supported_extensions)

        # Discover files
        files, labels = [], []
        for c in self.categories:
            for ext in self.config.supported_extensions:
                for fp in sorted((root / c).glob(f"*{ext}")):
                    files.append(fp)
                    labels.append(self.label_map[c])

        logger.info(
            "Discovered %d files | class distribution: %s",
            len(files),
            dict(Counter(self.categories[l] for l in labels)),
        )

        # NEW: Section 29 — compute and report class imbalance ratio
        class_counts = Counter(self.categories[l] for l in labels)
        if len(class_counts) > 0:
            counts_arr = np.array(list(class_counts.values()))
            imbalance_ratio = float(counts_arr.max() / max(1, counts_arr.min()))
            strategy = "none"
            if self.config.training.use_class_weights and self.config.training.use_weighted_sampler:
                strategy = "class_weights + weighted_sampler"
            elif self.config.training.use_class_weights:
                strategy = "class_weights"
            elif self.config.training.use_weighted_sampler:
                strategy = "weighted_sampler"
            loss_type = getattr(self.config.training, "loss_type", "cross_entropy")
            logger.info(
                "Class imbalance: ratio=%.2f (max/min) | strategy=%s | loss=%s",
                imbalance_ratio, strategy, loss_type,
            )
            if imbalance_ratio > 3.0 and strategy == "none":
                logger.warning(
                    "⚠ Imbalance ratio > 3.0 but no rebalancing strategy configured. "
                    "Consider use_class_weights=true or loss_type=focal."
                )

        # ── NEW: DATA AUDIT — prove every file is seen with real sizes ──
        try:
            from nanobio.utils.data_audit import audit_dataset, verify_hardlink_savings
            audit_df = audit_dataset(
                files, labels, self.categories,
                sampling_rate_hz=self.config.sampling.original_rate_hz,
            )
            audit_path = self.dirs["reports"] / f"{self.timestamp}_data_audit.csv"
            audit_df.to_csv(audit_path, index=False)
            logger.info("Data audit CSV saved: %s", audit_path)
        except ImportError:
            # Fallback inline audit if utils module not yet created
            logger.warning("nanobio.utils.data_audit not found — running inline audit")
            audit_rows = []
            for fp, y in zip(files, labels):
                try:
                    size_b = fp.stat().st_size
                    n_samp = size_b // 4  # float32
                    audit_rows.append({
                        "file_name": fp.name,
                        "category": self.categories[y],
                        "size_MB": round(size_b / (1024 ** 2), 2),
                        "n_samples": n_samp,
                        "duration_s": round(n_samp / self.config.sampling.original_rate_hz, 3),
                    })
                except OSError as e:
                    logger.warning("Cannot stat %s: %s", fp, e)
            if audit_rows:
                audit_df = pd.DataFrame(audit_rows)
                total_gb = audit_df["size_MB"].sum() / 1024
                total_hrs = audit_df["duration_s"].sum() / 3600
                logger.info("=" * 70)
                logger.info("DATA AUDIT (inline)")
                logger.info("=" * 70)
                logger.info("Total files    : %d", len(audit_df))
                logger.info("Total raw size : %.2f GB", total_gb)
                logger.info("Total duration : %.2f hours @ %d Hz",
                            total_hrs, self.config.sampling.original_rate_hz)
                logger.info("Total samples  : %s", f"{int(audit_df['n_samples'].sum()):,}")
                logger.info("-" * 70)
                for cat in sorted(audit_df["category"].unique()):
                    sub = audit_df[audit_df["category"] == cat]
                    logger.info(
                        "%-25s n=%3d  size=%.1f MB  samples=%s  hours=%.2f",
                        cat, len(sub), sub["size_MB"].sum(),
                        f"{int(sub['n_samples'].sum()):,}",
                        sub["duration_s"].sum() / 3600,
                    )
                logger.info(
                    "Per-file size: min=%.2f MB  median=%.2f MB  max=%.2f MB",
                    audit_df["size_MB"].min(),
                    audit_df["size_MB"].median(),
                    audit_df["size_MB"].max(),
                )
                logger.info(
                    "Per-file duration: min=%.2fs  median=%.2fs  max=%.2fs",
                    audit_df["duration_s"].min(),
                    audit_df["duration_s"].median(),
                    audit_df["duration_s"].max(),
                )
                logger.info("=" * 70)
                audit_path = self.dirs["reports"] / f"{self.timestamp}_data_audit.csv"
                audit_df.to_csv(audit_path, index=False)
                logger.info("Data audit CSV saved: %s", audit_path)

        # ── NEW: Group-leakage warning (device/session IDs in filenames) ──
        import re
        device_pattern = re.compile(r"([A-Za-z]{1,5}\d{1,3}[_-][A-Za-z0-9]+)")
        devices = set()
        for fp in files:
            m = device_pattern.search(fp.name)
            if m:
                devices.add(m.group(1))
        if devices and len(devices) < len(files) * 0.5:
            logger.warning("=" * 70)
            logger.warning(
                "⚠ POSSIBLE GROUP LEAKAGE: %d files span only %d device/session IDs",
                len(files), len(devices),
            )
            logger.warning(
                "Files from the same device may appear in BOTH train and test → "
                "file-level accuracy (e.g. 95%%) may be inflated."
            )
            logger.warning(
                "Enable group_aware=true with group_pattern=%r in config.",
                device_pattern.pattern,
            )
            logger.warning("Detected device IDs (sample): %s",
                           list(sorted(devices))[:10])
            logger.warning("=" * 70)

        # File-level stratified split
        self.splits = self.splitter.split(files, labels, self.categories)

        # ── Zero-copy browsable split tree (Hardlinks / Symlinks) ──
        self.split_link_root = None

        links_enabled = getattr(self.config.split, "links_enabled", True)
        if links_enabled:
            links_dir = getattr(self.config.split, "links_dir", "") or str(
                Path(self.config.data_root).parent / "split_links"
            )
            self.link_mgr = SplitLinkManager(
                Path(links_dir),
                mode=getattr(self.config.split, "link_mode", "auto"),
                verify_md5=getattr(self.config.split, "verify_md5", False),
            )

            # NEW: Force purge the split tree BEFORE building, if requested.
            # This ensures no stale files from previous runs contaminate the
            # new split (important because splits are random/seeded and
            # re-runs may produce different assignments).
            if getattr(self.config.split, "force_purge_split_tree", False):
                logger.info(
                    "force_purge_split_tree=True → wiping train/val/test tree "
                    "before rebuild"
                )
                self.link_mgr.force_purge()
            rep = self.link_mgr.build(
                splits={
                    "train": (self.splits.train_files, self.splits.train_labels),
                    "val":   (self.splits.val_files,   self.splits.val_labels),
                    "test":  (self.splits.test_files,  self.splits.test_labels),
                },
                categories=self.categories,
                ratios=(self.config.split.train_ratio,
                        self.config.split.val_ratio,
                        self.config.split.test_ratio),
                seed=self.config.split.seed,
                purge_on_change=getattr(self.config.split, "purge_on_change", True),
            )
            vrep = self.link_mgr.verify()
            self.split_link_root = Path(links_dir)
            logger.info("Manifest: %s", self.split_link_root / "split_manifest.csv")

            # ── NEW: Prove hardlinks are zero-copy (not full copies) ──
            try:
                from nanobio.utils.data_audit import verify_hardlink_savings
                verify_hardlink_savings(self.split_link_root)
            except ImportError:
                # Inline hardlink verification
                total_files, unique_inodes = 0, set()
                total_apparent = 0
                for f in self.split_link_root.rglob("*"):
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
                logger.info("=" * 70)
                logger.info("HARDLINK VERIFICATION (proves zero-copy)")
                logger.info("=" * 70)
                logger.info("Files in link tree  : %d", total_files)
                logger.info("Unique inodes       : %d", n_unique)
                logger.info("Apparent size (ls)  : %.2f MB", total_apparent / (1024 ** 2))
                if n_unique < total_files:
                    logger.info(
                        "✓ Confirmed: %d of %d files share inodes (hardlinked, 0 bytes extra)",
                        total_files - n_unique, total_files,
                    )
                elif total_files > 0:
                    logger.warning(
                        "⚠ No shared inodes detected — files may be COPIES "
                        "(check link_mode and filesystem). Run: du -sh %s",
                        self.split_link_root,
                    )
                logger.info("Verify disk usage manually: du -sh %s", self.split_link_root)
                logger.info("=" * 70)
        # NEW: Section 41 — write standalone QC report CSV (after all files touched)
        # Note: some files may not yet be QC-checked here if only Phase 2/3 loads them.
        # We write again at the end of Phase 3 for completeness.
        if self._qc_reports and getattr(self.config.quality, "save_qc_report", True):
            qc_path = self.dirs["reports"] / f"{self.timestamp}_qc_report_phase1.csv"
            write_qc_report(self._qc_reports, qc_path)


    # ────────────────────────────────────────────────────────────────────
    # Shared preprocessing per file
    # ────────────────────────────────────────────────────────────────────
    '''
    def _load_and_prepare(self, fp: Path):
        """Load raw signal, downsample to model rate, estimate baseline, detrend."""
        # --- Check memory cache to prevent double-reading the same 1 MHz file ---
        key = str(fp.resolve())
        if key in self._prep_memo:
            return self._prep_memo[key]

        raw = read_signal_file(fp)
        qr = self.qc.check(raw.reshape(-1, 1), str(fp))
        if not qr.passed:
            raise ValueError(f"QC failed: {qr.errors}")

        fr = self.signal_processor.downsample(
            raw, self.config.sampling.model_input_rate_hz, "LPF",
        )
        """br = self.baseline_estimator.estimate(
            fr.signal, method=self.config.baseline.method,
            window_seconds=self.config.baseline.window_seconds,
        )"""
        """br = self.baseline_estimator.estimate(
            fr.signal, method=self.config.baseline.method,
            window_seconds=self.config.baseline.window_seconds,
            compute_range=self.config.baseline.compute_baseline_range,
            compute_residual=self.config.baseline.compute_residual_std,
        )"""
        br = self.baseline_estimator.estimate(
            fr.signal,
            method=self.config.baseline.method,
            window_seconds=self.config.baseline.window_seconds,
            savgol_window=getattr(self.config.baseline, "savgol_window", 501),
            poly_order=getattr(self.config.baseline, "poly_order", 3),
            compute_range=getattr(self.config.baseline, "compute_baseline_range", True),
            compute_residual=getattr(self.config.baseline, "compute_residual_std", True),
            event_sigma=getattr(self.config.baseline, "event_sigma", 5.0),
            auto_candidates=getattr(
                self.config.baseline, "auto_candidates",
                ["constant", "linear", "rolling_median", "savgol"],
            ),
        )

        # Log auto-selection result once per file (if auto)
        if (self.config.baseline.method == "auto"
                and getattr(self.config.baseline, "log_selection_details", True)
                and br.all_candidates is not None):
            logger.info(
                "  %s baseline → %s (composite=%.4f)",
                fp.name, br.method, br.score.composite if br.score else -1,
            )
        detrended = self.detrender.apply(
            fr.signal,
            self.config.detrend.method if self.config.detrend.enabled else "none",
            br,
        )

        # --- Save to memory cache before returning ---
        result = (fr.signal, detrended, br)
        self._prep_memo[key] = result
        return result
    '''

    def _load_and_prepare(self, fp: Path):
        """Load raw signal, downsample to model rate, estimate baseline, detrend."""
        # --- Check memory cache to prevent double-reading the same 1 MHz file ---
        key = str(fp.resolve())
        if key in self._prep_memo:
            return self._prep_memo[key]

        raw = read_signal_file(fp)
        qr = self.qc.check(raw.reshape(-1, 1), str(fp))
        # NEW: Section 41 — collect QC results for standalone report
        self._qc_reports.append(qr)

        if qr.questionable:
            logger.warning(
                "Questionable file %s: %s",
                Path(fp).name, "; ".join(qr.warnings[:3]),
            )
        if not qr.passed:
            raise ValueError(f"QC failed: {qr.errors}")

        fr = self.signal_processor.downsample(
            raw, self.config.sampling.model_input_rate_hz, "LPF",
        )

        br = self.baseline_estimator.estimate(
            fr.signal,
            method=self.config.baseline.method,
            window_seconds=self.config.baseline.window_seconds,
            savgol_window=getattr(
                self.config.baseline, "savgol_window", 501
            ),
            poly_order=getattr(
                self.config.baseline, "poly_order", 3
            ),
            compute_range=getattr(
                self.config.baseline, "compute_baseline_range", True
            ),
            compute_residual=getattr(
                self.config.baseline, "compute_residual_std", True
            ),
            event_sigma=getattr(
                self.config.baseline, "event_sigma", 5.0
            ),
            auto_candidates=getattr(
                self.config.baseline,
                "auto_candidates",
                ["constant", "linear", "rolling_median", "savgol"],
            ),
        )

        # Log auto-selection result once per file (if auto)
        if (
            self.config.baseline.method == "auto"
            and getattr(
                self.config.baseline, "log_selection_details", True
            )
            and br.all_candidates is not None
        ):
            logger.info(
                "  %s baseline → %s (composite=%.4f)",
                fp.name,
                br.method,
                br.score.composite if br.score else -1,
            )

        # Attach selection metadata for later reporting
        if not hasattr(self, "_baseline_choices"):
            self._baseline_choices = []

        self._baseline_choices.append({
            "file": fp.name,
            "method": br.method,
            "composite": br.score.composite if br.score else None,
            "all_scores": (
                {k: v.to_dict() for k, v in br.all_candidates.items()}
                if br.all_candidates else None
            ),
        })

        detrended = self.detrender.apply(
            fr.signal,
            self.config.detrend.method if self.config.detrend.enabled else "none",
            br,
        )

        # NEW: Section 37 — persist per-file preprocessing metadata
        if getattr(self.config, "tracking", None) and self.config.tracking.save_per_file_preprocessing:
            self._per_file_preprocessing.append({
                "file": str(fp),
                "sampling_rate_hz": self.config.sampling.model_input_rate_hz,
                "filter_type": "LPF",
                "filter_impl": fr.filter_impl,
                "filter_order": fr.order,
                "filter_cutoff_hz": fr.cutoff_hz,
                "decimation_factor": fr.decimation_factor,
                "baseline_method": br.method,
                "baseline_composite_score": (
                    br.score.composite if getattr(br, "score", None) else None
                ),
                "detrend_method": (
                    self.config.detrend.method
                    if self.config.detrend.enabled else "none"
                ),
                "qc_passed": True,  # only reach here if passed
                "qc_questionable": self._qc_reports[-1].questionable
                    if self._qc_reports else False,
            })

        # --- Save to memory cache before returning ---
        result = (fr.signal, detrended, br)
        self._prep_memo[key] = result
        return result

    def _segment_and_features(self, detrended: np.ndarray):
        L = self.config.segment.segment_length_samples
        S = self.config.segment.stride_samples
        segs, phys, feats = [], [], []
        start = 0
        while start + L <= len(detrended):
            w = detrended[start : start + L].copy()
            if self.config.segment.normalize_per_segment:
                w = (w - w.mean()) / (w.std() + 1e-6)
            segs.append(w)
            phys.append(self.hybrid_extractor.extract(w))
            feats.append(self.feature_extractor.extract(w))
            start += S
        if not segs:
            return (
                np.zeros((0, L), np.float32),
                np.zeros((0, len(HybridFeatureExtractor.NAMES)), np.float32),
                np.zeros((0, len(FeatureExtractor.FEATURE_NAMES)), np.float32),
            )
        return (
            np.asarray(segs, np.float32),
            np.asarray(phys, np.float32),
            np.asarray(feats, np.float32),
        )

    def _save_baseline_selection_report(self) -> None:
        """Save per-file baseline method selection metadata for later reporting."""
        if not getattr(self, "_baseline_choices", None):
            return

        import json

        path = (
            self.dirs["reports"]
            / f"{self.timestamp}_baseline_selection.json"
        )

        path.write_text(
            json.dumps(self._baseline_choices, indent=2),
            encoding="utf-8",
        )

        logger.info("Baseline selection report: %s", path)


    # ────────────────────────────────────────────────────────────────────
    # Phase 2: per-file baseline visualization
    # ────────────────────────────────────────────────────────────────────

    def phase2_baseline_visualization(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 2: PER-FILE BASELINE VISUALIZATION")
        logger.info("=" * 70)
        if not self.config.baseline.save_visualizations:
            logger.info("Baseline viz disabled")
            return

        items, per_class = [], {c: 0 for c in self.categories}
        for fp, y in zip(self.splits.train_files, self.splits.train_labels):
            c = self.categories[y]
            if per_class[c] >= 3:
                continue
            try:
                sig, _, br = self._load_and_prepare(fp)
                items.append({
                    "signal": sig,
                    "baseline_result": br,
                    "sampling_rate_hz": self.config.sampling.model_input_rate_hz,
                    "file_name": fp.name,
                    "category": c,
                    "max_display_seconds": 20.0,
                })
                per_class[c] += 1
            except Exception as e:
                logger.warning("Baseline viz failed for %s: %s", fp.name, e)
        self.baseline_plotter.plot_batch(items, self.config.baseline.max_files_visualized)

        # Save automatic baseline-selection metadata after all
        # train/validation/test files have been processed.
        self._save_baseline_selection_report()

    # ────────────────────────────────────────────────────────────────────
    # Phase 3: build model-ready caches
    # ────────────────────────────────────────────────────────────────────

    def phase3_build_caches(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 3: BUILD MODEL CACHES")
        logger.info("=" * 70)

        need_classical = self.config.models.classical_enabled
        # NEW: DL cache is also needed by SOTA (TimesNet + MiniRocket)
        need_sota = getattr(self.config.models, "sota_enabled", False)
        need_dl = self.config.models.baseline_dl_enabled or need_sota
        need_proto = self.config.models.protonet_enabled

        for split_name, files, labels in [
            ("train", self.splits.train_files, self.splits.train_labels),
            ("val", self.splits.val_files, self.splits.val_labels),
            ("test", self.splits.test_files, self.splits.test_labels),
        ]:
            X_feat, y_feat = [], []
            fid_feat = []  # NEW: per-window file index for file-level aggregation
            X_raw, y_raw = [], []
            fid_raw = []   # NEW: per-window file index (DL / SOTA)
            proto_files = []

            for file_idx, (fp, y) in enumerate(zip(files, labels)):  # NEW: file_idx
                try:
                    '''
                    _, detrended, _ = self._load_and_prepare(fp)
                    segs, phys, feats = self._segment_and_features(detrended)
                    if len(segs) == 0:
                        continue

                    if need_classical:
                        for f in feats:
                            X_feat.append(f)
                            y_feat.append(y)
                            fid_feat.append(file_idx)  # NEW
                    '''
                    sig, detrended, br = self._load_and_prepare(fp)

                    segs, phys, feats = self._segment_and_features(detrended)

                    if len(segs) == 0:
                        continue

                    # --- NEW: S9, S13, S15 Event Features ---
                    file_event_feats = self.event_extractor.extract_file_level_features(
                        signal=sig,
                        fs=self.config.sampling.model_input_rate_hz,
                        baseline=br.baseline,
                    )

                    # Validate event-feature dimensionality before concatenation
                    expected_event_features = len(EventFeatureExtractor.get_feature_names())

                    if len(file_event_feats) != expected_event_features:
                        raise ValueError(
                            f"Event feature dimension mismatch: "
                            f"got {len(file_event_feats)}, "
                            f"expected {expected_event_features}"
                        )

                    if need_classical:
                        for f in feats:
                            # Concatenate window-level features with the
                            # file-level event features.
                            combined = np.concatenate([f, file_event_feats])

                            X_feat.append(combined)
                            y_feat.append(y)
                            fid_feat.append(file_idx)
                    if need_dl:
                        for s in segs:
                            X_raw.append(s)
                            y_raw.append(y)
                            fid_raw.append(file_idx)   # NEW
                    if need_proto:
                        proto_files.append({
                            "path": str(fp), "label": int(y),
                            "segments": segs, "phys_features": phys,
                        })
                except Exception as e:
                    logger.warning("Skipping %s: %s", fp.name, e)

            if need_classical:
                # NEW: cache stores 3-tuple (X, y, file_ids). Backwards compatible
                # via `_get_classical_split()` helper below.
                self.classical_data[split_name] = (
                    np.asarray(X_feat, np.float32),
                    np.asarray(y_feat, np.int64),
                    np.asarray(fid_feat, np.int64),
                )
                logger.info("classical %s: X=%s y=%s files=%d",
                            split_name,
                            self.classical_data[split_name][0].shape,
                            self.classical_data[split_name][1].shape,
                            len(np.unique(fid_feat)) if fid_feat else 0)
            if need_dl:
                self.dl_data[split_name] = (
                    np.asarray(X_raw, np.float32),
                    np.asarray(y_raw, np.int64),
                    np.asarray(fid_raw, np.int64),
                )
                logger.info("dl %s: X=%s y=%s files=%d",
                            split_name, self.dl_data[split_name][0].shape,
                            self.dl_data[split_name][1].shape,
                            len(np.unique(fid_raw)) if fid_raw else 0)
            if need_proto:
                self.proto_cache[split_name] = proto_files
                logger.info("proto %s: %d files", split_name, len(proto_files))

        # NEW: Section 41 — write final QC report after all files have been touched
        if self._qc_reports and getattr(self.config.quality, "save_qc_report", True):
            qc_path = self.dirs["reports"] / f"{self.timestamp}_qc_report_final.csv"
            write_qc_report(self._qc_reports, qc_path)

        # NEW: Section 37 — write per-file preprocessing metadata
        if self._per_file_preprocessing:
            pp_df = pd.DataFrame(self._per_file_preprocessing)
            pp_path = self.dirs["reports"] / f"{self.timestamp}_per_file_preprocessing.csv"
            pp_df.to_csv(pp_path, index=False)
            logger.info("Per-file preprocessing metadata → %s (%d files)",
                        pp_path, len(pp_df))
        
        # Save automatic baseline-selection metadata after all
        # train/validation/test files have been processed.
        self._save_baseline_selection_report()

    # ────────────────────────────────────────────────────────────────────
    # NEW: Phase 3B: Advanced Signal Analysis & Latent Space
    # ────────────────────────────────────────────────────────────────────
    def phase3b_signal_analysis_and_latent(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 3B: ADVANCED SIGNAL ANALYSIS & LATENT SPACE (Sections 7, 8, 11, 13, 16)")
        logger.info("=" * 70)
        
        from nanobio.viz.analysis_plots import SignalAnalysisPlotter
        from nanobio.viz.latent_space import LatentSpacePlotter
        
        analysis_plotter = SignalAnalysisPlotter(self.dirs["figures"], self.timestamp)
        
        # 1. Gather signals and events by class for Analysis Plots
        sigs_by_class = {c: [] for c in self.categories}
        events_by_class = {c: [] for c in self.categories}
        
        fs = self.config.sampling.model_input_rate_hz
        
        logger.info("Extracting data for PSD, ACF, PDF, and Event Kinetics...")
        for fp, lbl in zip(self.splits.train_files, self.splits.train_labels):
            cat = self.categories[lbl]
            if len(sigs_by_class[cat]) < 5: # Limit to 5 files per class for clean plots
                try:
                    _, detrended, _ = self._load_and_prepare(fp)
                    sigs_by_class[cat].append(detrended)
                    
                    # Run Event Detection (Ton/Toff/Dwell)
                    evts = self.event_detector.detect(detrended)
                    events_by_class[cat].extend([
                        {"duration_s": e.duration_s, "delta_i_over_i0": e.delta_i_over_i0} 
                        for e in evts
                    ])
                except Exception as e:
                    logger.warning(f"Analysis extraction failed for {fp.name}: {e}")

        # Generate Plots
        analysis_plotter.plot_class_psd(sigs_by_class, fs)
        analysis_plotter.plot_acf(sigs_by_class, fs)
        analysis_plotter.plot_pdf_cdf(sigs_by_class)
        analysis_plotter.plot_event_distributions(events_by_class)
        logger.info("✓ Generated PSD, ACF, PDF, CDF, and Event Kinetic distributions.")

        # 2. Latent Space Analysis (PCA, t-SNE, UMAP)
        if "train" in self.classical_data:
            X_tr, y_tr, _ = self._get_classical_split("train")
            if len(X_tr) > 10:
                latent_plotter = LatentSpacePlotter(self.dirs["figures"], self.timestamp, self.categories)
                
                # Standardize features before embeddings to prevent scaling artifacts
                from sklearn.preprocessing import StandardScaler
                X_scaled = StandardScaler().fit_transform(X_tr)
                
                latent_plotter.plot_all(X_scaled, y_tr)
                logger.info("✓ Generated PCA, t-SNE, and UMAP latent space projections.")
            else:
                logger.warning("Not enough training windows for Latent Space analysis.")


    def phase3c_comprehensive_analysis(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 3C: ADVANCED SIGNAL & LATENT ANALYSIS")
        logger.info("=" * 70)

        plotter = ComprehensivePlotter(
            self.dirs["figures"],
            self.timestamp,
        )

        class_signals = {c: [] for c in self.categories}

        # Gather sample signals.
        # Limit to 3 files per class to control memory usage.
        for fp, y in zip(
            self.splits.train_files,
            self.splits.train_labels,
        ):
            cat = self.categories[y]

            if len(class_signals[cat]) >= 3:
                continue

            try:
                _, detrended, _ = self._load_and_prepare(fp)
                class_signals[cat].append(detrended)
            except Exception as e:
                logger.warning(
                    "Comprehensive analysis extraction failed for %s: %s",
                    fp.name,
                    e,
                )

        fs = self.config.sampling.model_input_rate_hz

        # ── Sections 4/5: Multi-rate filtering ─────────────────────────────
        if self.categories and class_signals[self.categories[0]]:
            try:
                raw = read_signal_file(self.splits.train_files[0])

                plotter.plot_multirate_filters(
                    raw,
                    self.config.sampling.original_rate_hz,
                    self.signal_processor,
                    "Sample1",
                )
            except Exception as e:
                logger.warning(
                    "Multi-rate filter plot failed: %s",
                    e,
                )

        # ── Sections 7, 8, 11, 21 ──────────────────────────────────────────
        plotter.plot_psd(class_signals, fs)
        plotter.plot_acf_with_fits(class_signals, fs)
        plotter.plot_pdf_cdf(class_signals)

        # ── Sections 14, 20: Event-feature distributions ───────────────────
        if "train" in self.classical_data:
            X, y, _ = self._get_classical_split("train")

            event_names = EventFeatureExtractor.get_feature_names()

            if X.shape[1] < len(event_names):
                logger.warning(
                    "Cannot generate event violins: classical feature matrix "
                    "has %d columns but %d event features are required.",
                    X.shape[1],
                    len(event_names),
                )
            else:
                # Event features were appended to the end of the
                # classical feature vector in phase3_build_caches().
                event_X = X[:, -len(event_names):]

                df = pd.DataFrame(
                    event_X,
                    columns=event_names,
                )
                df["Category"] = [
                    self.categories[i]
                    for i in y
                ]

                plotter.plot_event_violins(df)

        # ── Section 16: Latent spaces ──────────────────────────────────────
        if "train" in self.classical_data:
            X, y, _ = self._get_classical_split("train")

            # Subsample to prevent t-SNE from becoming excessively expensive.
            if len(X) > 5000:
                idx = np.random.choice(
                    len(X),
                    5000,
                    replace=False,
                )
                X = X[idx]
                y = y[idx]

            plotter.plot_latent_spaces(
                X,
                y,
                self.categories,
            )

        logger.info("✓ Phase 3C comprehensive analysis complete.")


    # ────────────────────────────────────────────────────────────────────
    # NEW helpers: backwards-compatible accessors + metric recording
    # ────────────────────────────────────────────────────────────────────

    def _get_classical_split(self, split_name: str):
        """Return (X, y, file_ids) for a classical split, tolerating old 2-tuple caches."""
        entry = self.classical_data[split_name]
        if len(entry) == 3:
            return entry
        X, y = entry
        return X, y, np.arange(len(y), dtype=np.int64)

    def _get_dl_split(self, split_name: str):
        """Return (X, y, file_ids) for a DL split, tolerating old 2-tuple caches."""
        entry = self.dl_data[split_name]
        if len(entry) == 3:
            return entry
        X, y = entry
        return X, y, np.arange(len(y), dtype=np.int64)

    def _record_result(
        self, name: str,
        y_true: np.ndarray, y_pred: np.ndarray, y_prob,
        fit_time_s: float,
        fid_te: np.ndarray = None,
        granularity: str = "window",
        **extra,
    ) -> None:
        """
        Record window-level metrics AND, when possible, file-level metrics
        obtained by averaging predicted probabilities per file.
        """
        n_classes = len(self.categories)
        m = MetricEngine.compute_all(y_true, y_pred, y_prob, n_classes)
        self.results[name] = {
            "accuracy": m["accuracy"]["value"],
            "f1_weighted": m["f1_weighted"]["value"],
            "f1_macro": m["f1_macro"]["value"],
            "mcc": m["mcc"]["value"],
            "granularity": granularity,
            "time_s": fit_time_s,
            **extra,
        }
        logger.info("%s [%s] → acc=%.4f f1=%.4f",
                    name, granularity,
                    self.results[name]["accuracy"],
                    self.results[name]["f1_weighted"])

        # File-level aggregation for window-level models
        if (granularity == "window"
                and fid_te is not None and y_prob is not None
                and len(fid_te) > 0):
            n_files = int(fid_te.max()) + 1
            fp_probs = np.stack([
                y_prob[fid_te == f].mean(0) for f in range(n_files)
            ])
            yt_file = np.array([
                y_true[fid_te == f][0] for f in range(n_files)
            ])
            m2 = MetricEngine.compute_all(
                yt_file, fp_probs.argmax(1), fp_probs, n_classes,
            )
            key_file = f"{name} [file-level]"
            self.results[key_file] = {
                "accuracy": m2["accuracy"]["value"],
                "f1_weighted": m2["f1_weighted"]["value"],
                "f1_macro": m2["f1_macro"]["value"],
                "mcc": m2["mcc"]["value"],
                "granularity": "file",
                "n_test_files": n_files,
            }
            logger.info("%s → acc=%.4f f1=%.4f (n=%d files)",
                        key_file, m2["accuracy"]["value"],
                        m2["f1_weighted"]["value"], n_files)

        # NEW: Section 24 — ML visualizations per model
        ml_cfg = getattr(self.config, "ml_viz", None)
        if ml_cfg is None or not getattr(ml_cfg, "enabled", True):
            return
        try:
            n_classes = len(self.categories)
            if ml_cfg.save_confusion_matrix:
                self.ml_plotter.plot_confusion_matrix(
                    y_true, y_pred, self.categories, name,
                )
            if y_prob is not None:
                y_prob_n = MetricEngine._normalize_probabilities(
                    y_prob, n_classes, len(y_true),
                )
                if y_prob_n is not None:
                    if ml_cfg.save_roc:
                        self.ml_plotter.plot_roc(y_true, y_prob_n, self.categories, name)
                    if ml_cfg.save_pr:
                        self.ml_plotter.plot_pr(y_true, y_prob_n, self.categories, name)
                    if ml_cfg.save_calibration:
                        self.ml_plotter.plot_calibration(
                            y_true, y_prob_n, self.categories, name,
                        )
                    if ml_cfg.save_probability_dist:
                        self.ml_plotter.plot_probability_distribution(
                            y_true, y_prob_n, self.categories, name,
                        )
            if ml_cfg.save_classwise_csv:
                self.ml_plotter.save_classwise_metrics(
                    y_true, y_pred, self.categories, name, self.dirs["reports"],
                )
        except Exception as e:
            logger.warning("ML plot generation failed for %s: %s", name, e)

    def _dl_overfit_gate(self) -> bool:
        """
        128-sample overfit test. If DL cannot memorize a tiny batch, the
        DL data/label wiring is broken; DL phases are then skipped.
        Result is cached so we only run the gate once per pipeline run.
        """
        if self._dl_gate_passed is not None:
            return self._dl_gate_passed

        if "train" not in self.dl_data:
            self._dl_gate_passed = True
            return True

        Xtr, ytr, _ = self._get_dl_split("train")
        if len(Xtr) < 32:
            logger.warning("Overfit gate skipped: <32 training windows")
            self._dl_gate_passed = True
            return True

        logger.info("Running DL overfit gate (128 samples, up to 300 steps)...")
        ds = RawWindowDataset(Xtr[:128], ytr[:128])
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
        model = ResNet1D(len(self.categories)).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        acc = 0.0
        for step in range(300):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = Fn.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    acc = (model(xb).argmax(1) == yb).float().mean().item()
                logger.info("  gate step %d: loss=%.4f acc=%.3f",
                            step + 1, loss.item(), acc)
                if acc > 0.95:
                    logger.info("✓ DL overfit gate PASSED")
                    del model
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    self._dl_gate_passed = True
                    return True

        logger.error("✗ DL OVERFIT GATE FAILED (final acc=%.3f). "
                     "DL / TimesNet paths will be SKIPPED. "
                     "Fix label/tensor wiring before running DL.", acc)
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._dl_gate_passed = False
        return False

    # ────────────────────────────────────────────────────────────────────
    # Phase 4a: classical
    # ────────────────────────────────────────────────────────────────────

    def phase4a_classical(self) -> None:
        if not self.config.models.classical_enabled:
            return
        logger.info("=" * 70)
        logger.info("PHASE 4A: CLASSICAL MODELS (handcrafted features)")
        logger.info("=" * 70)

        # NEW: use 3-tuple accessor to also carry file_ids for file-level metrics
        Xtr, ytr, _fid_tr = self._get_classical_split("train")
        Xte, yte, fid_te = self._get_classical_split("test")

        scaler = RobustScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        # NEW: drop constant features before f_classif (fixes the recurring
        # 'Features [19] are constant' warning + divide-by-zero on high_band)
        vt = VarianceThreshold(0.0)
        Xtr_v = vt.fit_transform(Xtr_s)
        Xte_v = vt.transform(Xte_s)
        if Xtr_v.shape[1] < Xtr_s.shape[1]:
            logger.info("Dropped %d constant features",
                        Xtr_s.shape[1] - Xtr_v.shape[1])

        k = min(40, Xtr_v.shape[1])
        sel = SelectKBest(f_classif, k=k)
        Xtr_sel = sel.fit_transform(Xtr_v, ytr)
        Xte_sel = sel.transform(Xte_v)

        models = build_classical_models(
            self.config.models.classical_models, len(self.categories), len(Xtr_sel),
        )
        for name, clf in models.items():
            t0 = time.time()
            try:
                clf.fit(Xtr_sel, ytr)
                # NEW: Section 43 — extract feature importance if available
                if (getattr(self.config, "tracking", None)
                        and self.config.tracking.save_feature_importance):
                    try:
                        feat_names = list(FeatureExtractor.FEATURE_NAMES)
                        try:
                            from nanobio.features.event_features import (
                                EventFeatureExtractor,
                            )
                            feat_names.extend(
                                EventFeatureExtractor.get_feature_names()
                            )
                        except ImportError:
                            pass

                        # BUG-005 FIX: Apply the same masks that
                        # VarianceThreshold and SelectKBest applied
                        # to the feature matrix, so names align with
                        # the model's actual feature indices.
                        vt_mask = vt.get_support()
                        active_names = [
                            n for n, keep in zip(feat_names, vt_mask) if keep
                        ]
                        sel_mask = sel.get_support()
                        active_names = [
                            n for n, keep in zip(active_names, sel_mask) if keep
                        ]

                        extract_and_save_feature_importance(
                            clf, name, active_names,
                            self.dirs["reports"], self.timestamp,
                        )
                    except Exception as e:
                        logger.warning(
                            "Feature importance failed for %s: %s", name, e,
                        )
                pred = clf.predict(Xte_sel)
                # BUG-004 FIX: Ensure pred is 1D. XGBoost multi:softprob
                # returns 2D (n, n_classes); ravel() would double length.
                pred = np.asarray(pred)
                if pred.ndim > 1:
                    pred = pred.argmax(axis=1)

                prob = clf.predict_proba(Xte_sel) if hasattr(clf, "predict_proba") else None
                # NEW: unified recorder now handles window + file-level metrics
                self._record_result(
                    name, yte, pred, prob, time.time() - t0,
                    fid_te=fid_te, granularity="window",
                )
            
                # NEW: Save the trained classical model + preprocessing pipeline
                try:
                    import joblib
                    model_bundle = {
                        "model": clf,
                        "scaler": scaler,
                        "variance_threshold": vt,
                        "kbest_selector": sel,
                        "categories": self.categories,
                        "feature_names": FeatureExtractor.FEATURE_NAMES,
                        "config_snapshot": {
                            "model_input_rate_hz": self.config.sampling.model_input_rate_hz,
                            "segment_length": self.config.segment.segment_length_samples,
                            "stride": self.config.segment.stride_samples,
                            "baseline_method": self.config.baseline.method,
                            "baseline_window_seconds": self.config.baseline.window_seconds,
                            "detrend_method": self.config.detrend.method,
                        },
                        "trained_at": self.timestamp,
                        "n_train_samples": len(ytr),
                        "n_train_files": len(np.unique(self._get_classical_split("train")[2])),
                    }
                    save_path = self.dirs["models"] / f"{name}_bundle.joblib"
                    joblib.dump(model_bundle, save_path)
                    logger.info("Saved %s bundle to %s", name, save_path)
                except Exception as e:
                    logger.warning("Failed to save %s: %s", name, e)
                    #logger.error("%s failed: %s", name, e)

            except Exception as e:
                logger.error("%s failed: %s", name, e)


    # ────────────────────────────────────────────────────────────────────
    # Phase 4b: baseline DL
    # ────────────────────────────────────────────────────────────────────

    def phase4b_baseline_dl(self) -> None:
        if not self.config.models.baseline_dl_enabled:
            return
        logger.info("=" * 70)
        logger.info("PHASE 4B: BASELINE DL (raw windows)")
        logger.info("=" * 70)

        # NEW: overfit gate guards DL runs (opt-out via cfg.models.dl_overfit_gate=False)
        if getattr(self.config.models, "dl_overfit_gate", True):
            if not self._dl_overfit_gate():
                logger.error("Skipping ALL baseline-DL models (gate failed).")
                return

        Xtr, ytr, _ = self._get_dl_split("train")
        if "val" in self.dl_data:
            Xva, yva, _ = self._get_dl_split("val")
        else:
            Xva, yva = Xtr, ytr
        Xte, yte, fid_te = self._get_dl_split("test")
        if len(Xva) < 10:
            Xva, yva = Xte, yte

        n_classes = len(self.categories)
        factories = {
            "ResNet1D": lambda: ResNet1D(n_classes),
            "TCN": lambda: TCN(n_classes),
            "InceptionTime": lambda: InceptionTime(n_classes),
            "CNN-BiLSTM": lambda: CNNBiLSTM(n_classes),
            "Transformer": lambda: TransformerClassifier(n_classes),
        }

        for name in self.config.models.baseline_dl_models:
            if name not in factories:
                continue
            logger.info("Training %s ...", name)
            model = factories[name]()
            tr_ds = RawWindowDataset(Xtr, ytr)
            va_ds = RawWindowDataset(Xva, yva)
            te_ds = RawWindowDataset(Xte, yte)
            trainer = TorchClassifierTrainer(
                model, self.device, self.config.training, n_classes,
            )
            t0 = time.time()
            try:
                #trainer.fit(tr_ds, ytr, val_ds=va_ds)
                #pred, prob = trainer.predict(te_ds)
                history = trainer.fit(tr_ds, ytr, val_ds=va_ds)
                pred, prob = trainer.predict(te_ds)

                # NEW: Section 32 — save training history + curves + param count
                try:
                    n_params = sum(p.numel() for p in model.parameters())
                    logger.info("%s parameters: %s", name, f"{n_params:,}")
                    hist_df = pd.DataFrame(history.to_dict())
                    hist_df.to_csv(
                        self.dirs["reports"] / f"{self.timestamp}_history_{name}.csv",
                        index=False,
                    )
                    if getattr(self.config, "ml_viz", MLVizCfg()).save_training_curves:
                        self.ml_plotter.plot_training_curves(history.to_dict(), name)
                except Exception as e:
                    logger.warning("Training curve generation failed for %s: %s", name, e)
                # NEW: unified recorder for both granularities
                self._record_result(
                    name, yte, pred, prob, time.time() - t0,
                    fid_te=fid_te, granularity="window",
                )
                torch.save(model.state_dict(), self.dirs["models"] / f"{name}.pt")
            except Exception as e:
                logger.error("%s failed: %s", name, e)
            del model, trainer
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────
    # Phase 4c: ProtoNet
    # ────────────────────────────────────────────────────────────────────

    def phase4c_protonet(self) -> None:
        if not self.config.models.protonet_enabled:
            return
        logger.info("=" * 70)
        logger.info("PHASE 4C: HIERARCHICAL PROTONET (file-level episodes)")
        logger.info("=" * 70)

        cfg = self.config.protonet
        train_cache = self.proto_cache["train"]
        val_cache = self.proto_cache["val"] or self.proto_cache["test"]
        test_cache = self.proto_cache["test"]
        if len(train_cache) < 4:
            logger.error("Too few training files for ProtoNet")
            return

        train_labels = [e["label"] for e in train_cache]
        val_labels = [e["label"] for e in val_cache]
        test_labels = [e["label"] for e in test_cache]

        n_way = min(cfg.n_way, len(self.categories))
        phys_dim = len(HybridFeatureExtractor.NAMES)

        model = HierarchicalProtoNet(
            embed_dim=cfg.embed_dim, metric_dim=cfg.metric_dim,
            attention_hidden=cfg.attention_hidden,
            phys_dim=phys_dim, hybrid_enabled=cfg.hybrid_features_enabled,
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("ProtoNet parameters: %s", f"{n_params:,}")

        trainer = ProtoNetTrainer(
            model, self.device, cfg, checkpoint_dir=self.dirs["models"],
        )
        history = trainer.fit(
            train_cache, train_labels, val_cache, val_labels,
            n_way=n_way, seed=self.config.random_seed,
        )
        # NEW: Section 32 — ProtoNet training curves
        try:
            if getattr(self.config, "ml_viz", MLVizCfg()).save_training_curves:
                self.ml_plotter.plot_training_curves(history.to_dict(), "ProtoNet")
        except Exception as e:
            logger.warning("ProtoNet curves failed: %s", e)

        hist_df = pd.DataFrame(history.to_dict())
        hist_df.to_csv(
            self.dirs["reports"] / f"{self.timestamp}_protonet_history.csv",
            index=False,
        )

        test_metrics = trainer.evaluate(
            test_cache, test_labels, n_way=n_way,
            num_episodes=100, seed=self.config.random_seed + 999,
        )
        self.results["HierarchicalProtoNet"] = {
            "accuracy": test_metrics["test_acc"],
            "f1_weighted": test_metrics["test_f1_weighted"],
            "f1_macro": test_metrics["test_f1_macro"],
            "mcc": None,
            "granularity": "file",  # NEW: tag for leaderboard sorting
            "time_s": sum(history.epoch_time_s),
        }
        logger.info("ProtoNet TEST %s", test_metrics)

    # ────────────────────────────────────────────────────────────────────
    # NEW — Phase 4d: SOTA benchmarks (TimesNet | MiniRocket | TabPFN)
    # ────────────────────────────────────────────────────────────────────

    def phase4d_sota(self) -> None:
        if not getattr(self.config.models, "sota_enabled", False):
            return
        logger.info("=" * 70)
        logger.info("PHASE 4D: SOTA (TimesNet | MiniRocket | TabPFN)")
        logger.info("=" * 70)

        if "test" not in self.dl_data:
            logger.error("No DL cache built; skipping SOTA")
            return

        sc = getattr(self.config, "sota", None)
        if sc is None:
            logger.error("No `sota` config section; skipping SOTA phase")
            return

        n_classes = len(self.categories)
        Xtr, ytr, _ = self._get_dl_split("train")
        if "val" in self.dl_data:
            Xva, yva, _ = self._get_dl_split("val")
        else:
            Xva, yva = Xtr, ytr
        Xte, yte, fid_te = self._get_dl_split("test")

        # ── TimesNet (guarded by the same DL overfit gate) ──────────────
        if getattr(sc, "run_timesnet", True) and _TIMESNET_AVAILABLE:
            if (getattr(self.config.models, "dl_overfit_gate", True)
                    and not self._dl_overfit_gate()):
                logger.error("TimesNet skipped: DL overfit gate failed")
            else:
                logger.info("Training TimesNet ...")
                try:
                    model = TimesNetClassifier(
                        num_classes=n_classes,
                        d_model=getattr(sc, "timesnet_d_model", 32),
                        d_ff=getattr(sc, "timesnet_d_ff", 64),
                        top_k=getattr(sc, "timesnet_top_k", 3),
                        e_layers=getattr(sc, "timesnet_e_layers", 2),
                        num_kernels=getattr(sc, "timesnet_num_kernels", 6),
                    )
                    trainer = TorchClassifierTrainer(
                        model, self.device, self.config.training, n_classes,
                    )
                    t0 = time.time()
                    """trainer.fit(
                        RawWindowDataset(Xtr, ytr), ytr,
                        val_ds=RawWindowDataset(Xva, yva),
                    )"""
                    history = trainer.fit(
                        RawWindowDataset(Xtr, ytr), ytr,
                        val_ds=RawWindowDataset(Xva, yva),
                    )
                    # NEW: Section 32 for TimesNet
                    try:
                        n_params = sum(p.numel() for p in model.parameters())
                        logger.info("TimesNet parameters: %s", f"{n_params:,}")
                        pd.DataFrame(history.to_dict()).to_csv(
                            self.dirs["reports"] / f"{self.timestamp}_history_TimesNet.csv",
                            index=False,
                        )
                        if getattr(self.config, "ml_viz", MLVizCfg()).save_training_curves:
                            self.ml_plotter.plot_training_curves(history.to_dict(), "TimesNet")
                    except Exception as e:
                        logger.warning("TimesNet curve failed: %s", e)

                    pred, prob = trainer.predict(RawWindowDataset(Xte, yte))
                    self._record_result(
                        "TimesNet", yte, pred, prob, time.time() - t0,
                        fid_te=fid_te, granularity="window",
                    )
                    torch.save(model.state_dict(),
                               self.dirs["models"] / "TimesNet.pt")
                    del model, trainer
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error("TimesNet failed: %s", e)
        elif getattr(sc, "run_timesnet", True):
            logger.warning("TimesNet requested but module unavailable — skipping")

        # ── MiniRocket (no training loop; graceful if backend missing) ─
        if getattr(sc, "run_minirocket", True) and _SOTA_ADAPTERS_AVAILABLE:
            logger.info("Running MiniRocket ...")
            try:
                r = fit_predict_minirocket(Xtr, ytr, Xte)
                if r:
                    self._record_result(
                        "MiniRocket", yte, r["y_pred"], r["y_prob"],
                        r["fit_time_s"], fid_te=fid_te, granularity="window",
                        backend=r["backend"], pred_time_s=r["pred_time_s"],
                    )
                    joblib.dump(r["artifact"],
                                self.dirs["models"] / "minirocket.joblib")
            except Exception as e:
                logger.error("MiniRocket failed: %s", e)

        # ── TabPFN v2 window-level (38 handcrafted features) ───────────
        if getattr(sc, "run_tabpfn_window", True) and _SOTA_ADAPTERS_AVAILABLE:
            if self.classical_data:
                logger.info("Running TabPFN (window features) ...")
                Ftr, ytr_f, _ = self._get_classical_split("train")
                Fte, yte_f, fid_te_f = self._get_classical_split("test")
                try:
                    r = fit_predict_tabpfn(
                        Ftr, ytr_f, Fte,
                        max_train=getattr(sc, "tabpfn_max_train", 10000),
                        max_features=getattr(sc, "tabpfn_max_features", 500),
                        device=str(self.device),
                    )
                    if r:
                        self._record_result(
                            "TabPFN-window", yte_f, r["y_pred"], r["y_prob"],
                            r["fit_time_s"], fid_te=fid_te_f, granularity="window",
                        )
                except Exception as e:
                    logger.error("TabPFN-window failed: %s", e)
            else:
                logger.info("TabPFN-window skipped: classical features not built "
                            "(enable classical_enabled to activate)")

        # ── TabPFN v2 file-level (mean features per file) ──────────────
        if getattr(sc, "run_tabpfn_file", True) and _SOTA_ADAPTERS_AVAILABLE:
            if self.classical_data:
                logger.info("Running TabPFN (file-mean features) ...")
                Ftr, ytr_f, fid_tr_f = self._get_classical_split("train")
                Fte, yte_f, fid_te_f = self._get_classical_split("test")
                try:
                    n_tr = int(fid_tr_f.max()) + 1 if len(fid_tr_f) else 0
                    n_te = int(fid_te_f.max()) + 1 if len(fid_te_f) else 0
                    if n_tr < 2 or n_te < 1:
                        logger.warning("TabPFN-file skipped: not enough files")
                    else:
                        def _file_means(X, fids, n):
                            return np.stack([X[fids == f].mean(0) for f in range(n)])

                        Ftr_m = _file_means(Ftr, fid_tr_f, n_tr)
                        Fte_m = _file_means(Fte, fid_te_f, n_te)
                        yf_tr = np.array([ytr_f[fid_tr_f == f][0] for f in range(n_tr)])
                        yf_te = np.array([yte_f[fid_te_f == f][0] for f in range(n_te)])
                        r = fit_predict_tabpfn(
                            Ftr_m, yf_tr, Fte_m,
                            max_train=getattr(sc, "tabpfn_max_train", 10000),
                            max_features=getattr(sc, "tabpfn_max_features", 500),
                            device=str(self.device),
                        )
                        if r:
                            self._record_result(
                                "TabPFN-file", yf_te, r["y_pred"], r["y_prob"],
                                r["fit_time_s"], fid_te=None, granularity="file",
                            )
                except Exception as e:
                    logger.error("TabPFN-file failed: %s", e)
            else:
                logger.info("TabPFN-file skipped: classical features not built")

    # ────────────────────────────────────────────────────────────
    # Phase 6: Self-Supervised Representation Learning
    # ────────────────────────────────────────────────────────────

    def phase6_ssl(self) -> None:
        """Self-supervised contrastive learning on unlabeled windows."""
        rcfg = getattr(self.config, "research", None)
        if rcfg is None or not rcfg.enabled or not rcfg.ssl.enabled:
            return

        logger.info("=" * 70)
        logger.info("PHASE 6: SELF-SUPERVISED REPRESENTATION LEARNING")
        logger.info("=" * 70)

        from nanobio.models.resnet1d import ProjectionHead, build_ssl_encoder
        from nanobio.signal.augmentations import AugmentationComposer
        from nanobio.training.trainer import ContrastiveTrainer

        Xtr, ytr, _ = self._get_dl_split("train")
        logger.info("SSL training on %d windows (labels NOT used)", len(Xtr))

        enc_cfg = rcfg.ssl.encoder
        encoder = build_ssl_encoder(
            encoder_type=enc_cfg.encoder_type,
            embedding_dim=enc_cfg.embedding_dim,
            hidden_dim=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            kernel_size=enc_cfg.kernel_size,
            dropout=enc_cfg.dropout,
        )
        projection = ProjectionHead(
            enc_cfg.embedding_dim, enc_cfg.embedding_dim,
            rcfg.ssl.projection_dim,
        )

        augmenter = AugmentationComposer(rcfg.ssl.augmentation, seed=self.config.random_seed)

        trainer = ContrastiveTrainer(
            encoder=encoder,
            projection=projection,
            device=self.device,
            temperature=rcfg.ssl.temperature,
            learning_rate=rcfg.ssl.learning_rate,
            weight_decay=rcfg.ssl.weight_decay,
            epochs=rcfg.ssl.epochs,
            batch_size=rcfg.ssl.batch_size,
        )

        history = trainer.fit(Xtr, augmenter)

        # Compute and cache embeddings for all splits
        self._ssl_encoder = encoder
        for split_name in ("train", "val", "test"):
            if split_name in self.dl_data:
                X, y, fids = self._get_dl_split(split_name)
                embs = trainer.encode(X)
                self._ssl_embeddings[split_name] = (embs, y, fids)
                logger.info("SSL embeddings %s: %s", split_name, embs.shape)

        # Save encoder
        torch.save(encoder.state_dict(),
                   self.dirs["models"] / "ssl_encoder.pt")
        logger.info("SSL encoder saved.")

    # ────────────────────────────────────────────────────────────
    # Phase 7: Few-Shot Prototype Evaluation
    # ────────────────────────────────────────────────────────────

    def phase7_fewshot(self) -> None:
        """Episodic N-way K-shot evaluation on SSL embeddings."""
        rcfg = getattr(self.config, "research", None)
        if rcfg is None or not rcfg.enabled or not rcfg.few_shot.enabled:
            return
        if not hasattr(self, "_ssl_embeddings") or "test" not in self._ssl_embeddings:
            logger.warning("Phase 7 skipped: no SSL embeddings available.")
            return

        logger.info("=" * 70)
        logger.info("PHASE 7: FEW-SHOT PROTOTYPE EVALUATION")
        logger.info("=" * 70)

        from nanobio.training.episodic import run_episodic_evaluation

        fcfg = rcfg.few_shot
        all_emb, all_lbl = [], []
        for split in ("train", "test"):
            embs, y, _ = self._ssl_embeddings[split]
            all_emb.append(embs)
            all_lbl.append(y)
        all_emb = np.concatenate(all_emb)
        all_lbl = np.concatenate(all_lbl)

        for k_shot in rcfg.label_budgets:
            result = run_episodic_evaluation(
                all_emb, all_lbl,
                n_way=min(fcfg.n_way, len(self.categories)),
                k_shot=k_shot,
                q_query=fcfg.q_query,
                num_episodes=fcfg.num_eval_episodes,
                distance_metric=fcfg.distance_metric,
                normalize=fcfg.normalize_embeddings,
                seed=self.config.random_seed,
            )
            key = f"FewShot-{k_shot}shot"
            self.results[key] = {
                "accuracy": result["accuracy_mean"],
                "f1_weighted": result.get("f1_macro_mean", 0.0),
                "f1_macro": result.get("f1_macro_mean", 0.0),
                "mcc": None,
                "granularity": "episodic",
                "time_s": 0.0,
                **{f"episodic_{k}": v for k, v in result.items()},
            }

    # ────────────────────────────────────────────────────────────
    # Phase 8: Pseudo-Labeling + Semi-Supervised
    # ────────────────────────────────────────────────────────────

    def phase8_pseudo_labeling(self) -> None:
        """Confidence-aware pseudo-labeling on SSL embeddings."""
        rcfg = getattr(self.config, "research", None)
        if rcfg is None or not rcfg.enabled or not rcfg.pseudo_label.enabled:
            return
        if not hasattr(self, "_ssl_embeddings"):
            logger.warning("Phase 8 skipped: no SSL embeddings.")
            return

        logger.info("=" * 70)
        logger.info("PHASE 8: PSEUDO-LABELING + SEMI-SUPERVISED")
        logger.info("=" * 70)

        from nanobio.models.prototype import PrototypeClassifier
        from nanobio.training.pseudo_labeling import PseudoLabelManager

        pl_cfg = rcfg.pseudo_label
        train_emb, train_lbl, _ = self._ssl_embeddings["train"]
        test_emb, test_lbl, _ = self._ssl_embeddings["test"]

        # Sample small labeled support
        rng = np.random.default_rng(self.config.random_seed)
        support_idx = []
        for c in np.unique(train_lbl):
            c_idx = np.where(train_lbl == c)[0]
            k = min(rcfg.few_shot.k_shot, len(c_idx))
            support_idx.extend(rng.choice(c_idx, k, replace=False).tolist())

        support_emb = train_emb[support_idx]
        support_lbl = train_lbl[support_idx]
        unlabeled_mask = np.ones(len(train_emb), dtype=bool)
        unlabeled_mask[support_idx] = False
        unlabeled_emb = train_emb[unlabeled_mask]

        clf = PrototypeClassifier(
            distance_metric=rcfg.few_shot.distance_metric,
            normalize=rcfg.few_shot.normalize_embeddings,
        )
        clf.fit(support_emb, support_lbl)

        pl_manager = PseudoLabelManager(pl_cfg, clf)
        pseudo_emb, pseudo_lbl, stats = pl_manager.generate(
            unlabeled_emb, unlabeled_emb,
            support_emb, support_lbl,
        )

        if len(pseudo_emb) > 0:
            expanded_emb = np.concatenate([support_emb, pseudo_emb])
            expanded_lbl = np.concatenate([support_lbl, pseudo_lbl])
            clf.fit(expanded_emb, expanded_lbl)

            preds, probs, _ = clf.predict(test_emb)
            from sklearn.metrics import accuracy_score, f1_score
            self.results["PseudoLabel"] = {
                "accuracy": float(accuracy_score(test_lbl, preds)),
                "f1_weighted": float(f1_score(test_lbl, preds,
                                               average="weighted", zero_division=0)),
                "f1_macro": float(f1_score(test_lbl, preds,
                                            average="macro", zero_division=0)),
                "mcc": None,
                "granularity": "file",
                "time_s": 0.0,
                "n_pseudo": len(pseudo_emb),
                "pseudo_stats": stats.to_dict(),
            }
            logger.info("Pseudo-label result: acc=%.4f", self.results["PseudoLabel"]["accuracy"])

    # ────────────────────────────────────────────────────────────────────
    # Finalize
    # ────────────────────────────────────────────────────────────────────

    def phase5_report(self) -> None:
        logger.info("=" * 70)
        logger.info("PHASE 5: LEADERBOARD")
        logger.info("=" * 70)
        if not self.results:
            logger.warning("No results to report")
            return
        rows = [{"model": k, **v} for k, v in self.results.items()]
        df = pd.DataFrame(rows)

        # NEW: sort by granularity so file-level rows print first
        sort_cols = ["f1_weighted"]
        if "granularity" in df.columns:
            sort_cols = ["granularity", "f1_weighted"]
            df = df.sort_values(
                sort_cols, ascending=[True, False], na_position="last",
            )
        else:
            df = df.sort_values("f1_weighted", ascending=False, na_position="last")

        out = self.dirs["reports"] / f"{self.timestamp}_leaderboard.csv"
        df.to_csv(out, index=False)

        # NEW: split leaderboard printout by granularity when tagged
        if "granularity" in df.columns:
            file_rows = df[df["granularity"] == "file"]
            win_rows = df[df["granularity"] == "window"]
            if not file_rows.empty:
                logger.info("─── FILE-LEVEL RESULTS (scientifically defensible) ───")
                for _, row in file_rows.iterrows():
                    logger.info("  %-30s acc=%.4f f1=%.4f",
                                row["model"],
                                row.get("accuracy", float("nan")),
                                row.get("f1_weighted", float("nan")))
            if not win_rows.empty:
                logger.info("─── WINDOW-LEVEL RESULTS (autocorrelated) ───")
                for _, row in win_rows.iterrows():
                    logger.info("  %-30s acc=%.4f f1=%.4f",
                                row["model"],
                                row.get("accuracy", float("nan")),
                                row.get("f1_weighted", float("nan")))
        else:
            logger.info("Leaderboard:\n%s", df.to_string(index=False))

        # NEW: Section 43 — push results to the experiment registry
        if self.registry and self.results:
            try:
                best_model = max(
                    self.results.items(),
                    key=lambda x: x[1].get("f1_weighted", 0) or 0,
                )
                summary = {
                    "n_models_run": len(self.results),
                    "best_model": best_model[0],
                    "best_f1_weighted": best_model[1].get("f1_weighted"),
                    "best_accuracy": best_model[1].get("accuracy"),
                    "leaderboard_csv": str(out),
                }
                self.registry.update_results(self.experiment_id, summary)
            except Exception as e:
                logger.warning("Registry update failed: %s", e)
        logger.info("Leaderboard saved: %s", out)
    """
    def run(self) -> None:
        t0 = time.time()
        self.phase1_data_integrity()
        self.phase2_baseline_visualization()
        self.phase3_build_caches()
        self.phase4a_classical()
        self.phase4b_baseline_dl()
        self.phase4c_protonet()
        self.phase4d_sota()  # NEW
        self.phase5_report()
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE in %.1fs | exp=%s", time.time() - t0, self.experiment_id)
        logger.info("Figures: %s", self.dirs["figures"])
        logger.info("Reports: %s", self.dirs["reports"])
        logger.info("Models:  %s", self.dirs["models"])
        if self.split_link_root:  # NEW
            logger.info("Split tree: %s", self.split_link_root)
        logger.info("=" * 70)"""

    def run(self) -> None:
        t0 = time.time()
        self.phase1_data_integrity()
        self.phase2_baseline_visualization()
        self.phase3_build_caches()
        
        # --- NEW: Signal Analysis & Latent Space ---
        self.phase3b_signal_analysis_and_latent()
        # --- NEW: Comprehensive signal, event, and latent analysis ---
        self.phase3c_comprehensive_analysis()
        
        self.phase4a_classical()
        self.phase4b_baseline_dl()
        self.phase4c_protonet()
        self.phase4d_sota()
        # ── NEW: Research phases (only if enabled in config) ──
        self._ssl_embeddings = {}  # Initialize SSL embedding cache
        self.phase6_ssl()
        self.phase7_fewshot()
        self.phase8_pseudo_labeling()
        # ── END NEW ──
        self.phase5_report()
        
        logger.info("=" * 70)
        logger.info(
            "PIPELINE COMPLETE in %.1fs | exp=%s",
            time.time() - t0, 
            self.experiment_id
        )
        
        logger.info("Figures: %s", self.dirs["figures"])
        logger.info("Reports: %s", self.dirs["reports"])
        logger.info("Models:  %s", self.dirs["models"])

        if self.split_link_root:
            logger.info("Split tree: %s", self.split_link_root)

        logger.info("=" * 70)
        # ... rest of existing run() method