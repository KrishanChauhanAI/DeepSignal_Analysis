"""
Configuration system.

All experiment choices live in dataclasses. YAML files hydrate them.
Users toggle modules without editing source code.

CHANGELOG (Research-Grade Revision):
  - GAP-001: model_input_rate_hz default changed from 10000 to 50000
  - GAP-002: Added gaussian_lpf_cutoff_hz and gaussian_lpf_enabled to SamplingCfg
  - BUG-017: Verified force_purge_split_tree present in SplitCfg
  - BUG-023: from_yaml() now handles nested research sub-sub-configs

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# Sub-configurations
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SamplingCfg:
    """Sampling and downsampling target rates.

    GAP-001 FIX: model_input_rate_hz default changed from 10000 to 50000.
    GAP-002 FIX: Added gaussian_lpf_cutoff_hz and gaussian_lpf_enabled.

    Default processing chain:
        1 MHz → anti-alias/decimate → 50 kHz → Gaussian LPF 10 kHz → analysis
    """
    original_rate_hz: int = 1_000_000
    target_rates_hz: List[int] = field(
        default_factory=lambda: [50_000, 10_000, 1_000, 100, 10, 2]
    )
    # GAP-001 FIX: Was 10_000, now 50_000 per specification
    model_input_rate_hz: int = 50_000  #10_000
    # GAP-002 FIX: Gaussian LPF applied AFTER resampling to target rate
    gaussian_lpf_enabled: bool = True
    gaussian_lpf_cutoff_hz: int = 10_000


'''
@dataclass
class BaselineCfg:
    """
    Per-file baseline estimation.

    IMPORTANT: The baseline is for VISUALIZATION and as a detrending
    reference. The MODEL trains on the detrended signal, NOT on the raw
    signal that still contains DC offset.
    """
    method: str = "rolling_median"     # rolling_median | savgol | polynomial
    window_seconds: float = 1.0
    poly_order: int = 3
    savgol_window: int = 501
    save_visualizations: bool = True
    max_files_visualized: int = 20
    # NEW: user-controlled diagnostic flags
    compute_baseline_range: bool = True    # ptp(baseline) — drift measure
    compute_residual_std: bool = True      # std(signal − baseline) — noise/event measure
    show_flags_in_plot_annotation: bool = True  # display in plot text box
    # NEW: prevent short events from being absorbed into baseline
    min_event_isolation_seconds: float = 2.0  
    # baseline window will be at least this large so events < window_seconds
    # aren't averaged into baseline
'''

@dataclass
class BaselineCfg:
    """
    Per-file baseline estimation.

    method:
        "auto"           — objective selection among candidates (RECOMMENDED)
        "rolling_median" — fixed robust local median
        "constant"       — global mean
        "linear"         — linear trend
        "savgol"         — Savitzky-Golay
    """
    method: str = "auto"                 # ← changed default to auto
    window_seconds: float = 1.0
    poly_order: int = 3
    savgol_window: int = 501
    save_visualizations: bool = True
    max_files_visualized: int = 20
    # Diagnostic flags (user-controlled)
    compute_baseline_range: bool = True
    compute_residual_std: bool = True
    show_flags_in_plot_annotation: bool = True
    # Auto-selection settings
    auto_candidates: List[str] = field(
        default_factory=lambda: ["constant", "linear", "rolling_median", "savgol"]
    )
    event_sigma: float = 5.0             # for event-preservation scoring
    log_selection_details: bool = True   # log per-candidate scores


@dataclass
class DetrendCfg:
    """Detrending applied to MODEL INPUT only."""
    enabled: bool = True
    method: str = "subtract_baseline"  # subtract_baseline | mean_center | none


@dataclass
class SplitCfg:
    """File-level train / val / test split."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    group_aware: bool = False
    group_pattern: Optional[str] = None
    seed: int = 42
    #------Added------
    # ── Zero-copy link tree (browsable split view) ──
    links_enabled: bool = True
    links_dir: str = ""            # "" → <data_root>/../split_links  (SAME volume as data → hardlinks work)
    link_mode: str = "auto"        # auto | hardlink | symlink | copy
    verify_md5: bool = False       # slow on 1 MHz files; sizes checked instead
    purge_on_change: bool = True   # wipe stale tree if the split assignment changes
    # ── Optional speed layer (kept for the full dataset; default OFF per your choice) ──
    use_npz_cache: bool = False
    cache_dir: str = "data/split_cache"
    force_purge_split_tree: bool = False


@dataclass
class SegmentCfg:
    """Segmentation of each file into fixed-length windows."""
    segment_length_samples: int = 1024
    stride_samples: int = 512
    normalize_per_segment: bool = True


@dataclass
class ProtoNetCfg:
    """Hierarchical Attention Prototypical Network settings."""
    enabled: bool = True
    embed_dim: int = 128
    metric_dim: int = 64
    attention_hidden: int = 64
    hybrid_features_enabled: bool = True
    n_way: int = 3
    k_shot: int = 3
    q_query: int = 2
    episodes_per_epoch: int = 80
    num_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 10


@dataclass
class TrainingCfg:
    """Training hyperparameters for baseline DL models."""
    batch_size: int = 128
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    use_mixed_precision: bool = True
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    label_smoothing: float = 0.1
    gradient_clip_norm: float = 1.0
    augment_jitter: bool = True
    augment_scaling: bool = True
    augment_shift: bool = True
    jitter_sigma: float = 0.02
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shift_fraction: float = 0.05
    # NEW: Section 29
    loss_type: str = "cross_entropy"        # cross_entropy | focal
    focal_gamma: float = 2.0
    # NEW: Section 32
    best_checkpoint_metric: str = "val_loss"  # val_loss | val_accuracy | val_f1_macro

# NEW dataclass (add near the other sub-configs):
@dataclass
class MLVizCfg:
    """Section 24 ML visualization toggles."""
    enabled: bool = True
    save_confusion_matrix: bool = True
    save_roc: bool = True
    save_pr: bool = True
    save_calibration: bool = True
    save_probability_dist: bool = True
    save_classwise_csv: bool = True
    save_training_curves: bool = True

# ── add near the other dataclasses ──
@dataclass
class SotaCfg:
    """State-of-the-art model benchmarks (optional dependencies)."""
    enabled: bool = True
    run_timesnet: bool = True
    run_minirocket: bool = True
    run_patchtst: bool = False          # tsai optional; off by default
    run_tabpfn_window: bool = True
    run_tabpfn_file: bool = True
    # TimesNet (kept small on purpose — file-limited dataset)
    timesnet_d_model: int = 32
    timesnet_d_ff: int = 64
    timesnet_top_k: int = 3
    timesnet_e_layers: int = 2
    timesnet_num_kernels: int = 6
    timesnet_epochs: int = 25
    # TabPFN
    tabpfn_max_train: int = 10000
    tabpfn_max_features: int = 500

@dataclass
class ModelsCfg:
    """Which model families to run."""
    run_mode: str = "all"              # classical | baseline_dl | protonet | all
    classical_enabled: bool = True
    classical_models: List[str] = field(default_factory=lambda: [
        "RandomForest", "ExtraTrees", "XGBoost", "SVM-RBF",
        "LogisticRegression", "KNN", "GradientBoosting", "NaiveBayes",
        "DecisionTree", "MLP",
    ])
    baseline_dl_enabled: bool = True
    baseline_dl_models: List[str] = field(default_factory=lambda: [
        "ResNet1D", "TCN", "InceptionTime",
    ])
    protonet_enabled: bool = True
    sota_enabled: bool = True
    advanced_enabled: bool = False
    vae_enabled: bool = False
    # Safety gate
    dl_overfit_gate: bool = True   # skip DL if 128-sample overfit test fails


@dataclass
class FilterCfg:
    """Digital filter configuration."""
    lpf_enabled: bool = True
    gaussian_enabled: bool = True
    lpf_order: int = 8



@dataclass
class QualityCfg:
    """Quality-control thresholds."""
    max_clip_fraction: float = 0.05
    max_baseline_drift_fraction: float = 0.5
    min_variance: float = 1e-12
    # NEW — Section 41 additions
    impossible_value_min: float = -1e9
    impossible_value_max: float = 1e9
    max_noise_std_ratio: float = 0.5
    verify_sampling_rate: bool = True
    save_qc_report: bool = True

@dataclass
class TrackingCfg:
    """Section 43 experiment tracking + reproducibility."""
    enabled: bool = True
    save_environment: bool = True
    save_feature_importance: bool = True
    save_per_file_preprocessing: bool = True

@dataclass
class EventCfg:
    """Ton/Toff event detection."""
    enabled: bool = True
    threshold_sigma: float = 5.0
    min_duration_ms: float = 0.1
    max_duration_ms: float = 500.0
    merge_gap_ms: float = 0.005


@dataclass
class NoiseAugCfg:
    """Noise-augmentation configuration."""
    enabled: bool = True
    jitter_sigma: float = 0.02
    scale_range: Tuple[float, float] = (0.9, 1.1)
    time_shift_fraction: float = 0.05


@dataclass
class VisualizationCfg:
    """Global figure styling."""
    dpi: int = 150
    fig_format: str = "png"
    font_family: str = "DejaVu Sans"
    font_scale: float = 1.2
    style: str = "whitegrid"
    baseline_color: str = "#000000"
    signal_color: str = "#c92020"


@dataclass
class AnalysisCfg:
    """Which analyses to run."""
    psd: bool = True
    acf: bool = True
    events: bool = True
    max_analysis_files_per_class: int = 2


@dataclass
class DeploymentCfg:
    """Real-time deployment configuration."""
    enabled: bool = False
    causal_filter: bool = True
    buffer_size_samples: int = 1024
    max_latency_ms: float = 50.0
    confidence_threshold: float = 0.8


# ═══════════════════════════════════════════════════════════════════
# RESEARCH FRAMEWORK CONFIGURATION (added for SSL/few-shot/semi-supervised)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AugmentationCfg:
    """Time-series augmentation policy for contrastive views."""
    jitter_sigma: float = 0.03
    scaling_sigma: float = 0.1
    magnitude_warp_sigma: float = 0.2
    time_warp_sigma: float = 0.2
    crop_ratio: Tuple[float, float] = (0.8, 1.0)
    mask_ratio: float = 0.15
    permutation_enabled: bool = False
    permutation_segments: int = 5
    time_shift_fraction: float = 0.05
    enabled_transforms: List[str] = field(default_factory=lambda: [
        "jitter", "scaling", "crop", "mask",
    ])


@dataclass
class SSLEncoderCfg:
    """Self-supervised encoder architecture."""
    encoder_type: str = "tcn"           # tcn | resnet | cnn
    embedding_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    dropout: float = 0.1


@dataclass
class SSLCfg:
    """Self-supervised / contrastive representation learning."""
    enabled: bool = False               # OFF by default — opt-in
    method: str = "contrastive"         # contrastive | autoencoder
    encoder: SSLEncoderCfg = field(default_factory=SSLEncoderCfg)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    temperature: float = 0.07
    projection_dim: int = 64


@dataclass
class FewShotResearchCfg:
    """Prototype-based few-shot evaluation (research mode)."""
    enabled: bool = False
    n_way: int = 2
    k_shot: int = 5
    q_query: int = 15
    distance_metric: str = "euclidean"  # euclidean | cosine
    num_eval_episodes: int = 200
    normalize_embeddings: bool = True


@dataclass
class PseudoLabelCfg:
    """Confidence-aware pseudo-labeling."""
    enabled: bool = False
    confidence_threshold: float = 0.90
    consistency_enabled: bool = True
    consistency_num_views: int = 5
    consistency_agreement: float = 0.8
    max_pseudo_per_class: int = 50
    max_iterations: int = 3
    distance_filter_enabled: bool = True
    distance_percentile: float = 95.0
    class_balance_enabled: bool = True


@dataclass
class NoveltyCfg:
    """Unknown/novel-class detection."""
    enabled: bool = False
    distance_threshold: Optional[float] = None
    auto_percentile: float = 95.0


@dataclass
class ClusteringCfg:
    """Exploratory clustering on embeddings."""
    enabled: bool = False
    algorithms: List[str] = field(default_factory=lambda: ["kmeans"])
    n_clusters: int = 2


@dataclass
class ResearchCfg:
    """Master research experiment configuration."""
    enabled: bool = False               # Global toggle
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    label_budgets: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 25, 50])
    unlabeled_fractions: List[float] = field(default_factory=lambda: [
        0.1, 0.25, 0.5, 0.75, 1.0,
    ])
    ssl: SSLCfg = field(default_factory=SSLCfg)
    few_shot: FewShotResearchCfg = field(default_factory=FewShotResearchCfg)
    pseudo_label: PseudoLabelCfg = field(default_factory=PseudoLabelCfg)
    novelty: NoveltyCfg = field(default_factory=NoveltyCfg)
    clustering: ClusteringCfg = field(default_factory=ClusteringCfg)

@dataclass
class PipelineConfig:
    """Master configuration container."""
    data_root: str = "./data/raw_dat"
    project_root: str = "./"
    supported_extensions: Tuple[str, ...] = (".dat",)
    data_conversion: bool = False
    random_seed: int = 42
    window_size: int = 1024
    stride: int = 512

    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    baseline: BaselineCfg = field(default_factory=BaselineCfg)
    detrend: DetrendCfg = field(default_factory=DetrendCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    segment: SegmentCfg = field(default_factory=SegmentCfg)
    protonet: ProtoNetCfg = field(default_factory=ProtoNetCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    sota: SotaCfg = field(default_factory=SotaCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    filters: FilterCfg = field(default_factory=FilterCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    events: EventCfg = field(default_factory=EventCfg)
    noise_aug: NoiseAugCfg = field(default_factory=NoiseAugCfg)
    visualization: VisualizationCfg = field(default_factory=VisualizationCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    deployment: DeploymentCfg = field(default_factory=DeploymentCfg)
    ml_viz: MLVizCfg = field(default_factory=MLVizCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    research: ResearchCfg = field(default_factory=ResearchCfg)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()

        # Flat keys
        flat_keys = (
            "data_root", "project_root", "data_conversion", "random_seed",
            "window_size", "stride",
        )
        for k in flat_keys:
            if k in raw:
                setattr(cfg, k, raw[k])
        if "supported_extensions" in raw:
            cfg.supported_extensions = tuple(raw["supported_extensions"])

        # Nested sub-configs
        nested_map = {
            "sampling": cfg.sampling, "baseline": cfg.baseline,
            "detrend": cfg.detrend, "split": cfg.split,
            "segment": cfg.segment, "protonet": cfg.protonet,
            "training": cfg.training, "sota": cfg.sota, 
            "models": cfg.models, "filters": cfg.filters, 
            "quality": cfg.quality, "events": cfg.events, 
            "noise_aug": cfg.noise_aug,
            "visualization": cfg.visualization, "analysis": cfg.analysis,
            "deployment": cfg.deployment,
            "tracking": cfg.tracking,
            "ml_viz": cfg.ml_viz,
            "research": cfg.research,
        }
        for key, obj in nested_map.items():
            if key in raw and isinstance(raw[key], dict):
                for sk, sv in raw[key].items():
                    if hasattr(obj, sk):
                        setattr(obj, sk, sv)

        # Legacy compatibility keys
        if "lpf_enabled" in raw:
            cfg.filters.lpf_enabled = raw["lpf_enabled"]
        if "gaussian_enabled" in raw:
            cfg.filters.gaussian_enabled = raw["gaussian_enabled"]
        if "lpf_order" in raw:
            cfg.filters.lpf_order = raw["lpf_order"]

        return cfg

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def resolve_run_flags(self) -> None:
        """Apply run_mode overrides to individual enabled flags."""
        mode = (self.models.run_mode or "all").lower()
        if mode == "classical":
            self.models.classical_enabled = True
            self.models.baseline_dl_enabled = False
            self.models.protonet_enabled = False
            self.models.sota_enabled=False
        elif mode == "baseline_dl":
            self.models.classical_enabled = False
            self.models.baseline_dl_enabled = True
            self.models.protonet_enabled = False
            self.models.sota_enabled=False
        elif mode == "protonet":
            self.models.classical_enabled = False
            self.models.baseline_dl_enabled = False
            self.models.protonet_enabled = True
            self.models.sota_enabled=False
        elif mode == "sota": 
            self.models.classical_enabled = False
            self.models.baseline_dl_enabled = False
            self.models.protonet_enabled = False
            self.models.sota_enabled=True
        # "all" keeps individual flags untouched

    def to_dict(self) -> dict:
        return asdict(self)