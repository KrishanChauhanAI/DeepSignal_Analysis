"""Entry point for the nanobio pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from nanobio.config import PipelineConfig
from nanobio.pipeline import NanoBioPipeline

DEFAULT_YAML = """\
data_root: /data/In Vitro/
project_root: /data/project/
supported_extensions: [".dat"] #[".dat", ".csv", ".txt"]
data_conversion: false
random_seed: 42
window_size: 1024
stride: 512

sampling:
  original_rate_hz: 1000000
  target_rates_hz: [50000, 10000, 1000, 100, 10, 2]
  model_input_rate_hz: 10000

baseline:
  method: rolling_median    #auto                          # objective selection (or fix: rolling_median)
  window_seconds: 1.0
  poly_order: 3
  savgol_window: 501
  save_visualizations: true
  max_files_visualized: 20
  compute_baseline_range: true          # set false to hide from plots
  compute_residual_std: true            # set false to hide from plots
  show_flags_in_plot_annotation: true
  auto_candidates:
    - constant
    - linear
    - rolling_median
    - savgol
  event_sigma: 5.0
  log_selection_details: true

detrend:
  enabled: true
  method: subtract_baseline

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true
  seed: 42
  #------Added------
  links_enabled: true
  links_dir: ""          # empty = <data_root>/../split_links (recommended: stays on F: drive)
  link_mode: auto        # auto | hardlink | symlink | copy
  verify_md5: false
  purge_on_change: true
  use_npz_cache: false   # set true later for fast re-runs on the full dataset
  # auto-purge only if split assignment changed
  force_purge_split_tree: true      # NEW: always wipe train/val/test on every run

segment:
  segment_length_samples: 1024
  stride_samples: 512
  normalize_per_segment: true

models:
  run_mode: all          # classical | baseline_dl | protonet | all
  classical_enabled: true
  classical_models:
    - RandomForest
    - ExtraTrees
    - XGBoost
    - LogisticRegression
    - SVM-RBF
  baseline_dl_enabled: true
  baseline_dl_models:
    - ResNet1D
    - TCN
    - InceptionTime
  protonet_enabled: true
  advanced_enabled: false
  vae_enabled: false

protonet:
  enabled: true
  embed_dim: 128
  metric_dim: 64
  attention_hidden: 64
  hybrid_features_enabled: true
  n_way: 3
  k_shot: 3
  q_query: 2
  episodes_per_epoch: 80
  num_epochs: 40
  learning_rate: 0.001
  early_stopping_patience: 10

training:
  batch_size: 128
  num_epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 7
  use_class_weights: true
  use_weighted_sampler: true
  label_smoothing: 0.1
  gradient_clip_norm: 1.0
  loss_type: cross_entropy          # cross_entropy | focal
  focal_gamma: 2.0
  best_checkpoint_metric: val_loss  # val_loss | val_accuracy | val_f1_macro

ml_viz:
  enabled: true
  save_confusion_matrix: true
  save_roc: true
  save_pr: true
  save_calibration: true
  save_probability_dist: true
  save_classwise_csv: true
  save_training_curves: true

filters:
  lpf_enabled: true
  gaussian_enabled: true
  lpf_order: 8

quality:
  max_clip_fraction: 0.05
  max_baseline_drift_fraction: 0.5
  min_variance: 1.0e-12
  # NEW — Section 41
  impossible_value_min: -1e9
  impossible_value_max: 1e9
  max_noise_std_ratio: 0.5
  verify_sampling_rate: true
  save_qc_report: true

tracking:
  enabled: true
  save_environment: true            # Section 37
  save_feature_importance: true     # Section 43
  save_per_file_preprocessing: true # Section 37

events:
  enabled: true
  threshold_sigma: 5.0
  min_duration_ms: 0.1
  max_duration_ms: 500.0
  merge_gap_ms: 0.005

visualization:
  dpi: 150
  fig_format: png
  font_scale: 1.2
  style: whitegrid
  baseline_color: "#000000"
  signal_color: "#c92020"

deployment:
  enabled: false
  causal_filter: true
  buffer_size_samples: 1024
  max_latency_ms: 50.0
  confidence_threshold: 0.8

# Add to DEFAULT_YAML string in main.py, after the deployment section:

research:
  enabled: false              # Set true to run research phases
  seeds: [42, 123, 456]
  label_budgets: [1, 2, 5, 10, 25, 50]
  ssl:
    enabled: false
    method: contrastive
    epochs: 50
    batch_size: 256
    encoder:
      encoder_type: tcn
      embedding_dim: 128
  few_shot:
    enabled: false
    k_shot: 5
    num_eval_episodes: 200
  pseudo_label:
    enabled: false
    confidence_threshold: 0.90
    max_iterations: 3

"""


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", mode="a", encoding="utf-8"),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Nanobio pipeline entry point")
    ap.add_argument("--config", default="configs/protonet.yaml",
                    help="Path to YAML configuration file")
    args = ap.parse_args()

    setup_logging(Path("logs"))
    cfg_path = Path(args.config)

    if cfg_path.exists():
        config = PipelineConfig.from_yaml(cfg_path)
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(DEFAULT_YAML, encoding="utf-8")
        config = PipelineConfig.from_yaml(cfg_path)
        logging.info("Default config written to %s", cfg_path)

    NanoBioPipeline(config).run()


if __name__ == "__main__":
    main()