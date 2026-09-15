# Nanobio Signal Analysis Pipeline

Modular PyTorch pipeline for classifying 1 MHz raw electrical/current signals
from nanobiotechnology experiments (nanopore, nanoelectrode, single-molecule sensing).

## Features

- Handles `.dat`, `.csv`, `.txt` file formats
- Per-file baseline visualization (raw + baseline overlay)
- File-level train/val/test splitting with leakage checks
- Anti-alias LPF + Gaussian filter downsampling
- Three complementary model paths:
  - **Classical**: RandomForest / XGBoost / SVM on handcrafted features
  - **Baseline DL**: ResNet1D / TCN / InceptionTime on raw windows
  - **Hierarchical ProtoNet**: file-level episodic metric learning

## Installation

```bash
pip install -r requirements.txt


# Runs everything (classical + baseline DL + ProtoNet)
python main.py --config configs/protonet.yaml



---

## Running the Pipeline

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place your data
mkdir -p data/raw_dat
# Copy your category folders (e.g., InVivo_Control_Lysate, InVivo_WT_tau_Lysate) here.
# Or edit data_root in configs/protonet.yaml to point elsewhere.

# 3. Run all model families
python main.py --config configs/protonet.yaml

# 4. Run only ProtoNet
# Edit configs/protonet.yaml → models.run_mode: protonet
python main.py --config configs/protonet.yaml

# 5. Run tests
pytest tests/ -v

python main.py --config configs/protonet.yaml