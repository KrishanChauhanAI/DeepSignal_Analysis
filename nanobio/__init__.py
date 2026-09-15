"""
Nanobio: High-Frequency Signal Analysis Pipeline
=================================================
Modular pipeline for classifying 1 MHz raw electrical/current signals
from nanobiotechnology experiments.

Sub-packages:
    io       — File reading, conversion, and splitting
    signal   — Filtering, baseline, detrending, quality, events
    viz      — Visualization (baseline overlays, filter comparisons)
    features — Handcrafted feature extraction
    models   — Classical + deep learning + ProtoNet models
    training — Episodic sampler, trainer, metrics
    explain  — Attribution and interpretability
    deploy   — Real-time streaming inference
    pipeline — Master orchestrator
"""

__version__ = "1.0.0"

# Re-export key classes for convenience
from nanobio.config import PipelineConfig
from nanobio.pipeline import NanoBioPipeline

__all__ = ["PipelineConfig", "NanoBioPipeline", "__version__"]