"""Smoke tests for critical paths."""

import numpy as np
import pytest
import torch
from nanobio.config import PipelineConfig
from nanobio.features import FeatureExtractor, HybridFeatureExtractor
from nanobio.models import HierarchicalProtoNet, ResNet1D
from nanobio.signal import (
    BaselineEstimator,
    Detrender,
    FilterResult,
    SignalProcessor,
)
from nanobio.training import FileEpisodeSampler


def test_filter_result_default_construction():
    fr = FilterResult()
    assert isinstance(fr.signal, np.ndarray)
    assert fr.filter_type == "none"
    assert fr.filter_impl == "none"


def test_signal_processor_lpf():
    sp = SignalProcessor(1_000_000, 8)
    sig = np.random.randn(100_000).astype(np.float32)
    r = sp.downsample(sig, 10_000, "LPF")
    assert r.filter_type == "LPF"
    assert r.filter_impl == "butterworth"
    assert r.target_rate_hz == 10_000
    assert len(r.signal) == 1000


def test_baseline_removes_dc_step():
    fs = 1000
    sig = np.concatenate([
        np.random.randn(2 * fs) * 0.1 + 2.0,
        np.random.randn(2 * fs) * 0.1 + 5.0,
    ]).astype(np.float32)
    est = BaselineEstimator(fs)
    r = est.rolling_median(sig, window_seconds=0.5)
    assert abs(r.baseline[fs // 4] - 2.0) < 0.5
    assert abs(r.baseline[-fs // 4] - 5.0) < 0.5


def test_detrender():
    fs = 1000
    sig = np.random.randn(1000).astype(np.float32) + 100.0
    est = BaselineEstimator(fs)
    br = est.rolling_median(sig, 0.5)
    det = Detrender.apply(sig, "subtract_baseline", br)
    assert abs(det.mean()) < abs(sig.mean())


def test_feature_extractor():
    x = np.random.randn(1024).astype(np.float32)
    fe = FeatureExtractor()
    feats = fe.extract(x)
    assert feats.shape == (len(FeatureExtractor.FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_hybrid_extractor():
    x = np.random.randn(1024).astype(np.float32)
    hfe = HybridFeatureExtractor()
    feats = hfe.extract(x)
    assert feats.shape == (len(HybridFeatureExtractor.NAMES),)


def test_episode_sampler_no_overlap():
    labels = [0] * 20 + [1] * 20 + [2] * 20
    s = FileEpisodeSampler(labels, 3, 3, 2, 10, seed=42)
    for sf, sy, qf, qy in s:
        assert not (set(sf) & set(qf))
        assert len(sf) == 9
        assert len(qf) == 6


def test_protonet_shapes():
    m = HierarchicalProtoNet(
        embed_dim=64, metric_dim=32, attention_hidden=32,
        phys_dim=13, hybrid_enabled=True,
    )
    n_way, k_shot, q_query = 2, 2, 2
    L = 1024
    s_segs = [torch.randn(4, 1, L) for _ in range(n_way * k_shot)]
    s_phys = [torch.randn(4, 13) for _ in range(n_way * k_shot)]
    s_y = torch.tensor([0, 0, 1, 1])
    q_segs = [torch.randn(4, 1, L) for _ in range(n_way * q_query)]
    q_phys = [torch.randn(4, 13) for _ in range(n_way * q_query)]
    out = m(s_segs, s_phys, s_y, q_segs, q_phys, n_way)
    assert out.logits.shape == (n_way * q_query, n_way)


def test_resnet1d_forward():
    m = ResNet1D(3)
    x = torch.randn(2, 1, 1024)
    out = m(x)
    assert out.shape == (2, 3)