"""
Real-time streaming inference engine.

Uses causal filters only (no future samples). Produces per-window
predictions with configurable confidence thresholds.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from nanobio.config import DeploymentCfg
from nanobio.signal.filters import SignalProcessor

logger = logging.getLogger("nanobio.deploy.streaming")


class StreamingInferenceEngine:
    """
    Causal real-time inference.

    Consumes samples one chunk at a time, maintains a rolling buffer,
    and emits (prediction, confidence, latency_ms) tuples for each
    completed window.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DeploymentCfg,
        window_size: int,
        original_rate_hz: int,
        target_rate_hz: int,
        device: torch.device = None,
    ):
        self.model = model
        self.cfg = cfg
        self.window_size = window_size
        self.processor = SignalProcessor(original_rate_hz)
        self.target_rate = target_rate_hz
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.buffer: Deque[float] = deque(maxlen=cfg.buffer_size_samples * 10)
        self.decimation = max(1, original_rate_hz // target_rate_hz)
        self._sample_counter = 0

    def push(self, samples: np.ndarray) -> List[Tuple[int, float, float]]:
        """
        Push new samples. Returns list of (predicted_class, confidence, latency_ms).
        """
        t0 = time.time()
        self.buffer.extend(samples.tolist())
        outputs = []

        # Only run when buffer has enough for a full window at target rate
        needed = self.window_size * self.decimation
        while len(self.buffer) >= needed:
            raw_chunk = np.asarray([self.buffer.popleft() for _ in range(needed)],
                                    dtype=np.float32)
            # Causal downsampling (causal=True → sosfilt, not sosfiltfilt)
            result = self.processor.downsample(
                raw_chunk, self.target_rate, "LPF", causal=True,
            )
            window = result.signal[: self.window_size]
            window = (window - window.mean()) / (window.std() + 1e-6)

            with torch.no_grad():
                x = torch.from_numpy(window.astype(np.float32))
                x = x.unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
                latency_ms = (time.time() - t0) * 1000
                if conf.item() >= self.cfg.confidence_threshold:
                    outputs.append((int(pred.item()), float(conf.item()), latency_ms))
                self._sample_counter += 1
        return outputs