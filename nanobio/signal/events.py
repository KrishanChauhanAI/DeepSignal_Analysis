"""
Ton / Toff event detection.

Threshold-based detection is used because it is reproducible, fast,
and interpretable. Alternatives (CUSUM, HMM) can be added later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from nanobio.config import EventCfg

logger = logging.getLogger("nanobio.signal.events")


@dataclass
class DetectedEvent:
    """A single detected trapping / translocation event."""
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    amplitude: float
    baseline_before: float
    delta_i: float
    delta_i_over_i0: float


class EventDetector:
    """Threshold-based event detector."""

    def __init__(self, cfg: EventCfg, sampling_rate_hz: int):
        self.cfg = cfg
        self.fs = sampling_rate_hz

    def detect(self, signal: np.ndarray) -> List[DetectedEvent]:
        """Return list of detected events (below-baseline blockades)."""
        n_baseline = max(100, len(signal) // 10)
        baseline = float(np.median(signal[:n_baseline]))
        noise_std = float(np.std(signal[:n_baseline]))
        threshold = baseline - self.cfg.threshold_sigma * noise_std

        min_samples = max(1, int(self.cfg.min_duration_ms * 1e-3 * self.fs))
        max_samples = int(self.cfg.max_duration_ms * 1e-3 * self.fs)
        merge_samples = int(self.cfg.merge_gap_ms * 1e-3 * self.fs)

        below = signal < threshold
        events: List[DetectedEvent] = []
        in_event = False
        start = 0
        for i, b in enumerate(below):
            if b and not in_event:
                in_event = True
                start = i
            elif not b and in_event:
                in_event = False
                end = i
                dur = end - start
                if min_samples <= dur <= max_samples:
                    amp = float(np.mean(signal[start:end]))
                    pre_start = max(0, start - n_baseline)
                    pre_base = float(np.median(signal[pre_start:start]))
                    di = amp - pre_base
                    din = di / (abs(pre_base) + 1e-15)
                    events.append(DetectedEvent(
                        start_idx=start, end_idx=end,
                        start_time_s=start / self.fs, end_time_s=end / self.fs,
                        duration_s=dur / self.fs,
                        amplitude=amp, baseline_before=pre_base,
                        delta_i=di, delta_i_over_i0=din,
                    ))

        # Merge nearby events
        if merge_samples > 0 and len(events) > 1:
            merged = [events[0]]
            for e in events[1:]:
                gap = e.start_idx - merged[-1].end_idx
                if gap <= merge_samples:
                    p = merged[-1]
                    merged[-1] = DetectedEvent(
                        start_idx=p.start_idx, end_idx=e.end_idx,
                        start_time_s=p.start_time_s, end_time_s=e.end_time_s,
                        duration_s=(e.end_idx - p.start_idx) / self.fs,
                        amplitude=float(np.mean(signal[p.start_idx:e.end_idx])),
                        baseline_before=p.baseline_before,
                        delta_i=float(np.mean(signal[p.start_idx:e.end_idx])) - p.baseline_before,
                        delta_i_over_i0=(
                            float(np.mean(signal[p.start_idx:e.end_idx])) - p.baseline_before
                        ) / (abs(p.baseline_before) + 1e-15),
                    )
                else:
                    merged.append(e)
            events = merged

        return events

    def compute_ton_toff(self, events: List[DetectedEvent]) -> Dict[str, Any]:
        """Compute Ton (event duration) and Toff (inter-event gap) statistics."""
        if not events:
            return {"ton": [], "toff": [], "statistics": {}}
        ton = [e.duration_s for e in events]
        toff = [
            events[i].start_time_s - events[i - 1].end_time_s
            for i in range(1, len(events))
            if events[i].start_time_s > events[i - 1].end_time_s
        ]

        def _stats(vals, name):
            if not vals:
                return {}
            a = np.asarray(vals)
            return {
                f"{name}_mean": float(np.mean(a)),
                f"{name}_median": float(np.median(a)),
                f"{name}_std": float(np.std(a)),
                f"{name}_iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
                f"{name}_count": len(vals),
            }
        return {
            "ton": ton, "toff": toff,
            "statistics": {**_stats(ton, "ton"), **_stats(toff, "toff")},
        }