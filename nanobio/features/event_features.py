'''
"""
Physical event-based feature extraction (Sections 9, 13, 15).
"""
from typing import Dict, List

import numpy as np

from nanobio.signal.events import EventDetector


class EventFeatureExtractor:
    FEATURE_NAMES = [
        "n_events", "event_density_hz", "Ton_mean_ms", "Toff_mean_ms", 
        "delta_I_I0_mean", "delta_I_I0_max", "RMS_normalized", "RMS_total"
    ]

    def __init__(self, detector: EventDetector):
        self.detector = detector

    def extract(self, signal: np.ndarray, baseline: np.ndarray, fs: int) -> np.ndarray:
        residual = signal - baseline
        i0 = float(np.median(baseline)) if abs(np.median(baseline)) > 1e-9 else 1e-9
        
        rms_total = float(np.sqrt(np.mean(signal**2)))
        rms_base = float(np.sqrt(np.mean(baseline**2))) + 1e-9
        rms_norm = rms_total / rms_base
        
        events = self.detector.detect(residual)
        
        if not events:
            return np.array([0, 0, 0, 0, 0, 0, rms_norm, rms_total], dtype=np.float32)

        tons = [e.duration_s * 1000 for e in events]
        toffs = [events[i].start_time_s - events[i-1].end_time_s for i in range(1, len(events))]
        delta_i = [(e.amplitude - i0) / i0 for e in events]
        
        return np.nan_to_num(np.array([
            len(events),
            len(events) / (len(signal) / fs),
            np.mean(tons),
            np.mean(toffs) if toffs else 0.0,
            np.mean(delta_i),
            np.min(delta_i), # Traps are usually negative
            rms_norm,
            rms_total
        ], dtype=np.float32))
        
'''
"""
Extracts physical event-based features (Ton, Toff, ΔI/I₀, RMS_norm).
Fixes the critical gap of not using events for classification.
"""
from typing import Dict

import numpy as np

from nanobio.signal.events import EventDetector


class EventFeatureExtractor:
    def __init__(self, detector: EventDetector):
        self.detector = detector

    def extract_file_level_features(
        self, signal: np.ndarray, fs: int, baseline: np.ndarray, i0_method: str = "median"
    ) -> np.ndarray:
        """Returns a fixed-length vector of physical event statistics for ML."""
        # 1. Configurable I0 definition
        if i0_method == "median":
            i0 = np.median(signal)
        elif i0_method == "pre_event":
            i0 = np.mean(signal[:max(10, len(signal)//10)])
        else:
            i0 = np.mean(baseline)
            
        residual = signal - baseline
        
        # 2. Normalized RMS
        rms_total = np.sqrt(np.mean(signal**2))
        rms_base = np.sqrt(np.mean(baseline**2)) + 1e-9
        rms_norm = rms_total / rms_base
        
        # 3. Detect Events
        events = self.detector.detect(residual)
        stats = self.detector.compute_ton_toff(events)["statistics"]
        
        # 4. Compile vector
        n_events = len(events)
        ton_mean = stats.get("ton_mean", 0.0)
        toff_mean = stats.get("toff_mean", 0.0)
        
        delta_is = []
        for e in events:
            # ΔI/I₀ = (I - I₀)/I₀
            di_i0 = (e.amplitude - i0) / (abs(i0) + 1e-9)
            delta_is.append(di_i0)
            
        di_mean = np.mean(delta_is) if delta_is else 0.0
        di_max = np.min(delta_is) if delta_is else 0.0 # Traps are negative
        
        event_density = n_events / (len(signal) / fs)
        
        return np.nan_to_num(np.array([
            n_events, event_density, ton_mean, toff_mean, 
            di_mean, di_max, rms_norm, rms_total
        ], dtype=np.float32))

    @staticmethod
    def get_feature_names() -> list:
        return [
            "n_events", "event_density_hz", "Ton_mean_s", "Toff_mean_s", 
            "delta_I_I0_mean", "delta_I_I0_max", "RMS_normalized", "RMS_total"
        ]