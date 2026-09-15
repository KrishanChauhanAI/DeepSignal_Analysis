"""Signal processing: filters, baseline, detrending, quality, events."""

from nanobio.signal.baseline import BaselineEstimator, BaselineResult
from nanobio.signal.detrend import Detrender
from nanobio.signal.events import DetectedEvent, EventDetector
from nanobio.signal.filters import FilterResult, SignalProcessor
from nanobio.signal.quality import DataQualityChecker, QualityReport

__all__ = [
    "FilterResult", "SignalProcessor",
    "BaselineEstimator", "BaselineResult",
    "Detrender",
    "DataQualityChecker", "QualityReport",
    "DetectedEvent", "EventDetector",
]