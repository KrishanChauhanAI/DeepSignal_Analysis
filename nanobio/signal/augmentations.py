"""
Time-series augmentations for contrastive learning.

All augmentations preserve class semantics for biosignal data.
Permutation is disabled by default (destroys event structure).
"""

from __future__ import annotations

import logging
from typing import Callable, List

import numpy as np
from scipy.interpolate import CubicSpline

from nanobio.config import AugmentationCfg

logger = logging.getLogger("nanobio.signal.augmentations")


def jitter(x: np.ndarray, sigma: float = 0.03,
           rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return x + rng.normal(0, sigma, size=x.shape).astype(x.dtype)


def scaling(x: np.ndarray, sigma: float = 0.1,
            rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return (x * max(0.5, rng.normal(1.0, sigma))).astype(x.dtype)


def random_crop(x: np.ndarray, crop_ratio: tuple = (0.8, 1.0),
                rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    T = len(x)
    crop_len = max(1, int(T * rng.uniform(*crop_ratio)))
    start = rng.integers(0, T - crop_len + 1)
    cropped = x[start:start + crop_len]
    if len(cropped) < T:
        pad_l = (T - len(cropped)) // 2
        pad_r = T - len(cropped) - pad_l
        cropped = np.pad(cropped, (pad_l, pad_r), constant_values=0)
    return cropped.astype(x.dtype)


def temporal_mask(x: np.ndarray, mask_ratio: float = 0.15,
                  rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    T = len(x)
    out = x.copy()
    n = max(1, int(T * mask_ratio))
    start = rng.integers(0, max(1, T - n))
    out[start:start + n] = 0.0
    return out


def time_shift(x: np.ndarray, fraction: float = 0.05,
               rng: np.random.Generator = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    shift = rng.integers(-int(len(x) * fraction), int(len(x) * fraction) + 1)
    return np.roll(x, shift).astype(x.dtype)


_REGISTRY = {
    "jitter": jitter, "scaling": scaling, "crop": random_crop,
    "mask": temporal_mask, "time_shift": time_shift,
}


class AugmentationComposer:
    """Compose augmentations into a view generator."""

    def __init__(self, cfg: AugmentationCfg, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._transforms = []
        param_map = {
            "jitter": {"sigma": cfg.jitter_sigma},
            "scaling": {"sigma": cfg.scaling_sigma},
            "crop": {"crop_ratio": cfg.crop_ratio},
            "mask": {"mask_ratio": cfg.mask_ratio},
            "time_shift": {"fraction": cfg.time_shift_fraction},
        }
        for name in cfg.enabled_transforms:
            if name in _REGISTRY:
                self._transforms.append(
                    (name, _REGISTRY[name], param_map.get(name, {}))
                )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for _, fn, params in self._transforms:
            out = fn(out, rng=self.rng, **params)
        return out

    def generate_views(self, x: np.ndarray, n: int = 2) -> List[np.ndarray]:
        return [self(x) for _ in range(n)]