"""
Power Spectral Density and Autocorrelation Function helpers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import welch


def compute_psd(
    signal: np.ndarray, fs: int, nperseg: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate. Returns (frequencies, psd)."""
    if nperseg is None:
        nperseg = min(len(signal), 2048)
    return welch(signal, fs=fs, nperseg=nperseg, scaling="density")


def compute_acf(
    signal: np.ndarray, max_lag: int = None, normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Autocorrelation via FFT. Returns (lags, acf_values)."""
    n = len(signal)
    if max_lag is None:
        max_lag = min(n // 10, 10_000)
    x = signal - np.mean(signal)
    fft_x = np.fft.rfft(x, n=2 * n)
    acf_full = np.fft.irfft(fft_x * np.conj(fft_x))[:n]
    if normalize and acf_full[0] != 0:
        acf_full = acf_full / acf_full[0]
    lags = np.arange(min(max_lag, n))
    return lags, acf_full[: len(lags)]