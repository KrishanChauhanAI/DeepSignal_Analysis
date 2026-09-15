"""
Digital signal-processing filters with anti-alias downsampling.

GAP-002 FIX: Added apply_gaussian_lpf() for post-resampling Gaussian LPF.
GAP-003 FIX: Added compute_filter_diagnostics() for frequency-domain validation.
BUG-021 FIX: Documented Gaussian cutoff definition explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, freqz, sosfilt, sosfiltfilt

logger = logging.getLogger("nanobio.signal.filters")


@dataclass
class FilterResult:
    """Container for a filtered / downsampled signal with full provenance."""
    signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filter_type: str = "none"
    filter_impl: str = "none"
    order: Optional[int] = None
    cutoff_hz: Optional[float] = None
    sigma_samples: Optional[float] = None
    original_rate_hz: int = 1_000_000
    target_rate_hz: int = 1_000_000
    phase_mode: str = "zero_phase"
    decimation_factor: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FilterDiagnostics:
    """GAP-003 FIX: Frequency-domain validation data for reporting.

    Stores spectra before/after filtering, filter frequency response,
    and attenuation metrics for inclusion in PDF/Excel reports.
    """
    freq_axis_hz: Optional[np.ndarray] = None
    spectrum_before: Optional[np.ndarray] = None
    spectrum_after: Optional[np.ndarray] = None
    filter_freq_response_hz: Optional[np.ndarray] = None
    filter_freq_response_mag: Optional[np.ndarray] = None
    filter_freq_response_db: Optional[np.ndarray] = None
    attenuation_at_cutoff_db: Optional[float] = None
    attenuation_at_nyquist_db: Optional[float] = None
    effective_bandwidth_hz: Optional[float] = None


class SignalProcessor:
    """Anti-alias filtering + decimation + Gaussian LPF."""

    def __init__(self, original_rate_hz: int = 1_000_000, lpf_order: int = 8):
        self.original_rate = original_rate_hz
        self.lpf_order = lpf_order

    def design_lpf(self, cutoff_hz: float, fs_hz: int, order: int) -> np.ndarray:
        """Butterworth SOS low-pass filter."""
        nyquist = fs_hz / 2.0
        if cutoff_hz >= nyquist:
            logger.warning(
                "Cutoff %.1f Hz >= Nyquist %.1f Hz. Clamping.",
                cutoff_hz, nyquist,
            )
            cutoff_hz = 0.99 * nyquist
        return butter(order, cutoff_hz / nyquist, btype="low", output="sos")

    def apply_lpf(
        self, signal: np.ndarray, sos: np.ndarray, causal: bool = False,
    ) -> np.ndarray:
        if causal:
            return sosfilt(sos, signal).astype(signal.dtype)
        return sosfiltfilt(sos, signal).astype(signal.dtype)

    @staticmethod
    def sigma_for_cutoff(cutoff_hz: float, fs_hz: int) -> float:
        """Gaussian sigma that produces −3 dB at cutoff_hz.

        BUG-021 FIX: Explicit mathematical definition.

        The Gaussian filter in the frequency domain is:
            H(f) = exp(-2π²σ²f²)

        At −3 dB: H(f_c) = 1/√2, so:
            exp(-2π²σ²f_c²) = 1/√2
            σ = sqrt(ln(2)) / (π * f_c)   [in seconds]
            σ_samples = σ * fs

        This is the standard definition used throughout the pipeline.
        """
        sigma_seconds = np.sqrt(np.log(2)) / (np.pi * cutoff_hz)
        return max(1.0, sigma_seconds * fs_hz)

    def apply_gaussian(self, signal: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter1d(signal, sigma=sigma).astype(signal.dtype)

    def downsample(
        self,
        signal: np.ndarray,
        target_rate_hz: int,
        filter_type: str = "LPF",
        causal: bool = False,
    ) -> FilterResult:
        """Anti-alias filter and decimate (step 1 of the processing chain)."""
        if target_rate_hz >= self.original_rate:
            return FilterResult(
                signal=signal.copy(),
                filter_type="none",
                filter_impl="none",
                original_rate_hz=self.original_rate,
                target_rate_hz=self.original_rate,
                phase_mode="causal" if causal else "zero_phase",
            )

        decimation = max(1, self.original_rate // target_rate_hz)
        antialias_cutoff = 0.4 * target_rate_hz

        if filter_type == "LPF":
            sos = self.design_lpf(antialias_cutoff, self.original_rate, self.lpf_order)
            filtered = self.apply_lpf(signal, sos, causal=causal)
            downsampled = filtered[::decimation]
            return FilterResult(
                signal=downsampled,
                filter_type="LPF",
                filter_impl="butterworth",
                order=self.lpf_order,
                cutoff_hz=antialias_cutoff,
                original_rate_hz=self.original_rate,
                target_rate_hz=target_rate_hz,
                phase_mode="causal" if causal else "zero_phase",
                decimation_factor=decimation,
            )

        if filter_type == "Gaussian":
            sigma = self.sigma_for_cutoff(antialias_cutoff, self.original_rate)
            filtered = self.apply_gaussian(signal, sigma)
            downsampled = filtered[::decimation]
            return FilterResult(
                signal=downsampled,
                filter_type="Gaussian",
                filter_impl="gaussian",
                cutoff_hz=antialias_cutoff,
                sigma_samples=sigma,
                original_rate_hz=self.original_rate,
                target_rate_hz=target_rate_hz,
                phase_mode="zero_phase",
                decimation_factor=decimation,
            )

        raise ValueError(f"Unknown filter type: {filter_type!r}")

    # ── GAP-002 FIX: Post-resampling Gaussian LPF ──

    def apply_gaussian_lpf(
        self,
        signal: np.ndarray,
        cutoff_hz: float,
        fs_hz: int,
    ) -> Tuple[np.ndarray, float]:
        """Apply Gaussian LPF AFTER resampling (step 2 of the chain).

        This is separate from the anti-aliasing filter applied before
        decimation. The default pipeline is:

            1 MHz → Butterworth anti-alias → decimate to 50 kHz
                  → Gaussian LPF at 10 kHz → analysis

        Args:
            signal: Resampled signal at fs_hz.
            cutoff_hz: Gaussian −3 dB cutoff frequency.
            fs_hz: Sampling rate of the input signal.

        Returns:
            (filtered_signal, sigma_samples)
        """
        sigma = self.sigma_for_cutoff(cutoff_hz, fs_hz)
        filtered = self.apply_gaussian(signal, sigma)
        logger.info(
            "Gaussian LPF applied: cutoff=%d Hz, σ=%.2f samples, fs=%d Hz",
            cutoff_hz, sigma, fs_hz,
        )
        return filtered, sigma

    def downsample_and_filter(
        self,
        signal: np.ndarray,
        target_rate_hz: int,
        gaussian_cutoff_hz: Optional[int] = None,
        anti_alias_type: str = "LPF",
        causal: bool = False,
    ) -> Tuple[FilterResult, Optional[float]]:
        """Complete processing chain: anti-alias → decimate → Gaussian LPF.

        GAP-002 FIX: This is the recommended entry point for the
        default research pipeline:
            1 MHz → 50 kHz → 10 kHz Gaussian LPF

        Args:
            signal: Raw signal at self.original_rate.
            target_rate_hz: Target sampling rate after decimation.
            gaussian_cutoff_hz: Gaussian LPF cutoff (None = skip).
            anti_alias_type: "LPF" or "Gaussian" for anti-aliasing.
            causal: Use causal filtering (for streaming).

        Returns:
            (FilterResult, gaussian_sigma_or_None)
        """
        # Step 1: Anti-alias + decimate
        fr = self.downsample(signal, target_rate_hz, anti_alias_type, causal)

        # Step 2: Gaussian LPF (optional, applied at the resampled rate)
        gaussian_sigma = None
        if gaussian_cutoff_hz is not None and gaussian_cutoff_hz > 0:
            nyquist = target_rate_hz / 2.0
            if gaussian_cutoff_hz >= nyquist:
                logger.warning(
                    "Gaussian cutoff %d Hz >= Nyquist %d Hz at %d Hz rate. "
                    "Clamping to %.0f Hz.",
                    gaussian_cutoff_hz, nyquist, target_rate_hz, 0.95 * nyquist,
                )
                gaussian_cutoff_hz = int(0.95 * nyquist)
            fr.signal, gaussian_sigma = self.apply_gaussian_lpf(
                fr.signal, gaussian_cutoff_hz, target_rate_hz,
            )
        return fr, gaussian_sigma

    # ── GAP-003 FIX: Frequency-domain diagnostics ──

    def compute_filter_diagnostics(
        self,
        signal_before: np.ndarray,
        signal_after: np.ndarray,
        fs_before: int,
        fs_after: int,
        cutoff_hz: float,
    ) -> FilterDiagnostics:
        """Compute frequency-domain diagnostics for reporting.

        Returns spectra before/after and filter characteristics.
        """
        from scipy.fft import rfft, rfftfreq

        # Spectrum before (at fs_before)
        n_before = min(len(signal_before), 2 ** 16)  # cap for speed
        f_before = rfftfreq(n_before, d=1.0 / fs_before)
        s_before = np.abs(rfft(signal_before[:n_before])) ** 2

        # Spectrum after (at fs_after)
        n_after = min(len(signal_after), 2 ** 16)
        f_after = rfftfreq(n_after, d=1.0 / fs_after)
        s_after = np.abs(rfft(signal_after[:n_after])) ** 2

        # Gaussian filter frequency response
        sigma = self.sigma_for_cutoff(cutoff_hz, fs_after)
        f_resp = np.linspace(0, fs_after / 2, 1000)
        sigma_sec = sigma / fs_after
        h_mag = np.exp(-2 * np.pi**2 * sigma_sec**2 * f_resp**2)
        h_db = 20 * np.log10(h_mag + 1e-12)

        # Attenuation at key frequencies
        idx_cutoff = np.argmin(np.abs(f_resp - cutoff_hz))
        idx_nyquist = len(f_resp) - 1

        # Effective bandwidth (−3 dB)
        above_3db = h_db >= -3.0
        bw = float(f_resp[above_3db][-1]) if above_3db.any() else 0.0

        diag = FilterDiagnostics(
            freq_axis_hz=f_after,
            spectrum_before=s_before,
            spectrum_after=s_after,
            filter_freq_response_hz=f_resp,
            filter_freq_response_mag=h_mag,
            filter_freq_response_db=h_db,
            attenuation_at_cutoff_db=float(h_db[idx_cutoff]),
            attenuation_at_nyquist_db=float(h_db[idx_nyquist]),
            effective_bandwidth_hz=bw,
        )

        logger.info(
            "Filter diagnostics: cutoff=%d Hz, atten@cutoff=%.1f dB, "
            "atten@Nyquist=%.1f dB, effective BW=%.0f Hz",
            cutoff_hz, diag.attenuation_at_cutoff_db,
            diag.attenuation_at_nyquist_db, bw,
        )
        return diag