"""
Advanced Signal Analysis Visualizations (PSD, ACF, PDF, CDF, Events).
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import welch
from scipy.stats import gaussian_kde

logger = logging.getLogger("nanobio.viz.analysis_plots")

class SignalAnalysisPlotter:
    def __init__(self, output_dir: Path, timestamp: str, dpi: int = 150):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.dpi = dpi
        self._c = 0

    def _path(self, prefix: str) -> Path:
        self._c += 1
        return self.output_dir / f"{self.timestamp}_{prefix}_{self._c}.png"

    # --- 1. PSD (Power Spectral Density) ---
    def plot_class_psd(self, signals_by_class: Dict[str, List[np.ndarray]], fs: int):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cat, sigs in signals_by_class.items():
            if not sigs: continue
            psds = []
            for sig in sigs:
                f, p = welch(sig, fs=fs, nperseg=min(len(sig), 2048))
                psds.append(p)
            psds = np.array(psds)
            mean_psd = np.mean(psds, axis=0)
            std_psd = np.std(psds, axis=0)
            
            p = ax.semilogy(f, mean_psd, label=f"{cat} (Mean)")[0]
            ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd, color=p.get_color(), alpha=0.2)
        
        ax.set_title(f"Power Spectral Density (Welch) by Class | fs={fs}Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V^2/Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._path("PSD_Class_Comparison"), dpi=self.dpi)
        plt.close(fig)

    # --- 2. ACF (Autocorrelation) ---
    def plot_acf(self, signals_by_class: Dict[str, List[np.ndarray]], fs: int):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cat, sigs in signals_by_class.items():
            if not sigs: continue
            sig = sigs[0] # Plot representative file per class
            x = sig - np.mean(sig)
            fft_x = np.fft.rfft(x, n=2*len(x))
            acf = np.fft.irfft(fft_x * np.conj(fft_x))[:len(x)]
            if acf[0] != 0: acf = acf / acf[0]
            
            lags = np.arange(min(len(acf), fs // 10)) # Max lag 0.1s
            t_ms = lags / fs * 1000
            ax.plot(t_ms, acf[:len(lags)], label=f"{cat} (Rep file)", alpha=0.8)

        ax.set_title(f"Autocorrelation Function (ACF) by Class")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorrelation")
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._path("ACF_Comparison"), dpi=self.dpi)
        plt.close(fig)

    # --- 3. PDF & CDF ---
    def plot_pdf_cdf(self, signals_by_class: Dict[str, List[np.ndarray]]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        for cat, sigs in signals_by_class.items():
            if not sigs: continue
            # Concatenate a subset of data for distribution
            data = np.concatenate([s[:10000] for s in sigs]) 
            
            # PDF (KDE)
            sns.kdeplot(data, ax=ax1, label=cat, fill=True, alpha=0.3)
            
            # CDF
            x_sorted = np.sort(data)
            y_cdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
            ax2.plot(x_sorted, y_cdf, label=cat)

        ax1.set_title("Probability Density Function (PDF)")
        ax1.set_xlabel("Amplitude (Detrended)")
        ax1.set_ylabel("Density")
        ax1.legend()

        ax2.set_title("Cumulative Distribution Function (CDF)")
        ax2.set_xlabel("Amplitude (Detrended)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(self._path("PDF_CDF_Comparison"), dpi=self.dpi)
        plt.close(fig)

    # --- 4. Event Distributions (Ton, Toff, Dwell) ---
    def plot_event_distributions(self, events_by_class: Dict[str, List[dict]]):
        if not any(events_by_class.values()): return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for cat, events in events_by_class.items():
            if not events: continue
            ton = [e["duration_s"] * 1000 for e in events] # ms
            amp = [e["delta_i_over_i0"] for e in events]
            
            if ton: sns.kdeplot(ton, ax=axes[0], label=cat, fill=True, alpha=0.3)
            if amp: sns.kdeplot(amp, ax=axes[1], label=cat, fill=True, alpha=0.3)
            if ton and amp: axes[2].scatter(ton, amp, alpha=0.5, label=cat, s=10)

        axes[0].set_title("Dwell Time (Ton) Distribution")
        axes[0].set_xlabel("Duration (ms)")
        axes[1].set_title("Normalized Amplitude (ΔI/I₀) Distribution")
        axes[1].set_xlabel("ΔI/I₀")
        axes[2].set_title("Dwell Time vs Amplitude")
        axes[2].set_xlabel("Duration (ms)")
        axes[2].set_ylabel("ΔI/I₀")
        
        for ax in axes: ax.legend()
        fig.tight_layout()
        fig.savefig(self._path("Event_Kinetics_Comparison"), dpi=self.dpi)
        plt.close(fig)