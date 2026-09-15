"""
Comprehensive Scientific Visualizations.
Handles PSD, ACF (with decay fitting), PDF, CDF, Violins, and Latent Spaces.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.stats import ks_2samp, laplace, norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False

try:
    import pacmap
    PACMAP_OK = True
except ImportError:
    PACMAP_OK = False

logger = logging.getLogger("nanobio.viz.comprehensive")

class ComprehensivePlotter:
    def __init__(self, out_dir: Path, timestamp: str):
        self.out_dir = out_dir
        self.ts = timestamp
        self._c = 0
        
    def _path(self, name: str) -> Path:
        self._c += 1
        return self.out_dir / f"{self.ts}_{name}_{self._c}.png"

    # ─── SECTION 4 & 5: Filter Comparison ────────────────────────────────
    def plot_multirate_filters(self, raw_signal: np.ndarray, fs_raw: int, processor, file_id: str):
        rates = [50000, 10000, 1000, 100, 10, 2]
        fig, axes = plt.subplots(len(rates)+1, 1, figsize=(14, 12), sharex=True)
        
        t_raw = np.arange(min(10000, len(raw_signal))) / fs_raw
        axes[0].plot(t_raw, raw_signal[:len(t_raw)], color="black", lw=0.5)
        axes[0].set_title(f"Raw 1 MHz | {file_id}")
        
        for i, r in enumerate(rates):
            fr = processor.downsample(raw_signal, r, "LPF")
            t_f = np.arange(min(len(t_raw), len(fr.signal))) / r
            axes[i+1].plot(t_f, fr.signal[:len(t_f)], lw=0.8)
            axes[i+1].set_title(f"LPF {r} Hz")
            
        fig.tight_layout()
        fig.savefig(self._path(f"MultiRate_Filters_{file_id}"), dpi=150)
        plt.close(fig)

    # ─── SECTION 7: PSD ──────────────────────────────────────────────────
    def plot_psd(self, class_signals: dict, fs: int):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cat, sigs in class_signals.items():
            if not sigs: continue
            psds, freqs = [], None
            for s in sigs:
                f, p = welch(s, fs=fs, nperseg=2048)
                psds.append(p)
                freqs = f
            mean_psd = np.mean(psds, axis=0)
            std_psd = np.std(psds, axis=0)
            p_line = ax.semilogy(freqs, mean_psd, label=cat)[0]
            ax.fill_between(freqs, mean_psd-std_psd, mean_psd+std_psd, alpha=0.2, color=p_line.get_color())
            
        ax.set_title(f"Power Spectral Density | fs={fs}Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.legend()
        fig.savefig(self._path("PSD_Comparison"), dpi=150)
        plt.close(fig)

    # ─── SECTION 8: ACF Decay Fitting ────────────────────────────────────
    def _fit_acf(self, lags, acf, fs):
        t = lags / fs
        # 1. Exponential: A * exp(-t/tau)
        def exp_func(t, A, tau): return A * np.exp(-t / tau)
        # 2. Stretched: A * exp(-(t/tau)^beta)
        def str_func(t, A, tau, beta): return A * np.exp(-(t / tau)**beta)
        
        fits = {}
        try:
            popt, _ = curve_fit(exp_func, t, acf, p0=[1.0, 0.01], bounds=(0, [2.0, 10.0]))
            pred = exp_func(t, *popt)
            fits["Exponential"] = {"A": popt[0], "tau": popt[1], "rmse": np.sqrt(np.mean((acf-pred)**2))}
        except: pass
        
        try:
            popt, _ = curve_fit(str_func, t, acf, p0=[1.0, 0.01, 1.0], bounds=(0, [2.0, 10.0, 2.0]))
            pred = str_func(t, *popt)
            fits["Stretched"] = {"A": popt[0], "tau": popt[1], "beta": popt[2], "rmse": np.sqrt(np.mean((acf-pred)**2))}
        except: pass
        return fits

    def plot_acf_with_fits(self, class_signals: dict, fs: int):
        fig, ax = plt.subplots(figsize=(10, 6))
        all_fits = {}
        for cat, sigs in class_signals.items():
            if not sigs: continue
            sig = sigs[0] # Take one representative
            x = sig - np.mean(sig)
            acf = np.correlate(x, x, mode='full')[len(x)-1:]
            acf = acf / acf[0]
            lags = np.arange(min(len(acf), fs // 5)) # 0.2s lag
            ax.plot(lags/fs*1000, acf[:len(lags)], label=f"{cat} ACF", alpha=0.6)
            
            # Fit and plot best
            fits = self._fit_acf(lags, acf[:len(lags)], fs)
            all_fits[cat] = fits
            if fits:
                best = min(fits.keys(), key=lambda k: fits[k]["rmse"])
                t = lags/fs
                if best == "Exponential":
                    y_fit = fits[best]["A"] * np.exp(-t / fits[best]["tau"])
                else:
                    y_fit = fits[best]["A"] * np.exp(-(t / fits[best]["tau"])**fits[best]["beta"])
                ax.plot(lags/fs*1000, y_fit, '--', label=f"{cat} Fit ({best})")

        ax.set_title("Autocorrelation Function (ACF) Decay")
        ax.set_xlabel("Lag (ms)")
        ax.axhline(0, color='k', ls=':')
        ax.legend()
        fig.savefig(self._path("ACF_Decay"), dpi=150)
        plt.close(fig)
        
        with open(self.out_dir / f"{self.ts}_ACF_Fits.json", "w") as f:
            json.dump(all_fits, f, indent=2)

    # ─── SECTION 11 & 21: PDF / CDF / Parametric Fits ─────────────────────
    def plot_pdf_cdf(self, class_signals: dict):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        gof_stats = {}
        for cat, sigs in class_signals.items():
            if not sigs: continue
            data = np.concatenate([s[:5000] for s in sigs])
            data = (data - np.mean(data)) / np.std(data) # Normalize for comparison
            
            # PDF (KDE + Parametric Normal Fit)
            sns.kdeplot(data, ax=ax1, label=f"{cat} KDE", fill=True, alpha=0.3)
            mu, std = norm.fit(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax1.plot(x_range, norm.pdf(x_range, mu, std), '--', label=f"{cat} Gaussian Fit")
            
            # KS Test for Gaussianity
            d_stat, p_val = ks_2samp(data, np.random.normal(mu, std, len(data)))
            gof_stats[cat] = {"Gaussian_KS_Stat": d_stat, "p_value": p_val}

            # CDF
            x_sorted = np.sort(data)
            y_cdf = np.arange(1, len(x_sorted)+1) / len(x_sorted)
            ax2.plot(x_sorted, y_cdf, label=cat)

        ax1.set_title("PDF with Parametric Fits (Z-scored)")
        ax2.set_title("Empirical CDF")
        ax1.legend(); ax2.legend()
        fig.savefig(self._path("PDF_CDF_Distributions"), dpi=150)
        plt.close(fig)

    # ─── SECTION 14 & 20: Dwell Time / Violins ───────────────────────────
    def plot_event_violins(self, features_df: pd.DataFrame):
        if features_df.empty: return
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        sns.violinplot(data=features_df, x="Category", y="Ton_mean_s", inner="quartile", ax=axes[0])
        sns.violinplot(data=features_df, x="Category", y="delta_I_I0_mean", inner="quartile", ax=axes[1])
        sns.violinplot(data=features_df, x="Category", y="RMS_normalized", inner="quartile", ax=axes[2])
        
        axes[0].set_title("Dwell Time (Ton)")
        axes[1].set_title("Normalized ΔI/I₀")
        axes[2].set_title("Normalized RMS")
        fig.tight_layout()
        fig.savefig(self._path("Event_Kinetics_Violins"), dpi=150)
        plt.close(fig)

    # ─── SECTION 16: Latent Space 2D & 3D ────────────────────────────────
    def plot_latent_spaces(self, X: np.ndarray, y: np.ndarray, categories: list):
        # Prevent leakage: Standardize on X
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        labels = [categories[i] for i in y]
        
        def _plot_2d3d(embedding_2d, embedding_3d, title_prefix, name):
            fig = plt.figure(figsize=(14, 6))
            # 2D
            ax1 = fig.add_subplot(121)
            sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, palette="tab10", ax=ax1)
            ax1.set_title(f"{title_prefix} (2D)")
            # 3D
            ax2 = fig.add_subplot(122, projection='3d')
            for i, cat in enumerate(categories):
                mask = np.array(labels) == cat
                ax2.scatter(embedding_3d[mask, 0], embedding_3d[mask, 1], embedding_3d[mask, 2], label=cat)
            ax2.set_title(f"{title_prefix} (3D)")
            ax2.legend()
            fig.tight_layout()
            fig.savefig(self._path(name), dpi=150)
            plt.close(fig)

        # PCA
        _plot_2d3d(PCA(n_components=2).fit_transform(X), PCA(n_components=3).fit_transform(X), "PCA", "Latent_PCA")
        
        # t-SNE
        p = min(30, max(2, len(X)-1))
        _plot_2d3d(TSNE(n_components=2, perplexity=p).fit_transform(X), 
                   TSNE(n_components=3, perplexity=p).fit_transform(X), "t-SNE", "Latent_tSNE")

        # UMAP
        if UMAP_OK:
            import umap
            nn = min(15, max(2, len(X)-1))
            _plot_2d3d(umap.UMAP(n_components=2, n_neighbors=nn).fit_transform(X),
                       umap.UMAP(n_components=3, n_neighbors=nn).fit_transform(X), "UMAP", "Latent_UMAP")

        # PaCMAP
        if PACMAP_OK:
            import pacmap
            _plot_2d3d(pacmap.PaCMAP(n_components=2).fit_transform(X),
                       pacmap.PaCMAP(n_components=3).fit_transform(X), "PaCMAP", "Latent_PaCMAP")