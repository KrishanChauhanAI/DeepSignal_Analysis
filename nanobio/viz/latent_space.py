"""
Latent Space & Dimensionality Reduction (PCA, t-SNE, UMAP).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger("nanobio.viz.latent")

class LatentSpacePlotter:
    def __init__(self, output_dir: Path, timestamp: str, categories: list):
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.categories = categories

    def _plot_embedding(self, embedding: np.ndarray, labels: np.ndarray, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', alpha=0.7, s=15)
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i/10), markersize=8) 
                   for i in range(len(self.categories))]
        ax.legend(handles, self.categories, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{self.timestamp}_{filename}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_all(self, X: np.ndarray, y: np.ndarray):
        """Generates PCA, t-SNE, and UMAP (if available) on training data."""
        logger.info("Computing Latent Space Embeddings (PCA, t-SNE)...")
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        var_exp = np.sum(pca.explained_variance_ratio_) * 100
        self._plot_embedding(X_pca, y, f"PCA (Explained Var: {var_exp:.1f}%)", "Latent_PCA")

        # t-SNE
        # Perplexity must be less than n_samples. Handle small datasets safely.
        perp = min(30, max(2, len(X) - 1))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X)
        self._plot_embedding(X_tsne, y, "t-SNE Embedding", "Latent_tSNE")

        # UMAP
        if UMAP_AVAILABLE:
            logger.info("Computing UMAP Embedding...")
            n_neighbors = min(15, max(2, len(X) - 1))
            reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
            X_umap = reducer.fit_transform(X)
            self._plot_embedding(X_umap, y, "UMAP Embedding", "Latent_UMAP")
        else:
            logger.warning("UMAP not installed. Skipping UMAP plot. (pip install umap-learn)")