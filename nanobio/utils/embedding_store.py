"""
Embedding cache to avoid redundant encoder forward passes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("nanobio.utils.embedding_store")


class EmbeddingStore:
    """In-memory embedding cache with disk persistence."""

    def __init__(self):
        self._store: Dict[str, dict] = {}

    def put(self, key: str, embeddings: np.ndarray,
            labels: Optional[np.ndarray] = None) -> None:
        self._store[key] = {"embeddings": embeddings, "labels": labels}

    def get(self, key: str) -> dict:
        return self._store[key]

    def has(self, key: str) -> bool:
        return key in self._store

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for key, data in self._store.items():
            arrays = {k: v for k, v in data.items() if v is not None}
            np.savez_compressed(directory / f"emb_{key}.npz", **arrays)

    def load(self, directory: Path) -> None:
        for p in Path(directory).glob("emb_*.npz"):
            key = p.stem.replace("emb_", "")
            with np.load(p, allow_pickle=False) as f:
                self._store[key] = {k: f[k] for k in f.files}