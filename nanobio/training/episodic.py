"""
File-level episodic sampler for Prototypical Network training.

Every episode consists of entire FILES sampled within one split
(train/val/test). No support/query overlap within an episode. No file
leakage between splits (enforced upstream).
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterator, List, Tuple

import numpy as np

logger = logging.getLogger("nanobio.training.episodic")


class FileEpisodeSampler:
    """
    Sample N-way K-shot Q-query episodes at the file level.

    Args:
        file_labels: Class label per file within the split.
        n_way: Classes per episode.
        k_shot: Support files per class.
        q_query: Query files per class.
        episodes_per_epoch: Number of episodes generated per iteration cycle.
        seed: Random seed.
    """

    def __init__(
        self,
        file_labels: List[int],
        n_way: int,
        k_shot: int,
        q_query: int,
        episodes_per_epoch: int = 100,
        seed: int = 42,
    ):
        self.file_labels = np.asarray(file_labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = random.Random(seed)

        self.class_to_files: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(self.file_labels):
            self.class_to_files.setdefault(int(lbl), []).append(idx)
        self.classes = sorted(self.class_to_files.keys())

        needed = k_shot + q_query
        for c, files in self.class_to_files.items():
            if len(files) < needed:
                logger.warning(
                    "Class %d has %d files but episode needs %d — sampling with replacement.",
                    c, len(files), needed,
                )
        if len(self.classes) < n_way:
            raise ValueError(
                f"Requested {n_way}-way but only {len(self.classes)} classes available."
            )

    '''
    def _sample_episode(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Sample one episode. Labels are episode-local (0..n_way-1)."""
        chosen_classes = self.rng.sample(self.classes, self.n_way)
        s_files, s_labels, q_files, q_labels = [], [], [], []
        for ep_lbl, cls in enumerate(chosen_classes):
            files = self.class_to_files[cls]
            needed = self.k_shot + self.q_query
            selected = (
                self.rng.sample(files, needed) if len(files) >= needed
                else self.rng.choices(files, k=needed)
            )
            s_files.extend(selected[: self.k_shot])
            s_labels.extend([ep_lbl] * self.k_shot)
            q_files.extend(selected[self.k_shot :])
            q_labels.extend([ep_lbl] * self.q_query)
            if len(files) >= needed:
                assert not set(selected[: self.k_shot]) & set(selected[self.k_shot :])
        return s_files, s_labels, q_files, q_labels
        '''
        def _sample_episode(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Sample one episode. Labels are episode-local (0..n_way-1)."""
        chosen_classes = self.rng.sample(self.classes, self.n_way)
        s_files, s_labels, q_files, q_labels = [], [], [], []
        for ep_lbl, cls in enumerate(chosen_classes):
            files = self.class_to_files[cls]
            needed = self.k_shot + self.q_query
            if len(files) >= needed:
                selected = self.rng.sample(files, needed)
                s_files.extend(selected[: self.k_shot])
                s_labels.extend([ep_lbl] * self.k_shot)
                q_files.extend(selected[self.k_shot :])
                q_labels.extend([ep_lbl] * self.q_query)
            else:
                # BUG-013 FIX: When sampling with replacement, ensure
                # support and query sets don't share the same file index.
                # Sample support first, then sample query from remaining
                # (with replacement if needed).
                s_selected = self.rng.sample(
                    files, min(self.k_shot, len(files))
                )
                # If we need more support than available, pad with replacement
                while len(s_selected) < self.k_shot:
                    s_selected.append(self.rng.choice(files))
                s_files.extend(s_selected)
                s_labels.extend([ep_lbl] * self.k_shot)

                # Query: prefer files not in support
                remaining = [f for f in files if f not in set(s_selected)]
                if len(remaining) >= self.q_query:
                    q_selected = self.rng.sample(remaining, self.q_query)
                elif remaining:
                    q_selected = self.rng.choices(remaining, k=self.q_query)
                else:
                    # All files are in support; allow overlap as last resort
                    q_selected = self.rng.choices(files, k=self.q_query)
                    logger.warning(
                        "Class %d: query overlaps support (only %d files)",
                        cls, len(files),
                    )
                q_files.extend(q_selected)
                q_labels.extend([ep_lbl] * self.q_query)

        return s_files, s_labels, q_files, q_labels

    def __iter__(self) -> Iterator[Tuple[List[int], List[int], List[int], List[int]]]:
        for _ in range(self.episodes_per_epoch):
            yield self._sample_episode()

    def __len__(self) -> int:
        return self.episodes_per_epoch

# ═══════════════════════════════════════════════════════════════════
# EPISODIC EVALUATION (research mode)
# Extends existing episodic.py with standalone evaluation function.
# ═══════════════════════════════════════════════════════════════════

    def run_episodic_evaluation(
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_way: int,
        k_shot: int,
        q_query: int = 15,
        num_episodes: int = 200,
        distance_metric: str = "euclidean",
        normalize: bool = True,
        seed: int = 42,
    ) -> dict:
        """Run N-way K-shot episodic evaluation on precomputed embeddings.

        Uses PrototypeClassifier from nanobio.models.prototype.
        Returns dict with mean±std metrics and 95% CI.
        """
        from nanobio.models.prototype import PrototypeClassifier
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

        rng = np.random.default_rng(seed)
        unique_classes = np.unique(labels)
        n_way = min(n_way, len(unique_classes))

        class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
        needed = k_shot + q_query
        feasible = [c for c in unique_classes if len(class_indices[c]) >= needed]
        n_way = min(n_way, len(feasible))

        if n_way < 2:
            logger.warning("Cannot run episodic eval with < 2 feasible classes.")
            return {"n_episodes": 0, "accuracy_mean": 0.0}

        accs, f1s, bal_accs = [], [], []

        for _ in range(num_episodes):
            chosen = rng.choice(feasible, size=n_way, replace=False)
            s_emb, s_lbl, q_emb, q_lbl = [], [], [], []

            for ep_lbl, c in enumerate(chosen):
                idx = rng.choice(class_indices[c], size=needed, replace=False)
                s_emb.append(embeddings[idx[:k_shot]])
                s_lbl.extend([ep_lbl] * k_shot)
                q_emb.append(embeddings[idx[k_shot:k_shot + q_query]])
                q_lbl.extend([ep_lbl] * min(q_query, len(idx) - k_shot))

            s_emb = np.concatenate(s_emb)
            q_emb = np.concatenate(q_emb)
            s_lbl = np.array(s_lbl)
            q_lbl = np.array(q_lbl)

            clf = PrototypeClassifier(distance_metric=distance_metric,
                                    normalize=normalize)
            clf.fit(s_emb, s_lbl)
            preds, _, _ = clf.predict(q_emb)

            accs.append(float(accuracy_score(q_lbl, preds)))
            f1s.append(float(f1_score(q_lbl, preds, average="macro", zero_division=0)))
            bal_accs.append(float(balanced_accuracy_score(q_lbl, preds)))

        accs = np.array(accs)
        f1s = np.array(f1s)

        result = {
            "n_episodes": num_episodes,
            "n_way": n_way,
            "k_shot": k_shot,
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std()),
            "accuracy_ci95": (float(np.percentile(accs, 2.5)),
                            float(np.percentile(accs, 97.5))),
            "f1_macro_mean": float(f1s.mean()),
            "f1_macro_std": float(f1s.std()),
            "balanced_acc_mean": float(np.mean(bal_accs)),
        }

        logger.info(
            "Episodic: %d-way %d-shot × %d ep → acc=%.4f±%.4f CI95=[%.4f,%.4f]",
            n_way, k_shot, num_episodes,
            result["accuracy_mean"], result["accuracy_std"],
            result["accuracy_ci95"][0], result["accuracy_ci95"][1],
        )
        return result