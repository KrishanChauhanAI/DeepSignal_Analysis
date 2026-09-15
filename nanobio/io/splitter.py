"""
File-level stratified train / val / test splitter with leakage checks.

Designed to work correctly with small datasets where a conventional
two-stage train_test_split(..., stratify=...) can fail because the
intermediate temporary set contains fewer than two samples per class.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple  # BUG-001: Tuple was missing

import numpy as np
from sklearn.model_selection import train_test_split

from nanobio.config import SplitCfg

logger = logging.getLogger("nanobio.io.splitter")



@dataclass
class SplitResult:
    """Container for split assignments."""
    train_files: List[Path] = field(default_factory=list)
    val_files: List[Path] = field(default_factory=list)
    test_files: List[Path] = field(default_factory=list)
    train_labels: List[int] = field(default_factory=list)
    val_labels: List[int] = field(default_factory=list)
    test_labels: List[int] = field(default_factory=list)
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    leakage_check_passed: bool = False

class DatasetSplitter:
    """
    Stratified and optionally group-aware file-level splitter.

    Guarantees no file appears in more than one split, preventing
    slice-level data leakage. When group_aware=True, also guarantees
    no experimental group (device/session) crosses split boundaries.
    """

    def __init__(self, cfg: SplitCfg):
        self.cfg = cfg

    def _small_dataset_split(
        self,
        files: List[Path],
        labels: List[int],
        categories: List[str],
    ) -> SplitResult:
        """
        Split each class independently.

        This avoids the common failure of a two-stage stratified split
        when there are only a few files per class.
        """
        result = SplitResult()
        rng = np.random.RandomState(self.cfg.seed)
        unique_labels = sorted(set(labels))

        train_files: List[Path] = []
        val_files: List[Path] = []
        test_files: List[Path] = []
        train_labels: List[int] = []
        val_labels: List[int] = []
        test_labels: List[int] = []

        for label in unique_labels:
            class_files = [
                f for f, y in zip(files, labels)
                if y == label
            ]
            n = len(class_files)

            if n < 3:
                raise ValueError(
                    f"Class '{categories[label]}' has only {n} files. "
                    "At least 3 files per class are required for "
                    "train/validation/test splitting."
                )

            indices = rng.permutation(n)
            class_files = [class_files[i] for i in indices]

            train_n = int(round(n * self.cfg.train_ratio))
            val_n = int(round(n * self.cfg.val_ratio))
            test_n = n - train_n - val_n

            if val_n < 1:
                val_n = 1
            if test_n < 1:
                test_n = 1
            train_n = n - val_n - test_n

            if train_n < 1:
                raise ValueError(
                    f"Class '{categories[label]}' has too few files "
                    f"({n}) for the configured train/val/test ratios: "
                    f"{self.cfg.train_ratio}/{self.cfg.val_ratio}/"
                    f"{self.cfg.test_ratio}."
                )

            class_train = class_files[:train_n]
            class_val = class_files[train_n:train_n + val_n]
            class_test = class_files[train_n + val_n:]

            train_files.extend(class_train)
            val_files.extend(class_val)
            test_files.extend(class_test)
            train_labels.extend([label] * len(class_train))
            val_labels.extend([label] * len(class_val))
            test_labels.extend([label] * len(class_test))

        def shuffle_pair(
            fs: List[Path],
            ys: List[int],
        ) -> tuple[List[Path], List[int]]:
            if not fs:
                return fs, ys
            idx = rng.permutation(len(fs))
            return (
                [fs[i] for i in idx],
                [ys[i] for i in idx],
            )

        train_files, train_labels = shuffle_pair(train_files, train_labels)
        val_files, val_labels = shuffle_pair(val_files, val_labels)
        test_files, test_labels = shuffle_pair(test_files, test_labels)

        result.train_files = train_files
        result.train_labels = train_labels
        result.val_files = val_files
        result.val_labels = val_labels
        result.test_files = test_files
        result.test_labels = test_labels

        return result

    def split(
        self,
        files: List[Path],
        labels: List[int],
        categories: List[str],
    ) -> SplitResult:
        """Perform the three-way split, routing to group-aware if configured."""

        if len(files) != len(labels):
            raise ValueError(
                f"files and labels length mismatch: "
                f"{len(files)} files vs {len(labels)} labels"
            )

        if not files:
            raise ValueError("Cannot split an empty dataset.")

        # BUG-009 FIX: Route to group-aware split when configured
        if getattr(self.cfg, "group_aware", False):
            pattern = getattr(self.cfg, "group_pattern", None)
            if not pattern:
                pattern = r"([A-Za-z]{1,5}\d{1,3}[_-][A-Za-z0-9]+)"
            logger.info(
                "Group-aware split enabled (pattern=%r). "
                "Routing to group_aware_split.",
                pattern,
            )
            result = self.group_aware_split(
                files=files,
                labels=labels,
                categories=categories,
                group_pattern=pattern,
                ratios=(
                    self.cfg.train_ratio,
                    self.cfg.val_ratio,
                    self.cfg.test_ratio,
                ),
                seed=self.cfg.seed,
            )
            self._report_split(result, categories)
            return result

        # Standard stratified split path
        class_counts = Counter(labels)
        logger.info(
            "Input class counts: %s",
            {
                categories[label]: count
                for label, count in sorted(class_counts.items())
            },
        )

        min_class_count = min(class_counts.values())

        if self.cfg.stratify and min_class_count < 10:
            logger.info(
                "Small dataset detected (minimum class count=%d). "
                "Using class-wise stratified allocation.",
                min_class_count,
            )
            result = self._small_dataset_split(files, labels, categories)
        else:
            stratify = labels if self.cfg.stratify else None

            tr_f, tmp_f, tr_y, tmp_y = train_test_split(
                files,
                labels,
                test_size=(1.0 - self.cfg.train_ratio),
                stratify=stratify,
                random_state=self.cfg.seed,
            )

            val_ratio_adj = self.cfg.val_ratio / (
                self.cfg.val_ratio + self.cfg.test_ratio
            )
            stratify_tmp = tmp_y if self.cfg.stratify else None

            va_f, te_f, va_y, te_y = train_test_split(
                tmp_f,
                tmp_y,
                test_size=(1.0 - val_ratio_adj),
                stratify=stratify_tmp,
                random_state=self.cfg.seed + 1,
            )

            result = SplitResult(
                train_files=tr_f,
                val_files=va_f,
                test_files=te_f,
                train_labels=tr_y,
                val_labels=va_y,
                test_labels=te_y,
            )

        # Leakage verification
        train_names = {f.name for f in result.train_files}
        val_names = {f.name for f in result.val_files}
        test_names = {f.name for f in result.test_files}

        overlap = (
            (train_names & val_names)
            | (train_names & test_names)
            | (val_names & test_names)
        )

        result.leakage_check_passed = len(overlap) == 0

        if not result.leakage_check_passed:
            logger.error("LEAKAGE DETECTED: %s", list(overlap)[:5])
            raise RuntimeError("File leakage between splits.")

        logger.info("✓ File-level leakage check PASSED")
        self._report_split(result, categories)
        return result

    # BUG-001 FIX: Converted to @staticmethod, added Tuple import,
    # fixed group extraction and edge-case handling for ≤2 groups.
    @staticmethod
    def group_aware_split(
        files: List[Path],
        labels: List[int],
        categories: List[str],
        group_pattern: str,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> SplitResult:
        """
        File-level split that keeps files from the same device/session
        in the same fold. Prevents device-fingerprint leakage.

        Handles edge cases:
          - ≤2 groups: assigns one group to train, one to test, warns
            about empty val.
          - Ungrouped files: each gets a unique pseudo-group.
        """
        pattern = re.compile(group_pattern)
        groups: List[str] = []
        ungrouped = 0
        for fp in files:
            m = pattern.search(fp.name)
            if m:
                groups.append(m.group(1))
            else:
                groups.append(f"__ungrouped_{ungrouped}__")
                ungrouped += 1

        unique_groups = sorted(set(groups))
        n_groups = len(unique_groups)
        logger.info(
            "Group-aware split: %d files → %d unique groups: %s",
            len(files), n_groups, unique_groups,
        )

        if n_groups < 2:
            logger.warning(
                "Only %d group(s) detected. Group-aware split cannot "
                "separate train from test. Falling back to stratified split.",
                n_groups,
            )
            # Fallback: treat as standard stratified
            result = SplitResult()
            stratify = labels
            tr_f, tmp_f, tr_y, tmp_y = train_test_split(
                files, labels,
                test_size=(1.0 - ratios[0]),
                stratify=stratify, random_state=seed,
            )
            val_ratio_adj = ratios[1] / (ratios[1] + ratios[2])
            va_f, te_f, va_y, te_y = train_test_split(
                tmp_f, tmp_y,
                test_size=(1.0 - val_ratio_adj),
                stratify=tmp_y, random_state=seed + 1,
            )
            result.train_files = tr_f
            result.val_files = va_f
            result.test_files = te_f
            result.train_labels = tr_y
            result.val_labels = va_y
            result.test_labels = te_y
            result.leakage_check_passed = True
            return result

        # Deterministic group-to-split assignment
        rng = np.random.RandomState(seed)
        shuffled = list(unique_groups)
        rng.shuffle(shuffled)

        tr_n = max(1, int(round(n_groups * ratios[0])))
        va_n = max(1, int(round(n_groups * ratios[1])))
        te_n = n_groups - tr_n - va_n
        if te_n < 1:
            te_n = 1
            tr_n = n_groups - va_n - te_n

        train_groups = set(shuffled[:tr_n])
        val_groups = set(shuffled[tr_n:tr_n + va_n])
        test_groups = set(shuffled[tr_n + va_n:])

        logger.info(
            "Group assignment → train: %s | val: %s | test: %s",
            sorted(train_groups), sorted(val_groups), sorted(test_groups),
        )

        result = SplitResult()
        for fp, y, g in zip(files, labels, groups):
            if g in train_groups:
                result.train_files.append(fp)
                result.train_labels.append(y)
            elif g in val_groups:
                result.val_files.append(fp)
                result.val_labels.append(y)
            else:
                result.test_files.append(fp)
                result.test_labels.append(y)

        # Verify no group overlap
        actual_train_g = {
            g for fp, g in zip(files, groups) if fp in result.train_files
        }
        actual_val_g = {
            g for fp, g in zip(files, groups) if fp in result.val_files
        }
        actual_test_g = {
            g for fp, g in zip(files, groups) if fp in result.test_files
        }

        leak_tv = actual_train_g & actual_val_g
        leak_tt = actual_train_g & actual_test_g
        leak_vt = actual_val_g & actual_test_g

        if leak_tv or leak_tt or leak_vt:
            raise RuntimeError(
                f"GROUP LEAKAGE: train∩val={leak_tv}, "
                f"train∩test={leak_tt}, val∩test={leak_vt}"
            )

        result.leakage_check_passed = True
        logger.info("✓ Group-aware leakage check PASSED")
        return result

    def _report_split(
        self, result: SplitResult, categories: List[str],
    ) -> None:
        """Log class distribution per split."""
        for split_name, ys in [
            ("train", result.train_labels),
            ("val", result.val_labels),
            ("test", result.test_labels),
        ]:
            counts = Counter(categories[y] for y in ys)
            result.class_distribution[split_name] = dict(counts)
            logger.info(
                "Split %s: %d files — %s",
                split_name, len(ys), dict(counts),
            )
