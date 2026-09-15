"""
Adapters for optional SOTA dependencies. Every adapter degrades gracefully:
if a library is missing, the model is skipped with a clear log line — the
pipeline never crashes because an optional SOTA package is absent.

Backends tried, in order of preference:
    MiniRocket : tsai  ->  aeon  ->  sktime   (transformer) + sklearn head
    PatchTST   : tsai (defensive kwargs; skip on API mismatch)
    TabPFN v2  : pip 'tabpfn'  (zero-shot tabular FM; limits enforced)

Licensing note (deployment): TabPFN weights use the Prior Labs license —
free for research; commercial deployment requires an agreement. tsai: Apache-2.0.
aeon: BSD-3. Verify at adoption time; licenses can change.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("nanobio.models.sota_adapters")

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# MiniRocket
# ─────────────────────────────────────────────────────────────────────────────

def try_build_minirocket() -> Tuple[Optional[Any], str]:
    """
    Build a MiniRocket transformer from whichever backend is installed.

    Returns:
        (transformer_or_None, backend_name). The transformer exposes
        fit(X) / transform(X) with X of shape (N, C, L) float32.
    """
    # Backend 1: tsai
    try:
        from tsai.models.MINIROCKET import MiniRocket as TsaiMiniRocket
        tr = TsaiMiniRocket(random_state=SEED)
        return tr, "tsai"
    except Exception:
        pass
    # Backend 2: aeon
    try:
        from aeon.transformations.collection.rocket import MiniRocket as AeonMR
        tr = AeonMR(random_state=SEED)
        return tr, "aeon"
    except Exception:
        pass
    # Backend 3: sktime
    try:
        from sktime.transformations.panel.rocket import MiniRocket as SkMR
        return SkMR(random_state=SEED), "sktime"
    except Exception:
        pass
    return None, "none"


def fit_predict_minirocket(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xte: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    MiniRocket features + class-balanced LogisticRegression head.

    Args:
        Xtr/Xte: (N, T) raw windows (we add the channel dim internally).
        Returns dict with y_pred, y_prob, fit_time_s, pred_time_s, backend
        — or None if no backend is available.
    """
    tr, backend = try_build_minirocket()
    if tr is None:
        logger.warning("MiniRocket skipped: no backend (tsai/aeon/sktime) installed.")
        return None

    import time

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr3 = np.ascontiguousarray(Xtr[:, None, :], dtype=np.float32)
    Xte3 = np.ascontiguousarray(Xte[:, None, :], dtype=np.float32)

    t0 = time.time()
    if backend == "tsai":
        Ftr = tr.fit(Xtr3)                 # tsai returns features from fit()
        Fte = tr.transform(Xte3)
    else:
        tr.fit(Xtr3)
        Ftr = tr.transform(Xtr3)
        Fte = tr.transform(Xte3)
    fit_time = time.time() - t0

    scaler = StandardScaler()
    Ftr = scaler.fit_transform(np.asarray(Ftr, dtype=np.float64))
    Fte = scaler.transform(np.asarray(Fte, dtype=np.float64))

    clf = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=1.0, random_state=SEED,
    )
    t0 = time.time()
    clf.fit(Ftr, ytr)
    fit_time += time.time() - t0
    t0 = time.time()
    prob = clf.predict_proba(Fte)
    pred_time = time.time() - t0

    return {
        "y_pred": prob.argmax(1), "y_prob": prob,
        "fit_time_s": fit_time, "pred_time_s": pred_time,
        "backend": backend,
        "artifact": {"transformer": tr, "scaler": scaler, "clf": clf},
    }


# ─────────────────────────────────────────────────────────────────────────────
# TabPFN v2 (zero-shot tabular foundation model)
# ─────────────────────────────────────────────────────────────────────────────

TABPFN_MAX_TRAIN = 10_000      # v2 limit (verify against installed version docs)
TABPFN_MAX_FEATURES = 500
TABPFN_MAX_CLASSES = 10


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, max_n: int, seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= max_n:
        return X, y
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(
        np.arange(len(X)), train_size=max_n, stratify=y, random_state=seed,
    )
    logger.info("TabPFN: subsampled %d -> %d training rows (limit %d)",
                len(X), max_n, max_n)
    return X[keep], y[keep]


def fit_predict_tabpfn(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
    device: str = "cpu",
    max_train: int = 10_000,           # NEW
    max_features: int = 500,           # NEW
) -> Optional[Dict[str, Any]]:
    """
    Zero-shot TabPFN v2. Enforces documented limits with stratified
    subsampling; defensive kwargs so minor API differences don't crash.
    """
    if not tabpfn_available():
        logger.warning("TabPFN skipped: pip install tabpfn")
        return None
    if Xtr.shape[1] > TABPFN_MAX_FEATURES:
        logger.warning("TabPFN skipped: %d features > limit %d",
                       Xtr.shape[1], TABPFN_MAX_FEATURES)
        return None
    if len(np.unique(ytr)) > TABPFN_MAX_CLASSES:
        logger.warning("TabPFN skipped: >%d classes", TABPFN_MAX_CLASSES)
        return None
    if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(Xte)):
        logger.warning("TabPFN skipped: non-finite features")
        return None

    import time

    from tabpfn import TabPFNClassifier

    #Xtr, ytr = _stratified_subsample(Xtr, ytr, TABPFN_MAX_TRAIN)
    Xtr, ytr = _stratified_subsample(Xtr, ytr, max_train)

    clf = None
    for kwargs in (
        {"device": device},
        {"device": device, "ignore_pretraining_limits": True},
        {},
    ):
        try:
            clf = TabPFNClassifier(**kwargs)
            break
        except TypeError:
            continue
    if clf is None:
        logger.warning("TabPFN skipped: could not construct classifier")
        return None

    t0 = time.time()
    clf.fit(Xtr.astype(np.float32), ytr.astype(np.int64))
    fit_time = time.time() - t0
    t0 = time.time()
    prob = clf.predict_proba(Xte.astype(np.float32))
    pred_time = time.time() - t0

    return {
        "y_pred": np.asarray(prob).argmax(1), "y_prob": np.asarray(prob),
        "fit_time_s": fit_time, "pred_time_s": pred_time,
        "backend": "tabpfn_v2", "artifact": clf,
    }