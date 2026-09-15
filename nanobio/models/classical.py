"""Scikit-learn classical model zoo."""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

SEED = 42


def build_classical_models(
    names: List[str], n_classes: int, n_train_samples: int,
) -> Dict[str, Any]:
    """Instantiate the requested classical classifiers."""
    k_neighbors = min(25, max(2, n_train_samples - 1))
    base_tree = DecisionTreeClassifier(max_depth=2, random_state=SEED)

    zoo = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=30, class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="saga",
            n_jobs=-1, random_state=SEED,
        ),
        "SVM-RBF": SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=SEED,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, random_state=SEED,
        ),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15, random_state=SEED,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=base_tree, n_estimators=100, random_state=SEED,
        ),
        "SGD": SGDClassifier(
            loss="log_loss", n_jobs=-1, random_state=SEED,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=80,
            early_stopping=True, random_state=SEED,
        ),
    }
    if XGB_AVAILABLE:
        # BUG-014 FIX: Use binary:logistic for 2-class to ensure
        # predict() returns 1D array. multi:softprob returns 2D
        # which causes ravel() to double the length in MetricEngine.
        if n_classes == 2:
            xgb_objective = "binary:logistic"
            xgb_eval = "logloss"
            xgb_kwargs = {}
        else:
            xgb_objective = "multi:softprob"
            xgb_eval = "mlogloss"
            xgb_kwargs = {"num_class": n_classes}

        zoo["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            objective=xgb_objective,
            eval_metric=xgb_eval,
            tree_method="hist",
            random_state=SEED,
            **xgb_kwargs,
        )
    return {n: zoo[n] for n in names if n in zoo}