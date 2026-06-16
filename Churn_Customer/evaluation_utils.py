"""
Shared evaluation framework for all Churn_Customer models.

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation_utils import evaluate_model_cv, print_evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def _metrics_at_threshold(y, y_proba, threshold, roc_auc, pr_auc, pos_label=1):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_1": precision_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "recall_1": recall_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "f1_1": f1_score(y, y_pred, pos_label=pos_label, zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def evaluate_predictions(
    y: pd.Series,
    y_proba: np.ndarray,
    thresholds: list = None,
    pos_label: int = 1,
) -> dict:
    """
    Evaluate already-computed probabilities against true labels.
    Use this for a single train/test (holdout) check — no cross-validation.

    Parameters
    ----------
    y          : true binary labels
    y_proba    : positive-class predicted probabilities
    thresholds : list of classification thresholds to evaluate. Defaults to [0.5].
    pos_label  : positive class label (default 1)

    Returns
    -------
    dict with keys:
        "thresholds" -> {threshold -> metrics_dict}
        "y_proba"    -> the probabilities passed in
    """
    if thresholds is None:
        thresholds = [0.5]

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba, pos_label=pos_label)

    threshold_results = {
        t: _metrics_at_threshold(y, y_proba, t, roc_auc, pr_auc, pos_label=pos_label)
        for t in thresholds
    }

    return {
        "thresholds": threshold_results,
        "y_proba": y_proba,
    }


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    thresholds: list = None,
    pos_label: int = 1,
) -> dict:
    """
    Run cross-validated evaluation for a sklearn-compatible model.

    Parameters
    ----------
    model      : sklearn Pipeline or estimator
    X          : feature DataFrame
    y          : binary target Series
    cv         : StratifiedKFold (or any sklearn CV splitter)
    thresholds : list of classification thresholds to evaluate.
                 Defaults to [0.5].
    pos_label  : positive class label (default 1)

    Returns
    -------
    dict with keys:
        "thresholds" -> {threshold -> metrics_dict}
        "y_proba"    -> 1-D array of OOF positive-class probabilities
        "y_oof"      -> 1-D array of OOF predictions at threshold=0.5
    """
    if thresholds is None:
        thresholds = [0.5]

    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    y_oof = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=1)

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba, pos_label=pos_label)

    threshold_results = {
        t: _metrics_at_threshold(y, y_proba, t, roc_auc, pr_auc, pos_label=pos_label)
        for t in thresholds
    }

    return {
        "thresholds": threshold_results,
        "y_proba": y_proba,
        "y_oof": y_oof,
    }


def print_evaluation(results: dict, label: str = "") -> None:
    """
    Print evaluation results in a standardised format.

    Parameters
    ----------
    results : dict returned by evaluate_model_cv
    label   : optional header label (e.g. experiment name or threshold description)
    """
    header = f"Evaluation Results" + (f" - {label}" if label else "")
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    for threshold, metrics in results["thresholds"].items():
        print(f"\n  Threshold = {threshold:.4f}")
        print(f"  {'-' * 40}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (Ch=1):  {metrics['precision_1']:.4f}")
        print(f"  Recall    (Ch=1):  {metrics['recall_1']:.4f}")
        print(f"  F1        (Ch=1):  {metrics['f1_1']:.4f}")
        print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
        print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:            {metrics['pr_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(f"    {row}")
