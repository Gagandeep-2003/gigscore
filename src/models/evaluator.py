"""
src/models/evaluator.py — Comprehensive Model Evaluation

Computes ALL relevant metrics for credit scoring model evaluation.
NEVER reports only accuracy — with 8% default rate, a model that predicts
all zeros gets 92% accuracy and is completely useless.

Metrics computed:
    - AUC-ROC: PRIMARY metric — insensitive to class imbalance
    - AUC-PR: Better than ROC for very imbalanced datasets
    - F1 Score: Harmonic mean of precision and recall
    - Precision: Of predicted defaults, how many actually defaulted
    - Recall: Of actual defaults, how many did we catch
    - KS Statistic: Standard in credit scoring (see detailed comment below)
    - Gini Coefficient: = 2*AUC - 1, industry standard in BFSI
    - Brier Score: Probability calibration quality
    - Optimal Threshold: Threshold maximizing F1 (not always 0.5!)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from loguru import logger


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Comprehensive model evaluation with all credit-scoring-relevant metrics.

    Args:
        model: Fitted model with predict_proba method
        X_test: Test features (preprocessed)
        y_test: Test labels
        model_name: Human-readable model name for logging

    Returns:
        Dictionary with all computed metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold (maximizes F1 score)
    optimal_threshold = _find_optimal_threshold(y_test, y_pred_proba)

    # Binary predictions at optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # ── Core Metrics ──
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_pred_proba)

    # ── KS Statistic ──
    # The KS (Kolmogorov-Smirnov) statistic measures how well the model
    # separates defaulters from non-defaulters by comparing their cumulative
    # score distributions. It's computed as the maximum difference between
    # the CDF of scores for defaults and non-defaults.
    #
    # In the credit industry:
    #   KS > 40 is considered good
    #   KS > 50 is very good
    #   KS > 60 is excellent
    #
    # Recruiters from PhonePe, CRED, HDFC, or any BFSI company will
    # recognize this metric immediately — it's their primary model metric.
    ks_statistic = _compute_ks_statistic(y_test, y_pred_proba)

    # ── Gini Coefficient ──
    # Gini = 2 * AUC - 1
    # This is the standard metric used by every BFSI company.
    # It ranges from 0 (random model) to 1 (perfect model).
    # A Gini of 0.60+ is considered good for alternative credit scoring.
    gini_coefficient = 2 * auc_roc - 1

    # ── Confusion Matrix & Classification Report ──
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Repaid', 'Defaulted'])

    # ── ROC Curve data (for plotting) ──
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)

    # ── Precision-Recall Curve data (for plotting) ──
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

    results = {
        'model_name': model_name,
        'auc_roc': round(auc_roc, 4),
        'auc_pr': round(auc_pr, 4),
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'ks_statistic': round(ks_statistic, 2),
        'gini_coefficient': round(gini_coefficient, 4),
        'brier_score': round(brier, 4),
        'confusion_matrix': cm,
        'classification_report': report,
        'optimal_threshold': round(optimal_threshold, 4),
        # Curve data for visualization
        'roc_curve': {'fpr': fpr, 'tpr': tpr},
        'pr_curve': {'precision': pr_precision, 'recall': pr_recall},
    }

    # Log summary
    logger.info(f"  AUC-ROC: {results['auc_roc']:.4f}")
    logger.info(f"  Gini:    {results['gini_coefficient']:.4f}")
    logger.info(f"  KS Stat: {results['ks_statistic']:.1f}")
    logger.info(f"  F1:      {results['f1_score']:.4f}")
    logger.info(f"  Prec:    {results['precision']:.4f}")
    logger.info(f"  Recall:  {results['recall']:.4f}")
    logger.info(f"  Brier:   {results['brier_score']:.4f}")
    logger.info(f"  Optimal threshold: {results['optimal_threshold']:.4f}")

    return results


def _compute_ks_statistic(y_true, y_pred_proba) -> float:
    """
    Computes the Kolmogorov-Smirnov statistic.

    KS = max|CDF_defaults(score) - CDF_non_defaults(score)|

    This is the maximum vertical distance between the cumulative
    distribution functions of the predicted scores for the two classes.

    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        KS statistic (0-100 scale, as used in industry)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr) * 100  # Scale to 0-100
    return ks


def _find_optimal_threshold(y_true, y_pred_proba) -> float:
    """
    Finds the probability threshold that maximizes F1 score.

    The default threshold of 0.5 is almost never optimal for imbalanced
    data. With 8% default rate, the optimal threshold is typically
    around 0.15-0.30.

    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Creates a comparison table of all evaluated models.

    Args:
        results_list: List of result dicts from evaluate_model()

    Returns:
        Formatted comparison DataFrame
    """
    comparison = pd.DataFrame([
        {
            'Model': r['model_name'],
            'AUC-ROC': r['auc_roc'],
            'Gini': r['gini_coefficient'],
            'KS Stat': r['ks_statistic'],
            'F1': r['f1_score'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'Brier': r['brier_score'],
            'Threshold': r['optimal_threshold'],
        }
        for r in results_list
    ])

    # Sort by AUC-ROC descending
    comparison = comparison.sort_values('AUC-ROC', ascending=False)

    # Print formatted table
    logger.info("\n" + "=" * 100)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 100)
    logger.info(
        f"{'Model':<25} {'AUC-ROC':>8} {'Gini':>7} {'KS Stat':>8} "
        f"{'F1':>6} {'Prec':>6} {'Recall':>7} {'Brier':>7}"
    )
    logger.info("-" * 100)
    for _, row in comparison.iterrows():
        logger.info(
            f"{row['Model']:<25} {row['AUC-ROC']:>8.4f} {row['Gini']:>7.4f} "
            f"{row['KS Stat']:>8.1f} {row['F1']:>6.4f} {row['Precision']:>6.4f} "
            f"{row['Recall']:>7.4f} {row['Brier']:>7.4f}"
        )
    logger.info("=" * 100)

    return comparison
