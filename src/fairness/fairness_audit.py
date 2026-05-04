"""
src/fairness/fairness_audit.py — Model Fairness Audit

Audits the GigScore model for bias across protected and proxy attributes:
    1. Gender (CODE_GENDER / gender field)
    2. Region risk rating (REGION_RATING_CLIENT: 1=urban, 3=rural — correlates with poverty)
    3. Income bracket (low/medium/high income thirds)
    4. Data source (home_credit / give_me_credit / synthetic_india_gig)

Uses Fairlearn's MetricFrame for demographic-level metric computation
and reports standard fairness metrics:
    - Demographic Parity Difference: Difference in approval rates between groups
    - Equalized Odds Difference: Max difference in TPR or FPR between groups

Thresholds (based on industry practice and US EEOC 80% rule analog):
    < 0.05: FAIR ✅ (within acceptable range)
    0.05-0.10: REVIEW ⚠️ (needs monitoring, potentially acceptable)
    > 0.10: BIASED ❌ (requires model remediation)
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score
)
from loguru import logger

try:
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Fairlearn not installed. Fairness audit will use manual computations.")
    FAIRLEARN_AVAILABLE = False


def run_fairness_audit(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_features: pd.DataFrame,
    threshold: float = 0.5,
    save_path: str = None,
) -> dict:
    """
    Comprehensive fairness audit of the GigScore model.

    Audits bias across multiple sensitive/proxy attributes and computes
    per-group metrics to identify disparate impact.

    Args:
        model: Fitted model with predict_proba method
        X_test: Preprocessed test features
        y_test: True labels
        sensitive_features: DataFrame with columns for each sensitive attribute
            (e.g., 'gender', 'region_risk_rating', 'income_bracket')
        threshold: Classification threshold for binary predictions
        save_path: Optional path to save fairness_report.json

    Returns:
        Comprehensive fairness audit report as dictionary
    """
    logger.info("=" * 60)
    logger.info("FAIRNESS AUDIT")
    logger.info("=" * 60)

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # "Approved" = NOT predicted to default (y_pred == 0)
    y_approved = 1 - y_pred

    report = {
        'audit_summary': {},
        'group_metrics': {},
        'fairness_metrics': {},
        'overall_metrics': {
            'total_samples': len(y_test),
            'overall_approval_rate': float(y_approved.mean()),
            'overall_default_rate': float(y_test.mean()),
            'threshold_used': threshold,
        }
    }

    # Audit each sensitive attribute
    for col in sensitive_features.columns:
        logger.info(f"\n--- Auditing: {col} ---")
        group_values = sensitive_features[col].values

        group_report = _audit_single_attribute(
            y_test, y_pred, y_approved, y_pred_proba, group_values, col
        )
        report['group_metrics'][col] = group_report['group_metrics']
        report['fairness_metrics'][col] = group_report['fairness_metrics']

        # Summarize
        dpd = group_report['fairness_metrics'].get('demographic_parity_difference', 0)
        eod = group_report['fairness_metrics'].get('equalized_odds_difference', 0)
        report['audit_summary'][col] = {
            'demographic_parity_diff': dpd,
            'dpd_status': _get_fairness_status(dpd),
            'equalized_odds_diff': eod,
            'eod_status': _get_fairness_status(eod),
        }

    # Print summary
    _print_audit_summary(report)

    # Save report
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nFairness report saved to: {save_path}")

    return report


def _audit_single_attribute(
    y_true, y_pred, y_approved, y_pred_proba, group_values, attribute_name
) -> dict:
    """
    Audits model fairness for a single sensitive attribute.

    Args:
        y_true: True labels
        y_pred: Binary predictions (1=default)
        y_approved: Binary approval (1=approved, 0=declined)
        y_pred_proba: Predicted probabilities
        group_values: Array of group memberships
        attribute_name: Name of the attribute being audited

    Returns:
        Dict with group metrics and fairness metrics
    """
    groups = pd.Series(group_values)
    unique_groups = groups.unique()

    # Per-group metrics
    group_metrics = {}
    for group in sorted(unique_groups, key=str):
        mask = groups == group
        n = mask.sum()
        if n == 0:
            continue

        # True Positive Rate (Recall): Of actual defaults, how many did we catch?
        tpr = recall_score(y_true[mask], y_pred[mask], zero_division=0)

        # False Positive Rate: Of actual non-defaults, how many did we wrongly flag?
        tn = ((y_pred[mask] == 0) & (y_true[mask] == 0)).sum()
        fp = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum()
        fpr = fp / max(fp + tn, 1)

        # Selection/Approval rate
        approval_rate = float(y_approved[mask].mean())

        # Precision
        prec = precision_score(y_true[mask], y_pred[mask], zero_division=0)

        group_metrics[str(group)] = {
            'count': int(n),
            'default_rate': float(y_true[mask].mean()),
            'approval_rate': approval_rate,
            'true_positive_rate': round(tpr, 4),
            'false_positive_rate': round(fpr, 4),
            'precision': round(prec, 4),
        }

        logger.info(
            f"  Group '{group}': n={n:,}, approval={approval_rate:.2%}, "
            f"TPR={tpr:.3f}, FPR={fpr:.3f}"
        )

    # Compute fairness metrics
    fairness_metrics = {}

    # Demographic Parity Difference
    approval_rates = [m['approval_rate'] for m in group_metrics.values()]
    if len(approval_rates) >= 2:
        dpd = max(approval_rates) - min(approval_rates)
        fairness_metrics['demographic_parity_difference'] = round(dpd, 4)

    # Equalized Odds Difference (max of TPR diff and FPR diff)
    tpr_values = [m['true_positive_rate'] for m in group_metrics.values()]
    fpr_values = [m['false_positive_rate'] for m in group_metrics.values()]
    if len(tpr_values) >= 2:
        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        eod = max(tpr_diff, fpr_diff)
        fairness_metrics['equalized_odds_difference'] = round(eod, 4)
        fairness_metrics['tpr_difference'] = round(tpr_diff, 4)
        fairness_metrics['fpr_difference'] = round(fpr_diff, 4)

    return {
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
    }


def _get_fairness_status(value: float) -> str:
    """
    Returns fairness status based on standard thresholds.

    Args:
        value: Fairness metric value (0-1)

    Returns:
        Status string with emoji
    """
    if abs(value) < 0.05:
        return "FAIR ✅"
    elif abs(value) < 0.10:
        return "REVIEW ⚠️"
    else:
        return "BIASED ❌"


def _print_audit_summary(report: dict):
    """Prints a formatted fairness audit summary."""
    logger.info("\n" + "=" * 60)
    logger.info("FAIRNESS AUDIT SUMMARY")
    logger.info("=" * 60)

    for attr, summary in report['audit_summary'].items():
        logger.info(f"\n  {attr}:")
        logger.info(
            f"    Demographic Parity Diff: {summary['demographic_parity_diff']:.4f} "
            f"[{summary['dpd_status']}]"
        )
        logger.info(
            f"    Equalized Odds Diff:     {summary['equalized_odds_diff']:.4f} "
            f"[{summary['eod_status']}]"
        )

    logger.info("\n" + "=" * 60)


def prepare_sensitive_features(
    X_raw: pd.DataFrame,
    y: np.ndarray = None,
) -> pd.DataFrame:
    """
    Extracts and prepares sensitive features for fairness audit from raw data.

    Creates standardized groupings:
        - gender: M/F
        - region_risk_rating: 1/2/3
        - income_bracket: low/medium/high (tertiles)

    Args:
        X_raw: Raw (un-preprocessed) feature DataFrame
        y: Optional labels (not used but for consistency)

    Returns:
        DataFrame with sensitive feature columns
    """
    sensitive = pd.DataFrame(index=X_raw.index)

    # Gender
    if 'gender' in X_raw.columns:
        sensitive['gender'] = X_raw['gender'].fillna('Unknown')
    elif 'CODE_GENDER' in X_raw.columns:
        sensitive['gender'] = X_raw['CODE_GENDER'].fillna('Unknown')

    # Region risk rating
    if 'region_risk_rating' in X_raw.columns:
        sensitive['region_risk_rating'] = X_raw['region_risk_rating'].fillna(2).astype(int)
    elif 'REGION_RATING_CLIENT' in X_raw.columns:
        sensitive['region_risk_rating'] = X_raw['REGION_RATING_CLIENT'].fillna(2).astype(int)

    # Income bracket (tertiles)
    income_col = None
    for col in ['monthly_earnings_proxy', 'monthly_earnings_inr', 'AMT_INCOME_TOTAL']:
        if col in X_raw.columns:
            income_col = col
            break

    if income_col:
        income = X_raw[income_col].fillna(X_raw[income_col].median())
        tertiles = income.quantile([0.33, 0.66])
        conditions = [
            income <= tertiles.iloc[0],
            (income > tertiles.iloc[0]) & (income <= tertiles.iloc[1]),
            income > tertiles.iloc[1],
        ]
        sensitive['income_bracket'] = np.select(conditions, ['low', 'medium', 'high'], default='medium')

    # Data source
    if 'data_source' in X_raw.columns:
        sensitive['data_source'] = X_raw['data_source'].fillna('unknown')

    return sensitive
