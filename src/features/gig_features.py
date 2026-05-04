"""
src/features/gig_features.py — Gig Worker-Specific Engineered Features

These features are the CORE DIFFERENTIATOR of GigScore from traditional
credit scoring. They capture behavioral signals specific to gig workers:
    - Payment reliability from installment history
    - Platform engagement and consistency
    - Income stability composites
    - Financial resilience indicators

Traditional credit models miss these entirely because they were designed
for salaried employees with formal employer verification.
"""

import numpy as np
import pandas as pd
from loguru import logger


def engineer_gig_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers gig-worker-specific features from synthetic data and
    installments payment history.

    Features created (9 total):
        FROM INSTALLMENTS DATA:
        1. payment_on_time_ratio — behavioral reliability (already from merger)
        2. payment_amount_ratio — disciplined payer signal (already from merger)
        3. payment_streak_max — consistency streak (already from merger)
        4. payment_regularity_cv — payment consistency (already from merger)

        FROM SYNTHETIC/UNIFIED DATA:
        5. income_stability_score — composite income reliability
        6. platform_reliability_score — composite platform engagement
        7. income_experience_ratio — earnings efficiency
        8. financial_resilience_score — buffer capacity
        9. work_intensity_score — work volume proxy

    Args:
        df: Merged DataFrame with all data sources

    Returns:
        DataFrame with gig-specific features added
    """
    df = df.copy()
    logger.info("Engineering gig-specific features...")

    # ── 1. Income Stability Score (COMPOSITE) ──
    # Combines income volatility and income gap data into a single [0,1] score.
    # High score = stable, reliable income pattern.
    # This is the most important gig-specific feature because it directly
    # addresses the core objection banks have: "gig income is too irregular."
    if ('income_volatility_coefficient' in df.columns and
            'income_gap_months_last_year' in df.columns):
        df['income_stability_score'] = (
            (1 - df['income_volatility_coefficient'].fillna(0.5)) *
            (1 - df['income_gap_months_last_year'].fillna(2) / 12)
        )
        df['income_stability_score'] = df['income_stability_score'].clip(0, 1)
        logger.info(f"  income_stability_score: mean={df['income_stability_score'].mean():.3f}")

    # ── 2. Platform Reliability Score (COMPOSITE) ──
    # Weighted combination of platform rating, work consistency, and cancellation rate.
    # Weights: rating=40%, consistency=30%, low-cancellation=30%
    # Rationale: Platform rating is the strongest signal (direct customer feedback),
    # but consistency and cancellation behavior add temporal reliability data.
    if all(c in df.columns for c in ['platform_rating', 'weekly_work_consistency', 'cancellation_rate']):
        df['platform_reliability_score'] = (
            (df['platform_rating'].fillna(3.5) / 5) * 0.4 +
            df['weekly_work_consistency'].fillna(0.5) * 0.3 +
            (1 - df['cancellation_rate'].fillna(0.2)) * 0.3
        )
        df['platform_reliability_score'] = df['platform_reliability_score'].clip(0, 1)
        logger.info(f"  platform_reliability_score: mean={df['platform_reliability_score'].mean():.3f}")

    # ── 3. Income-to-Experience Ratio ──
    # Earnings per month of experience on the platform.
    # IMPROVING over time is positive — shows the worker is building reputation
    # and earning more per unit of experience.
    if 'monthly_earnings_inr' in df.columns and 'platform_tenure_months' in df.columns:
        df['income_experience_ratio'] = (
            df['monthly_earnings_inr'] / (df['platform_tenure_months'].fillna(1) + 1)
        )
        # Normalize to reasonable range
        df['income_experience_ratio'] = df['income_experience_ratio'].clip(upper=50000)

    # ── 4. Financial Resilience Score (COMPOSITE) ──
    # Measures the worker's financial buffer — ability to absorb income shocks.
    # Components:
    #   - Has savings account (40%) — emergency fund access
    #   - Low rent-to-income ratio (35%) — not housing-cost-burdened
    #   - Has life insurance (25%) — financial planning behavior
    components = []
    weights = []

    if 'has_savings_account' in df.columns:
        components.append(df['has_savings_account'].fillna(0).astype(float) * 0.4)
        weights.append(0.4)
    if 'rent_to_income_ratio' in df.columns:
        components.append((1 - df['rent_to_income_ratio'].fillna(0.4)) * 0.35)
        weights.append(0.35)
    if 'has_life_insurance' in df.columns:
        components.append(df['has_life_insurance'].fillna(0).astype(float) * 0.25)
        weights.append(0.25)

    if components:
        df['financial_resilience_score'] = sum(components)
        # Renormalize if not all components are present
        if sum(weights) < 1.0:
            df['financial_resilience_score'] = df['financial_resilience_score'] / sum(weights)
        df['financial_resilience_score'] = df['financial_resilience_score'].clip(0, 1)
        logger.info(f"  financial_resilience_score: mean={df['financial_resilience_score'].mean():.3f}")

    # ── 5. Work Intensity Score ──
    # Proxy for how actively the worker is engaged with the platform.
    # Combines total trips with consistency to get a quality-adjusted volume metric.
    if 'monthly_trips_or_orders' in df.columns and 'weekly_work_consistency' in df.columns:
        # Normalize trips to [0,1] range
        trips_norm = df['monthly_trips_or_orders'].fillna(100) / 700
        trips_norm = trips_norm.clip(0, 1)
        df['work_intensity_score'] = (
            trips_norm * 0.6 + df['weekly_work_consistency'].fillna(0.5) * 0.4
        )
        df['work_intensity_score'] = df['work_intensity_score'].clip(0, 1)

    # ── 6. Payment Behavior Score (for Home Credit data) ──
    # Combines installment payment features into a single behavioral score.
    if 'payment_on_time_ratio' in df.columns:
        payment_components = []

        # On-time ratio (strongest signal)
        payment_components.append(
            df['payment_on_time_ratio'].fillna(0.5) * 0.4
        )

        # Low average days late
        if 'avg_days_late' in df.columns:
            max_late = df['avg_days_late'].quantile(0.95)
            late_norm = 1 - (df['avg_days_late'].fillna(0) / max(max_late, 1))
            payment_components.append(late_norm.clip(0, 1) * 0.3)

        # Payment amount ratio (paying ≥ required amount)
        if 'payment_amount_ratio' in df.columns:
            payment_components.append(
                df['payment_amount_ratio'].fillna(1.0).clip(0, 2) / 2 * 0.3
            )

        df['payment_behavior_score'] = sum(payment_components)
        df['payment_behavior_score'] = df['payment_behavior_score'].clip(0, 1)
        logger.info(f"  payment_behavior_score: mean={df['payment_behavior_score'].mean():.3f}")

    # ── 7. Income Momentum (INTERACTION FEATURE) ──
    # Captures the interaction between income trend and volatility.
    # A positive trend WITH low volatility = genuinely improving worker.
    # A positive trend WITH high volatility = noise, not real improvement.
    # This is a non-linear interaction that tree models can exploit
    # but logistic regression cannot — key to beating the baseline.
    if 'income_trend_6m' in df.columns and 'income_volatility_coefficient' in df.columns:
        trend = df['income_trend_6m'].fillna(0.0)
        volatility = df['income_volatility_coefficient'].fillna(0.5)
        df['income_momentum'] = (trend * (1 - volatility)).clip(-1, 1)
        logger.info(f"  income_momentum: mean={df['income_momentum'].mean():.3f}")

    # ── 8. Income Diversification Score ──
    # Multi-platform workers have more diversified income streams.
    # Log scaling ensures diminishing returns (2→3 matters more than 3→4).
    if 'active_platforms_count' in df.columns:
        platforms = df['active_platforms_count'].fillna(1).clip(1, 4)
        df['income_diversification_score'] = (
            np.log1p(platforms - 1) / np.log1p(3)
        ).clip(0, 1)
        logger.info(f"  income_diversification_score: mean={df['income_diversification_score'].mean():.3f}")

    # ── 9. Debt Stress Score (INTERACTION FEATURE) ──
    # Captures the interaction between credit burden and income instability.
    # High debt ratio alone may be okay IF income is stable.
    # High debt ratio WITH unstable income = acute stress.
    if 'credit_income_ratio' in df.columns:
        credit_ratio = df['credit_income_ratio'].fillna(0).clip(0, 10)
        stability = df.get('income_stability_score',
                           pd.Series(0.5, index=df.index))
        df['debt_stress_score'] = (credit_ratio * (1 - stability)).clip(0, 5)
        logger.info(f"  debt_stress_score: mean={df['debt_stress_score'].mean():.3f}")
    elif 'rent_to_income_ratio' in df.columns:
        # Fallback for synthetic data — use rent ratio as debt proxy
        rent = df['rent_to_income_ratio'].fillna(0.3)
        stability = df.get('income_stability_score',
                           pd.Series(0.5, index=df.index))
        df['debt_stress_score'] = (rent * 3 * (1 - stability)).clip(0, 5)
        logger.info(f"  debt_stress_score (rent-based): mean={df['debt_stress_score'].mean():.3f}")

    # ── 10. Earnings Efficiency Score ──
    # How much does the worker earn relative to their work intensity?
    # High earnings per trip/order = skilled/efficient worker = lower risk.
    if ('monthly_earnings_inr' in df.columns and
            'monthly_trips_or_orders' in df.columns):
        trips = df['monthly_trips_or_orders'].fillna(100).clip(lower=1)
        df['earnings_efficiency'] = np.log1p(
            df['monthly_earnings_inr'].fillna(25000) / trips
        )
        logger.info(f"  earnings_efficiency: mean={df['earnings_efficiency'].mean():.3f}")

    n_features = sum(1 for c in [
        'income_stability_score', 'platform_reliability_score',
        'income_experience_ratio', 'financial_resilience_score',
        'work_intensity_score', 'payment_behavior_score',
        'income_momentum', 'income_diversification_score',
        'debt_stress_score', 'earnings_efficiency',
    ] if c in df.columns)
    logger.info(f"  Gig-specific features created: {n_features}")

    return df
