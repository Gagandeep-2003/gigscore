"""
src/features/india_features.py — India-Specific UPI/Mobile-Based Features

Engineers features specific to India's digital economy landscape:
    - UPI transaction patterns (digital financial participation proxy)
    - Migration stability indicators
    - Lifestyle risk factors
    - Multi-platform income diversification

These features leverage India's unique digital payments infrastructure
(UPI) and demographic patterns to assess creditworthiness of gig workers
who don't have traditional credit histories.

Source context:
    - India has 300M+ UPI users (RBI data, 2023)
    - UPI processed ₹14.05 lakh crore in March 2023 alone
    - Gig workers heavily rely on UPI for both earning and spending
"""

import numpy as np
import pandas as pd
from loguru import logger


def engineer_india_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers India-specific features for gig worker credit assessment.

    Features created (4 total):
        1. digital_engagement_score — UPI usage as financial literacy proxy
        2. settlement_stability_score — geographic stability indicator
        3. lifestyle_risk_score — lifestyle-based risk factors
        4. multi_platform_bonus — income diversification signal

    These features are primarily computed from synthetic gig data columns
    but are designed to work with real UPI data when available.

    Args:
        df: Merged DataFrame with all data sources

    Returns:
        DataFrame with India-specific features added
    """
    df = df.copy()
    logger.info("Engineering India-specific features...")

    # ── 1. Digital Engagement Score ──
    # UPI engagement as a proxy for financial literacy and participation.
    # Components:
    #   - Transaction volume (50%): More UPI transactions = more digital participation
    #   - Transaction consistency (50%): Regular usage = stable financial behavior
    #
    # Why this matters: In India, UPI adoption is the strongest indicator of
    # financial system participation for the unbanked/underbanked population.
    # Someone with high, consistent UPI usage is demonstrably engaged with
    # the financial system even without a formal credit history.
    if ('upi_transaction_count_monthly' in df.columns and
            'upi_transaction_consistency_score' in df.columns):
        # Normalize transaction count to [0,1] — cap at 300
        upi_volume_norm = (df['upi_transaction_count_monthly'].fillna(20) / 300).clip(0, 1)

        df['digital_engagement_score'] = (
            upi_volume_norm * 0.5 +
            df['upi_transaction_consistency_score'].fillna(0.5) * 0.5
        )
        df['digital_engagement_score'] = df['digital_engagement_score'].clip(0, 1)
        logger.info(f"  digital_engagement_score: mean={df['digital_engagement_score'].mean():.3f}")

    # ── 2. Settlement Stability Score ──
    # Geographic stability + financial buffer proxy.
    # Components:
    #   - Years in city (60%): Longer residency = established network, stable earnings
    #   - Low mobile recharge frequency (40%): Fewer, larger recharges = better budgeting
    #     (High recharge frequency = many small top-ups = living expense-to-expense)
    #
    # Why this matters: Inter-city migration is extremely common among Indian
    # gig workers. Workers who have been in the same city for longer tend to
    # have more stable earnings, housing, and social networks.
    components = []

    if 'years_in_city' in df.columns:
        # Saturate at 10 years (beyond that, marginal stability gain is minimal)
        city_stability = (df['years_in_city'].fillna(2) / 10).clip(0, 1)
        components.append(city_stability * 0.6)

    if 'mobile_recharge_frequency' in df.columns:
        # Inverse relationship: lower frequency = larger, planned recharges = better
        recharge_score = (1 / df['mobile_recharge_frequency'].fillna(2)).clip(0, 1)
        components.append(recharge_score * 0.4)

    if components:
        df['settlement_stability_score'] = sum(components)
        total_weight = 0.6 * ('years_in_city' in df.columns) + 0.4 * ('mobile_recharge_frequency' in df.columns)
        if total_weight > 0 and total_weight < 1:
            df['settlement_stability_score'] = df['settlement_stability_score'] / total_weight
        df['settlement_stability_score'] = df['settlement_stability_score'].clip(0, 1)
        logger.info(f"  settlement_stability_score: mean={df['settlement_stability_score'].mean():.3f}")

    # ── 3. Lifestyle Risk Score ──
    # Higher = more lifestyle risk factors.
    # Components:
    #   - Late night work ratio (50%): Working 10pm-5am = health/safety risks
    #   - Dependents count normalized (30%): More dependents = more financial pressure
    #   - No premium smartphone (20%): Lower device quality = lower earning capacity
    #
    # NOTE: This is a RISK score (higher = worse) unlike other scores.
    # It will have negative SHAP values for people with low risk.
    risk_components = []

    if 'late_night_work_ratio' in df.columns:
        risk_components.append(df['late_night_work_ratio'].fillna(0.15) * 0.5)

    if 'dependents_count' in df.columns:
        dep_norm = (df['dependents_count'].fillna(1) / 6).clip(0, 1)
        risk_components.append(dep_norm * 0.3)

    if 'owns_smartphone_above_10k' in df.columns:
        risk_components.append((1 - df['owns_smartphone_above_10k'].fillna(0.5)) * 0.2)

    if risk_components:
        df['lifestyle_risk_score'] = sum(risk_components)
        total_weight = (0.5 * ('late_night_work_ratio' in df.columns) +
                       0.3 * ('dependents_count' in df.columns) +
                       0.2 * ('owns_smartphone_above_10k' in df.columns))
        if total_weight > 0 and total_weight < 1:
            df['lifestyle_risk_score'] = df['lifestyle_risk_score'] / total_weight
        df['lifestyle_risk_score'] = df['lifestyle_risk_score'].clip(0, 1)
        logger.info(f"  lifestyle_risk_score: mean={df['lifestyle_risk_score'].mean():.3f}")

    # ── 4. Multi-Platform Bonus ──
    # Active on multiple gig platforms = diversified income = lower default risk.
    # log1p scaling ensures diminishing returns (2→3 platforms matters more than 3→4).
    if 'active_platforms_count' in df.columns:
        df['multi_platform_bonus'] = (
            np.log1p(df['active_platforms_count'].fillna(1) - 1) / np.log1p(3)
        )
        df['multi_platform_bonus'] = df['multi_platform_bonus'].clip(0, 1)
        logger.info(f"  multi_platform_bonus: mean={df['multi_platform_bonus'].mean():.3f}")

    n_features = sum(1 for c in [
        'digital_engagement_score', 'settlement_stability_score',
        'lifestyle_risk_score', 'multi_platform_bonus'
    ] if c in df.columns)
    logger.info(f"  India-specific features created: {n_features}")

    return df
