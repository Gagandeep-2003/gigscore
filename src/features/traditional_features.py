"""
src/features/traditional_features.py — Traditional Credit Features

Engineers features from the Home Credit base dataset that mirror
traditional credit scoring signals (income ratios, credit terms,
employment history). These are the features that banks normally use.

For GigScore, these features serve as a baseline — they work well
for traditional applicants but miss the behavioral signals that
differentiate good vs. risky gig workers. That's why we supplement
with gig_features.py and india_features.py.
"""

import numpy as np
import pandas as pd
from loguru import logger


def engineer_traditional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers traditional credit features from Home Credit base columns.

    Features created (10 total):
        1. credit_income_ratio — loan amount vs income (overextension risk)
        2. annuity_income_ratio — monthly repayment burden
        3. credit_term_months — effective loan duration
        4. income_per_person — real disposable income per family member
        5. ext_source_mean — average of external credit scores
        6. ext_source_std — disagreement between external sources
        7. ext_source_available_count — credit file thickness (already from cleaner)
        8. days_employed_percent — career stability
        9. documents_provided_count — documentation transparency
        10. enquiry_score_7d — credit shopping recency/desperation

    Args:
        df: Cleaned and harmonized DataFrame

    Returns:
        DataFrame with new feature columns added
    """
    df = df.copy()
    logger.info("Engineering traditional credit features...")

    # ── 1. Credit-to-Income Ratio ──
    # Interpretation: How large is the requested loan relative to income?
    # High ratio = borrower is taking on more debt than income supports = risky.
    # Industry standard threshold: ratio > 8 is considered dangerous.
    if 'AMT_CREDIT' in df.columns and 'monthly_earnings_proxy' in df.columns:
        df['credit_income_ratio'] = (
            df['AMT_CREDIT'] / df['monthly_earnings_proxy'].replace(0, np.nan)
        ).fillna(0)
        # Cap extreme values to prevent model distortion
        df['credit_income_ratio'] = df['credit_income_ratio'].clip(upper=50)
        logger.info(f"  credit_income_ratio: mean={df['credit_income_ratio'].mean():.2f}")

    # ── 2. Annuity-to-Income Ratio ──
    # Interpretation: What % of monthly income goes to loan repayment?
    # Rule of thumb: > 40% is dangerous for low-income borrowers.
    if 'AMT_ANNUITY' in df.columns and 'monthly_earnings_proxy' in df.columns:
        df['annuity_income_ratio'] = (
            df['AMT_ANNUITY'] / df['monthly_earnings_proxy'].replace(0, np.nan)
        ).fillna(0)
        df['annuity_income_ratio'] = df['annuity_income_ratio'].clip(upper=2)
        logger.info(f"  annuity_income_ratio: mean={df['annuity_income_ratio'].mean():.2f}")

    # ── 3. Credit Term in Months ──
    # Interpretation: Effective loan duration = total credit / monthly payment.
    # Longer terms mean smaller monthly payments but more total interest.
    if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
        df['credit_term_months'] = (
            df['AMT_CREDIT'] / df['AMT_ANNUITY'].replace(0, np.nan)
        ).fillna(0)
        df['credit_term_months'] = df['credit_term_months'].clip(upper=120)

    # ── 4. Income Per Person ──
    # Interpretation: Real disposable income adjusted for family size.
    # A ₹50,000 earner with 5 dependents is very different from a single earner.
    if 'monthly_earnings_proxy' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['income_per_person'] = (
            df['monthly_earnings_proxy'] / (df['CNT_FAM_MEMBERS'].fillna(1) + 1)
        )
    elif 'monthly_earnings_proxy' in df.columns and 'dependents_count' in df.columns:
        df['income_per_person'] = (
            df['monthly_earnings_proxy'] / (df['dependents_count'].fillna(0) + 1)
        )

    # ── 5 & 6. External Source Aggregations ──
    # Mean of available external scores — composite credit file signal.
    # Std measures disagreement between sources — high std = uncertain profile.
    ext_cols = ['ext_source_1', 'external_credit_score', 'ext_source_3']
    available_ext = [c for c in ext_cols if c in df.columns]
    if available_ext:
        df['ext_source_mean'] = df[available_ext].mean(axis=1, skipna=True)
        df['ext_source_std'] = df[available_ext].std(axis=1, skipna=True).fillna(0)
        logger.info(f"  ext_source_mean: mean={df['ext_source_mean'].mean():.3f}")

    # ── 7. Days Employed Percent ──
    # Interpretation: What fraction of their life have they been working?
    # Low for young workers (normal), low for older workers (concerning).
    if 'employment_duration_days' in df.columns and 'days_birth' in df.columns:
        # Both are negative in original data, so ratio is positive
        df['days_employed_percent'] = (
            df['employment_duration_days'] / df['days_birth'].replace(0, np.nan)
        ).fillna(0)
        df['days_employed_percent'] = df['days_employed_percent'].clip(0, 1)

    # ── 8. Documents Provided Count ──
    # Interpretation: More documents submitted = more transparency = lower risk.
    # Gig workers typically submit fewer documents (they don't have salary slips).
    flag_doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    if flag_doc_cols:
        df['documents_provided_count'] = df[flag_doc_cols].sum(axis=1)
        logger.info(f"  documents_provided_count: mean={df['documents_provided_count'].mean():.1f}")

    # ── 9. Enquiry Score (7-day) ──
    # Interpretation: Ratio of recent (7-day) to annual credit bureau enquiries.
    # High ratio = the person is desperately seeking credit right now = risky.
    # This is a known credit risk signal used by every bank.
    if ('AMT_REQ_CREDIT_BUREAU_WEEK' in df.columns and
            'AMT_REQ_CREDIT_BUREAU_YEAR' in df.columns):
        df['enquiry_score_7d'] = (
            df['AMT_REQ_CREDIT_BUREAU_WEEK'] /
            (df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0) + 1)
        )

    n_features = sum(1 for c in ['credit_income_ratio', 'annuity_income_ratio',
                                  'credit_term_months', 'income_per_person',
                                  'ext_source_mean', 'ext_source_std',
                                  'days_employed_percent', 'documents_provided_count',
                                  'enquiry_score_7d'] if c in df.columns)
    logger.info(f"  Traditional features created: {n_features}")

    return df
