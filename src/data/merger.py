"""
src/data/merger.py — Dataset Merger & Column Harmonization

Combines three datasets into one unified training set:
    1. Home Credit (filtered for gig-worker-like profiles)
    2. Give Me Some Credit (thin-file applicants)
    3. Synthetic India Gig Data

Alignment strategy:
    - All datasets share: income features, payment behavior features, target variable
    - Dataset-specific features get NaN for other datasets (handled by feature pipeline)
    - Adds data_source column for tracking and fairness analysis
    - Final dataset will have ~80,000–120,000 rows depending on filter thresholds

Column harmonization ensures a unified schema across datasets that were
collected with different naming conventions and scales.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Column Harmonization Mappings
# These map dataset-specific column names to our unified GigScore schema.
# Columns not in these maps are kept as-is (dataset-specific features).
# ─────────────────────────────────────────────────────────────────────

HOME_CREDIT_COLUMN_MAP = {
    'AMT_INCOME_TOTAL': 'monthly_earnings_proxy',
    'DAYS_EMPLOYED': 'employment_duration_days',
    'EXT_SOURCE_1': 'ext_source_1',
    'EXT_SOURCE_2': 'external_credit_score',
    'EXT_SOURCE_3': 'ext_source_3',
    'REGION_RATING_CLIENT': 'region_risk_rating',
    'CNT_CHILDREN': 'dependents_count',
    'FLAG_OWN_CAR': 'has_vehicle',
    'FLAG_OWN_REALTY': 'has_realty',
    'CODE_GENDER': 'gender',
    'DAYS_BIRTH': 'days_birth',
    'TARGET': 'target',
}

GIVE_ME_CREDIT_COLUMN_MAP = {
    'serious_dlqin_2yrs': 'target',
    'revolving_utilization': 'credit_utilization_ratio',
    'age': 'age',
    'late_30_59_count': 'late_30_59_count',
    'debt_ratio': 'debt_ratio',
    'monthly_income': 'monthly_earnings_proxy',
    'number_of_dependents': 'dependents_count',
    'open_credit_lines': 'open_credit_lines',
    'times_90_days_late': 'times_90_days_late',
    'real_estate_loans': 'real_estate_loans',
    'late_60_89_count': 'late_60_89_count',
    'has_verifiable_income': 'has_verifiable_income',
}

# Gig worker occupations for filtering Home Credit data
GIG_OCCUPATIONS = [
    'Drivers', 'Laborers', 'Sales staff', 'Cooking staff',
    'Security staff', 'Low-skill Laborers', 'Waiters/barmen staff',
    'Cleaning staff', 'Private service staff',
]


def filter_home_credit_gig_workers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters Home Credit data to keep only gig-worker-like profiles.

    Selection criteria (OR logic — keep if ANY condition is met):
        1. NAME_INCOME_TYPE IN ('Commercial associate', 'Working') AND
           OCCUPATION_TYPE is in the gig occupation list
        2. DAYS_EMPLOYED indicates short tenure (< 365 days) — indicating
           contract or gig-like employment patterns
        3. is_unemployed flag is set (formerly DAYS_EMPLOYED = 365243) —
           these are exactly our target population (informal work)

    This filtering reduces ~307k rows to ~45k-55k relevant rows.

    Args:
        df: Cleaned Home Credit DataFrame

    Returns:
        Filtered DataFrame with only gig-worker-like profiles
    """
    logger.info(f"Filtering Home Credit for gig-worker profiles (starting: {len(df):,} rows)")

    # Condition 1: Income type + occupation match
    income_type_mask = df['NAME_INCOME_TYPE'].isin(['Commercial associate', 'Working'])
    occupation_mask = df['OCCUPATION_TYPE'].isin(GIG_OCCUPATIONS)
    condition_1 = income_type_mask & occupation_mask

    # Condition 2: Short employment tenure (< 1 year = < 365 days)
    # DAYS_EMPLOYED is negative (days before application), so -365 means 1 year
    condition_2 = (df['DAYS_EMPLOYED'].notna()) & (df['DAYS_EMPLOYED'] > -365)

    # Condition 3: Unemployed flag (sentinel value = informal/gig work)
    condition_3 = df.get('is_unemployed', pd.Series(False, index=df.index)) == 1

    # Apply OR logic
    filtered = df[condition_1 | condition_2 | condition_3].copy()

    logger.info(f"  Condition 1 (income+occupation match): {condition_1.sum():,}")
    logger.info(f"  Condition 2 (short tenure <1 year): {condition_2.sum():,}")
    logger.info(f"  Condition 3 (unemployed/informal): {condition_3.sum():,}")
    logger.info(f"  After filtering (union): {len(filtered):,} rows")
    logger.info(f"  Default rate in filtered set: {filtered['TARGET'].mean():.2%}")

    return filtered


def harmonize_home_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonizes Home Credit columns to unified GigScore schema.

    Key transformations:
        - Rename columns per HOME_CREDIT_COLUMN_MAP
        - Convert AMT_INCOME_TOTAL to monthly (original is annual)
        - Convert DAYS_BIRTH to age in years
        - Encode FLAG_OWN_CAR from Y/N to 1/0
        - Add data_source tag

    Args:
        df: Filtered Home Credit DataFrame

    Returns:
        Harmonized DataFrame
    """
    df = df.copy()

    # Convert annual income to monthly for consistency with other datasets
    if 'AMT_INCOME_TOTAL' in df.columns:
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'] / 12.0

    # Convert DAYS_BIRTH to age in years (DAYS_BIRTH is negative)
    if 'DAYS_BIRTH' in df.columns:
        df['age'] = (-df['DAYS_BIRTH'] / 365.25).astype(int)

    # Encode binary car ownership
    if 'FLAG_OWN_CAR' in df.columns:
        df['FLAG_OWN_CAR'] = (df['FLAG_OWN_CAR'] == 'Y').astype(int)

    # Encode binary realty ownership
    if 'FLAG_OWN_REALTY' in df.columns:
        df['FLAG_OWN_REALTY'] = (df['FLAG_OWN_REALTY'] == 'Y').astype(int)

    # Rename columns to unified schema
    df = df.rename(columns=HOME_CREDIT_COLUMN_MAP)

    # Add data source tracking
    df['data_source'] = 'home_credit'

    return df


def harmonize_give_me_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonizes Give Me Some Credit columns to unified GigScore schema.

    Args:
        df: Cleaned Give Me Some Credit DataFrame

    Returns:
        Harmonized DataFrame
    """
    df = df.copy()

    # Rename columns to unified schema
    df = df.rename(columns=GIVE_ME_CREDIT_COLUMN_MAP)

    # Add data source tracking
    df['data_source'] = 'give_me_credit'

    return df


def harmonize_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonizes synthetic gig data columns.
    Most columns already have the right names, just need to add
    the monthly_earnings_proxy alias and ensure consistency.

    Args:
        df: Synthetic gig worker DataFrame

    Returns:
        Harmonized DataFrame
    """
    df = df.copy()

    # Map monthly_earnings_inr to unified name
    df['monthly_earnings_proxy'] = df['monthly_earnings_inr']

    # Map region_rating to unified name
    if 'region_rating' in df.columns:
        df['region_risk_rating'] = df['region_rating']

    # data_source already set in generator

    return df


def merge_all_datasets(
    home_credit_df: pd.DataFrame,
    give_me_credit_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    installments_df: pd.DataFrame = None,
    bureau_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merges all datasets into one unified training set.

    Pipeline:
        1. Filter Home Credit for gig-worker profiles
        2. Harmonize column names across all datasets
        3. If installments data provided, aggregate payment features per applicant
        4. If bureau data provided, aggregate credit history features per applicant
        5. Concatenate all datasets (union of columns, NaN where missing)
        6. Add data_source column for fairness analysis

    Args:
        home_credit_df: Cleaned Home Credit application DataFrame
        give_me_credit_df: Cleaned Give Me Some Credit DataFrame
        synthetic_df: Generated synthetic gig worker DataFrame
        installments_df: Optional installments payment data
        bureau_df: Optional bureau credit data

    Returns:
        Merged DataFrame with unified schema and ~80k-120k rows
    """
    logger.info("=" * 60)
    logger.info("MERGING ALL DATASETS")
    logger.info("=" * 60)

    # ── Step 1: Filter Home Credit ──
    hc_filtered = filter_home_credit_gig_workers(home_credit_df)

    # ── Step 2: Aggregate installments features (if available) ──
    if installments_df is not None:
        logger.info("Aggregating installments payment features...")
        installment_features = _aggregate_installment_features(installments_df)
        hc_filtered = hc_filtered.merge(
            installment_features,
            on='SK_ID_CURR',
            how='left'
        )
        logger.info(f"  Merged installment features: {len(installment_features.columns)-1} columns")

    # ── Step 3: Aggregate bureau features (if available) ──
    if bureau_df is not None:
        logger.info("Aggregating bureau credit features...")
        bureau_features = _aggregate_bureau_features(bureau_df)
        hc_filtered = hc_filtered.merge(
            bureau_features,
            on='SK_ID_CURR',
            how='left'
        )
        logger.info(f"  Merged bureau features: {len(bureau_features.columns)-1} columns")

    # ── Step 4: Harmonize columns ──
    hc_harmonized = harmonize_home_credit(hc_filtered)
    gmc_harmonized = harmonize_give_me_credit(give_me_credit_df)
    synth_harmonized = harmonize_synthetic(synthetic_df)

    logger.info(f"  Home Credit harmonized: {hc_harmonized.shape}")
    logger.info(f"  Give Me Credit harmonized: {gmc_harmonized.shape}")
    logger.info(f"  Synthetic harmonized: {synth_harmonized.shape}")

    # ── Step 5: Concatenate ──
    # Using outer join — dataset-specific features get NaN for other datasets.
    # This is intentional and handled by the feature pipeline's imputer.
    merged = pd.concat(
        [hc_harmonized, gmc_harmonized, synth_harmonized],
        axis=0,
        ignore_index=True,
        sort=False
    )

    # ── Step 6: Final cleanup ──
    # Ensure target is integer
    merged['target'] = merged['target'].astype(int)

    # Drop duplicate ID columns that don't matter for modeling
    drop_cols = ['SK_ID_CURR', 'SK_ID_PREV']
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors='ignore')

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info(f"  Total rows: {len(merged):,}")
    logger.info(f"  Total columns: {len(merged.columns)}")
    logger.info(f"  Default rate: {merged['target'].mean():.2%}")
    logger.info(f"  Data source breakdown:")
    for source, count in merged['data_source'].value_counts().items():
        logger.info(f"    {source}: {count:,} ({count/len(merged)*100:.1f}%)")
    logger.info("=" * 60)

    return merged


def _aggregate_installment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates installments_payments.csv into per-applicant features.

    Features engineered:
        - payment_on_time_ratio: % of payments made on or before due date
        - avg_days_late: Average days late (0 if always on time)
        - payment_amount_ratio: Average (amount_paid / amount_due)
        - total_installments: Total number of installment payments
        - payment_regularity_cv: Coefficient of variation of payment amounts

    Args:
        df: Raw installments DataFrame

    Returns:
        DataFrame with SK_ID_CURR and aggregated features
    """
    # Calculate days late (positive = late, negative = early)
    df = df.copy()
    df['days_late'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['is_on_time'] = (df['days_late'] <= 0).astype(int)
    df['days_late_positive'] = df['days_late'].clip(lower=0)
    df['payment_ratio'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT'].replace(0, np.nan)

    agg = df.groupby('SK_ID_CURR').agg(
        payment_on_time_ratio=('is_on_time', 'mean'),
        avg_days_late=('days_late_positive', 'mean'),
        payment_amount_ratio=('payment_ratio', 'mean'),
        total_installments=('is_on_time', 'count'),
        payment_amount_std=('AMT_PAYMENT', 'std'),
        payment_amount_mean=('AMT_PAYMENT', 'mean'),
    ).reset_index()

    # Coefficient of variation = std / mean (lower = more consistent)
    agg['payment_regularity_cv'] = (
        agg['payment_amount_std'] / agg['payment_amount_mean'].replace(0, np.nan)
    ).fillna(0)

    # Calculate payment streak (longest consecutive on-time payments)
    streak_df = df.sort_values(['SK_ID_CURR', 'DAYS_INSTALMENT'])
    streak_agg = streak_df.groupby('SK_ID_CURR')['is_on_time'].apply(
        _longest_streak
    ).reset_index()
    streak_agg.columns = ['SK_ID_CURR', 'payment_streak_max']

    agg = agg.merge(streak_agg, on='SK_ID_CURR', how='left')

    # Drop intermediate columns
    agg = agg.drop(columns=['payment_amount_std', 'payment_amount_mean'])

    return agg


def _longest_streak(series: pd.Series) -> int:
    """Calculate the longest consecutive streak of 1s in a binary series."""
    max_streak = 0
    current_streak = 0
    for val in series:
        if val == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def _aggregate_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates bureau.csv into per-applicant credit history features.

    Features engineered:
        - total_past_credit_count: Total number of previous credits
        - has_overdue_history: Binary — ever had overdue payments
        - credit_type_diversity: Number of distinct credit types used
        - active_credits_count: Number of currently active credits
        - total_credit_sum: Total credit amount across all past credits
        - max_overdue_amount: Maximum overdue amount (severity indicator)

    Args:
        df: Raw bureau DataFrame

    Returns:
        DataFrame with SK_ID_CURR and aggregated features
    """
    agg = df.groupby('SK_ID_CURR').agg(
        total_past_credit_count=('SK_ID_BUREAU', 'count'),
        has_overdue_history=('CREDIT_DAY_OVERDUE', lambda x: int((x > 0).any())),
        credit_type_diversity=('CREDIT_TYPE', 'nunique'),
        active_credits_count=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        total_credit_sum=('AMT_CREDIT_SUM', 'sum'),
        max_overdue_amount=('AMT_CREDIT_SUM_OVERDUE', 'max'),
    ).reset_index()

    # Fill NaN in max_overdue_amount with 0 (no overdue)
    agg['max_overdue_amount'] = agg['max_overdue_amount'].fillna(0)

    return agg
