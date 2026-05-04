"""
src/data/cleaner.py — Data Cleaning & Preprocessing Module

Handles all data cleaning operations for GigScore:
    - Missing value imputation (strategy varies by column type and meaning)
    - Outlier treatment (IQR-based capping for skewed financial columns)
    - Sentinel value handling (DAYS_EMPLOYED = 365243)
    - Feature-type-specific cleaning logic
    - Gig worker proxy flag creation

DESIGN DECISIONS:
    1. Missing values are NEVER just dropped — missingness is information,
       especially for thin-file/gig worker applicants.
    2. Outliers are CAPPED, not removed — extreme values are real data points
       in credit scoring, they just need bounded influence.
    3. Sentinel values (365243) are decoded into meaningful features before imputation.
    4. Categorical unknowns get the explicit string "UNKNOWN" — never NaN.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Gig Worker Occupation Mapping
# Maps Home Credit OCCUPATION_TYPE to binary gig-worker-like proxy.
# These occupations mirror the type of work done in India's gig economy:
#   Drivers → Ola/Uber, Laborers → construction gig, Sales staff → retail gig,
#   Cooking staff → Swiggy/Zomato cloud kitchens, Security → contract security, etc.
# ─────────────────────────────────────────────────────────────────────
GIG_OCCUPATION_MAP = {
    'Drivers': 1,
    'Laborers': 1,
    'Sales staff': 1,
    'Cooking staff': 1,
    'Security staff': 1,
    'Low-skill Laborers': 1,
    'Waiters/barmen staff': 1,
    'Cleaning staff': 1,
    'Private service staff': 1,
}


def clean_home_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Home Credit application_train.csv dataset.

    Cleaning steps (in order):
        1. Handle DAYS_EMPLOYED sentinel value (365243)
        2. Handle EXT_SOURCE missingness (create count + group imputation)
        3. Create gig worker proxy flag from OCCUPATION_TYPE
        4. Log-transform AMT_INCOME_TOTAL
        5. Treat outliers in financial columns (IQR capping)
        6. Fill remaining missing values (categorical → "UNKNOWN", numerical → median)

    Args:
        df: Raw Home Credit DataFrame from loader

    Returns:
        Cleaned DataFrame with new engineered cleaning features
    """
    df = df.copy()
    logger.info("Cleaning Home Credit dataset...")

    # ── Step 1: DAYS_EMPLOYED sentinel value ──
    # 365243 is NOT a real employment duration — it's Home Credit's encoding
    # for "unemployed / not applicable". If left as-is, it will dominate any
    # model as an extreme outlier (~1000 years employed).
    # Strategy: Replace with NaN, create binary flag, THEN impute.
    anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
    logger.info(f"  DAYS_EMPLOYED anomaly (365243): {anomaly_count:,} rows ({anomaly_count/len(df)*100:.1f}%)")

    df['is_unemployed'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan

    # ── Step 2: EXT_SOURCE handling ──
    # EXT_SOURCE_1/2/3 are normalized external credit bureau scores (0-1).
    # They are the MOST predictive features in the dataset BUT are DELIBERATELY
    # missing for many applicants — especially gig-worker-like profiles with no
    # formal credit history. The MISSINGNESS PATTERN ITSELF is a feature.
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']

    # Count how many external scores are available (0, 1, 2, or 3)
    # This captures "credit file thickness" — thin-file = fewer scores available
    df['ext_source_available_count'] = df[ext_cols].notna().sum(axis=1)
    logger.info(f"  EXT_SOURCE availability distribution:")
    logger.info(f"    {df['ext_source_available_count'].value_counts().sort_index().to_dict()}")

    # Impute missing EXT_SOURCE with median BY income type group.
    # Rationale: A driver with missing EXT_SOURCE_1 should be imputed with the
    # median of OTHER drivers (who share similar credit profiles), not the
    # global median which is dominated by salaried employees.
    for col in ext_cols:
        group_medians = df.groupby('NAME_INCOME_TYPE')[col].transform('median')
        df[col] = df[col].fillna(group_medians)
        # If entire group is NaN (rare), fall back to global median
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median if not np.isnan(global_median) else 0.5)

    # ── Step 3: Gig worker proxy flag ──
    # Map occupation types to binary gig-worker-like indicator
    df['is_gig_worker_proxy'] = df['OCCUPATION_TYPE'].map(GIG_OCCUPATION_MAP).fillna(0).astype(int)
    gig_count = df['is_gig_worker_proxy'].sum()
    logger.info(f"  Gig worker proxy flagged: {gig_count:,} rows ({gig_count/len(df)*100:.1f}%)")

    # ── Step 4: Income log transformation ──
    # AMT_INCOME_TOTAL has heavy right skew (few very high earners).
    # log1p(x) = log(1+x) handles the skew while preserving zeros.
    # Store both: raw for interpretability, log for modeling.
    df['income_raw'] = df['AMT_INCOME_TOTAL'].copy()
    df['income_log'] = np.log1p(df['AMT_INCOME_TOTAL'])

    # ── Step 5: Outlier treatment (IQR capping) ──
    # Only apply to continuous financial columns where extreme values are
    # likely data errors or at least should have bounded model influence.
    # Using 3x IQR (not 1.5x) to be conservative — credit data legitimately
    # has high-income individuals, we only want to cap true extremes.
    # DO NOT apply to count variables — they have natural discrete distributions.
    outlier_columns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
    for col in outlier_columns:
        if col in df.columns:
            df = _cap_outliers_iqr(df, col, multiplier=3.0)

    # ── Step 6: Fill remaining missing values ──
    # Categorical: "UNKNOWN" — never drop, unknowns carry information
    # (if occupation is unknown, it means informal/unrecorded employment)
    # Numerical: Median — robust to the skewed distributions common in credit data
    # IMPORTANT: Apply median imputation AFTER outlier treatment, not before,
    # so that extreme outliers don't pull the imputation values.
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('UNKNOWN')

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"  Cleaning complete. Final shape: {df.shape}")
    logger.info(f"  Remaining missing values: {df.isnull().sum().sum()}")

    return df


def clean_give_me_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Give Me Some Credit (cs-training.csv) dataset.

    Cleaning steps:
        1. MonthlyIncome: Create verifiable income flag + age-group median imputation
        2. RevolvingUtilization: Cap at 1.0 (values > 1 are data errors)
        3. Late payment sentinel values: Cap at 10 (96, 98 are sentinel codes)
        4. NumberOfDependents: Fill missing with 0 (reasonable assumption)
        5. DebtRatio: Cap extreme values

    Args:
        df: Raw Give Me Some Credit DataFrame from loader

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    logger.info("Cleaning Give Me Some Credit dataset...")

    # ── Step 1: MonthlyIncome ──
    # 19.8% missing — these are NOT random. They represent applicants WITHOUT
    # formal income documentation. In our context, these ARE the gig workers
    # (informal income, no salary slips). The MISSINGNESS is a feature.
    income_missing = df['monthly_income'].isnull().sum()
    logger.info(f"  MonthlyIncome missing: {income_missing:,} ({income_missing/len(df)*100:.1f}%)")

    # Binary flag: does this person have verifiable income?
    df['has_verifiable_income'] = df['monthly_income'].notna().astype(int)

    # Impute with median of same age group (±5 years).
    # Rationale: A 25-year-old with no income proof likely earns differently
    # than a 55-year-old with no income proof. Age-grouped imputation is
    # more realistic than a single global median.
    df['age_group'] = (df['age'] // 10) * 10  # Decade buckets: 20s, 30s, etc.
    age_group_medians = df.groupby('age_group')['monthly_income'].transform('median')
    df['monthly_income'] = df['monthly_income'].fillna(age_group_medians)
    # Fallback for any remaining NaN (edge cases in very sparse age groups)
    df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())
    df = df.drop(columns=['age_group'])  # Cleanup temp column

    # ── Step 2: RevolvingUtilization ──
    # Values > 1.0 mean "borrower owes more than their credit limit" which can
    # occur due to fees/interest but values like 50000+ are clearly data errors.
    # Cap at 1.0 for modeling stability.
    pre_cap = (df['revolving_utilization'] > 1.0).sum()
    df['revolving_utilization'] = df['revolving_utilization'].clip(upper=1.0)
    logger.info(f"  RevolvingUtilization capped (>1.0→1.0): {pre_cap:,} rows")

    # ── Step 3: Late payment sentinel values ──
    # Values 96 and 98 in late payment count columns are NOT real counts —
    # they are sentinel codes (likely meaning "unknown" or "many").
    # Cap at 10 to prevent these from dominating the model.
    late_cols = ['late_30_59_count', 'late_60_89_count', 'times_90_days_late']
    for col in late_cols:
        if col in df.columns:
            sentinel_count = (df[col] > 10).sum()
            df[col] = df[col].clip(upper=10)
            if sentinel_count > 0:
                logger.info(f"  {col} sentinel values capped (>10→10): {sentinel_count:,} rows")

    # ── Step 4: NumberOfDependents ──
    # Fill missing with 0. Assumption: if not reported, likely no dependents.
    # This is a reasonable default and documented as an assumption.
    dep_missing = df['number_of_dependents'].isnull().sum()
    df['number_of_dependents'] = df['number_of_dependents'].fillna(0).astype(int)
    logger.info(f"  NumberOfDependents missing filled with 0: {dep_missing:,} rows")

    # ── Step 5: DebtRatio ──
    # Some values are extremely high (>100), suggesting data quality issues.
    # Cap at 5.0 (500% debt-to-income is already extreme but plausible with
    # short-term debt vs monthly income framing).
    pre_cap = (df['debt_ratio'] > 5.0).sum()
    df['debt_ratio'] = df['debt_ratio'].clip(upper=5.0)
    if pre_cap > 0:
        logger.info(f"  DebtRatio capped (>5.0→5.0): {pre_cap:,} rows")

    logger.info(f"  Cleaning complete. Final shape: {df.shape}")

    return df


def _cap_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Caps outliers using the IQR method.

    Values below Q1 - multiplier*IQR are set to Q1 - multiplier*IQR.
    Values above Q3 + multiplier*IQR are set to Q3 + multiplier*IQR.

    Using 3x IQR (Tukey's outer fence) instead of 1.5x (inner fence) because
    financial data legitimately has high variance. We only want to cap the
    most extreme outliers, not genuine high earners.

    Args:
        df: DataFrame
        column: Column name to cap
        multiplier: IQR multiplier (default 3.0)

    Returns:
        DataFrame with capped values
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    capped_low = (df[column] < lower_bound).sum()
    capped_high = (df[column] > upper_bound).sum()

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    if capped_low + capped_high > 0:
        logger.info(
            f"  Outlier capping [{column}]: {capped_low} low, {capped_high} high "
            f"(bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}])"
        )

    return df
