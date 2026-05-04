"""
src/data/loader.py — Data Loading & Validation Module

Loads and validates all raw datasets for the GigScore pipeline.
Performs initial type enforcement and schema validation.
Never modifies data — only validates and loads.
Raises clear errors if files are missing or schema doesn't match.

Datasets supported:
    1. Home Credit Default Risk (application_train.csv, installments_payments.csv, bureau.csv)
    2. Give Me Some Credit (cs-training.csv)
"""

import os
import pandas as pd
import numpy as np
from loguru import logger


def validate_dataset(df: pd.DataFrame, required_columns: list, name: str) -> None:
    """
    Validates that a DataFrame contains all required columns.
    Logs dataset shape and missing value summary.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        name: Human-readable dataset name for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset '{name}' is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns[:20])}..."
        )

    # Log dataset summary
    logger.info(f"✅ Dataset '{name}' loaded successfully")
    logger.info(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Missing value summary
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    cols_with_missing = missing_pct[missing_pct > 0]
    if len(cols_with_missing) > 0:
        logger.info(f"   Columns with missing values: {len(cols_with_missing)}")
        for col in cols_with_missing.head(5).index:
            logger.info(f"     - {col}: {cols_with_missing[col]:.1f}% missing")
        if len(cols_with_missing) > 5:
            logger.info(f"     ... and {len(cols_with_missing) - 5} more")
    else:
        logger.info("   No missing values found")


def load_home_credit_application(path: str) -> pd.DataFrame:
    """
    Loads the Home Credit application_train.csv dataset.

    This is the PRIMARY dataset for GigScore. Each row represents one loan
    application with demographic info, income, employment type, credit amount,
    and TARGET (0=repaid, 1=defaulted).

    Args:
        path: Path to application_train.csv

    Returns:
        DataFrame with dtypes enforced

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing or TARGET is not binary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Home Credit application file not found at: {path}\n"
            f"Download from: https://www.kaggle.com/c/home-credit-default-risk/data\n"
            f"Place application_train.csv in data/raw/"
        )

    logger.info(f"Loading Home Credit application data from: {path}")
    df = pd.read_csv(path)

    # Required columns for GigScore pipeline
    required_cols = [
        'SK_ID_CURR', 'TARGET', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'DAYS_EMPLOYED', 'DAYS_BIRTH',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'REGION_RATING_CLIENT', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER'
    ]
    validate_dataset(df, required_cols, "Home Credit Application")

    # Validate TARGET is binary (0 or 1)
    unique_targets = df['TARGET'].dropna().unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError(
            f"TARGET column must be binary (0/1). Found values: {unique_targets}"
        )
    logger.info(f"   Target distribution: {df['TARGET'].value_counts().to_dict()}")
    logger.info(f"   Default rate: {df['TARGET'].mean():.2%}")

    # Enforce key dtypes
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    df['TARGET'] = df['TARGET'].astype(int)
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].astype(float)
    df['AMT_CREDIT'] = df['AMT_CREDIT'].astype(float)

    return df


def load_installments(path: str) -> pd.DataFrame:
    """
    Loads the installments_payments.csv dataset from Home Credit.

    Contains payment history for previous loans. Each row is one payment.
    Key columns: SK_ID_CURR (for joining), DAYS_INSTALMENT (when due),
    DAYS_ENTRY_PAYMENT (when paid), AMT_INSTALMENT (amount due),
    AMT_PAYMENT (amount actually paid).

    Used to engineer behavioral payment features:
        - payment_on_time_ratio
        - avg_days_late
        - payment_consistency_score

    Args:
        path: Path to installments_payments.csv

    Returns:
        DataFrame with payment records
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Installments file not found at: {path}\n"
            f"Download from: https://www.kaggle.com/c/home-credit-default-risk/data\n"
            f"Place installments_payments.csv in data/raw/"
        )

    logger.info(f"Loading installments data from: {path}")
    df = pd.read_csv(path)

    required_cols = [
        'SK_ID_PREV', 'SK_ID_CURR', 'NUM_INSTALMENT_VERSION',
        'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT',
        'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'
    ]
    validate_dataset(df, required_cols, "Installments Payments")

    # Enforce dtypes
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

    return df


def load_bureau(path: str) -> pd.DataFrame:
    """
    Loads the bureau.csv dataset from Home Credit.

    Contains information about client's previous credits at other institutions.
    Each row is one credit bureau record.

    Used to engineer:
        - total_past_credit_count
        - has_overdue_history
        - credit_type_diversity

    Args:
        path: Path to bureau.csv

    Returns:
        DataFrame with bureau records
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bureau file not found at: {path}\n"
            f"Download from: https://www.kaggle.com/c/home-credit-default-risk/data\n"
            f"Place bureau.csv in data/raw/"
        )

    logger.info(f"Loading bureau data from: {path}")
    df = pd.read_csv(path)

    required_cols = [
        'SK_ID_CURR', 'SK_ID_BUREAU', 'CREDIT_ACTIVE',
        'DAYS_CREDIT', 'CREDIT_DAY_OVERDUE',
        'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
        'AMT_CREDIT_SUM_OVERDUE', 'CREDIT_TYPE'
    ]
    validate_dataset(df, required_cols, "Bureau")

    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

    return df


def load_give_me_some_credit(path: str) -> pd.DataFrame:
    """
    Loads the Give Me Some Credit (cs-training.csv) dataset.

    This is the SECONDARY/VALIDATION dataset. Each row is one credit applicant.
    The missing MonthlyIncome field (~20% missing) is strategically important —
    it represents our gig worker target population (no formal income proof).

    Args:
        path: Path to cs-training.csv

    Returns:
        DataFrame with columns renamed to snake_case

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Give Me Some Credit file not found at: {path}\n"
            f"Download from: https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            f"Place cs-training.csv in data/raw/"
        )

    logger.info(f"Loading Give Me Some Credit data from: {path}")
    df = pd.read_csv(path)

    # The first column is often an unnamed index — drop it
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Rename columns to snake_case for consistency
    rename_map = {
        'SeriousDlqin2yrs': 'serious_dlqin_2yrs',
        'RevolvingUtilizationOfUnsecuredLines': 'revolving_utilization',
        'age': 'age',
        'NumberOfTime30-59DaysPastDueNotWorse': 'late_30_59_count',
        'DebtRatio': 'debt_ratio',
        'MonthlyIncome': 'monthly_income',
        'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines',
        'NumberOfTimes90DaysLate': 'times_90_days_late',
        'NumberRealEstateLoansOrLines': 'real_estate_loans',
        'NumberOfTime60-89DaysPastDueNotWorse': 'late_60_89_count',
        'NumberOfDependents': 'number_of_dependents'
    }
    df = df.rename(columns=rename_map)

    required_cols = ['serious_dlqin_2yrs', 'monthly_income', 'age']
    validate_dataset(df, required_cols, "Give Me Some Credit")

    # Log the critical missing income statistic
    income_missing = df['monthly_income'].isnull().sum()
    income_missing_pct = income_missing / len(df) * 100
    logger.info(
        f"   ⚠️  MonthlyIncome missing: {income_missing:,} rows ({income_missing_pct:.1f}%) "
        f"— these represent 'thin-file' applicants (proxy for gig workers)"
    )

    return df


def load_all_datasets(data_dir: str) -> dict:
    """
    Convenience function to load all datasets at once.

    Args:
        data_dir: Path to the data/raw/ directory

    Returns:
        Dictionary with keys:
            'application': Home Credit application DataFrame
            'installments': Installments payments DataFrame
            'bureau': Bureau DataFrame
            'give_me_credit': Give Me Some Credit DataFrame
    """
    logger.info("=" * 60)
    logger.info("LOADING ALL DATASETS")
    logger.info("=" * 60)

    datasets = {}

    datasets['application'] = load_home_credit_application(
        os.path.join(data_dir, 'application_train.csv')
    )
    datasets['installments'] = load_installments(
        os.path.join(data_dir, 'installments_payments.csv')
    )
    datasets['bureau'] = load_bureau(
        os.path.join(data_dir, 'bureau.csv')
    )
    datasets['give_me_credit'] = load_give_me_some_credit(
        os.path.join(data_dir, 'cs-training.csv')
    )

    logger.info("=" * 60)
    logger.info("ALL DATASETS LOADED SUCCESSFULLY")
    logger.info("=" * 60)

    return datasets
