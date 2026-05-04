"""
src/features/feature_pipeline.py — Scikit-learn Feature Pipeline

Builds a complete preprocessing pipeline using scikit-learn's Pipeline
and ColumnTransformer. This ensures:
    1. NO DATA LEAKAGE: Pipeline is fit ONLY on training data, then used
       to transform validation and test data.
    2. REPRODUCIBILITY: All preprocessing steps are encapsulated as objects
       that can be serialized (joblib) and reused for inference.
    3. CONSISTENCY: Same transformations applied during training and inference.

Pipeline architecture:
    Numerical features:  Impute (median) → Scale (StandardScaler)
    Categorical features: Impute (constant 'UNKNOWN') → OrdinalEncode
    Binary features:      Impute (constant 0) → Passthrough
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Feature Definitions
# Categorized by type for appropriate preprocessing.
# ─────────────────────────────────────────────────────────────────────

# Numerical features that need imputation + scaling
NUMERICAL_FEATURES = [
    # Traditional credit features
    'monthly_earnings_proxy',
    'credit_income_ratio',
    'annuity_income_ratio',
    'credit_term_months',
    'income_per_person',
    'ext_source_mean',
    'ext_source_std',
    'days_employed_percent',
    'documents_provided_count',
    'enquiry_score_7d',
    'external_credit_score',
    'income_log',

    # Gig-specific features
    'income_stability_score',
    'platform_reliability_score',
    'income_experience_ratio',
    'financial_resilience_score',
    'work_intensity_score',
    'payment_behavior_score',
    'payment_on_time_ratio',
    'avg_days_late',
    'payment_amount_ratio',
    'payment_streak_max',
    'payment_regularity_cv',
    'income_momentum',
    'income_diversification_score',
    'debt_stress_score',
    'earnings_efficiency',

    # India-specific features
    'digital_engagement_score',
    'settlement_stability_score',
    'lifestyle_risk_score',
    'multi_platform_bonus',

    # Raw synthetic features (available for synthetic data)
    'monthly_earnings_inr',
    'income_volatility_coefficient',
    'income_trend_6m',
    'platform_tenure_months',
    'platform_rating',
    'weekly_work_consistency',
    'cancellation_rate',
    'upi_transaction_count_monthly',
    'upi_transaction_consistency_score',
    'digital_wallet_balance_avg',
    'rent_to_income_ratio',
    'late_night_work_ratio',
    'years_in_city',
    'age',

    # From Give Me Some Credit
    'credit_utilization_ratio',
    'debt_ratio',
    'late_30_59_count',
    'late_60_89_count',
    'times_90_days_late',
    'open_credit_lines',
    'real_estate_loans',
]

# Categorical features that need imputation + encoding
CATEGORICAL_FEATURES = [
    'vehicle_owned',
    'data_source',
]

# Binary features that just need 0-fill for missing values
BINARY_FEATURES = [
    'has_savings_account',
    'has_life_insurance',
    'owns_smartphone_above_10k',
    'has_vehicle',
    'has_realty',
    'is_unemployed',
    'is_gig_worker_proxy',
    'has_verifiable_income',
    'has_overdue_history',
    'ext_source_available_count',
    'active_platforms_count',
    'dependents_count',
    'mobile_recharge_frequency',
    'monthly_trips_or_orders',
    'income_gap_months_last_year',
    'region_risk_rating',
    'total_past_credit_count',
    'credit_type_diversity',
    'active_credits_count',
]


def get_available_features(df: pd.DataFrame) -> tuple:
    """
    Determines which features are actually available in the DataFrame.
    Returns only features that exist in the data (handles dataset-specific columns).

    Args:
        df: DataFrame to check

    Returns:
        Tuple of (numerical_cols, categorical_cols, binary_cols)
    """
    available_cols = set(df.columns)

    numerical = [c for c in NUMERICAL_FEATURES if c in available_cols]
    categorical = [c for c in CATEGORICAL_FEATURES if c in available_cols]
    binary = [c for c in BINARY_FEATURES if c in available_cols]

    return numerical, categorical, binary


def build_feature_pipeline(
    numerical_features: list,
    categorical_features: list,
    binary_features: list,
) -> Pipeline:
    """
    Builds a complete scikit-learn preprocessing pipeline.

    Architecture:
        Numerical:    SimpleImputer(median) → StandardScaler
        Categorical:  SimpleImputer(constant='UNKNOWN') → OrdinalEncoder
        Binary:       SimpleImputer(constant=0) → Passthrough

    CRITICAL: This pipeline must be FIT ONLY ON TRAINING DATA.
    Fitting on full data before split causes data leakage —
    the scaler would learn the mean/std of the test set, and the
    imputer would use test-set medians, artificially inflating metrics.

    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        binary_features: List of binary column names

    Returns:
        sklearn Pipeline ready for fit/transform
    """
    logger.info("Building feature pipeline...")
    logger.info(f"  Numerical features: {len(numerical_features)}")
    logger.info(f"  Categorical features: {len(categorical_features)}")
    logger.info(f"  Binary features: {len(binary_features)}")
    logger.info(f"  Total features: {len(numerical_features) + len(categorical_features) + len(binary_features)}")

    # Numerical pipeline: Impute missing → Scale to zero mean, unit variance
    # Using median for imputation because financial data is heavily skewed —
    # mean would be pulled by outliers.
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # Categorical pipeline: Fill missing with 'UNKNOWN' → Encode to integers
    # OrdinalEncoder is used instead of OneHotEncoder because:
    #   1. Tree-based models (LightGBM, XGBoost) handle ordinal encoding natively
    #   2. OneHot would create too many sparse columns for high-cardinality features
    #   3. LightGBM's categorical handling is more effective with ordinal encoding
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    # Binary pipeline: Fill missing with 0 → No transformation needed
    # Binary features are already 0/1 and don't need scaling
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, numerical_features),
            ('categorical', categorical_pipeline, categorical_features),
            ('binary', binary_pipeline, binary_features),
        ],
        remainder='drop',  # Drop any columns not in our feature lists
        verbose_feature_names_out=False,
    )

    # Wrap in a Pipeline for clean API
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
    ])

    return pipeline


def get_feature_names(pipeline: Pipeline,
                      numerical_features: list,
                      categorical_features: list,
                      binary_features: list) -> list:
    """
    Returns the full list of feature names after pipeline transformation.
    Maintains the same order as the pipeline output.

    Args:
        pipeline: Fitted pipeline
        numerical_features: Original numerical column names
        categorical_features: Original categorical column names
        binary_features: Original binary column names

    Returns:
        List of feature names in pipeline output order
    """
    return numerical_features + categorical_features + binary_features


def print_feature_summary(df: pd.DataFrame,
                          numerical_features: list,
                          categorical_features: list,
                          binary_features: list) -> pd.DataFrame:
    """
    Prints a formatted feature summary table showing:
        feature_name | source | type | missingness_% | notes

    Args:
        df: DataFrame with all features
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        binary_features: List of binary feature names

    Returns:
        Summary DataFrame
    """
    all_features = numerical_features + categorical_features + binary_features
    summary_rows = []

    for feat in all_features:
        if feat not in df.columns:
            continue

        # Determine source
        if feat in ['credit_income_ratio', 'annuity_income_ratio', 'credit_term_months',
                     'income_per_person', 'ext_source_mean', 'ext_source_std',
                     'days_employed_percent', 'documents_provided_count',
                     'enquiry_score_7d', 'external_credit_score']:
            source = 'Home Credit (engineered)'
        elif feat in ['income_stability_score', 'platform_reliability_score',
                       'financial_resilience_score', 'work_intensity_score',
                       'income_experience_ratio', 'payment_behavior_score']:
            source = 'Gig (engineered)'
        elif feat in ['digital_engagement_score', 'settlement_stability_score',
                       'lifestyle_risk_score', 'multi_platform_bonus']:
            source = 'India (engineered)'
        elif feat in ['credit_utilization_ratio', 'debt_ratio', 'late_30_59_count']:
            source = 'Give Me Credit'
        elif feat in ['monthly_earnings_inr', 'platform_tenure_months', 'platform_rating',
                       'upi_transaction_count_monthly']:
            source = 'Synthetic'
        else:
            source = 'Mixed/Multiple'

        # Feature type
        if feat in numerical_features:
            ftype = 'numerical'
        elif feat in categorical_features:
            ftype = 'categorical'
        else:
            ftype = 'binary'

        # Missingness
        missing_pct = df[feat].isnull().mean() * 100

        summary_rows.append({
            'feature_name': feat,
            'source': source,
            'type': ftype,
            'missingness_pct': round(missing_pct, 1),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Print formatted table
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE SUMMARY TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Feature':<35} {'Source':<22} {'Type':<12} {'Missing%':>8}")
    logger.info("-" * 80)
    for _, row in summary_df.iterrows():
        logger.info(
            f"{row['feature_name']:<35} {row['source']:<22} "
            f"{row['type']:<12} {row['missingness_pct']:>7.1f}%"
        )
    logger.info("=" * 80)
    logger.info(f"Total features: {len(summary_df)}")

    return summary_df
