"""
config.py — Centralized Configuration for GigScore

All constants, file paths, feature explanations, credit limit bands,
and UI settings live here. Import from this file instead of hardcoding
values throughout the codebase.
"""

import os

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "xgboost": os.path.join(ROOT, "artifacts", "xgboost_model.pkl"),
    "lightgbm": os.path.join(ROOT, "artifacts", "lightgbm_model.pkl"),
    "ensemble": os.path.join(ROOT, "artifacts", "ensemble_model.pkl"),
    "baseline": os.path.join(ROOT, "artifacts", "baseline_model.pkl"),
}
FEATURE_PIPELINE_PATH = os.path.join(ROOT, "artifacts", "feature_pipeline.pkl")
SHAP_EXPLAINER_PATH = os.path.join(ROOT, "artifacts", "shap_explainer.pkl")
MODEL_METADATA_PATH = os.path.join(ROOT, "artifacts", "model_metadata.json")
FEATURE_IMPORTANCE_PATH = os.path.join(ROOT, "artifacts", "feature_importance.csv")

DATA_PATH = os.path.join(ROOT, "data", "processed", "final_training_data.csv")
FAIRNESS_REPORT_PATH = os.path.join(ROOT, "data", "reports", "fairness_report.json")

# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────
SCORE_SCALE = 100
DEFAULT_MODEL = "ensemble"

CREDIT_LIMIT_BANDS = [
    {"min": 0,  "max": 39,  "decision": "Rejected",    "max_limit": 0,       "label": "Very Poor",  "emoji": "⛔", "color": "#7f1d1d"},
    {"min": 40, "max": 54,  "decision": "Conditional",  "max_limit": 25000,   "label": "Poor",       "emoji": "🔴", "color": "#ef4444"},
    {"min": 55, "max": 69,  "decision": "Approved",     "max_limit": 75000,   "label": "Fair",       "emoji": "🟠", "color": "#f59e0b"},
    {"min": 70, "max": 84,  "decision": "Approved",     "max_limit": 150000,  "label": "Good",       "emoji": "🟡", "color": "#84cc16"},
    {"min": 85, "max": 100, "decision": "Approved",     "max_limit": 300000,  "label": "Excellent",  "emoji": "🟢", "color": "#22c55e"},
]

# ─────────────────────────────────────────────────────────────────────
# Feature Explanations — used in "Why This Score?" section
# Maps feature names to human-readable explanations
# ─────────────────────────────────────────────────────────────────────
FEATURE_EXPLANATIONS = {
    # Income & Stability
    "income_stability_score": "Consistent monthly income reduces repayment risk",
    "monthly_earnings_inr": "Monthly earning capacity relative to credit requested",
    "monthly_earnings_proxy": "Monthly earnings demonstrate earning capacity",
    "income_volatility_coefficient": "High income swings increase default probability",
    "income_gap_months_last_year": "Months with no earnings indicate earning instability",
    "income_trend_6m": "6-month income trajectory shows improving or declining capacity",
    "income_log": "Log-transformed income captures diminishing returns of higher income",

    # Platform Behavior
    "platform_tenure_months": "Long platform tenure demonstrates commitment and reliability",
    "platform_rating": "High customer ratings reflect service quality",
    "weekly_work_consistency": "Regular work patterns correlate with stable income",
    "cancellation_rate": "High cancellation rates signal platform instability",
    "monthly_trips_or_orders": "Work volume demonstrates active platform engagement",
    "active_platforms_count": "Multiple platforms diversify income sources",

    # Digital Finance
    "upi_transaction_count_monthly": "Regular digital transactions indicate financial activity",
    "upi_transaction_consistency_score": "Steady UPI patterns demonstrate financial stability",
    "digital_wallet_balance_avg": "Digital wallet usage indicates financial engagement",
    "mobile_recharge_frequency": "Regular recharge shows consistent digital behaviour",

    # Financial Health
    "has_savings_account": "Savings account signals financial planning behaviour",
    "has_life_insurance": "Life insurance indicates long-term financial planning",
    "rent_to_income_ratio": "High rent burden reduces repayment capacity",
    "owns_smartphone_above_10k": "Smartphone ownership indicates digital access",
    "dependents_count": "More dependents increase financial obligations",
    "years_in_city": "Longer city residency indicates settled lifestyle",
    "age": "Age contributes to experience and earning stability",

    # Engineered Composites
    "income_stability_score": "Composite of income volatility and gap months",
    "platform_reliability_score": "Weighted combination of rating, consistency, and cancellations",
    "income_experience_ratio": "Earnings relative to platform experience",
    "financial_resilience_score": "Savings + low rent + insurance composite buffer",
    "work_intensity_score": "Quality-adjusted work volume metric",
    "income_momentum": "Income trend weighted by stability — genuine vs noisy growth",
    "income_diversification_score": "Multi-platform income diversification",
    "debt_stress_score": "Debt burden multiplied by income instability",
    "earnings_efficiency": "Income per trip/order — skilled worker proxy",
    "digital_engagement_score": "UPI volume and consistency composite",
    "settlement_stability_score": "City residency and recharge pattern composite",
    "lifestyle_risk_score": "Late-night work, dependents, and asset risk factors",
    "multi_platform_bonus": "Diversification bonus from multiple platforms",

    # Traditional Credit
    "credit_income_ratio": "Loan amount relative to income — overextension risk",
    "annuity_income_ratio": "Monthly repayment as percentage of income",
    "ext_source_mean": "Average external credit bureau scores",
    "payment_on_time_ratio": "Historical on-time payment percentage",
    "payment_behavior_score": "Composite of payment discipline signals",

    # Other
    "vehicle_owned": "Vehicle ownership indicates asset base",
    "data_source": "Data origin (synthetic/kaggle)",
    "region_risk_rating": "Regional economic risk classification",
    "late_night_work_ratio": "Proportion of late-night work shifts",
}

# ─────────────────────────────────────────────────────────────────────
# AI Assistant
# ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_CALL_DELAY = 4  # seconds between calls (15 RPM free tier)

# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────
APP_TITLE = "GigScore — Alternative Credit Scoring"
APP_ICON = "🏦"
THEME = "dark"
