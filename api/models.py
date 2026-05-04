"""
api/models.py — Pydantic Request/Response Schemas

Defines the data contracts for the GigScore API.
Pydantic provides:
    - Automatic validation (type checking, range enforcement)
    - Auto-generated Swagger documentation
    - Serialization/deserialization
    - Clear error messages for invalid inputs
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class GigWorkerInput(BaseModel):
    """
    Request schema for the /predict endpoint.
    Represents a single gig worker applicant's profile.

    All fields have validation constraints to catch invalid inputs
    before they reach the model — garbage in, garbage out.
    """
    # Income features
    monthly_earnings_inr: float = Field(
        ..., ge=0, le=1000000,
        description="Monthly earnings in INR"
    )
    income_volatility_coefficient: float = Field(
        ..., ge=0, le=1,
        description="Income volatility (0=stable, 1=very volatile)"
    )
    income_gap_months_last_year: int = Field(
        ..., ge=0, le=12,
        description="Number of months with zero/very low earnings in past year"
    )
    income_trend_6m: float = Field(
        default=0.0, ge=-1, le=1,
        description="6-month income trend (positive=growing, negative=declining)"
    )

    # Platform features
    platform_tenure_months: int = Field(
        ..., ge=1, le=120,
        description="Months active on gig platform(s)"
    )
    platform_rating: float = Field(
        ..., ge=1.0, le=5.0,
        description="Customer/client rating on platform (1-5)"
    )
    active_platforms_count: int = Field(
        default=1, ge=1, le=10,
        description="Number of gig platforms the worker is active on"
    )
    weekly_work_consistency: float = Field(
        ..., ge=0, le=1,
        description="Ratio of weeks with regular activity (0-1)"
    )
    cancellation_rate: float = Field(
        ..., ge=0, le=1,
        description="Job/order cancellation rate (0=no cancellations, 1=always cancels)"
    )
    monthly_trips_or_orders: int = Field(
        default=100, ge=0, le=1000,
        description="Monthly trips or orders completed"
    )

    # India-specific features
    upi_transaction_count_monthly: int = Field(
        ..., ge=0, le=500,
        description="Number of UPI transactions per month"
    )
    upi_transaction_consistency_score: float = Field(
        ..., ge=0, le=1,
        description="Consistency of monthly UPI activity (0-1)"
    )
    has_savings_account: bool = Field(
        ..., description="Whether the applicant has a savings bank account"
    )
    rent_to_income_ratio: float = Field(
        ..., ge=0, le=1,
        description="Monthly rent as fraction of monthly income"
    )
    years_in_city: float = Field(
        ..., ge=0, le=50,
        description="Years of residency in current city"
    )
    dependents_count: int = Field(
        ..., ge=0, le=10,
        description="Number of financial dependents"
    )

    # Optional features
    has_life_insurance: bool = Field(
        default=False, description="Has life insurance policy"
    )
    owns_smartphone_above_10k: bool = Field(
        default=True, description="Owns smartphone valued above ₹10,000"
    )
    vehicle_owned: str = Field(
        default='two_wheeler',
        description="Vehicle type: none, bicycle, two_wheeler, or car"
    )
    mobile_recharge_frequency: int = Field(
        default=2, ge=1, le=10,
        description="Times per month prepaid mobile is recharged"
    )
    late_night_work_ratio: float = Field(
        default=0.1, ge=0, le=1,
        description="Fraction of work done between 10pm-5am"
    )
    digital_wallet_balance_avg: float = Field(
        default=2000, ge=0,
        description="Average digital wallet balance in INR"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "monthly_earnings_inr": 32000,
                "income_volatility_coefficient": 0.35,
                "income_gap_months_last_year": 1,
                "income_trend_6m": 0.08,
                "platform_tenure_months": 24,
                "platform_rating": 4.3,
                "active_platforms_count": 2,
                "weekly_work_consistency": 0.78,
                "cancellation_rate": 0.06,
                "monthly_trips_or_orders": 180,
                "upi_transaction_count_monthly": 38,
                "upi_transaction_consistency_score": 0.82,
                "has_savings_account": True,
                "rent_to_income_ratio": 0.28,
                "years_in_city": 4.5,
                "dependents_count": 2,
                "has_life_insurance": False,
                "owns_smartphone_above_10k": True,
                "vehicle_owned": "two_wheeler",
                "mobile_recharge_frequency": 2,
                "late_night_work_ratio": 0.1,
                "digital_wallet_balance_avg": 2500,
            }
        }


class GigScorePrediction(BaseModel):
    """Response schema for the /predict endpoint."""
    gigscore: int = Field(..., ge=0, le=100, description="GigScore (0-100)")
    default_probability: float = Field(..., description="Raw model default probability")
    score_band: str = Field(..., description="Score band: EXCELLENT/GOOD/FAIR/POOR/VERY POOR")
    score_color: str = Field(..., description="Hex color code for the score band")
    credit_limit_recommendation: int = Field(..., description="Recommended credit limit in INR")
    loan_decision: str = Field(..., description="APPROVED/CONDITIONAL/DECLINED")
    decision_detail: str = Field(..., description="Detailed decision explanation")

    top_positive_factors: list = Field(
        default=[], description="Top 3 factors helping the score"
    )
    top_negative_factors: list = Field(
        default=[], description="Top 3 factors hurting the score"
    )
    improvement_tips: list = Field(
        default=[], description="Actionable suggestions to improve score"
    )

    model_version: str = Field(default="1.0.0", description="Model version")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    disclaimer: str = Field(..., description="Legal disclaimer")


class BatchPredictionRequest(BaseModel):
    """Request schema for /batch-predict endpoint."""
    applicants: List[GigWorkerInput] = Field(
        ..., min_length=1, max_length=100,
        description="List of applicant profiles (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for /batch-predict endpoint."""
    predictions: List[GigScorePrediction]
    total_processed: int
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    """Response schema for /model-info endpoint."""
    model_name: str
    model_version: str
    training_date: str
    auc_roc: float
    gini_coefficient: float
    ks_statistic: float
    optimal_threshold: float
    total_features: int
    training_samples: int
    default_rate: float


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    model_loaded: bool
    model_version: str
    feature_count: int
    uptime_seconds: float
