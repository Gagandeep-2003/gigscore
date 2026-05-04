"""
src/scoring/score_calculator.py — GigScore Calculation Engine

Converts the model's raw default probability (0.0 to 1.0) into a
GigScore (0 to 100) with credit band, loan limit recommendation,
and decision outcome.

Score architecture:
    GigScore = round((1 - default_probability) * 100)

    The score is INVERSELY related to default probability:
        - Low default prob → High GigScore → Good credit
        - High default prob → Low GigScore → Poor credit

Banding system designed to map to actionable lending decisions:
    85-100: EXCELLENT — Approve up to ₹3,00,000
    70-84:  GOOD — Approve up to ₹1,50,000
    55-69:  FAIR — Approve up to ₹75,000
    40-54:  POOR — Approve with guarantor up to ₹25,000
    0-39:   VERY POOR — Decline or require 6-month proof
"""

from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Score Bands Configuration
# ─────────────────────────────────────────────────────────────────────
SCORE_BANDS = {
    (85, 100): {
        'label': 'EXCELLENT',
        'color': '#22c55e',
        'emoji': '🟢',
        'max_credit_limit': 300000,
        'income_multiple': 6.0,
        'decision': 'APPROVED',
        'decision_detail': 'Full approval with standard terms',
    },
    (70, 84): {
        'label': 'GOOD',
        'color': '#84cc16',
        'emoji': '🟡',
        'max_credit_limit': 150000,
        'income_multiple': 3.0,
        'decision': 'APPROVED',
        'decision_detail': 'Approved with standard monitoring',
    },
    (55, 69): {
        'label': 'FAIR',
        'color': '#f59e0b',
        'emoji': '🟠',
        'max_credit_limit': 75000,
        'income_multiple': 1.5,
        'decision': 'CONDITIONAL',
        'decision_detail': 'Approved with enhanced monitoring and lower limit',
    },
    (40, 54): {
        'label': 'POOR',
        'color': '#ef4444',
        'emoji': '🔴',
        'max_credit_limit': 25000,
        'income_multiple': 0.5,
        'decision': 'CONDITIONAL',
        'decision_detail': 'Requires guarantor or collateral. Maximum ₹25,000',
    },
    (0, 39): {
        'label': 'VERY POOR',
        'color': '#7f1d1d',
        'emoji': '⛔',
        'max_credit_limit': 0,
        'income_multiple': 0.0,
        'decision': 'DECLINED',
        'decision_detail': 'Declined. Recommend 6-month income proof before reapplication',
    },
}


def probability_to_gigscore(default_probability: float) -> int:
    """
    Converts model's default probability to GigScore (0-100).

    The mapping is simple and linear:
        GigScore = round((1 - default_probability) * 100)

    A linear mapping is chosen for transparency — it's easy to explain
    to a customer that "your score is 100 minus your default risk percent."

    Args:
        default_probability: Model's predicted probability of default (0.0 to 1.0)

    Returns:
        Integer GigScore from 0 to 100
    """
    # Clamp probability to valid range
    prob = max(0.0, min(1.0, default_probability))
    score = round((1 - prob) * 100)
    return max(0, min(100, score))


def get_score_band(score: int) -> dict:
    """
    Returns the complete band information for a given GigScore.

    Args:
        score: GigScore integer (0-100)

    Returns:
        Dictionary with band_label, color_hex, emoji, max_credit_limit,
        income_multiple, decision, decision_detail
    """
    score = max(0, min(100, score))

    for (low, high), band_info in SCORE_BANDS.items():
        if low <= score <= high:
            return {
                'score': score,
                **band_info,
            }

    # Fallback (should never reach here)
    return SCORE_BANDS[(0, 39)].copy()


def calculate_credit_limit(
    score: int,
    monthly_income: float = None,
) -> int:
    """
    Calculates recommended credit limit based on GigScore and income.

    Formula:
        credit_limit = min(income_multiple × monthly_income, max_credit_limit)

    The credit limit is the LOWER of:
        - The income-based calculation (responsible lending)
        - The band cap (risk-based ceiling)

    Args:
        score: GigScore (0-100)
        monthly_income: Monthly income in INR (optional)

    Returns:
        Recommended credit limit in INR
    """
    band = get_score_band(score)

    if monthly_income and monthly_income > 0:
        income_based_limit = int(band['income_multiple'] * monthly_income)
        return min(income_based_limit, band['max_credit_limit'])

    return band['max_credit_limit']


def get_full_score_result(
    default_probability: float,
    monthly_income: float = None,
) -> dict:
    """
    Generates the complete scoring result for an applicant.

    This is the primary function called by the API endpoint.
    Returns everything needed for the dashboard display.

    Args:
        default_probability: Model's predicted default probability
        monthly_income: Monthly income in INR (optional, for credit limit)

    Returns:
        Complete scoring result dictionary
    """
    score = probability_to_gigscore(default_probability)
    band = get_score_band(score)
    credit_limit = calculate_credit_limit(score, monthly_income)

    result = {
        'gigscore': score,
        'default_probability': round(default_probability, 4),
        'score_band': band['label'],
        'score_color': band['color'],
        'score_emoji': band['emoji'],
        'credit_limit_recommendation': credit_limit,
        'loan_decision': band['decision'],
        'decision_detail': band['decision_detail'],
        'disclaimer': (
            'This score is indicative only and does not constitute a credit approval. '
            'Final lending decisions are subject to additional verification by the '
            'lending institution. GigScore is designed as an alternative assessment '
            'tool and should be used in conjunction with other evaluation criteria.'
        ),
    }

    return result
