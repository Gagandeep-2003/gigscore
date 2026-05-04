"""
src/explainability/reason_codes.py — Plain-English Reason Code Mapper

Maps feature names to human-readable explanations for why a particular
feature helped or hurt an applicant's GigScore.

Every credit decision must be explainable — both for regulatory compliance
(RBI fair lending guidelines) and for customer trust. A score without
explanation is just a black box.

Reason codes follow the pattern:
    - Positive: "Your [behavior] demonstrates [positive attribute]"
    - Negative: "Your [behavior] indicates [risk factor]"
"""

from loguru import logger


# ─────────────────────────────────────────────────────────────────────
# Reason Code Map
# Each feature maps to positive and negative plain-English explanations.
# These are shown to the applicant in the dashboard under
# "Why This Score?" section.
# ─────────────────────────────────────────────────────────────────────

REASON_CODE_MAP = {
    # ── Income & Stability Features ──
    'income_stability_score': {
        'positive': 'Consistent monthly earnings demonstrate reliable repayment capacity',
        'negative': 'Significant income volatility increases repayment uncertainty',
    },
    'monthly_earnings_proxy': {
        'positive': 'Monthly earnings are sufficient to support the requested credit',
        'negative': 'Monthly earnings may be insufficient for the requested credit amount',
    },
    'monthly_earnings_inr': {
        'positive': 'Your monthly earnings demonstrate strong earning capacity',
        'negative': 'Current monthly earnings suggest limited repayment buffer',
    },
    'income_volatility_coefficient': {
        'positive': 'Low income variability indicates predictable earning patterns',
        'negative': 'High income variability makes repayment reliability uncertain',
    },
    'income_gap_months_last_year': {
        'positive': 'Minimal income gaps in the past year indicate continuous earning',
        'negative': 'Multiple income gap months suggest earning instability',
    },
    'income_trend_6m': {
        'positive': 'Upward income trend over the past 6 months shows growing capacity',
        'negative': 'Declining income trend raises concerns about future repayment ability',
    },

    # ── Platform Behavior Features ──
    'platform_reliability_score': {
        'positive': 'Strong platform ratings and work consistency show professional reliability',
        'negative': 'Low platform engagement or high cancellation rate signals instability',
    },
    'platform_tenure_months': {
        'positive': 'Long platform tenure demonstrates commitment and earning stability',
        'negative': 'Short platform history limits assessment of income reliability',
    },
    'platform_rating': {
        'positive': 'High customer/client ratings reflect service quality and professionalism',
        'negative': 'Low platform ratings may indicate service quality concerns',
    },
    'weekly_work_consistency': {
        'positive': 'Regular weekly work patterns show disciplined earning behavior',
        'negative': 'Irregular work patterns suggest inconsistent earning commitment',
    },
    'cancellation_rate': {
        'positive': 'Low cancellation rate indicates reliable and committed work behavior',
        'negative': 'High cancellation rate signals potential workplace or reliability issues',
    },
    'work_intensity_score': {
        'positive': 'High work volume demonstrates active platform engagement',
        'negative': 'Low work volume limits earning potential assessment',
    },

    # ── Payment Behavior Features ──
    'payment_on_time_ratio': {
        'positive': 'Consistent on-time payments on previous obligations',
        'negative': 'History of delayed payments indicates repayment difficulty',
    },
    'payment_behavior_score': {
        'positive': 'Strong historical payment discipline across previous loans',
        'negative': 'Past payment behavior indicates potential repayment challenges',
    },
    'avg_days_late': {
        'positive': 'Historically timely on payments with minimal delays',
        'negative': 'Average payment delays suggest cash flow management challenges',
    },
    'payment_amount_ratio': {
        'positive': 'Consistently meeting or exceeding required payment amounts',
        'negative': 'Tendency to underpay on installments signals financial strain',
    },
    'payment_streak_max': {
        'positive': 'Long streak of consecutive on-time payments shows discipline',
        'negative': 'Short payment streaks suggest intermittent financial difficulties',
    },

    # ── India-Specific Digital Features ──
    'digital_engagement_score': {
        'positive': 'Regular UPI usage demonstrates financial participation and digital literacy',
        'negative': 'Limited digital financial activity reduces assessment confidence',
    },
    'upi_transaction_count_monthly': {
        'positive': 'Active UPI transaction volume shows robust digital financial participation',
        'negative': 'Low UPI transaction volume limits digital financial activity assessment',
    },
    'upi_transaction_consistency_score': {
        'positive': 'Steady monthly digital transaction volume shows financial stability',
        'negative': 'Irregular digital payment patterns suggest income inconsistency',
    },
    'settlement_stability_score': {
        'positive': 'Established city residency and stable lifestyle indicate low migration risk',
        'negative': 'Recent migration or frequent moves may indicate settling challenges',
    },
    'years_in_city': {
        'positive': 'Long residency in current city shows established local network',
        'negative': 'Short city residency may indicate recent migration with limited stability',
    },

    # ── Financial Health Features ──
    'financial_resilience_score': {
        'positive': 'Savings account and low rent burden provide financial buffer',
        'negative': 'High housing cost relative to income and no savings creates vulnerability',
    },
    'has_savings_account': {
        'positive': 'Having a savings account indicates financial planning behavior',
        'negative': 'Absence of savings account limits emergency financial buffer',
    },
    'rent_to_income_ratio': {
        'positive': 'Manageable rent-to-income ratio leaves room for loan repayment',
        'negative': 'High rent burden relative to income leaves little room for loan repayment',
    },
    'lifestyle_risk_score': {
        'positive': 'Low lifestyle risk factors indicate stable living conditions',
        'negative': 'Lifestyle factors such as dependents and work patterns increase risk',
    },

    # ── Traditional Credit Features ──
    'credit_income_ratio': {
        'positive': 'Requested loan amount is proportionate to income level',
        'negative': 'Loan amount is large relative to income — high repayment burden',
    },
    'annuity_income_ratio': {
        'positive': 'Monthly repayment amount is affordable given current income',
        'negative': 'Monthly repayment would consume a significant portion of income',
    },
    'ext_source_mean': {
        'positive': 'External credit bureau scores indicate good credit behavior',
        'negative': 'External credit scores suggest previous repayment difficulties',
    },
    'ext_source_available_count': {
        'positive': 'Available credit bureau data enables more confident assessment',
        'negative': 'No external credit score available — assessment relies on behavioral data',
    },
    'external_credit_score': {
        'positive': 'External credit score is in the favorable range',
        'negative': 'External credit score indicates elevated default risk',
    },
    'credit_utilization_ratio': {
        'positive': 'Low credit utilization indicates responsible credit management',
        'negative': 'High credit utilization suggests heavy reliance on borrowed funds',
    },
    'debt_ratio': {
        'positive': 'Manageable debt-to-income ratio leaves capacity for new obligations',
        'negative': 'High debt-to-income ratio limits capacity for additional borrowing',
    },
    'multi_platform_bonus': {
        'positive': 'Active on multiple platforms indicates diversified income sources',
        'negative': 'Single platform reliance creates income concentration risk',
    },
    'dependents_count': {
        'positive': 'Few dependents means more disposable income for repayment',
        'negative': 'Higher number of dependents increases financial obligations',
    },
}


def get_reason_codes(feature_contributions: dict, top_n: int = 3) -> dict:
    """
    Converts SHAP feature contributions to plain-English reason codes.

    Args:
        feature_contributions: Dict of {feature_name: shap_value} from explainer
        top_n: Number of positive and negative reasons to return

    Returns:
        Dictionary with:
            - positive_reasons: List of top N positive reason strings
            - negative_reasons: List of top N negative reason strings
            - improvement_tips: Actionable suggestions based on negative factors
    """
    positive_reasons = []
    negative_reasons = []

    # Sort by absolute SHAP value — most impactful first
    sorted_feats = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    for feat_name, shap_val in sorted_feats:
        if feat_name not in REASON_CODE_MAP:
            continue

        if shap_val < 0 and len(positive_reasons) < top_n:
            # Negative SHAP = reduces default probability = GOOD for the applicant
            positive_reasons.append({
                'feature': feat_name,
                'reason': REASON_CODE_MAP[feat_name]['positive'],
                'impact_strength': abs(shap_val),
            })
        elif shap_val > 0 and len(negative_reasons) < top_n:
            # Positive SHAP = increases default probability = BAD for the applicant
            negative_reasons.append({
                'feature': feat_name,
                'reason': REASON_CODE_MAP[feat_name]['negative'],
                'impact_strength': abs(shap_val),
            })

        if len(positive_reasons) >= top_n and len(negative_reasons) >= top_n:
            break

    # Generate improvement tips from negative factors
    improvement_tips = _generate_improvement_tips(negative_reasons)

    return {
        'positive_reasons': positive_reasons,
        'negative_reasons': negative_reasons,
        'improvement_tips': improvement_tips,
    }


def _generate_improvement_tips(negative_reasons: list) -> list:
    """
    Generates actionable improvement tips based on the top negative factors.
    Each tip includes an estimated score impact in GigScore points.

    Args:
        negative_reasons: List of negative reason dicts

    Returns:
        List of improvement tip dicts with 'tip' and 'impact' keys
    """
    IMPROVEMENT_MAP = {
        'income_stability_score': {
            'tip': 'Maintain consistent weekly earnings — avoid extended gaps between work periods',
            'impact': '+5 to +12 points',
        },
        'income_volatility_coefficient': {
            'tip': 'Try to maintain stable monthly earnings by working consistent hours',
            'impact': '+3 to +8 points',
        },
        'income_gap_months_last_year': {
            'tip': 'Reduce income gap months by staying active on platforms during slow periods',
            'impact': '+4 to +10 points',
        },
        'platform_reliability_score': {
            'tip': 'Improve your platform rating and maintain regular weekly work schedules',
            'impact': '+5 to +10 points',
        },
        'platform_tenure_months': {
            'tip': 'Continue building tenure on your platforms — longer history improves your score',
            'impact': '+2 to +6 points',
        },
        'platform_rating': {
            'tip': 'Focus on improving your platform ratings through quality service',
            'impact': '+3 to +7 points',
        },
        'cancellation_rate': {
            'tip': 'Reduce job cancellations to demonstrate reliability',
            'impact': '+2 to +5 points',
        },
        'weekly_work_consistency': {
            'tip': 'Work consistently each week rather than in sporadic bursts',
            'impact': '+3 to +8 points',
        },
        'payment_on_time_ratio': {
            'tip': 'Prioritize on-time payments on all existing obligations',
            'impact': '+5 to +12 points',
        },
        'digital_engagement_score': {
            'tip': 'Use UPI regularly for transactions to build your digital financial footprint',
            'impact': '+3 to +7 points',
        },
        'upi_transaction_consistency_score': {
            'tip': 'Maintain steady UPI transaction patterns month over month',
            'impact': '+2 to +5 points',
        },
        'financial_resilience_score': {
            'tip': 'Open a savings account and try to reduce rent burden relative to income',
            'impact': '+4 to +9 points',
        },
        'has_savings_account': {
            'tip': 'Open a savings account to build an emergency financial buffer',
            'impact': '+3 to +6 points',
        },
        'rent_to_income_ratio': {
            'tip': 'Consider housing that keeps rent below 30% of your monthly earnings',
            'impact': '+2 to +5 points',
        },
        'credit_income_ratio': {
            'tip': 'Consider applying for a smaller loan amount relative to your income',
            'impact': '+3 to +8 points',
        },
        'ext_source_available_count': {
            'tip': 'Build credit history through small formal loans to establish a credit file',
            'impact': '+5 to +15 points',
        },
        'multi_platform_bonus': {
            'tip': 'Register on additional gig platforms to diversify your income sources',
            'impact': '+2 to +5 points',
        },
        'dependents_count': {
            'tip': 'This factor reflects family obligations — focus on increasing income to offset',
            'impact': '+1 to +3 points',
        },
        'lifestyle_risk_score': {
            'tip': 'Reducing late-night work and building assets can lower this risk factor',
            'impact': '+2 to +5 points',
        },
        'income_momentum': {
            'tip': 'Build a consistent earnings growth trend over 6+ months with low volatility',
            'impact': '+3 to +7 points',
        },
        'debt_stress_score': {
            'tip': 'Reduce debt obligations while maintaining stable income to lower stress risk',
            'impact': '+4 to +9 points',
        },
        'earnings_efficiency': {
            'tip': 'Focus on higher-value orders or surge-time work to increase per-trip earnings',
            'impact': '+2 to +5 points',
        },
    }

    tips = []
    for reason in negative_reasons:
        feat = reason['feature']
        if feat in IMPROVEMENT_MAP:
            tips.append(IMPROVEMENT_MAP[feat])

    # Add a generic tip if we have fewer than 1
    if len(tips) < 1:
        tips.append({
            'tip': 'Maintain consistent platform activity and on-time payments to improve your score',
            'impact': '+3 to +8 points',
        })

    return tips
