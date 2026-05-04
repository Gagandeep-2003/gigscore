"""
dashboard/components/metrics_card.py — Reusable Metric Display Components

Provides styled metric cards for the Streamlit dashboard using
custom HTML/CSS for a professional dark-theme appearance.
"""

import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    color: str = "#6366f1",
    icon: str = "📊",
):
    """
    Renders a styled metric card with icon, label, value, and subtitle.

    Args:
        label: Metric label
        value: Metric value (formatted string)
        subtitle: Optional subtitle text
        color: Accent color (hex)
        icon: Emoji icon
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid {color};
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    ">
        <div style="font-size: 14px; color: #94a3b8; margin-bottom: 6px;">
            {icon} {label}
        </div>
        <div style="font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;">
            {value}
        </div>
        <div style="font-size: 12px; color: #64748b;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_badge(score: int, band: str, color: str):
    """
    Renders a large score badge with band label.

    Args:
        score: GigScore (0-100)
        band: Score band text
        color: Band color
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <div style="
            display: inline-block;
            background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
            border: 2px solid {color};
            border-radius: 20px;
            padding: 16px 32px;
        ">
            <div style="font-size: 72px; font-weight: 800; color: {color};
                         font-family: 'Inter', sans-serif; line-height: 1.1;">
                {score}
            </div>
            <div style="font-size: 18px; font-weight: 600; color: {color};
                         letter-spacing: 2px; margin-top: 4px;">
                {band}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_decision_badge(decision: str, detail: str = ""):
    """
    Renders a loan decision badge.

    Args:
        decision: APPROVED / CONDITIONAL / DECLINED
        detail: Decision detail text
    """
    colors = {
        'APPROVED': ('#22c55e', '✅'),
        'CONDITIONAL': ('#f59e0b', '⚠️'),
        'DECLINED': ('#ef4444', '❌'),
    }
    color, emoji = colors.get(decision, ('#64748b', '❓'))

    st.markdown(f"""
    <div style="
        text-align: center;
        background: {color}15;
        border: 1px solid {color};
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    ">
        <div style="font-size: 24px; font-weight: 700; color: {color};">
            {emoji} {decision}
        </div>
        <div style="font-size: 13px; color: #94a3b8; margin-top: 8px;">
            {detail}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_credit_limit(amount: int, currency: str = "₹"):
    """
    Renders the recommended credit limit.

    Args:
        amount: Credit limit in INR
        currency: Currency symbol
    """
    formatted = f"{currency}{amount:,}"

    st.markdown(f"""
    <div style="
        text-align: center;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    ">
        <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;
                     letter-spacing: 1px;">
            Recommended Credit Limit
        </div>
        <div style="font-size: 32px; font-weight: 700; color: #6366f1;
                     margin-top: 6px;">
            {formatted}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _hex_to_rgba(hex_color: str, alpha: float = 0.06) -> str:
    """Converts a hex color to rgba() string for reliable cross-browser rendering."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def render_factor_list(title: str, factors: list, color: str, icon: str):
    """
    Renders a list of positive or negative factors.

    Args:
        title: Section title
        factors: List of reason strings
        color: Theme color
        icon: Emoji for each item
    """
    items_html = ""
    bg_color = _hex_to_rgba(color, 0.06)
    for factor in factors:
        items_html += f"""
        <div style="padding: 8px 12px; margin: 6px 0;
                     background: {bg_color}; border-left: 3px solid {color};
                     border-radius: 0 8px 8px 0; font-size: 13px; color: #e2e8f0;">
            {icon} {factor}
        </div>
        """

    st.markdown(f"""
    <div style="margin: 12px 0;">
        <div style="font-size: 15px; font-weight: 600; color: {color};
                     margin-bottom: 8px;">
            {title}
        </div>
        {items_html if items_html else '<div style="font-size: 13px; color: #64748b;">No data available</div>'}
    </div>
    """, unsafe_allow_html=True)
