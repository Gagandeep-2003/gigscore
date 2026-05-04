"""
dashboard/components/gauge_chart.py — Animated GigScore Gauge Chart

Creates a Plotly gauge chart that visually represents the GigScore (0-100).
Color zones indicate score bands:
    0-39: Red (VERY POOR)
    40-54: Dark orange (POOR)
    55-69: Orange (FAIR)
    70-84: Light green (GOOD)
    85-100: Green (EXCELLENT)
"""

import plotly.graph_objects as go


def create_gigscore_gauge(score: int, band: str, color: str) -> go.Figure:
    """
    Creates an animated circular gauge chart for GigScore display.

    The gauge transitions from 0 to the final score with a smooth animation.
    Color zones on the gauge bar indicate risk tiers.

    Args:
        score: GigScore (0-100)
        band: Score band label (EXCELLENT, GOOD, etc.)
        color: Hex color for the score

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={
            'font': {'size': 72, 'color': color, 'family': 'Inter, sans-serif'},
            'suffix': '',
        },
        title={
            'text': f"<b>{band}</b>",
            'font': {'size': 20, 'color': color, 'family': 'Inter, sans-serif'},
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': '#334155',
                'dtick': 10,
                'tickfont': {'size': 12, 'color': '#94a3b8'},
            },
            'bar': {
                'color': color,
                'thickness': 0.75,
            },
            'bgcolor': '#1e293b',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 39], 'color': 'rgba(127, 29, 29, 0.3)'},     # Very Poor
                {'range': [39, 54], 'color': 'rgba(239, 68, 68, 0.3)'},     # Poor
                {'range': [54, 69], 'color': 'rgba(245, 158, 11, 0.3)'},    # Fair
                {'range': [69, 84], 'color': 'rgba(132, 204, 22, 0.3)'},    # Good
                {'range': [84, 100], 'color': 'rgba(34, 197, 94, 0.3)'},    # Excellent
            ],
            'threshold': {
                'line': {'color': '#e2e8f0', 'width': 3},
                'thickness': 0.8,
                'value': score,
            },
        },
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'},
    )

    return fig


def create_mini_gauge(score: int, label: str, color: str) -> go.Figure:
    """
    Creates a smaller gauge for inline display (e.g., in metrics cards).

    Args:
        score: Value (0-100)
        label: Label text
        color: Hex color

    Returns:
        Plotly Figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 36, 'color': color}},
        title={'text': label, 'font': {'size': 14, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#1e293b',
            'borderwidth': 0,
        },
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig
