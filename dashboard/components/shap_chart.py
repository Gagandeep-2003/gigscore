"""
dashboard/components/shap_chart.py — SHAP Visualization Components

Creates interactive SHAP waterfall and bar charts using Plotly
for the Streamlit dashboard. These charts show feature-level
contributions to individual predictions.
"""

import plotly.graph_objects as go
import numpy as np


def create_shap_waterfall(
    feature_names: list,
    shap_values: list,
    base_value: float = 0,
    max_features: int = 12,
) -> go.Figure:
    """
    Creates a horizontal waterfall chart showing SHAP feature contributions.

    Positive SHAP values (push toward default) shown in red.
    Negative SHAP values (push away from default) shown in green.

    Args:
        feature_names: List of feature names
        shap_values: Corresponding SHAP values
        base_value: Model baseline value
        max_features: Maximum features to display

    Returns:
        Plotly Figure
    """
    # Sort by absolute value and take top N
    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:max_features]

    # Reverse for bottom-to-top display
    pairs = pairs[::-1]

    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    # Format feature names for display (replace underscores, title case)
    display_names = [n.replace('_', ' ').title() for n in names]

    # Colors: green for negative SHAP (reduces risk), red for positive (increases risk)
    colors = ['#22c55e' if v < 0 else '#ef4444' for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=display_names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition='outside',
        textfont={'size': 11, 'color': '#e2e8f0'},
        hovertemplate=(
            '<b>%{y}</b><br>'
            'SHAP Value: %{x:.4f}<br>'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title={
            'text': 'Feature Contributions to Score',
            'font': {'size': 16, 'color': '#f1f5f9'},
            'x': 0.5,
        },
        xaxis={
            'title': {'text': 'SHAP Value (impact on default probability)',
                      'font': {'color': '#94a3b8', 'size': 12}},
            'gridcolor': '#334155',
            'zerolinecolor': '#64748b',
            'zerolinewidth': 2,
            'tickfont': {'color': '#94a3b8'},
        },
        yaxis={
            'tickfont': {'color': '#e2e8f0', 'size': 11},
        },
        height=max(350, len(pairs) * 35 + 100),
        margin=dict(l=180, r=60, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        showlegend=False,
    )

    # Add annotations for direction
    fig.add_annotation(
        x=max(values) * 0.7 if max(values) > 0 else 0.1,
        y=len(pairs) - 0.5,
        text="← Reduces Risk | Increases Risk →",
        showarrow=False,
        font={'size': 10, 'color': '#64748b'},
    )

    return fig


def create_global_importance_bar(
    feature_names: list,
    importance_values: list,
    top_n: int = 20,
) -> go.Figure:
    """
    Creates a horizontal bar chart of global feature importance
    (mean |SHAP| values).

    Args:
        feature_names: List of feature names
        importance_values: Mean absolute SHAP values
        top_n: Number of top features to display

    Returns:
        Plotly Figure
    """
    # Sort and take top N
    pairs = sorted(zip(feature_names, importance_values), key=lambda x: x[1], reverse=True)
    pairs = pairs[:top_n]
    pairs = pairs[::-1]  # Reverse for bottom-to-top

    names = [p[0].replace('_', ' ').title() for p in pairs]
    values = [p[1] for p in pairs]

    # Color gradient from light to dark
    max_val = max(values) if values else 1
    colors = [
        f'rgba(99, 102, 241, {0.4 + 0.6 * v / max_val})' for v in values
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition='outside',
        textfont={'size': 10, 'color': '#e2e8f0'},
    ))

    fig.update_layout(
        title={
            'text': f'Top {top_n} Features by Mean |SHAP| Value',
            'font': {'size': 16, 'color': '#f1f5f9'},
            'x': 0.5,
        },
        xaxis={
            'title': {'text': 'Mean |SHAP Value|',
                      'font': {'color': '#94a3b8', 'size': 12}},
            'gridcolor': '#334155',
            'tickfont': {'color': '#94a3b8'},
        },
        yaxis={'tickfont': {'color': '#e2e8f0', 'size': 11}},
        height=max(400, top_n * 30 + 100),
        margin=dict(l=200, r=80, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        showlegend=False,
    )

    return fig
