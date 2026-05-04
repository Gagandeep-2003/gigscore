"""
dashboard/pages/dataset_explorer.py — Tab 4: Dataset Explorer

Interactive EDA of the training data:
    - Dataset source breakdown pie chart
    - Default rate by income type
    - Default rate by platform tenure
    - Income distribution by default status
    - Correlation heatmap
    - Feature distribution explorer
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dashboard.components.metrics_card import render_metric_card


def render_dataset_explorer_page():
    """Renders the Dataset Explorer page."""

    st.markdown("""
    <h1 style="text-align: center; color: #f1f5f9; margin-bottom: 4px;">
        🔍 Dataset Explorer
    </h1>
    <p style="text-align: center; color: #64748b; font-size: 14px; margin-bottom: 24px;">
        Explore the training data: distributions, relationships, and patterns
    </p>
    """, unsafe_allow_html=True)

    st.info("""
    **About this dataset:** GigScore is trained on a synthetic dataset of 50,000 gig workers
    designed to reflect the real distribution of India's informal gig economy — income patterns,
    platform tenure, UPI usage, and default rates based on published RBI and NASSCOM data.
    Real gig worker transaction data is not publicly available due to privacy regulations.
    The synthetic generation process was validated to match known population statistics.
    """)

    # Load data
    df = _load_training_data()

    if df is None:
        st.info("📝 Generating synthetic demo data for exploration...")
        from src.data.synthetic_generator import generate_synthetic_gig_data
        df = generate_synthetic_gig_data(n_samples=10000, seed=42)

    # ── Overview Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📊 Total Records", value=f"{len(df):,}")
    with col2:
        default_rate = df['target'].mean() if 'target' in df.columns else 0
        st.metric(label="⚠️ Default Rate", value=f"{default_rate:.1%}")
    with col3:
        n_features = len([c for c in df.columns if c not in ['target', 'data_source']])
        st.metric(label="🔧 Features", value=str(n_features))
    with col4:
        sources = df['data_source'].nunique() if 'data_source' in df.columns else 1
        st.metric(label="📁 Data Sources", value=str(sources))

    st.markdown("---")

    # ── Data Source Breakdown ──
    col_pie, col_default = st.columns(2)

    with col_pie:
        st.markdown("### 📁 Data Source Breakdown")
        if 'data_source' in df.columns:
            source_counts = df['data_source'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=source_counts.index,
                values=source_counts.values,
                hole=0.4,
                marker={'colors': ['#6366f1', '#8b5cf6', '#a78bfa']},
                textfont={'color': '#e2e8f0'},
            )])
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font={'color': '#e2e8f0'}),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, key="source_pie")
        else:
            st.info("data_source column not available")

    with col_default:
        st.markdown("### 📈 Default Rate by Platform Tenure")
        if 'platform_tenure_months' in df.columns and 'target' in df.columns:
            df['tenure_bucket'] = pd.cut(
                df['platform_tenure_months'],
                bins=[0, 6, 12, 24, 48, 100],
                labels=['0-6m', '6-12m', '1-2yr', '2-4yr', '4+yr']
            )
            tenure_default = df.groupby('tenure_bucket', observed=False)['target'].mean()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tenure_default.index.astype(str),
                y=tenure_default.values * 100,
                marker_color='#6366f1',
                text=[f"{v:.1f}%" for v in tenure_default.values * 100],
                textposition='outside',
                textfont={'color': '#e2e8f0'},
            ))
            fig.update_layout(
                yaxis_title='Default Rate (%)',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.5)',
                xaxis={'tickfont': {'color': '#e2e8f0'}},
                yaxis={'gridcolor': '#1e293b', 'tickfont': {'color': '#94a3b8'}},
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, key="tenure_default")
            df = df.drop(columns=['tenure_bucket'], errors='ignore')

    # ── Income Distribution ──
    st.markdown("### 💰 Income Distribution by Default Status")
    income_col = None
    for col in ['monthly_earnings_inr', 'monthly_earnings_proxy', 'AMT_INCOME_TOTAL']:
        if col in df.columns:
            income_col = col
            break

    if income_col and 'target' in df.columns:
        fig = go.Figure()
        for label, color, name in [(0, '#22c55e', 'Repaid'), (1, '#ef4444', 'Defaulted')]:
            subset = df[df['target'] == label][income_col].dropna()
            fig.add_trace(go.Histogram(
                x=subset,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
            ))
        fig.update_layout(
            barmode='overlay',
            xaxis_title=f'{income_col} (₹)',
            yaxis_title='Count',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.5)',
            xaxis={'gridcolor': '#1e293b', 'tickfont': {'color': '#94a3b8'}},
            yaxis={'gridcolor': '#1e293b', 'tickfont': {'color': '#94a3b8'}},
            legend=dict(font={'color': '#e2e8f0'}, bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, key="income_dist")

    # ── Correlation Heatmap ──
    st.markdown("### 🔥 Feature Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Select top features by variance
    top_cols = [c for c in numeric_cols if c != 'target'][:15]
    if 'target' in numeric_cols:
        top_cols.append('target')

    if len(top_cols) > 3:
        corr = df[top_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[c.replace('_', ' ').title()[:20] for c in corr.columns],
            y=[c.replace('_', ' ').title()[:20] for c in corr.columns],
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={'size': 9, 'color': '#e2e8f0'},
        ))
        fig.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis={'tickfont': {'color': '#94a3b8', 'size': 10}, 'tickangle': 45},
            yaxis={'tickfont': {'color': '#94a3b8', 'size': 10}},
            margin=dict(l=120, r=20, t=20, b=100),
        )
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

    # ── Feature Explorer ──
    st.markdown("### 🔎 Feature Distribution Explorer")
    feature_options = [c for c in numeric_cols if c != 'target']
    if feature_options:
        selected_feature = st.selectbox(
            "Select a feature to explore:",
            options=feature_options,
            index=0,
        )

        if selected_feature and 'target' in df.columns:
            fig = go.Figure()
            for label, color, name in [(0, '#22c55e', 'Repaid'), (1, '#ef4444', 'Defaulted')]:
                subset = df[df['target'] == label][selected_feature].dropna()
                fig.add_trace(go.Histogram(
                    x=subset, name=name,
                    marker_color=color, opacity=0.7, nbinsx=40,
                ))
            fig.update_layout(
                barmode='overlay',
                xaxis_title=selected_feature.replace('_', ' ').title(),
                yaxis_title='Count',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.5)',
                xaxis={'gridcolor': '#1e293b', 'tickfont': {'color': '#94a3b8'}},
                yaxis={'gridcolor': '#1e293b', 'tickfont': {'color': '#94a3b8'}},
                legend=dict(font={'color': '#e2e8f0'}, bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, key="feature_explorer")

    # ── Raw Data Preview ──
    with st.expander("🗂️ Raw Data Preview"):
        st.dataframe(df.head(100), use_container_width=True, height=400)


def _load_training_data() -> pd.DataFrame:
    """Loads the processed training data if available."""
    data_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'final_training_data.csv'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'synthetic_india_gig.csv'),
    ]

    for path in data_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return None
