"""
SHAP integration for Streamlit dashboard
Provides interactive model explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import shap
import io
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.explainability.shap_explainer import CreditRiskExplainer

def render_shap_dashboard_tab(explainer: CreditRiskExplainer, X_sample: pd.DataFrame):
    """
    Render SHAP explainability tab in Streamlit dashboard
    
    Args:
        explainer: Initialized CreditRiskExplainer
        X_sample: Sample data for analysis
    """
    st.markdown('<div class="sub-header">🔍 Model Explainability with SHAP</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
    <strong>Why Explainability Matters in Finance:</strong> Under Basel II regulations,
    banks must be able to explain why credit decisions are made. SHAP (SHapley Additive exPlanations)
    shows exactly how each customer characteristic influences their risk score.
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate SHAP values if not already done
    with st.spinner("Calculating SHAP values for model explanations..."):
        if explainer.shap_values is None:
            explainer.calculate_shap_values(X_sample)
    
    # Create tabs within the explainability section
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Global Feature Importance", 
        "📊 Summary Plot",
        "🎯 Individual Prediction Explanation",
        "📈 Feature Dependence"
    ])
    
    with tab1:
        render_global_importance(explainer)
    
    with tab2:
        render_summary_plot(explainer, X_sample)
    
    with tab3:
        render_local_explanations(explainer, X_sample)
    
    with tab4:
        render_dependence_plots(explainer, X_sample)

def render_global_importance(explainer: CreditRiskExplainer):
    """Render global feature importance"""
    st.markdown("### Which Factors Most Influence Credit Risk?")
    
    st.markdown("""
    This chart shows the average impact of each feature on model predictions. 
    Features with higher values have a stronger influence on risk scores.
    """)
    
    # Get feature importance
    importance_df = explainer.get_global_feature_importance()
    
    # Create interactive Plotly bar chart
    fig = go.Figure()
    
    colors = ['#EF4444' if i < 2 else '#F59E0B' if i < 4 else '#3B82F6' 
              for i in range(len(importance_df))]
    
    fig.add_trace(go.Bar(
        y=importance_df['feature'],
        x=importance_df['importance'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        ),
        text=importance_df['importance'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.3f}<br>Importance: %{customdata:.1f}%<extra></extra>',
        customdata=importance_df['importance_pct']
    ))
    
    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Mean |SHAP Value| (Impact Magnitude)",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance interpretation
    st.markdown("#### 🔑 Key Insights:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_feature = importance_df.iloc[0]
        st.info(f"**Most Important: {top_feature['feature']}**  \n"
                f"Accounts for {top_feature['importance_pct']:.1f}% of model decisions")
    
    with col2:
        st.success("**Business Implication:**  \n"
                  "Focus data collection efforts on the most influential features "
                  "to improve model accuracy.")

def render_summary_plot(explainer: CreditRiskExplainer, X_sample: pd.DataFrame):
    """Render SHAP summary plot (beeswarm)"""
    st.markdown("### Feature Impact Distribution")
    
    st.markdown("""
    This beeswarm plot shows how each feature affects predictions across all customers:
    - **Red dots**: High feature values
    - **Blue dots**: Low feature values
    - **X-axis position**: Impact on risk score (positive = higher risk)
    """)
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        explainer.shap_values, 
        X_sample, 
        feature_names=explainer.feature_names,
        show=False,
        plot_size=(10, 6)
    )
    plt.tight_layout()
    
    # Convert to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    
    st.image(img, use_container_width=True)
    plt.close()
    
    # Add interpretation
    st.markdown("#### 📊 Interpretation Guide:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Recency:**
        - High values (recent transactions) → Lower risk (blue)
        - Low values (inactive customers) → Higher risk (red)
        
        **For Frequency:**
        - High frequency → Lower risk (blue)
        - Low frequency → Higher risk (red)
        """)
    
    with col2:
        st.markdown("""
        **For Monetary:**
        - High spending → Lower risk (blue)
        - Low spending → Higher risk (red)
        
        **Pattern:** Lower engagement consistently indicates higher risk
        """)

def render_local_explanations(explainer: CreditRiskExplainer, X_sample: pd.DataFrame):
    """Render local explanations for individual predictions"""
    st.markdown("### Explain Individual Customer Decisions")
    
    st.markdown("""
    Select a customer to understand why they received their specific risk score.
    This meets Basel II requirements for explaining credit decisions to customers and regulators.
    """)
    
    # Customer selection
    customer_options = [f"Customer {i}: {X_sample.iloc[i][:3].values}" 
                       for i in range(min(20, len(X_sample)))]
    
    selected_idx = st.selectbox(
        "Choose a customer to explain:",
        options=range(len(customer_options)),
        format_func=lambda x: customer_options[x]
    )
    
    # Get explanation
    explanation = explainer.explain_prediction(X_sample, selected_idx)
    
    # Display prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_color = {
            'Low Risk': '#10B981',
            'Medium Risk': '#F59E0B',
            'High Risk': '#DC2626'
        }.get(explanation['risk_category'], '#6B7280')
        
        st.markdown(
            f'<div style="background-color:{risk_color}20; padding:15px; border-radius:10px; text-align:center">'
            f'<h3 style="color:{risk_color}">{explanation["risk_category"]}</h3>'
            f'<h2>{explanation["probability"]:.1%}</h2>'
            f'<p>Risk Probability</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div style="background-color:#F3F4F6; padding:15px; border-radius:10px; text-align:center">'
            f'<h4>Base Risk</h4>'
            f'<h3>{explanation["base_value"]:.3f}</h3>'
            f'<p>Average population risk</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        net_effect = explanation['final_value'] - explanation['base_value']
        effect_color = '#10B981' if net_effect < 0 else '#DC2626'
        
        st.markdown(
            f'<div style="background-color:#F3F4F6; padding:15px; border-radius:10px; text-align:center">'
            f'<h4>Net Effect</h4>'
            f'<h3 style="color:{effect_color}">{net_effect:+.3f}</h3>'
            f'<p>Deviation from average</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Feature contributions
    st.markdown("### 📊 Feature Contributions to This Decision")
    
    # Create waterfall chart
    contributions = explanation['contributions']
    
    # Prepare data for waterfall
    base = explanation['base_value']
    final = explanation['final_value']
    
    # Create cumulative values
    cumulative = [base]
    for i, contrib in enumerate(contributions):
        cumulative.append(cumulative[-1] + contrib['shap_value'])
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="SHAP", orientation="v",
        measure=["absolute"] + ["relative"] * len(contributions) + ["total"],
        x=["Base Value"] + [c['feature'] for c in contributions] + ["Final Score"],
        y=[base] + [c['shap_value'] for c in contributions] + [0],
        text=[f"{base:.3f}"] + 
             [f"{c['shap_value']:+.3f}<br>({c['value']:.1f})" for c in contributions] +
             [f"{final:.3f}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#DC2626"}},  # Red for risk-increasing
        decreasing={"marker": {"color": "#10B981"}},  # Green for risk-decreasing
    ))
    
    fig.update_layout(
        title="Waterfall Chart: How Features Drove This Risk Score",
        xaxis_title="Feature",
        yaxis_title="Risk Score Contribution",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed contribution table
    st.markdown("### Detailed Feature Breakdown")
    
    contrib_df = pd.DataFrame(contributions)
    contrib_df['impact_direction'] = contrib_df['shap_value'].apply(
        lambda x: '⬆️ Increases Risk' if x > 0 else '⬇️ Decreases Risk'
    )
    contrib_df['shap_value'] = contrib_df['shap_value'].round(3)
    contrib_df['value'] = contrib_df['value'].round(2)
    
    st.dataframe(
        contrib_df[['feature', 'value', 'shap_value', 'impact_direction']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'feature': 'Feature',
            'value': 'Customer Value',
            'shap_value': 'SHAP Contribution',
            'impact_direction': 'Impact'
        }
    )
    
    # Business narrative
    st.markdown("#### 💡 What This Means for the Business:")
    
    top_positive = next((c for c in contributions if c['shap_value'] > 0), None)
    top_negative = next((c for c in contributions if c['shap_value'] < 0), None)
    
    narrative = f"""
    This customer's risk score of **{explanation['probability']:.1%}** places them in the 
    **{explanation['risk_category']}** category.
    
    - **Main risk driver:** {top_positive['feature']} ({top_positive['value']:.1f}) 
      increases risk by {top_positive['shap_value']:.3f}
    - **Main risk mitigator:** {top_negative['feature']} ({top_negative['value']:.1f}) 
      decreases risk by {abs(top_negative['shap_value']):.3f}
    
    **Recommendation:** {'Consider approving with monitoring' if explanation['probability'] < 0.6 
                        else 'Review additional documentation before approval'}
    """
    
    st.info(narrative)

def render_dependence_plots(explainer: CreditRiskExplainer, X_sample: pd.DataFrame):
    """Render feature dependence plots"""
    st.markdown("### How Feature Values Affect Risk")
    
    st.markdown("""
    These plots show how changing a feature's value impacts the risk prediction,
    and how features interact with each other.
    """)
    
    # Feature selection
    feature = st.selectbox(
        "Select feature to analyze:",
        options=explainer.feature_names,
        index=0
    )
    
    # Interaction feature selection
    interaction_options = ['None'] + explainer.feature_names
    interaction = st.selectbox(
        "Color by (interaction feature):",
        options=interaction_options,
        index=1 if len(explainer.feature_names) > 1 else 0
    )
    
    # Create dependence plot
    feature_idx = explainer.feature_names.index(feature)
    
    if interaction != 'None':
        interaction_idx = explainer.feature_names.index(interaction)
    else:
        interaction_idx = None
    
    # Get SHAP values and feature values
    shap_values_column = explainer.shap_values[:, feature_idx]
    feature_values = X_sample[feature].values
    
    if interaction_idx is not None:
        interaction_values = X_sample[interaction].values
    else:
        interaction_values = None
    
    # Create scatter plot
    fig = go.Figure()
    
    if interaction_values is not None:
        # Color by interaction feature
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=shap_values_column,
            mode='markers',
            marker=dict(
                size=8,
                color=interaction_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=interaction)
            ),
            text=[f"{feature}: {v:.1f}<br>SHAP: {s:.3f}<br>{interaction}: {iv:.1f}" 
                  for v, s, iv in zip(feature_values, shap_values_column, interaction_values)],
            hovertemplate='%{text}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=shap_values_column,
            mode='markers',
            marker=dict(size=8, color='#3B82F6'),
            text=[f"{feature}: {v:.1f}<br>SHAP: {s:.3f}" 
                  for v, s in zip(feature_values, shap_values_column)],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add trend line
    z = np.polyfit(feature_values, shap_values_column, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(feature_values), max(feature_values), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Trend'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_color='gray', line_width=1, opacity=0.5)
    
    fig.update_layout(
        title=f"SHAP Dependence Plot: {feature}",
        xaxis_title=feature,
        yaxis_title=f"SHAP value for {feature}",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("#### 📈 Pattern Interpretation:")
    
    # Calculate correlation
    correlation = np.corrcoef(feature_values, shap_values_column)[0, 1]
    
    if correlation > 0.5:
        trend = "strong positive correlation"
    elif correlation > 0.2:
        trend = "weak positive correlation"
    elif correlation < -0.5:
        trend = "strong negative correlation"
    elif correlation < -0.2:
        trend = "weak negative correlation"
    else:
        trend = "no clear linear trend"
    
    st.info(f"""
    **{feature}** shows {trend} with risk impact (correlation: {correlation:.2f}).
    
    {'Higher values of this feature tend to INCREASE risk.' if correlation > 0.2 else ''}
    {'Higher values of this feature tend to DECREASE risk.' if correlation < -0.2 else ''}
    
    {'The interaction with ' + interaction + ' suggests that the effect depends on other customer characteristics.' 
     if interaction != 'None' else ''}
    """)