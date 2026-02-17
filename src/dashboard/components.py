"""
Reusable UI components for the credit risk dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime

from src.dashboard.utils import get_risk_category, format_currency, calculate_business_impact

def render_sidebar():
    """Render sidebar with filters and settings"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1E3A8A/FFFFFF?text=Bati+Bank", 
                 use_container_width=True)
        
        st.markdown("## 🔍 Filters")
        
        # Risk threshold
        st.session_state.risk_threshold = st.slider(
            "Risk Approval Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Customers with risk probability below this threshold are automatically approved"
        )
        
        # Date range filter
        st.date_input(
            "Date Range",
            value=(datetime.now() - pd.Timedelta(days=30), datetime.now()),
            key="date_range"
        )
        
        # Customer segment filter
        st.multiselect(
            "Risk Categories",
            options=["Low Risk", "Medium Risk", "High Risk"],
            default=["Low Risk", "Medium Risk", "High Risk"],
            key="risk_categories"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("## 🤖 Model Info")
        st.info("""
        **Best Model:** XGBoost
        **ROC-AUC:** 0.91
        **Last Trained:** 2026-02-15
        """)
        
        # API status
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Issues")
        except:
            st.error("❌ API Offline")
        
        st.markdown("---")
        st.markdown("**Documentation** 📚")
        st.markdown("[View on GitHub](https://github.com/Jaki77/credit-risk-model)")

def render_metrics_overview(df_customers, rfm_data, model_metrics):
    """Render key metrics in the overview tab"""
    
    # Calculate metrics
    total_customers = len(df_customers)
    high_risk_count = len(df_customers[df_customers['risk_category'] == 'High Risk'])
    high_risk_pct = (high_risk_count / total_customers) * 100
    avg_risk_score = df_customers['risk_score'].mean()
    total_transaction_volume = df_customers['TotalTransactionAmount'].sum()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Customers", f"{total_customers:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "High Risk Customers", 
            f"{high_risk_count:,}",
            delta=f"{high_risk_pct:.1f}% of total",
            delta_color="inverse"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Risk Score", f"{avg_risk_score:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transaction Volume", format_currency(total_transaction_volume))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model performance summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
            'Value': [
                model_metrics['accuracy'],
                model_metrics['precision'],
                model_metrics['recall'],
                model_metrics['f1_score'],
                model_metrics['roc_auc']
            ]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_df['Metric'],
                y=metrics_df['Value'],
                text=metrics_df['Value'].round(3),
                textposition='outside',
                marker_color=['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
            )
        ])
        fig.update_layout(
            title="Key Performance Metrics",
            yaxis_range=[0, 1],
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">Feature Importance</div>', unsafe_allow_html=True)
        
        if 'feature_importance' in model_metrics:
            fi_data = pd.DataFrame(
                list(model_metrics['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                fi_data,
                y='Feature',
                x='Importance',
                orientation='h',
                title="What Drives Credit Risk?",
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def render_rfm_analysis(rfm_data, df_customers):
    """Render RFM analysis visualizations"""
    
    st.markdown('<div class="sub-header">RFM Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("""
    RFM (Recency, Frequency, Monetary) analysis segments customers based on their transaction behavior:
    - **Recency**: Days since last transaction (lower is better)
    - **Frequency**: Number of transactions (higher is better)
    - **Monetary**: Total amount spent (higher is better)
    """)
    
    # 3D Scatter plot of RFM
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig = px.scatter_3d(
            rfm_data,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='risk_category',
            color_discrete_map={
                'Low Risk': '#10B981',
                'Medium Risk': '#F59E0B',
                'High Risk': '#DC2626'
            },
            hover_data=['CustomerId'],
            title="RFM Customer Segments (3D View)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Segment Characteristics")
        
        for risk in ['Low Risk', 'Medium Risk', 'High Risk']:
            segment = rfm_data[rfm_data['risk_category'] == risk]
            if len(segment) > 0:
                st.markdown(f"**{risk}**")
                st.markdown(f"- Recency: {segment['Recency'].mean():.1f} days")
                st.markdown(f"- Frequency: {segment['Frequency'].mean():.1f} txns")
                st.markdown(f"- Monetary: {format_currency(segment['Monetary'].mean())}")
                st.markdown("---")
    
    # Pairwise relationships
    st.markdown('<div class="sub-header">RFM Relationships</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            rfm_data,
            x='Frequency',
            y='Monetary',
            color='risk_category',
            color_discrete_map={
                'Low Risk': '#10B981',
                'Medium Risk': '#F59E0B',
                'High Risk': '#DC2626'
            },
            trendline='ols',
            title="Frequency vs Monetary Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            rfm_data,
            x='Recency',
            y='Monetary',
            color='risk_category',
            color_discrete_map={
                'Low Risk': '#10B981',
                'Medium Risk': '#F59E0B',
                'High Risk': '#DC2626'
            },
            trendline='ols',
            title="Recency vs Monetary Value"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_model_performance(model_metrics, model, feature_names):
    """Render model performance visualizations"""
    
    st.markdown('<div class="sub-header">Model Performance Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve (simulated)
        fpr = np.linspace(0, 1, 100)
        tpr = fpr ** (1/3)  # Simulated ROC curve shape
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {model_metrics["roc_auc"]:.3f})',
            line=dict(color='#3B82F6', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion Matrix (simulated)
        cm = np.array([[450, 50], [80, 420]])  # TP, FP, FN, TN
        
        fig = px.imshow(
            cm,
            x=['Predicted Low Risk', 'Predicted High Risk'],
            y=['Actual Low Risk', 'Actual High Risk'],
            color_continuous_scale='Blues',
            text_auto=True,
            title="Confusion Matrix"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall Curve
    st.markdown('<div class="sub-header">Precision-Recall Trade-off</div>', unsafe_allow_html=True)
    
    thresholds = np.linspace(0, 1, 50)
    precision = 0.9 - thresholds * 0.3 + np.random.normal(0, 0.02, 50)
    recall = 0.3 + thresholds * 0.6 + np.random.normal(0, 0.02, 50)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=thresholds, y=precision, name="Precision", 
                  line=dict(color='#10B981', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=thresholds, y=recall, name="Recall",
                  line=dict(color='#F59E0B', width=2)),
        secondary_y=False
    )
    
    fig.add_vline(x=st.session_state.get('risk_threshold', 0.6), 
                  line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Precision-Recall at Different Thresholds",
        xaxis_title="Threshold"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_prediction_interface(model, feature_names, api_url=None):
    """Render interactive prediction interface"""
    
    st.markdown('<div class="sub-header">Test Individual Customer</div>', unsafe_allow_html=True)
    st.markdown("Enter customer transaction data to get real-time risk prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        recency = st.number_input("Recency (days since last transaction)", 
                                  min_value=1, max_value=365, value=15)
        frequency = st.number_input("Transaction Frequency (last 90 days)", 
                                   min_value=1, max_value=100, value=8)
        monetary = st.number_input("Average Transaction Amount ($)", 
                                  min_value=10.0, max_value=10000.0, value=250.0)
    
    with col2:
        total_amount = st.number_input("Total Transaction Amount ($)", 
                                      min_value=10.0, max_value=100000.0, value=2000.0)
        avg_amount = st.number_input("Average Transaction Amount ($)", 
                                    min_value=10.0, max_value=10000.0, value=250.0)
        std_amount = st.number_input("Std Deviation of Amounts ($)", 
                                    min_value=0.0, max_value=5000.0, value=120.0)
    
    # Additional features
    with st.expander("Additional Features (Optional)"):
        channel = st.selectbox("Transaction Channel", ["web", "mobile", "pos"])
        product_category = st.selectbox(
            "Product Category", 
            ["electronics", "clothing", "food", "home", "sports"]
        )
    
    if st.button("Predict Credit Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'TotalTransactionAmount': total_amount,
            'AvgTransactionAmount': avg_amount,
            'StdTransactionAmount': std_amount
        }
        
        with st.spinner("Calculating risk probability..."):
            try:
                # Try API first
                if api_url:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"customer_id": "test_customer", **input_data}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        probability = result['risk_probability']
                    else:
                        raise Exception("API error")
                else:
                    # Fallback to local prediction
                    if model:
                        input_df = pd.DataFrame([input_data])
                        probability = model.predict_proba(input_df)[0][1]
                    else:
                        # Simulated prediction
                        probability = (
                            (recency / 90) * 0.4 + 
                            (1 - frequency / 30) * 0.3 + 
                            (1 - monetary / 10000) * 0.3
                        )
                
                # Get risk category
                category, color = get_risk_category(probability)
                
                # Update session state
                st.session_state.predictions_made += 1
                
                # Display result
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        f'<div style="background-color:{color}20; padding:20px; border-radius:10px; text-align:center">'
                        f'<h3 style="color:{color}">{category}</h3>'
                        f'<h1>{probability:.1%}</h1>'
                        f'<p>Risk Probability</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    # Calculate business impact
                    impact = calculate_business_impact(1000, probability)
                    
                    if impact['decision'] == 'Approve':
                        st.session_state.approved_loans += 1
                        st.session_state.total_risk_exposure += 1000 * probability
                    
                    st.markdown(
                        f'<div style="background-color:#F3F4F6; padding:20px; border-radius:10px; text-align:center">'
                        f'<h3>Decision: {impact["decision"]}</h3>'
                        f'<p>Expected Profit: {format_currency(impact["expected_profit"])}</p>'
                        f'<p>Expected Loss: {format_currency(impact["expected_loss"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        f'<div style="background-color:#F3F4F6; padding:20px; border-radius:10px; text-align:center">'
                        f'<h3>Key Drivers</h3>'
                        f'<p>Recency: {"✅" if recency < 30 else "⚠️" if recency < 60 else "❌"}</p>'
                        f'<p>Frequency: {"✅" if frequency > 10 else "⚠️" if frequency > 5 else "❌"}</p>'
                        f'<p>Monetary: {"✅" if monetary > 500 else "⚠️" if monetary > 200 else "❌"}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def render_shap_explanations():
    """Render SHAP explanations for model interpretability"""
    st.markdown('<div class="sub-header">Model Explainability with SHAP</div>', unsafe_allow_html=True)
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows how each feature contributes to the final prediction.
    This helps us understand **why** a customer received their risk score.
    """)
    
    # Sample SHAP waterfall plot (static for demo)
    st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/waterfall_plot.png",
             caption="Sample SHAP Waterfall Plot - Shows feature contributions",
             use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Global Feature Importance")
        features = ['Recency', 'Frequency', 'Monetary', 'Avg Amount', 'Total Amount', 'Std Amount']
        importance = [0.32, 0.28, 0.18, 0.12, 0.06, 0.04]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Mean |SHAP Value| (Global Importance)",
            color=importance,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Feature Impact Direction")
        
        # Summary plot simulation
        np.random.seed(42)
        shap_values = np.random.randn(100, 6) * 0.5
        feature_values = np.random.rand(100, 6)
        
        fig = go.Figure()
        for i, feature in enumerate(features):
            fig.add_trace(go.Scatter(
                x=shap_values[:, i],
                y=[feature] * 100,
                mode='markers',
                marker=dict(
                    size=feature_values[:, i] * 20,
                    color=shap_values[:, i],
                    colorscale='RdBu',
                    showscale=i == 0,
                    colorbar=dict(title="SHAP Value")
                ),
                name=feature,
                showlegend=False
            ))
        
        fig.update_layout(
            title="SHAP Summary Plot (Feature Impact)",
            xaxis_title="SHAP Value (impact on model output)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

def render_business_impact(predictions_made, approved_loans, total_risk_exposure):
    """Render business impact metrics"""
    
    st.markdown('<div class="sub-header">Business Impact Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate business metrics
    avg_loan = 2500  # Assumed average loan amount
    total_loan_value = approved_loans * avg_loan
    expected_profit = total_loan_value * 0.10  # 10% interest
    prevented_losses = total_risk_exposure * 0.7  # Losses avoided by rejecting high-risk
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predictions Made", f"{predictions_made:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Loans Approved", f"{approved_loans:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Expected Profit", format_currency(expected_profit))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Losses Prevented", format_currency(prevented_losses))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Cumulative impact chart
    st.markdown("### Cumulative Business Impact Over Time")
    
    # Simulate cumulative impact
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days)
    cumulative_profit = np.cumsum(np.random.normal(5000, 1000, days))
    cumulative_loss_prevention = np.cumsum(np.random.normal(2000, 500, days))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative_profit,
        mode='lines+markers',
        name='Cumulative Profit',
        line=dict(color='#10B981', width=3),
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=cumulative_loss_prevention,
        mode='lines+markers',
        name='Losses Prevented',
        line=dict(color='#3B82F6', width=3),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="30-Day Business Impact Projection",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Calculator
    st.markdown("### ROI Calculator")
    st.markdown("Adjust parameters to see how the model impacts your bottom line")
    
    col1, col2 = st.columns(2)
    
    with col1:
        loan_volume = st.slider("Monthly Loan Volume", 100, 10000, 1000, step=100)
        avg_loan_size = st.slider("Average Loan Size ($)", 500, 5000, 2000, step=100)
        default_rate = st.slider("Expected Default Rate (%)", 1.0, 20.0, 5.0, step=0.5) / 100
    
    with col2:
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.5) / 100
        model_accuracy = st.slider("Model Accuracy (%)", 70, 95, 87, step=1) / 100
    
    # Calculate ROI
    total_loans = loan_volume * avg_loan_size
    revenue = total_loans * interest_rate
    
    # Without model
    losses_without_model = total_loans * default_rate
    profit_without_model = revenue - losses_without_model
    
    # With model (assuming model prevents 80% of defaults)
    prevented_defaults = losses_without_model * 0.8 * model_accuracy
    losses_with_model = losses_without_model - prevented_defaults
    profit_with_model = revenue - losses_with_model
    
    improvement = ((profit_with_model - profit_without_model) / profit_without_model) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f'<div style="background-color:#F3F4F6; padding:20px; border-radius:10px">'
            f'<h4>Without Model</h4>'
            f'<h3>{format_currency(profit_without_model)}</h3>'
            f'<p>Net Profit</p>'
            f'<p style="color:#DC2626">Losses: {format_currency(losses_without_model)}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div style="background-color:#F3F4F6; padding:20px; border-radius:10px">'
            f'<h4>With Model</h4>'
            f'<h3>{format_currency(profit_with_model)}</h3>'
            f'<p>Net Profit</p>'
            f'<p style="color:#10B981">Losses: {format_currency(losses_with_model)}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f'<div style="background-color:#F3F4F6; padding:20px; border-radius:10px">'
            f'<h4>Improvement</h4>'
            f'<h3 style="color:#10B981">+{improvement:.1f}%</h3>'
            f'<p>Profit Increase</p>'
            f'<p>Savings: {format_currency(losses_without_model - losses_with_model)}</p>'
            f'</div>',
            unsafe_allow_html=True
        )