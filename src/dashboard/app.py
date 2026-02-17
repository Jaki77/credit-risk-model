"""
Credit Risk Model Dashboard
Interactive Streamlit application for exploring credit risk predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import joblib
import json
from datetime import datetime
import requests

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dashboard.utils import (
    load_sample_data,
    load_model_and_metadata,
    get_risk_category,
    calculate_business_impact
)
from src.dashboard.components import (
    render_sidebar,
    render_metrics_overview,
    render_rfm_analysis,
    render_model_performance,
    render_prediction_interface,
    render_shap_explanations,
    render_business_impact
)

from src.dashboard.shap_integration import render_shap_dashboard_tab
from src.explainability.shap_explainer import CreditRiskExplainer

# Page configuration
st.set_page_config(
    page_title="Credit Risk Model Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-low {
        color: #10B981;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6B7280;
        border-top: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'total_risk_exposure' not in st.session_state:
    st.session_state.total_risk_exposure = 0
if 'approved_loans' not in st.session_state:
    st.session_state.approved_loans = 0

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🏦 Bati Bank Credit Risk Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <strong>Buy Now, Pay Later - Risk Assessment Tool</strong><br>
        This dashboard enables real-time credit risk assessment using alternative data from e-commerce transactions.
        Make data-driven lending decisions with explainable AI.
        </div>
    """, unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading model and data..."):
        try:
            # Try to load from API first
            api_url = "http://localhost:8000"
            model_available = requests.get(f"{api_url}/health").status_code == 200
        except:
            model_available = False
            st.warning("⚠️ API not available. Using local model for predictions.")
        
        # Load sample data for demonstration
        df_customers, rfm_data = load_sample_data()
        
        # Load model and metadata
        model, feature_names, model_metrics = load_model_and_metadata()
    
    # Sidebar
    render_sidebar()

    # After loading model and data
    try:
        explainer = CreditRiskExplainer(
            model_path="models/best_model.pkl",
            feature_names_path="models/feature_names.json"
        )
        explainer.prepare_background_data(df_customers[feature_names].head(100))
        st.sidebar.success("✅ SHAP explainer ready")
    except Exception as e:
        explainer = None
        st.sidebar.warning(f"⚠️ SHAP explainer not available: {str(e)}")

    # Main content - Tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview & KPIs", 
        "👥 RFM Customer Analysis", 
        "🤖 Model Performance",
        "🔮 Risk Prediction",
        "🔍 Model Explainability",  # New SHAP tab
        "💼 Business Impact"
    ])
    
    with tab1:
        render_metrics_overview(df_customers, rfm_data, model_metrics)
        
        # Key visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="sub-header">Risk Distribution</div>', 
                       unsafe_allow_html=True)
            
            # Calculate risk distribution
            risk_counts = df_customers['risk_category'].value_counts().reset_index()
            risk_counts.columns = ['Risk Category', 'Count']
            
            colors = {'Low Risk': '#10B981', 'Medium Risk': '#F59E0B', 'High Risk': '#DC2626'}
            
            fig = px.pie(
                risk_counts, 
                values='Count', 
                names='Risk Category',
                color='Risk Category',
                color_discrete_map=colors,
                hole=0.4,
                title="Customer Risk Segmentation"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="sub-header">Transaction Activity</div>', 
                       unsafe_allow_html=True)
            
            # Monthly transaction trends
            if 'TransactionMonth' in df_customers.columns:
                monthly_data = df_customers.groupby('TransactionMonth').agg({
                    'CustomerId': 'count',
                    'TotalTransactionAmount': 'sum'
                }).reset_index()
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_data['TransactionMonth'],
                        y=monthly_data['CustomerId'],
                        name="Active Customers",
                        marker_color='#3B82F6'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_data['TransactionMonth'],
                        y=monthly_data['TotalTransactionAmount'],
                        name="Transaction Volume",
                        mode='lines+markers',
                        line=dict(color='#F59E0B', width=3),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Customer Activity & Transaction Volume Over Time",
                    hovermode='x unified'
                )
                
                fig.update_yaxes(title_text="Active Customers", secondary_y=False)
                fig.update_yaxes(title_text="Transaction Volume ($)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        render_rfm_analysis(rfm_data, df_customers)
    
    with tab3:
        render_model_performance(model_metrics, model, feature_names)
    
    with tab4:
        render_prediction_interface(model, feature_names, api_url if model_available else None)
    
    with tab5:
        if explainer is not None:
            render_shap_dashboard_tab(explainer, df_customers[feature_names].head(200))
        else:
            st.error("SHAP explainer not available. Please train a model first.")
            st.info("Run the model training script to generate a model for explanations.")

    with tab6:
        render_business_impact(
            st.session_state.predictions_made,
            st.session_state.approved_loans,
            st.session_state.total_risk_exposure
        )
    
    # Footer
    st.markdown('<div class="footer">© 2026 Bati Bank - Credit Risk Modeling Team</div>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()