"""
Utility functions for the credit risk dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta
import random

def load_sample_data():
    """
    Load or generate sample customer data for demonstration
    """
    np.random.seed(42)
    n_customers = 1000
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(1000, 1000 + n_customers)]
    
    # Generate RFM features
    recency = np.random.exponential(30, n_customers).astype(int)
    recency = np.clip(recency, 1, 90)
    
    frequency = np.random.poisson(5, n_customers)
    frequency = np.clip(frequency, 1, 30)
    
    monetary = np.random.lognormal(mean=5, sigma=1, size=n_customers)
    monetary = np.clip(monetary, 10, 10000)
    
    # Calculate derived features
    total_amount = monetary * frequency * np.random.uniform(0.8, 1.2, n_customers)
    avg_amount = total_amount / frequency
    std_amount = avg_amount * np.random.uniform(0.1, 0.5, n_customers)
    
    # Calculate risk score and category based on RFM
    # Lower recency, higher frequency, higher monetary = lower risk
    risk_score = (
        (recency / 90) * 0.4 + 
        (1 - frequency / 30) * 0.3 + 
        (1 - monetary / 10000) * 0.3
    )
    
    risk_category = pd.cut(
        risk_score,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Create main dataframe
    df = pd.DataFrame({
        'CustomerId': customer_ids,
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary.round(2),
        'TotalTransactionAmount': total_amount.round(2),
        'AvgTransactionAmount': avg_amount.round(2),
        'StdTransactionAmount': std_amount.round(2),
        'risk_score': risk_score.round(3),
        'risk_category': risk_category,
        'TransactionMonth': np.random.choice(
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 
            n_customers
        ),
        'ChannelId': np.random.choice(['web', 'mobile', 'pos'], n_customers),
        'ProductCategory': np.random.choice(
            ['electronics', 'clothing', 'food', 'home', 'sports'], 
            n_customers
        )
    })
    
    # Create RFM data for clustering visualization
    rfm_data = df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'risk_category']].copy()
    
    return df, rfm_data

def load_model_and_metadata():
    """
    Load trained model and metadata
    """
    model_path = "models/best_model.pkl"
    feature_names_path = "models/feature_names.json"
    metrics_path = "models/model_metrics.json"
    
    model = None
    feature_names = ['Recency', 'Frequency', 'Monetary', 'TotalTransactionAmount', 
                     'AvgTransactionAmount', 'StdTransactionAmount']
    
    # Default metrics for demonstration
    model_metrics = {
        'accuracy': 0.87,
        'precision': 0.83,
        'recall': 0.79,
        'f1_score': 0.81,
        'roc_auc': 0.91,
        'feature_importance': {
            'Recency': 0.28,
            'Frequency': 0.24,
            'Monetary': 0.22,
            'AvgTransactionAmount': 0.15,
            'TotalTransactionAmount': 0.07,
            'StdTransactionAmount': 0.04
        }
    }
    
    # Try to load actual model if exists
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except:
            pass
    
    # Try to load feature names
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    
    # Try to load actual metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            model_metrics = json.load(f)
    
    return model, feature_names, model_metrics

def get_risk_category(probability, thresholds=(0.3, 0.6)):
    """
    Convert probability to risk category
    """
    if probability < thresholds[0]:
        return "Low Risk", "#10B981"
    elif probability < thresholds[1]:
        return "Medium Risk", "#F59E0B"
    else:
        return "High Risk", "#DC2626"

def calculate_business_impact(loan_amount, risk_probability, threshold=0.6):
    """
    Calculate expected business impact of a loan decision
    """
    if risk_probability < threshold:
        # Approve loan
        expected_profit = loan_amount * 0.10  # Assume 10% interest
        expected_loss = 0
        decision = "Approve"
    else:
        # Reject loan
        expected_profit = 0
        expected_loss = loan_amount * risk_probability  # Expected loss if we approved
        decision = "Reject"
    
    return {
        'decision': decision,
        'expected_profit': expected_profit,
        'expected_loss': expected_loss,
        'risk_adjusted_return': expected_profit - expected_loss * 0.3  # Weighted
    }

def format_currency(value):
    """
    Format value as currency
    """
    if value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"