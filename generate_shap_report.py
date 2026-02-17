#!/usr/bin/env python3
"""
Script to generate comprehensive SHAP explainability report
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.explainability.shap_explainer import CreditRiskExplainer
from src.data_processing import load_and_preprocess_data

def main():
    """Generate SHAP explainability report"""
    
    print("=" * 60)
    print("🔍 Generating SHAP Explainability Report")
    print("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"reports/shap_report_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Load data
    print("\n📊 Loading data...")
    try:
        # Try to load from processed data
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        print(f"✅ Loaded training data: {X_train.shape}")
        print(f"✅ Loaded test data: {X_test.shape}")
    except:
        print("⚠️ Processed data not found. Generating sample data...")
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        X_train = pd.DataFrame({
            'Recency': np.random.exponential(30, n_samples),
            'Frequency': np.random.poisson(5, n_samples),
            'Monetary': np.random.lognormal(5, 1, n_samples),
            'TotalTransactionAmount': np.random.lognormal(6, 1, n_samples),
            'AvgTransactionAmount': np.random.lognormal(5, 0.5, n_samples),
            'StdTransactionAmount': np.random.lognormal(4, 0.5, n_samples)
        })
        X_test = X_train.iloc[:200].copy()
    
    # Initialize explainer
    print("\n🤖 Initializing SHAP explainer...")
    try:
        explainer = CreditRiskExplainer(
            model_path="models/best_model.pkl",
            feature_names_path="models/feature_names.json"
        )
        print("✅ Explainer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize explainer: {e}")
        return
    
    # Prepare background data
    print("\n📚 Preparing background data...")
    explainer.prepare_background_data(X_train, n_samples=100)
    
    # Generate report
    print("\n📈 Generating explainability report...")
    report = explainer.generate_explainability_report(X_test, output_dir=output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 Report Summary")
    print("=" * 60)
    
    print("\n🔝 Top 5 Most Important Features:")
    print(report['importance_df'].head(5).to_string(index=False))
    
    print("\n📊 Sample Explanations Generated:")
    for i, exp in enumerate(report['explanations'][:3]):
        print(f"\n  Customer {exp['sample_idx']}:")
        print(f"    Risk: {exp['risk_category']} ({exp['probability']:.1%})")
        print(f"    Top driver: {exp['contributions'][0]['feature']} "
              f"({exp['contributions'][0]['shap_value']:+.3f})")
    
    print(f"\n✅ Report saved to: {output_dir}")
    print("\n📁 Generated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
    
    print("\n" + "=" * 60)
    print("🎉 SHAP explainability report complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()