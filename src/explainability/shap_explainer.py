"""
SHAP-based model explainability module for credit risk prediction
Provides global and local interpretations of model decisions
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class CreditRiskExplainer:
    """
    SHAP explainer for credit risk model
    """
    
    def __init__(self, model_path: str = "models/best_model.pkl", 
                 feature_names_path: str = "models/feature_names.json"):
        """
        Initialize the explainer with trained model
        
        Args:
            model_path: Path to trained model
            feature_names_path: Path to feature names JSON
        """
        self.model = self._load_model(model_path)
        self.feature_names = self._load_feature_names(feature_names_path)
        self.explainer = None
        self.shap_values = None
        self.background_data = None
        
    def _load_model(self, model_path: str):
        """Load trained model"""
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    def _load_feature_names(self, feature_names_path: str) -> List[str]:
        """Load feature names"""
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                return json.load(f)
        else:
            # Default feature names if file not found
            return ['Recency', 'Frequency', 'Monetary', 
                    'TotalTransactionAmount', 'AvgTransactionAmount', 
                    'StdTransactionAmount']
    
    def prepare_background_data(self, X: pd.DataFrame, n_samples: int = 100):
        """
        Prepare background data for SHAP explainer
        
        Args:
            X: Training data
            n_samples: Number of background samples to use
        """
        if len(X) > n_samples:
            self.background_data = shap.sample(X, n_samples)
        else:
            self.background_data = X
        
        # Create explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            # For scikit-learn models
            self.explainer = shap.TreeExplainer(self.model) if 'tree' in str(type(self.model)).lower() else \
                             shap.KernelExplainer(self.model.predict_proba, self.background_data)
        else:
            # For other model types
            self.explainer = shap.Explainer(self.model, self.background_data)
        
        return self
    
    def calculate_shap_values(self, X: pd.DataFrame):
        """
        Calculate SHAP values for predictions
        
        Args:
            X: Data to explain
        """
        if self.explainer is None:
            self.prepare_background_data(X)
        
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            # For tree models with multiple outputs
            self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
        
        return self.shap_values
    
    def get_global_feature_importance(self) -> pd.DataFrame:
        """
        Calculate global feature importance (mean |SHAP value|)
        
        Returns:
            DataFrame with feature importance sorted
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values first.")
        
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap,
            'importance_pct': mean_shap / mean_shap.sum() * 100
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_global_feature_importance(self, save_path: Optional[str] = None):
        """
        Plot global feature importance bar chart
        
        Args:
            save_path: Optional path to save the figure
        """
        importance_df = self.get_global_feature_importance()
        
        plt.figure(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
        
        bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
        plt.xlabel('Mean |SHAP Value| (Impact on Model Output)')
        plt.title('Global Feature Importance - What Drives Credit Risk?')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create SHAP summary plot (beeswarm)
        
        Args:
            X: Data used for SHAP calculation
            save_path: Optional path to save the figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_dependence(self, feature: str, X: pd.DataFrame, 
                       interaction_feature: Optional[str] = None,
                       save_path: Optional[str] = None):
        """
        Create SHAP dependence plot for a specific feature
        
        Args:
            feature: Feature to plot
            X: Data used for SHAP calculation
            interaction_feature: Feature for color interaction
            save_path: Optional path to save the figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        feature_idx = self.feature_names.index(feature)
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature:
            interaction_idx = self.feature_names.index(interaction_feature)
            shap.dependence_plot(feature_idx, self.shap_values, X, 
                               feature_names=self.feature_names,
                               interaction_index=interaction_idx, show=False)
        else:
            shap.dependence_plot(feature_idx, self.shap_values, X,
                               feature_names=self.feature_names, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def explain_prediction(self, X_sample: pd.DataFrame, 
                          sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            X_sample: Sample data
            sample_idx: Index of sample to explain
        
        Returns:
            Dictionary with explanation details
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_sample)
        
        # Get single instance
        if isinstance(X_sample, pd.DataFrame):
            x_instance = X_sample.iloc[sample_idx:sample_idx+1]
        else:
            x_instance = X_sample[sample_idx:sample_idx+1]
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(x_instance)[0][1]
            prediction = self.model.predict(x_instance)[0]
        else:
            proba = self.model.predict(x_instance)[0]
            prediction = 1 if proba > 0.5 else 0
        
        # Get SHAP values for this instance
        if isinstance(self.shap_values, list):
            instance_shap = self.shap_values[1][sample_idx] if len(self.shap_values) > 1 else self.shap_values[0][sample_idx]
        else:
            instance_shap = self.shap_values[sample_idx]
        
        # Create feature contributions
        contributions = []
        for i, feature in enumerate(self.feature_names):
            contributions.append({
                'feature': feature,
                'value': x_instance[feature].values[0] if isinstance(x_instance, pd.DataFrame) else x_instance[0][i],
                'shap_value': instance_shap[i],
                'impact': 'positive' if instance_shap[i] > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        contributions = sorted(contributions, 
                             key=lambda x: abs(x['shap_value']), 
                             reverse=True)
        
        # Calculate base value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        return {
            'sample_idx': sample_idx,
            'prediction': int(prediction),
            'probability': float(proba),
            'risk_category': 'High Risk' if proba > 0.6 else 'Medium Risk' if proba > 0.3 else 'Low Risk',
            'base_value': float(expected_value),
            'final_value': float(expected_value + instance_shap.sum()),
            'contributions': contributions
        }
    
    def plot_waterfall(self, X_sample: pd.DataFrame, sample_idx: int = 0,
                      save_path: Optional[str] = None):
        """
        Create waterfall plot for a single prediction
        
        Args:
            X_sample: Sample data
            sample_idx: Index of sample to explain
            save_path: Optional path to save the figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_sample)
        
        # Get single instance
        if isinstance(X_sample, pd.DataFrame):
            x_instance = X_sample.iloc[sample_idx:sample_idx+1]
        else:
            x_instance = X_sample[sample_idx:sample_idx+1]
        
        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        # Get SHAP values for this instance
        if isinstance(self.shap_values, list):
            instance_shap = self.shap_values[1][sample_idx] if len(self.shap_values) > 1 else self.shap_values[0][sample_idx]
        else:
            instance_shap = self.shap_values[sample_idx]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        feature_names = self.feature_names
        feature_values = x_instance.values[0] if isinstance(x_instance, pd.DataFrame) else x_instance[0]
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(instance_shap))[::-1]
        
        # Plot waterfall
        fig = shap.plots._waterfall.waterfall_legacy(expected_value, instance_shap, 
                                                      features=feature_values,
                                                      feature_names=feature_names,
                                                      max_display=10, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_explainability_report(self, X: pd.DataFrame, 
                                       output_dir: str = "reports/shap"):
        """
        Generate comprehensive explainability report
        
        Args:
            X: Data to analyze
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # 1. Global feature importance
        importance_df = self.get_global_feature_importance()
        importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
        
        # 2. Global feature importance plot
        self.plot_global_feature_importance(save_path=f"{output_dir}/global_importance.png")
        
        # 3. Summary plot
        self.plot_summary(X, save_path=f"{output_dir}/summary_plot.png")
        
        # 4. Dependence plots for top features
        top_features = importance_df['feature'].head(3).tolist()
        for feature in top_features:
            self.plot_dependence(feature, X, save_path=f"{output_dir}/dependence_{feature}.png")
        
        # 5. Explain a few sample predictions
        explanations = []
        for idx in [0, 10, 25, 50, 75, 99]:  # Sample different indices
            if idx < len(X):
                explanation = self.explain_prediction(X, idx)
                explanations.append(explanation)
        
        # Save explanations
        with open(f"{output_dir}/sample_explanations.json", 'w') as f:
            # Convert to serializable format
            serializable_explanations = []
            for exp in explanations:
                exp_copy = exp.copy()
                # Convert contributions to serializable
                exp_copy['contributions'] = [
                    {k: float(v) if isinstance(v, np.floating) else v 
                     for k, v in contrib.items()}
                    for contrib in exp['contributions']
                ]
                serializable_explanations.append(exp_copy)
            
            json.dump(serializable_explanations, f, indent=2)
        
        print(f"✅ Explainability report generated in {output_dir}")
        return {
            'importance_df': importance_df,
            'explanations': explanations,
            'output_dir': output_dir
        }