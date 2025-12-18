"""
Prediction Module for Credit Risk Model
Handles inference and probability to score conversion
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    """Main class for making predictions and converting to credit scores"""
    
    def __init__(self, model_path: str = None, processor_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            processor_path: Path to data processor
        """
        self.model = None
        self.model_info = None
        self.processor = None
        self.feature_columns = None
        
        if model_path:
            self.load_model(model_path)
        
        if processor_path:
            self.load_processor(processor_path)
        
        # Score mapping parameters
        self.PDO = 20  # Points to Double Odds
        self.base_score = 600  # Base score at odds of 50:1
        self.base_odds = 50  # Base odds
    
    def load_model(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Try to load model info
        info_path = model_path.replace('.joblib', '_info.joblib')
        try:
            self.model_info = joblib.load(info_path)
            self.feature_columns = self.model_info.get('feature_columns', [])
            logger.info(f"Loaded model info: {self.model_info.get('model_name', 'Unknown')}")
        except:
            logger.warning(f"Could not load model info from {info_path}")
    
    def load_processor(self, processor_path: str):
        """Load data processor"""
        logger.info(f"Loading processor from {processor_path}")
        self.processor = joblib.load(processor_path)
    
    def prepare_features(self, customer_data: pd.DataFrame, 
                        transaction_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            customer_data: DataFrame with customer information
            transaction_data: Optional raw transaction data for feature engineering
        
        Returns:
            Prepared feature DataFrame
        """
        logger.info("Preparing features for prediction")
        
        if transaction_data is not None and self.processor is not None:
            # Use processor to create features from transaction data
            df = self.processor.preprocess_raw_data(transaction_data)
            rfm_features = self.processor.calculate_rfm(df)
            
            # Get customer IDs from input
            customer_ids = customer_data['CustomerId'].unique()
            
            # Filter to only relevant customers
            rfm_features = rfm_features[rfm_features['CustomerId'].isin(customer_ids)]
            
            # Create all features
            aggregate_features = self.processor.create_aggregate_features(df)
            temporal_features = self.processor.create_temporal_features(df)
            behavioral_features = self.processor.create_behavioral_features(df)
            
            # Merge all features
            features = rfm_features.merge(aggregate_features, on='CustomerId', how='left')
            features = features.merge(temporal_features, on='CustomerId', how='left')
            features = features.merge(behavioral_features, on='CustomerId', how='left')
            
            # Ensure we have the expected columns
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(features.columns)
                if missing_cols:
                    logger.warning(f"Adding missing columns: {missing_cols}")
                    for col in missing_cols:
                        features[col] = 0
            
            # Select only the columns used during training
            if self.feature_columns:
                features = features[['CustomerId'] + self.feature_columns]
            
            return features
        
        else:
            # Use provided customer data directly
            if 'CustomerId' not in customer_data.columns:
                raise ValueError("Customer data must contain 'CustomerId' column")
            
            # Ensure we have required columns
            required_cols = ['Recency', 'Frequency', 'Monetary']
            for col in required_cols:
                if col not in customer_data.columns:
                    raise ValueError(f"Customer data must contain '{col}' column")
            
            # Use provided data as features
            features = customer_data.copy()
            
            # Add log-transformed RFM features if not present
            for col in required_cols:
                log_col = f'{col}_log'
                if log_col not in features.columns:
                    features[log_col] = np.log1p(features[col].clip(lower=0))
            
            # Ensure we have the expected columns
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(features.columns)
                if missing_cols:
                    logger.warning(f"Adding missing columns: {missing_cols}")
                    for col in missing_cols:
                        features[col] = 0
            
            # Select only the columns used during training
            if self.feature_columns:
                features = features[['CustomerId'] + self.feature_columns]
            
            return features
    
    def predict_risk_probability(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk probability for customers
        
        Args:
            features: Prepared feature DataFrame
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Making predictions for {len(features)} customers")
        
        # Extract features (excluding CustomerId)
        if 'CustomerId' in features.columns:
            X = features.drop('CustomerId', axis=1)
            customer_ids = features['CustomerId'].tolist()
        else:
            X = features.copy()
            customer_ids = list(range(len(features)))
        
        # Ensure columns are in correct order
        if hasattr(self.model, 'feature_names_in_'):
            expected_cols = self.model.feature_names_in_
            missing_cols = set(expected_cols) - set(X.columns)
            extra_cols = set(X.columns) - set(expected_cols)
            
            if missing_cols:
                logger.warning(f"Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0
            
            if extra_cols:
                logger.warning(f"Removing extra columns: {extra_cols}")
                X = X[expected_cols]
        
        # Make predictions
        try:
            probabilities = self.model.predict_proba(X)
            
            # For binary classification, get probability of positive class (high risk)
            if probabilities.shape[1] == 2:
                risk_probabilities = probabilities[:, 1]
            else:
                risk_probabilities = probabilities[:, 0]
            
            # Create results DataFrame
            results = pd.DataFrame({
                'CustomerId': customer_ids,
                'risk_probability': risk_probabilities,
                'prediction': (risk_probabilities >= 0.5).astype(int)
            })
            
            logger.info(f"Predictions completed. High risk: {results['prediction'].sum()}")
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def probability_to_score(self, probability: float) -> int:
        """
        Convert risk probability to credit score using log-odds scaling
        
        Args:
            probability: Risk probability (0 to 1)
        
        Returns:
            Credit score (typically 300-850)
        """
        # Avoid edge cases
        eps = 1e-10
        probability = np.clip(probability, eps, 1 - eps)
        
        # Convert probability to odds
        odds = (1 - probability) / probability
        
        # Convert odds to score
        score = self.base_score + (self.PDO / np.log(2)) * np.log(odds / self.base_odds)
        
        # Clip to reasonable range (300-850)
        score = np.clip(score, 300, 850)
        
        return int(round(score))
    
    def predict_loan_terms(self, customer_id: str, risk_probability: float, 
                          monthly_income: Optional[float] = None) -> Dict:
        """
        Predict optimal loan amount and duration based on risk
        
        Args:
            customer_id: Customer identifier
            risk_probability: Risk probability (0-1)
            monthly_income: Optional monthly income for income-based calculation
        
        Returns:
            Dictionary with loan recommendations
        """
        logger.info(f"Predicting loan terms for customer {customer_id}")
        
        # Base loan parameters
        base_amount = 1000  # Base loan amount
        base_duration = 12  # Base duration in months
        
        # Adjust based on risk
        risk_adjustment = 1 - risk_probability  # Lower risk = higher adjustment
        
        # Calculate recommended amount
        if monthly_income:
            # Income-based calculation (typically up to 50% of annual income)
            max_amount = monthly_income * 12 * 0.5
            recommended_amount = min(base_amount * risk_adjustment * 2, max_amount)
        else:
            # Risk-based calculation
            recommended_amount = base_amount * risk_adjustment * 2
        
        # Calculate recommended duration (6-60 months)
        min_duration = 6
        max_duration = 60
        recommended_duration = int(base_duration * risk_adjustment * 2)
        recommended_duration = np.clip(recommended_duration, min_duration, max_duration)
        
        # Determine interest rate based on risk
        if risk_probability < 0.2:
            interest_rate = 0.08  # 8% for low risk
        elif risk_probability < 0.5:
            interest_rate = 0.12  # 12% for medium risk
        else:
            interest_rate = 0.18  # 18% for high risk
        
        # Calculate monthly payment
        monthly_rate = interest_rate / 12
        monthly_payment = (recommended_amount * monthly_rate * 
                         (1 + monthly_rate) ** recommended_duration) / \
                         ((1 + monthly_rate) ** recommended_duration - 1)
        
        # Create recommendation
        recommendation = {
            'customer_id': customer_id,
            'risk_probability': float(risk_probability),
            'credit_score': self.probability_to_score(risk_probability),
            'risk_category': self._get_risk_category(risk_probability),
            'recommended_loan_amount': round(recommended_amount, 2),
            'recommended_duration_months': recommended_duration,
            'estimated_interest_rate': round(interest_rate * 100, 2),
            'estimated_monthly_payment': round(monthly_payment, 2),
            'debt_to_income_ratio': round(monthly_payment / monthly_income * 100, 2) 
            if monthly_income else None,
            'approval_recommendation': 'APPROVE' if risk_probability < 0.5 else 'REVIEW',
            'confidence_score': round(1 - risk_probability, 4),
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendation
    
    def _get_risk_category(self, probability: float) -> str:
        """Convert probability to risk category"""
        if probability < 0.2:
            return 'LOW RISK'
        elif probability < 0.4:
            return 'MEDIUM-LOW RISK'
        elif probability < 0.6:
            return 'MEDIUM RISK'
        elif probability < 0.8:
            return 'MEDIUM-HIGH RISK'
        else:
            return 'HIGH RISK'
    
    def batch_predict(self, customer_data: pd.DataFrame, 
                     transaction_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make batch predictions for multiple customers
        
        Args:
            customer_data: Customer information DataFrame
            transaction_data: Optional transaction data
        
        Returns:
            DataFrame with predictions and recommendations
        """
        logger.info(f"Starting batch prediction for {len(customer_data)} customers")
        
        # Prepare features
        features = self.prepare_features(customer_data, transaction_data)
        
        # Predict risk probabilities
        predictions = self.predict_risk_probability(features)
        
        # Convert to scores and get loan recommendations
        results = []
        for _, row in predictions.iterrows():
            recommendation = self.predict_loan_terms(
                customer_id=row['CustomerId'],
                risk_probability=row['risk_probability']
            )
            results.append(recommendation)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        logger.info(f"Batch prediction completed. Results shape: {results_df.shape}")
        return results_df
    
    def save_predictions(self, predictions: pd.DataFrame, filepath: str):
        """Save predictions to file"""
        predictions.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")


def main():
    """Main function for standalone prediction"""
    # Example usage
    predictor = CreditRiskPredictor(
        model_path='../models/best_model.joblib',
        processor_path='../data/processed/processor.joblib'
    )
    
    # Example customer data
    example_customers = pd.DataFrame({
        'CustomerId': ['CUST001', 'CUST002', 'CUST003'],
        'Recency': [30, 180, 10],
        'Frequency': [5, 2, 20],
        'Monetary': [1000, 200, 5000]
    })
    
    # Make predictions
    results = predictor.batch_predict(example_customers)
    
    # Display results
    print("\n" + "="*60)
    print("CREDIT RISK PREDICTION RESULTS")
    print("="*60)
    print(results[['customer_id', 'credit_score', 'risk_category', 
                   'recommended_loan_amount', 'approval_recommendation']].to_string())
    print("="*60)
    
    # Save results
    predictor.save_predictions(results, '../data/predictions/sample_predictions.csv')


if __name__ == "__main__":
    main()