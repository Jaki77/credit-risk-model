"""
Data Processing Module for Credit Risk Model
Handles feature engineering, preprocessing, and RFM analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import category_encoders as ce
from xverse.transformer import WOE
import joblib
import logging
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main class for data processing and feature engineering"""
    
    def __init__(self, snapshot_date: str = '2025-12-31'):
        """
        Initialize DataProcessor
        
        Args:
            snapshot_date: Date to calculate recency from (YYYY-MM-DD)
        """
        self.snapshot_date = pd.to_datetime(snapshot_date)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.preprocessor = None
        self.feature_columns = None
        self.categorical_cols = None
        self.numerical_cols = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw data from CSV"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial data cleaning and type conversion"""
        logger.info("Starting raw data preprocessing")
        
        # Convert TransactionStartTime to datetime and remove timezone info if present
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
            try:
                # .dt.tz is None for tz-naive Series, non-None for tz-aware
                if df['TransactionStartTime'].dt.tz is not None:
                    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_convert(None)
            except Exception:
                # fallback: ensure all timestamps are tz-naive
                df['TransactionStartTime'] = df['TransactionStartTime'].apply(
                    lambda ts: ts.tz_convert(None) if hasattr(ts, 'tz') and ts is not pd.NaT and ts.tz is not None else ts
                )
        
        # Ensure Amount is numeric
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Create Value column if not present
        if 'Value' not in df.columns and 'Amount' in df.columns:
            df['Value'] = df['Amount'].abs()
        
        # Handle missing CustomerId
        if 'CustomerId' in df.columns:
            df = df.dropna(subset=['CustomerId'])
            df['CustomerId'] = df['CustomerId'].astype(str)
        
        logger.info(f"After preprocessing: {df.shape}")
        return df
    
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer
        
        Recency: Days since last transaction
        Frequency: Number of transactions
        Monetary: Total transaction value
        """
        logger.info("Calculating RFM metrics")
        
        # Group by customer
        rfm_df = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (self.snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Value': 'Monetary'
        }).reset_index()
        
        # Handle edge cases
        rfm_df['Recency'] = rfm_df['Recency'].clip(lower=0)
        rfm_df['Frequency'] = rfm_df['Frequency'].clip(lower=1)
        rfm_df['Monetary'] = rfm_df['Monetary'].clip(lower=0.01)
        
        # Log transform to reduce skewness
        rfm_df['Recency_log'] = np.log1p(rfm_df['Recency'])
        rfm_df['Frequency_log'] = np.log1p(rfm_df['Frequency'])
        rfm_df['Monetary_log'] = np.log1p(rfm_df['Monetary'])
        
        logger.info(f"RFM calculated for {len(rfm_df)} customers")
        return rfm_df
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features per customer"""
        logger.info("Creating aggregate features")
        
        agg_features = df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'Value': ['sum', 'mean', 'std'],
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['CustomerId'] + [
            f'{col[0]}_{col[1]}' for col in agg_features.columns[1:]
        ]
        
        # Fill NaN values
        for col in agg_features.columns:
            if agg_features[col].dtype in ['float64', 'int64']:
                agg_features[col] = agg_features[col].fillna(agg_features[col].median())
        
        logger.info(f"Aggregate features created: {agg_features.shape[1]} columns")
        return agg_features
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from transaction data"""
        logger.info("Creating temporal features")
        
        temporal_features = []
        
        if 'TransactionStartTime' in df.columns:
            # Extract time components
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
            df['TransactionWeekOfYear'] = df['TransactionStartTime'].dt.isocalendar().week
            
            # Time-based aggregations
            temporal_agg = df.groupby('CustomerId').agg({
                'TransactionHour': ['mean', 'std', lambda x: x.mode()[0] if not x.mode().empty else 12],
                'TransactionDayOfWeek': ['mean', 'std'],
            }).reset_index()
            
            # Flatten columns
            temporal_agg.columns = ['CustomerId'] + [
                f'{col[0]}_{col[1]}' for col in temporal_agg.columns[1:]
            ]
            
            temporal_features.append(temporal_agg)
        
        if temporal_features:
            result = temporal_features[0]
            for tf in temporal_features[1:]:
                result = result.merge(tf, on='CustomerId', how='left')
            return result
        return pd.DataFrame({'CustomerId': df['CustomerId'].unique()})
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features"""
        logger.info("Creating behavioral features")
        
        # Calculate time between transactions
        df_sorted = df.sort_values(['CustomerId', 'TransactionStartTime'])
        df_sorted['TimeSinceLast'] = df_sorted.groupby('CustomerId')['TransactionStartTime'].diff().dt.total_seconds() / 3600  # hours
        
        behavioral_agg = df_sorted.groupby('CustomerId').agg({
            'TimeSinceLast': ['mean', 'std', 'min', 'max'],
            'ProductCategory': lambda x: x.nunique() if 'ProductCategory' in df.columns else 0,
            'ProviderId': lambda x: x.nunique() if 'ProviderId' in df.columns else 0,
            'ChannelId': lambda x: x.nunique() if 'ChannelId' in df.columns else 0,
        }).reset_index()
        
        behavioral_agg.columns = ['CustomerId'] + [
            f'Behavior_{col[0]}_{col[1]}' if col[1] != '<lambda>' else f'Behavior_{col[0]}_nunique'
            for col in behavioral_agg.columns[1:]
        ]
        
        return behavioral_agg
    
    def identify_high_risk_clusters(self, rfm_df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """
        Use KMeans clustering to identify high-risk customers
        
        Args:
            rfm_df: DataFrame with RFM metrics
            n_clusters: Number of clusters to create
        
        Returns:
            DataFrame with cluster labels
        """
        logger.info(f"Performing KMeans clustering with {n_clusters} clusters")
        
        # Use log-transformed features for clustering
        features = ['Recency_log', 'Frequency_log', 'Monetary_log']
        X = rfm_df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm_df['Cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Analyze clusters to identify high-risk
        cluster_stats = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        logger.info("Cluster Statistics:")
        logger.info(f"\n{cluster_stats}")
        
        # Identify high-risk cluster (high recency, low frequency, low monetary)
        cluster_stats['RiskScore'] = (
            cluster_stats['Recency'].rank(ascending=False) +  # High recency = risky
            cluster_stats['Frequency'].rank(ascending=True) +  # Low frequency = risky
            cluster_stats['Monetary'].rank(ascending=True)     # Low monetary = risky
        )
        
        high_risk_cluster = cluster_stats['RiskScore'].idxmax()
        logger.info(f"High-risk cluster identified: {high_risk_cluster}")
        
        # Create binary target
        rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
        
        risk_distribution = rfm_df['is_high_risk'].value_counts(normalize=True)
        logger.info(f"Risk distribution: {risk_distribution[1]:.2%} high risk")
        
        return rfm_df[['CustomerId', 'Cluster', 'is_high_risk']]
    
    def build_preprocessing_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """Build preprocessing pipeline with ColumnTransformer"""
        logger.info("Building preprocessing pipeline")
        
        # Identify column types
        self.categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove CustomerId if present
        if 'CustomerId' in self.categorical_cols:
            self.categorical_cols.remove('CustomerId')
        if 'CustomerId' in self.numerical_cols:
            self.numerical_cols.remove('CustomerId')
        
        logger.info(f"Categorical columns: {len(self.categorical_cols)}")
        logger.info(f"Numerical columns: {len(self.numerical_cols)}")
        
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', ce.TargetEncoder())
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        return self.preprocessor
    
    def apply_woe_encoding(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply Weight of Evidence encoding"""
        logger.info("Applying WoE encoding")
        
        # Initialize WoE transformer
        woe_transformer = WOE()
        
        # Fit and transform
        X_woe = woe_transformer.fit_transform(X, y)
        
        # Calculate Information Value
        iv_df = pd.DataFrame({
            'Variable': list(woe_transformer.iv_df['Variable']),
            'IV': list(woe_transformer.iv_df['IV'])
        })
        
        logger.info("Information Value Summary:")
        logger.info(f"\n{iv_df.sort_values('IV', ascending=False).head(10)}")
        
        # Select features with IV > 0.02 (at least weak predictive power)
        strong_features = iv_df[iv_df['IV'] > 0.02]['Variable'].tolist()
        logger.info(f"Selected {len(strong_features)} features with IV > 0.02")
        
        return X_woe[strong_features]
    
    def prepare_training_data(self, 
                             df: pd.DataFrame, 
                             target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare final training dataset by merging all features
        
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        logger.info("Preparing final training dataset")
        
        # Calculate all features
        rfm_features = self.calculate_rfm(df)
        aggregate_features = self.create_aggregate_features(df)
        temporal_features = self.create_temporal_features(df)
        behavioral_features = self.create_behavioral_features(df)
        
        # Merge all features
        X = rfm_features.merge(aggregate_features, on='CustomerId', how='left')
        X = X.merge(temporal_features, on='CustomerId', how='left')
        X = X.merge(behavioral_features, on='CustomerId', how='left')
        
        # Merge with target
        X = X.merge(target_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='inner')
        
        # Separate features and target
        y = X['is_high_risk']
        X = X.drop(['CustomerId', 'is_high_risk'], axis=1)
        
        # Store feature names
        self.feature_columns = X.columns.tolist()
        
        logger.info(f"Final dataset shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y, self.feature_columns
    
    def save_processor(self, filepath: str):
        """Save the processor object"""
        joblib.dump(self, filepath)
        logger.info(f"Processor saved to {filepath}")
    
    @staticmethod
    def load_processor(filepath: str) -> 'DataProcessor':
        """Load processor object"""
        return joblib.load(filepath)


def main():
    """Main function for data processing"""
    # Example usage
    processor = DataProcessor()
    
    # Load data
    df = processor.load_data('./data/raw/data.csv')
    
    # Preprocess
    df = processor.preprocess_raw_data(df)
    
    # Calculate RFM and identify high-risk
    rfm_df = processor.calculate_rfm(df)
    target_df = processor.identify_high_risk_clusters(rfm_df)
    
    # Prepare training data
    X, y, features = processor.prepare_training_data(df, target_df)
    
    # Save processed data
    X.to_csv('./data/processed/features.csv', index=False)
    y.to_csv('./data/processed/target.csv', index=False)
    
    # Save processor
    processor.save_processor('./data/processed/processor.joblib')
    logger.info("Data processing completed successfully!")


if __name__ == "__main__":
    main()