"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import DataProcessor


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing"""
    np.random.seed(42)

    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(100)]

    data = {
        "TransactionId": [f"TX{i:03d}" for i in range(100)],
        "CustomerId": [f"CUST{np.random.randint(1, 11):03d}" for _ in range(100)],
        "TransactionStartTime": dates,
        "Amount": np.random.uniform(-500, 500, 100),
        "ProductCategory": np.random.choice(
            ["tv", "airtime", "rata_bundles", "transport"], 100
        ),
        "ProviderId": np.random.choice(
            ["ProviderId_1", "ProviderId_2", "ProviderId_3", "ProviderId_4"], 100
        ),
        "ChannelId": np.random.choice(
            ["ChannelId_1", "ChannelId_2", "ChannelId_3"], 100
        ),
    }

    df = pd.DataFrame(data)
    df["Value"] = df["Amount"].abs()
    return df


@pytest.fixture
def data_processor():
    """Create DataProcessor instance"""
    return DataProcessor(snapshot_date="2024-12-31")


def test_preprocess_raw_data(data_processor, sample_transaction_data):
    """Test raw data preprocessing"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data.copy())

    # Check data types
    assert processed["TransactionStartTime"].dtype == "datetime64[ns]"
    assert processed["Amount"].dtype == "float64"
    assert processed["CustomerId"].dtype == "object"

    # Check no missing CustomerId
    assert processed["CustomerId"].isnull().sum() == 0

    # Check Value column created
    assert "Value" in processed.columns
    assert (processed["Value"] >= 0).all()


def test_calculate_rfm(data_processor, sample_transaction_data):
    """Test RFM calculation"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    rfm_df = data_processor.calculate_rfm(processed)

    # Check columns
    expected_columns = [
        "CustomerId",
        "Recency",
        "Frequency",
        "Monetary",
        "Recency_log",
        "Frequency_log",
        "Monetary_log",
    ]
    assert all(col in rfm_df.columns for col in expected_columns)

    # Check values are positive
    assert (rfm_df["Recency"] >= 0).all()
    assert (rfm_df["Frequency"] >= 1).all()
    assert (rfm_df["Monetary"] > 0).all()

    # Check unique customers
    assert (
        rfm_df["CustomerId"].nunique()
        == sample_transaction_data["CustomerId"].nunique()
    )


def test_create_aggregate_features(data_processor, sample_transaction_data):
    """Test aggregate feature creation"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    agg_features = data_processor.create_aggregate_features(processed)

    # Check columns
    assert "CustomerId" in agg_features.columns
    assert "Amount_sum" in agg_features.columns
    assert "Amount_mean" in agg_features.columns
    assert "Amount_std" in agg_features.columns

    # Check no NaN values
    assert agg_features.isnull().sum().sum() == 0

    # Check customer count
    assert agg_features["CustomerId"].nunique() == processed["CustomerId"].nunique()


def test_create_temporal_features(data_processor, sample_transaction_data):
    """Test temporal feature creation"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    temporal_features = data_processor.create_temporal_features(processed)

    # Check columns
    assert "CustomerId" in temporal_features.columns

    # Check no NaN in key columns
    assert temporal_features["CustomerId"].isnull().sum() == 0

    # Check customer count
    assert (
        temporal_features["CustomerId"].nunique() == processed["CustomerId"].nunique()
    )


def test_identify_high_risk_clusters(data_processor, sample_transaction_data):
    """Test high-risk cluster identification"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    rfm_df = data_processor.calculate_rfm(processed)

    # Test with 3 clusters
    target_df = data_processor.identify_high_risk_clusters(rfm_df, n_clusters=3)

    # Check columns
    assert "CustomerId" in target_df.columns
    assert "Cluster" in target_df.columns
    assert "is_high_risk" in target_df.columns

    # Check cluster labels
    assert target_df["Cluster"].nunique() == 3

    # Check binary target
    assert set(target_df["is_high_risk"].unique()).issubset({0, 1})

    # Check all customers have target
    assert len(target_df) == rfm_df["CustomerId"].nunique()


def test_prepare_training_data(data_processor, sample_transaction_data):
    """Test training data preparation"""
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    rfm_df = data_processor.calculate_rfm(processed)
    target_df = data_processor.identify_high_risk_clusters(rfm_df)

    X, y, features = data_processor.prepare_training_data(processed, target_df)

    # Check shapes match
    assert len(X) == len(y)
    assert X.shape[1] == len(features)

    # Check target distribution
    assert y.nunique() == 2  # Binary classification
    assert set(y.unique()).issubset({0, 1})

    # Check feature names
    assert all(feat in X.columns for feat in features)

    # Check no NaN in features
    assert X.isnull().sum().sum() == 0


def test_save_load_processor(data_processor, sample_transaction_data, tmp_path):
    """Test processor serialization"""
    # Process some data first
    processed = data_processor.preprocess_raw_data(sample_transaction_data)
    data_processor.calculate_rfm(processed)  # Fit scaler

    # Save processor
    save_path = tmp_path / "processor.joblib"
    data_processor.save_processor(str(save_path))

    # Load processor
    loaded_processor = DataProcessor.load_processor(str(save_path))

    # Check attributes
    assert loaded_processor.snapshot_date == data_processor.snapshot_date
    assert hasattr(loaded_processor, "scaler")
    assert hasattr(loaded_processor, "kmeans") or loaded_processor.kmeans is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
