import pytest
import pandas as pd
import numpy as np
import joblib
import os
from src.predict import load_model, predict_risk


class TestModel:
    @pytest.fixture
    def sample_model_input(self):
        """Create sample input for model prediction"""
        return pd.DataFrame(
            {
                "Recency": [5, 30],
                "Frequency": [10, 2],
                "Monetary": [500.0, 50.0],
                "TotalTransactionAmount": [5000.0, 100.0],
                "AvgTransactionAmount": [500.0, 50.0],
                "StdTransactionAmount": [100.0, 0.0],
            }
        )

    def test_model_loads_successfully(self):
        """Test that model can be loaded from expected path"""
        model_path = "models/best_model.pkl"

        # Skip if model doesn't exist (for CI environment)
        if not os.path.exists(model_path):
            pytest.skip(f"Model not found at {model_path}")

        model = load_model(model_path)
        assert model is not None

    def test_prediction_output_range(self, sample_model_input):
        """Test that predictions are probabilities between 0 and 1"""
        model_path = "models/best_model.pkl"

        if not os.path.exists(model_path):
            pytest.skip(f"Model not found at {model_path}")

        model = load_model(model_path)
        predictions = predict_risk(model, sample_model_input)

        assert all(0 <= pred <= 1 for pred in predictions)

    def test_prediction_reproducibility(self, sample_model_input):
        """Test that same input produces same output"""
        model_path = "models/best_model.pkl"

        if not os.path.exists(model_path):
            pytest.skip(f"Model not found at {model_path}")

        model = load_model(model_path)
        predictions_1 = predict_risk(model, sample_model_input)
        predictions_2 = predict_risk(model, sample_model_input)

        assert np.array_equal(predictions_1, predictions_2)

    def test_model_feature_importance_exists(self):
        """Test that model has feature importance attribute if applicable"""
        model_path = "models/best_model.pkl"

        if not os.path.exists(model_path):
            pytest.skip(f"Model not found at {model_path}")

        model = load_model(model_path)

        # Check for feature importance if model type supports it
        if hasattr(model, "feature_importances_"):
            assert len(model.feature_importances_) > 0
        elif hasattr(model, "coef_"):
            assert len(model.coef_[0]) > 0
