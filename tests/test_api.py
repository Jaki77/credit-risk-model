"""
Unit tests for API
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app
from src.api.pydantic_models import CustomerData, PredictionResponse

client = TestClient(app)


class TestAPI:
    def test_health_endpoint(self):
        """Test that health check endpoint works"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_predict_endpoint_returns_expected_structure(self):
        """Test that prediction endpoint returns correct response structure"""
        test_input = {
            "customer_id": "test_123",
            "recency": 5,
            "frequency": 10,
            "monetary": 500.0,
            "total_transaction_amount": 5000.0,
            "avg_transaction_amount": 500.0,
            "std_transaction_amount": 100.0,
        }

        response = client.post("/predict", json=test_input)
        assert response.status_code == 200

        data = response.json()
        assert "customer_id" in data
        assert "risk_probability" in data
        assert "risk_category" in data
        assert "prediction_timestamp" in data

    def test_predict_endpoint_validates_input(self):
        """Test that API validates required fields"""
        # Missing required field
        invalid_input = {
            "customer_id": "test_123",
            "recency": 5,
            # Missing frequency, monetary, etc.
        }

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint if implemented"""
        test_input = {
            "customers": [
                {
                    "customer_id": "test_1",
                    "recency": 5,
                    "frequency": 10,
                    "monetary": 500.0,
                    "total_transaction_amount": 5000.0,
                    "avg_transaction_amount": 500.0,
                    "std_transaction_amount": 100.0,
                },
                {
                    "customer_id": "test_2",
                    "recency": 30,
                    "frequency": 2,
                    "monetary": 50.0,
                    "total_transaction_amount": 100.0,
                    "avg_transaction_amount": 50.0,
                    "std_transaction_amount": 0.0,
                },
            ]
        }

        response = client.post("/batch_predict", json=test_input)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
