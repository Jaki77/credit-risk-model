"""
FastAPI Application for Credit Risk Model API
Provides REST endpoints for model inference
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import CreditRiskPredictor
from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    LoanRecommendation,
    HealthCheck,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Model API",
    description="API for credit risk prediction and loan recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


def get_predictor():
    """Dependency to get predictor instance"""
    global predictor
    if predictor is None:
        try:
            # Load model and processor
            model_path = os.getenv("MODEL_PATH", "../models/best_model.joblib")
            processor_path = os.getenv(
                "PROCESSOR_PATH", "../data/processed/processor.joblib"
            )

            predictor = CreditRiskPredictor(model_path, processor_path)
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model initialization failed",
            )
    return predictor


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup"""
    logger.info("Starting Credit Risk Model API")
    get_predictor()


@app.get("/", response_model=HealthCheck, tags=["Health"])
async def root():
    """Root endpoint for health check"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="credit-risk-api",
        version="1.0.0",
    )


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Verify predictor is loaded
        pred = get_predictor()

        # Simple test prediction
        test_data = pd.DataFrame(
            {
                "CustomerId": ["TEST001"],
                "Recency": [30],
                "Frequency": [5],
                "Monetary": [1000],
            }
        )

        results = pred.batch_predict(test_data)

        return HealthCheck(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            service="credit-risk-api",
            version="1.0.0",
            model_loaded=True,
            model_name=(
                pred.model_info.get("model_name", "unknown")
                if pred.model_info
                else "unknown"
            ),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest, predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Predict credit risk for a single customer

    - **customer_id**: Unique customer identifier
    - **recency**: Days since last transaction
    - **frequency**: Number of transactions
    - **monetary**: Total transaction value
    - **monthly_income**: Optional monthly income for loan calculation
    """
    try:
        logger.info(f"Prediction request for customer: {request.customer_id}")

        # Convert request to DataFrame
        customer_data = pd.DataFrame(
            [
                {
                    "CustomerId": request.customer_id,
                    "Recency": request.recency,
                    "Frequency": request.frequency,
                    "Monetary": request.monetary,
                }
            ]
        )

        # Prepare features and predict
        features = predictor.prepare_features(customer_data)
        predictions = predictor.predict_risk_probability(features)

        if predictions.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to generate predictions",
            )

        # Get risk probability
        risk_prob = predictions.iloc[0]["risk_probability"]

        # Get loan recommendation
        recommendation = predictor.predict_loan_terms(
            customer_id=request.customer_id,
            risk_probability=risk_prob,
            monthly_income=request.monthly_income,
        )

        # Prepare response
        response = PredictionResponse(
            customer_id=request.customer_id,
            risk_probability=float(risk_prob),
            credit_score=recommendation["credit_score"],
            risk_category=recommendation["risk_category"],
            approval_recommendation=recommendation["approval_recommendation"],
            loan_recommendation=LoanRecommendation(
                amount=recommendation["recommended_loan_amount"],
                duration_months=recommendation["recommended_duration_months"],
                interest_rate=recommendation["estimated_interest_rate"],
                monthly_payment=recommendation["estimated_monthly_payment"],
            ),
            confidence_score=recommendation["confidence_score"],
            timestamp=recommendation["timestamp"],
        )

        logger.info(
            f"Prediction completed for {request.customer_id}: "
            f"score={response.credit_score}, risk={response.risk_category}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"]
)
async def batch_predict(
    request: BatchPredictionRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
):
    """
    Batch predict credit risk for multiple customers

    - **customers**: List of customer data for prediction
    """
    try:
        logger.info(f"Batch prediction request for {len(request.customers)} customers")

        # Convert request to DataFrame
        customer_data = pd.DataFrame(
            [
                {
                    "CustomerId": customer.customer_id,
                    "Recency": customer.recency,
                    "Frequency": customer.frequency,
                    "Monetary": customer.monetary,
                }
                for customer in request.customers
            ]
        )

        # Make predictions
        results = predictor.batch_predict(customer_data)

        # Convert to response format
        predictions = []
        for _, row in results.iterrows():
            predictions.append(
                PredictionResponse(
                    customer_id=row["customer_id"],
                    risk_probability=row["risk_probability"],
                    credit_score=row["credit_score"],
                    risk_category=row["risk_category"],
                    approval_recommendation=row["approval_recommendation"],
                    loan_recommendation=LoanRecommendation(
                        amount=row["recommended_loan_amount"],
                        duration_months=row["recommended_duration_months"],
                        interest_rate=row["estimated_interest_rate"],
                        monthly_payment=row["estimated_monthly_payment"],
                    ),
                    confidence_score=row["confidence_score"],
                    timestamp=row["timestamp"],
                )
            )

        response = BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=sum(1 for p in predictions if p.risk_probability >= 0.5),
            average_credit_score=np.mean([p.credit_score for p in predictions]),
            timestamp=datetime.now().isoformat(),
        )

        logger.info(
            f"Batch prediction completed: {response.high_risk_count} high risk customers"
        )

        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info(predictor: CreditRiskPredictor = Depends(get_predictor)):
    """Get information about the loaded model"""
    try:
        if predictor.model_info:
            return {
                "model_name": predictor.model_info.get("model_name", "unknown"),
                "feature_count": (
                    len(predictor.feature_columns) if predictor.feature_columns else 0
                ),
                "training_timestamp": predictor.model_info.get("timestamp", "unknown"),
                "metrics": predictor.model_info.get("metrics", {}),
                "score_parameters": {
                    "base_score": predictor.base_score,
                    "PDO": predictor.PDO,
                    "base_odds": predictor.base_odds,
                },
            }
        else:
            return {
                "message": "Model info not available",
                "model_loaded": predictor.model is not None,
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@app.get("/features", tags=["Model"])
async def get_feature_list(predictor: CreditRiskPredictor = Depends(get_predictor)):
    """Get list of features used by the model"""
    try:
        if predictor.feature_columns:
            return {
                "features": predictor.feature_columns,
                "feature_count": len(predictor.feature_columns),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Feature list not available",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get features: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    # Run the API
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
