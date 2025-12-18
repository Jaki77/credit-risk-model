"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk categories for credit scoring"""
    LOW_RISK = "LOW RISK"
    MEDIUM_LOW_RISK = "MEDIUM-LOW RISK"
    MEDIUM_RISK = "MEDIUM RISK"
    MEDIUM_HIGH_RISK = "MEDIUM-HIGH RISK"
    HIGH_RISK = "HIGH RISK"


class ApprovalRecommendation(str, Enum):
    """Loan approval recommendations"""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class CustomerData(BaseModel):
    """Single customer data for prediction"""
    customer_id: str = Field(..., description="Unique customer identifier")
    recency: float = Field(..., ge=0, description="Days since last transaction")
    frequency: float = Field(..., ge=1, description="Number of transactions")
    monetary: float = Field(..., ge=0, description="Total transaction value")
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income (optional)")
    
    @validator('recency', 'frequency', 'monetary')
    def validate_positive(cls, v, field):
        if v < 0:
            raise ValueError(f"{field.name} must be positive")
        return v


class LoanRecommendation(BaseModel):
    """Loan recommendation details"""
    amount: float = Field(..., description="Recommended loan amount")
    duration_months: int = Field(..., description="Recommended loan duration in months")
    interest_rate: float = Field(..., description="Estimated interest rate (%)")
    monthly_payment: float = Field(..., description="Estimated monthly payment")
    
    @validator('amount', 'monthly_payment')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount must be positive")
        return round(v, 2)
    
    @validator('interest_rate')
    def validate_interest_rate(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Interest rate must be between 0 and 100")
        return round(v, 2)


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    customer_id: str = Field(..., description="Unique customer identifier")
    recency: float = Field(..., ge=0, description="Days since last transaction")
    frequency: float = Field(..., ge=1, description="Number of transactions")
    monetary: float = Field(..., ge=0, description="Total transaction value")
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income (optional)")


class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    customer_id: str = Field(..., description="Unique customer identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of high risk (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_category: RiskCategory = Field(..., description="Risk category")
    approval_recommendation: ApprovalRecommendation = Field(..., description="Loan approval recommendation")
    loan_recommendation: LoanRecommendation = Field(..., description="Loan recommendation details")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence score")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "risk_probability": 0.15,
                "credit_score": 750,
                "risk_category": "LOW RISK",
                "approval_recommendation": "APPROVE",
                "loan_recommendation": {
                    "amount": 15000.0,
                    "duration_months": 24,
                    "interest_rate": 8.5,
                    "monthly_payment": 678.92
                },
                "confidence_score": 0.92,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    customers: List[CustomerData] = Field(..., description="List of customer data")
    
    @validator('customers')
    def validate_customers(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 customers")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total number of customers processed")
    high_risk_count: int = Field(..., description="Number of high-risk customers")
    average_credit_score: float = Field(..., description="Average credit score")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total_customers": 3,
                "high_risk_count": 1,
                "average_credit_score": 650.0,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_loaded: Optional[bool] = Field(None, description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Name of loaded model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "service": "credit-risk-api",
                "version": "1.0.0",
                "model_loaded": True,
                "model_name": "xgboost"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Recency must be positive",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }