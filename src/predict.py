
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
CHAMPION_MODEL_PATH = MODEL_DIR / "champion_model.joblib"


# ---------------------------------------------------------
# Global model objects
# ---------------------------------------------------------
preprocessor = None
champion_model = None


# ---------------------------------------------------------
# Load models when API starts
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor
    global champion_model

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    if not CHAMPION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found: {CHAMPION_MODEL_PATH}"
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    champion_model = joblib.load(CHAMPION_MODEL_PATH)

    print("Preprocessor loaded successfully.")
    print("Champion model loaded successfully.")

    yield

    preprocessor = None
    champion_model = None


# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict whether a customer is likely to churn "
        "using the champion machine-learning model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------
class CustomerInput(BaseModel):
    age: int = Field(
        ...,
        ge=18,
        le=100,
        examples=[29],
    )

    monthly_charges: float = Field(
        ...,
        ge=0,
        examples=[85.50],
    )

    tenure_months: int = Field(
        ...,
        ge=0,
        examples=[4],
    )

    support_calls: int = Field(
        ...,
        ge=0,
        examples=[5],
    )

    contract_type: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    ]


# ---------------------------------------------------------
# Response schema
# ---------------------------------------------------------
class PredictionResponse(BaseModel):
    churn_prediction: int
    prediction_label: str
    churn_probability: float | None
    model_name: str


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "documentation": "/docs",
    }


# ---------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    models_loaded = (
        preprocessor is not None
        and champion_model is not None
    )

    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": champion_model is not None,
    }


# ---------------------------------------------------------
# Model-information endpoint
# ---------------------------------------------------------
@app.get("/model-info")
def model_information():
    if champion_model is None:
        raise HTTPException(
            status_code=503,
            detail="Champion model is not loaded.",
        )

    return {
        "model_class": champion_model.__class__.__name__,
        "preprocessor_class": preprocessor.__class__.__name__,
        "model_file": CHAMPION_MODEL_PATH.name,
        "preprocessor_file": PREPROCESSOR_PATH.name,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_churn(customer: CustomerInput):
    if preprocessor is None or champion_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model service is not ready.",
        )

    try:
        # Convert validated request into DataFrame
        input_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Apply saved preprocessing
        processed_data = preprocessor.transform(
            input_data
        )

        # Generate prediction
        prediction = int(
            champion_model.predict(
                processed_data
            )[0]
        )

        # Generate probability
        churn_probability = None

        if hasattr(champion_model, "predict_proba"):
            churn_probability = float(
                champion_model.predict_proba(
                    processed_data
                )[0][1]
            )

        prediction_label = (
            "Churn"
            if prediction == 1
            else "No Churn"
        )

        return PredictionResponse(
            churn_prediction=prediction,
            prediction_label=prediction_label,
            churn_probability=churn_probability,
            model_name=champion_model.__class__.__name__,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error