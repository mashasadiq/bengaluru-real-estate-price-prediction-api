
from pathlib import Path

import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
import logging

from app.schemas import HouseFeatures, PredictionResponse

from contextlib import asynccontextmanager

from app.services import ModelService

#FastAPI application, owns HTTP routing, request handling and HTTP error responses

logger = logging.getLogger(__name__)


# Locate and load the trained model artifact.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the project root.
load_dotenv(BASE_DIR / ".env")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

MODEL_PATH = BASE_DIR / "artifacts" / "real_estate_model.joblib"


model_service = ModelService(MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_model()
        logger.info("Prediction model loaded successfully.")

    except Exception:
        logger.exception("Failed to load prediction model.")
        raise
    yield



app = FastAPI(
    title="Bengaluru Real Estate Price Prediction API",
    description="""
    ## Overview
    An API that predicts Bengaluru residential property prices
    using a trained Random Forest regression pipeline.

    ## Features
    - Property price prediction.
    - Input validation using Pydantic.
    - Supported location validation.
    - Health monitoring.

    ## Prediction output
    Returns the predicted property price in Indian lakhs.

    ## Limitations
    - Predictions are based on the training dataset.
    - Only locations learned during training are supported.
    - Predictions are estimates, not guaranteed market prices.
    """,
    version="1.0.0",
    lifespan=lifespan
)




@app.get(
    "/",
    summary="API welcome endpoint",
    description="Confirms that the Real Estate Price Prediction API is running."
)
def home():
    return {"message": "Real Estate Price Prediction API is running."}

@app.get(
    "/health",
    summary="Check API health",
    description="Checks whether the prediction service and trained model are available."
)


def health():
    if not model_service.is_ready:
        logger.error("Health check failed: model is not loaded.")

        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable."
        )

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict property price",
    description="""
    Accepts residential property details and returns an estimated
    property price in Indian lakhs.

    Input validation is performed using the HouseFeatures schema.
    Unsupported locations are rejected with HTTP 422.
    Unexpected prediction failures return HTTP 500.
    """
)
def predict_price(house: HouseFeatures):
    #explicitly rejects unknown locations before they reach the model
    if house.location not in model_service.supported_locations:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported location: {house.location}"
        )


    try:
        prediction = model_service.predict_price(house)

        return PredictionResponse(
            predicted_price_lakhs=round(float(prediction), 2)
        )

    except Exception:
        logger.exception("Unexpected error during price prediction.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error."
        )

