# Bengaluru Real Estate Price Prediction API

A machine learning API that predicts residential property prices in Bengaluru, India, using a trained Random Forest regression pipeline.

The application is built with **Python, Scikit-learn, FastAPI, Pydantic and Pandas**.

## 1. Project Overview

This project exposes a trained machine learning model through REST API endpoints, allowing client applications to submit property details and receive estimated property prices.

The model uses property characteristics such as:
- Area type
- Location
- Number of bedrooms
- Whether the property is an RK unit
- Total area in square feet
- Number of bathrooms
- Number of balconies

The trained preprocessing and prediction pipeline is stored as a Joblib artifact and loaded by the FastAPI application.

## 2. Technology Stack

- Python 3.12
- FastAPI
- Scikit-learn
- Pandas
- Pydantic
- Joblib
- Uvicorn

## 3. Project Structure

real_estate_price_prediction_api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── artifacts/
│   └── real_estate_model.joblib
├── tests/
│   └── test_main.py
├── requirements.txt
├── README.md
└── .gitignore

## 4. Model Information

The API uses a trained Random Forest regression pipeline.

The pipeline includes:
- Numerical feature preprocessing and imputation.
- Categorical feature preprocessing.
- One-hot encoding.
- Random Forest regression.

### Model evaluation

| Metric | Result |
|---|---:|
| MAE | 36.726 lakhs |
| RMSE | 110.058 lakhs |
| R² | 0.586 |

These results are from the model's held-out evaluation dataset.

## 5. Geographic Scope and Limitations

### Geographic scope
- The model was developed using Bengaluru residential property data.
- Predictions are restricted to locations learned during model training.
- Requests containing unsupported locations are rejected with HTTP 422.
- Supported locations include both frequent and infrequent categories learned by the fitted encoder.

### Model limitations
- Predictions are estimates and are not guaranteed market prices.
- Model accuracy depends on the quality and representativeness of the training dataset.
- Unusual properties and expensive properties may have larger prediction errors.
- Rare location categories are grouped by the fitted encoder, which limits the model's ability to learn their individual effects.
- The model has not been established as a real-time property valuation system.

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Checks service and model availability |
| POST | `/predict` | Predicts a property's estimated price |

### Prediction request

Example request:

```json
{
    "area_type": "Built-up Area",
    "location": "Whitefield",
    "bedrooms": 2,
    "is_rk": false,
    "total_sqft": 1200,
    "bath": 2,
    "balcony": 1
}
```

### Prediction response

```json
{
    "predicted_price_lakhs": 62.95
}
```

The response value is illustrative. Actual predictions depend on the submitted property details.

**Price unit:** Indian lakhs (1 lakh = INR 100,000).

### Input validation

The API validates:
- Area type against the four supported categories.
- Location against categories learned during training.
- Bedrooms: 1–10.
- Total area: 100–20,000 square feet.
- Bathrooms: 1–10, when provided.
- Balconies: 0–3, when provided.
- Unexpected request fields are rejected.

Optional bathroom and balcony values may be omitted.

## 7. Running Locally

### Prerequisites

- Python 3.12
- Git
- The trained model artifact in `artifacts/real_estate_model.joblib`

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

### Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

Run from the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 8. API Documentation

FastAPI automatically generates interactive API documentation.

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Use Swagger UI to submit prediction requests and inspect responses.

## 9. HTTP Status Codes

| Status code | Meaning |
|---|---|
| 200 | Request successful |
| 422 | Invalid input or unsupported location |
| 500 | Unexpected prediction error |
| 503 | Prediction service unavailable |


## 10. Automated Testing

The API uses pytest and FastAPI TestClient for automated testing.

The test suite currently contains 10 passing tests covering:

- API welcome endpoint.
- Model health checks.
- Successful price predictions.
- Missing required fields.
- Invalid area types.
- Invalid numerical inputs.
- Unexpected request fields.
- Unsupported locations.
- Internal prediction errors.
- Model unavailability.

Run the tests from the project root:

```bash
python -m pytest -v
```

## 11. Future Improvements

- Improved model evaluation and data quality.
- Model versioning and monitoring.
- Docker containerisation.
- CI/CD automation.
- Cloud deployment.
- Frontend integration.

## 12. Project Status

**Version:** 1.0.0

This is an experimental machine learning application developed for learning and demonstrating machine learning model development, API engineering and deployment workflows.