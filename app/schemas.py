
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#Request and response validation


class HouseFeatures(BaseModel):
    """Validated input features for a house-price prediction."""

    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "area_type": "Built-up Area",
                    "location": "Whitefield",
                    "bedrooms": 2,
                    "is_rk": False,
                    "total_sqft": 1200,
                    "bath": 2,
                    "balcony": 1
                }
            ]
        }
    )


    area_type: Literal[
        "Super built-up Area",
        "Built-up Area",
        "Plot Area",
        "Carpet Area"
    ]

    location: str = Field(min_length=1)

    bedrooms: int = Field(ge=1, le=10)

    is_rk: bool

    total_sqft: float = Field(ge=100, le=20_000)

    bath: float | None = Field(default=None, ge=1, le=10)

    balcony: int | None = Field(default=None, ge=0, le=3)


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    predicted_price_lakhs: float
