from pathlib import Path

import joblib
import pandas as pd

#owns the trained model, its supported locations and prediction operations/logic.


class ModelService:
    """Handles loading the trained model and generating predictions."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.supported_locations = set()

    @property
    def is_ready(self) -> bool:
        """Check whether the model has loaded successfully."""
        return self.model is not None

    def load_model(self) -> None:
        """Load the model and extract locations supported by its encoder."""
        model = joblib.load(self.model_path)

        location_encoder = (
            model.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        locations = {
            str(location).strip()
            for location in location_encoder.categories_[1]
        }

        # Assign only after loading and extraction have succeeded.
        self.model = model
        self.supported_locations = locations

    def predict_price(self, house) -> float:
        """Generate a price prediction for a validated house input."""
        if not self.is_ready:
            raise RuntimeError("The prediction model is not loaded.")

        input_data = {
            "area_type": house.area_type,
            "location": house.location,
            "bedrooms": house.bedrooms,
            "is_rk": int(house.is_rk),
            "total_sqft": house.total_sqft,
            "bath": house.bath,
            "balcony": house.balcony,
            "balcony_missing": int(house.balcony is None),
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=self.model.feature_names_in_,
        )

        prediction = self.model.predict(input_df)[0]

        return float(prediction)