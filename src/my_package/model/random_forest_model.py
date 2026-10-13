import pandas as pd
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from .base_model import BaseModel


class RandomForestTitanicModel(BaseModel):
    def __init__(self):
        self.pipeline = None

    def _build_pipeline(self) -> Pipeline:
        """Recreate the preprocessing pipeline with a Random Forest classifier."""
        num_cat_transformation = ColumnTransformer(
            transformers=[
                ("scaling", MinMaxScaler(), ["Age", "Fare"]),
                ("onehot1", OneHotEncoder(handle_unknown="ignore"), ["Embarked", "Pclass"]),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    ["Sex"],
                ),
                (
                    "onehot2",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Title", "Family_size"],
                ),
            ],
            remainder="passthrough",
        )

        bins = ColumnTransformer(
            transformers=[
                (
                    "kbins",
                    KBinsDiscretizer(
                        n_bins=15, encode="ordinal", strategy="quantile"
                    ),
                    [0, 1],
                ),
            ],
            remainder="passthrough",
        )

        # Using Random Forest instead of Logistic Regression
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        return Pipeline(
            [
                ("num_cat_transformation", num_cat_transformation),
                ("bins", bins),
                ("classifier", classifier),
            ]
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train the model on the provided data."""
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        return self.pipeline
        
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions using the trained model."""
        if self.pipeline is None:
            raise ValueError("The model must be trained before predicting.")
        return self.pipeline.predict(X)