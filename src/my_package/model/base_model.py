from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class BaseModel(ABC):
    """Abstract base class for the machine learning model."""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train the model on the provided data."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions using the trained model."""
        pass