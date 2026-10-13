from abc import ABC, abstractmethod
import pandas as pd

class BaseFeaturesComputer(ABC):
    """Abstract base class for the feature engineering step."""
    
    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a preprocessed DataFrame and return one with engineered features."""