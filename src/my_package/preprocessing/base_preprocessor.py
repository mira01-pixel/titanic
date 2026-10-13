from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    """Abstract base class for the preprocessing step."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a DataFrame and return a transformed DataFrame."""