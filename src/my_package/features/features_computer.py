import pandas as pd
from .base_features_computer import BaseFeaturesComputer

class TitanicFeaturesComputer(BaseFeaturesComputer):
    
    @staticmethod
    def _categorize_family_size(size: int) -> str:
        """Categorize family size into Alone, Small, or Large."""
        if size == 1:
            return "Alone"
        elif 1 < size < 5:
            return "Small"
        else:
            return "Large"

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Extract Title from Name
        if "Name" in df.columns:
            df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
            
            rare_titles = [
                "Lady", "the Countess", "Capt", "Col", "Don", 
                "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"
            ]
            df["Title"] = df["Title"].replace(rare_titles, "Rare")
            df["Title"] = df["Title"].replace("Mlle", "Miss")
            df["Title"] = df["Title"].replace("Ms", "Miss")
            df["Title"] = df["Title"].replace("Mme", "Mrs")

        # 2. Calculate Family_size
        if "SibSp" in df.columns and "Parch" in df.columns:
            df["Family_size"] = df["SibSp"] + df["Parch"] + 1
            df["Family_size"] = df["Family_size"].apply(self._categorize_family_size)

        # 3. Drop redundant columns
        drop_cols = ["Name", "Parch", "SibSp", "Ticket", "PassengerId"]
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df