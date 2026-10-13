import pandas as pd
from .base_preprocessor import BasePreprocessor

class TitanicPreprocessor(BasePreprocessor):
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Drop Cabin
        if "Cabin" in df.columns:
            df = df.drop(columns=["Cabin"])

        # 2. Fill missing Embarked with 'S'
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna("S")

        # 3. Fill missing Fare with mean
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

        # 4. Impute missing Age with median of (Sex, Pclass) group
        if "Age" in df.columns and "Sex" in df.columns and "Pclass" in df.columns:
            df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )
            df["Age"] = df["Age"].astype("int64")

        return df