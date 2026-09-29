import argparse
from pathlib import Path
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean raw Titanic dataset")
    p.add_argument("--input", required=True, help="Path to input raw CSV file")
    p.add_argument("--output", required=True, help="Path to save cleaned CSV file")
    return p


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Drop Cabin (Cell In[8])
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])

    # 2. Fill missing Embarked with 'S' (Cell In[11])
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna("S")

    # 3. Fill missing Fare with mean (Cell In[12])
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    # 4. Impute missing Age with median of (Sex, Pclass) group (Cells In[18] & In[27])
    if "Age" in df.columns and "Sex" in df.columns and "Pclass" in df.columns:
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Age"] = df["Age"].astype("int64")

    return df


def main():
    args = build_parser().parse_args()

    # Load raw data
    raw_df = pd.read_csv(args.input)

    # Clean data
    cleaned_df = preprocess_data(raw_df)

    # Save cleaned data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Preprocessed data successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
