import argparse
from pathlib import Path
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract features for Titanic model")
    p.add_argument("--input", required=True, help="Path to preprocessed CSV file")
    p.add_argument("--output", required=True, help="Path to save featurized CSV file")
    return p


def categorize_family_size(size: int) -> str:
    """Categorize family size into Alone, Small, or Large."""
    if size == 1:
        return "Alone"
    elif 1 < size < 5:
        return "Small"
    else:
        return "Large"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Extract Title from Name (Cells In[20] & In[22])
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

    # 2. Calculate Family_size (Cells In[24] & In[26])
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["Family_size"] = df["SibSp"] + df["Parch"] + 1
        df["Family_size"] = df["Family_size"].apply(categorize_family_size)

    # 3. Drop redundant columns (Cells In[25] & In[35])
    drop_cols = ["Name", "Parch", "SibSp", "Ticket", "PassengerId"]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def main():
    args = build_parser().parse_args()

    # Load cleaned data
    df = pd.read_csv(args.input)

    # Featurize
    featured_df = extract_features(df)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_path, index=False)
    print(f"Featurized dataset saved to: {output_path}")


if __name__ == "__main__":
    main()