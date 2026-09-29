import argparse
from pathlib import Path
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Titanic survival baseline model")
    p.add_argument(
        "--input", required=True, help="Path to featurized training CSV file"
    )
    p.add_argument(
        "--output", required=True, help="Path to save the trained model pickle file"
    )
    return p


def build_pipeline() -> Pipeline:
    """Recreate the exact preprocessing and binning pipeline from the notebook."""
    # 1. Scaling, One-Hot, and Ordinal encodings (Cell In[38])
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

    # 2. Quantile binning for Age and Fare (Cell In[39])
    # Features 0 and 2 after num_cat_transformation correspond to scaled Age and Fare
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

    # 3. Baseline Classifier (Cell In[44] & Lab guidelines)
    classifier = LogisticRegression(max_iter=1000, random_state=42)

    # 4. Chain into full pipeline (Cell In[42])
    return Pipeline(
        [
            ("num_cat_transformation", num_cat_transformation),
            ("bins", bins),
            ("classifier", classifier),
        ]
    )


def main():
    args = build_parser().parse_args()

    # Load featurized dataset
    df = pd.read_csv(args.input)

    # Separate features and target (Cell In[36])
    if "Survived" not in df.columns:
        raise ValueError("Training dataset must contain 'Survived' column.")

    X_train = df.drop(columns=["Survived"])
    y_train = df["Survived"].astype(int)

    # Build and fit pipeline
    pipeline = build_pipeline()
    print("Training baseline model...")
    pipeline.fit(X_train, y_train)

    # Save fitted model with pickle
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
