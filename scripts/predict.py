import argparse
from pathlib import Path
import pickle
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run inference using trained Titanic model")
    p.add_argument(
        "--model", required=True, help="Path to saved model pickle file"
    )
    p.add_argument(
        "--input", required=True, help="Path to input features CSV file (unlabeled)"
    )
    p.add_argument(
        "--output", required=True, help="Path to save predictions CSV file"
    )
    return p


def main():
    args = build_parser().parse_args()

    # 1. Load trained model pipeline
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # 2. Load unlabeled test features
    df = pd.read_csv(args.input)

    # Ensure no label column leaks into inference
    X_test = df.drop(columns=["Survived"], errors="ignore")

    # 3. Generate predictions (0 = Did not survive, 1 = Survived)
    predictions = model.predict(X_test)

    # 4. Save predictions to CSV
    output_df = pd.DataFrame({"prediction": predictions})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Predictions successfully saved to: {output_path}")


if __name__ == "__main__":
    main()
