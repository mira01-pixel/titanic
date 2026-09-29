import argparse
import json
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained Titanic model")
    p.add_argument(
        "--model", required=True, help="Path to saved model pickle file"
    )
    p.add_argument(
        "--input", required=True, help="Path to evaluation featurized CSV file"
    )
    p.add_argument(
        "--output", required=False, help="Optional path to save metrics JSON"
    )
    return p


def main():
    args = build_parser().parse_args()

    # 1. Load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # 2. Load evaluation dataset
    df = pd.read_csv(args.input)
    if "Survived" not in df.columns:
        raise ValueError("Evaluation data must contain the 'Survived' column.")

    X_eval = df.drop(columns=["Survived"])
    y_eval = df["Survived"].astype(int)

    # 3. Predict and compute accuracy (Cell In[52])
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_eval, y_pred)

    metrics = {"accuracy": float(acc)}
    print(f"Evaluation Accuracy: {acc:.4f}")

    # 4. Save metrics to JSON if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
