import argparse
import json
from pathlib import Path
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate and compare trained Titanic models")
    p.add_argument(
        "--models", nargs="+", required=True, help="Path(s) to saved model pickle file(s)"
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

    # Load evaluation dataset
    df = pd.read_csv(args.input)
    if "Survived" not in df.columns:
        raise ValueError("Evaluation data must contain the 'Survived' column.")

    X_eval = df.drop(columns=["Survived"])
    y_eval = df["Survived"].astype(int)

    metrics = {}
    print("\n--- Model Comparison ---")
    
    # Loop through all provided model paths
    for model_path_str in args.models:
        model_path = Path(model_path_str)
        model_name = model_path.stem  # Extracts 'xgboost_model' from 'models/xgboost_model.pkl'
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Predict and compute accuracy
        y_pred = model.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        
        metrics[model_name] = {"accuracy": float(acc)}
        print(f"{model_name:25s}: Accuracy = {acc:.4f}")

    print("------------------------\n")

    # Save all metrics to a single JSON if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()