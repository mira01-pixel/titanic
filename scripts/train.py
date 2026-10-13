import argparse
from pathlib import Path
import pickle
import pandas as pd

# Import all three models from your package
from my_package.model.model import TitanicModel
from my_package.model.random_forest_model import RandomForestTitanicModel
from my_package.model.xgboost_model import XGBoostTitanicModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Titanic survival model")
    p.add_argument(
        "--input", required=True, help="Path to featurized training CSV file"
    )
    p.add_argument(
        "--output", required=True, help="Path to save the trained model pickle file"
    )
    # Update choices to include xgboost
    p.add_argument(
        "--model-type", 
        choices=["logistic", "rf", "xgboost"], 
        default="logistic", 
        help="Type of model to train: 'logistic', 'rf', or 'xgboost'"
    )
    return p

def main():
    args = build_parser().parse_args()

    # Load featurized dataset
    df = pd.read_csv(args.input)

    # Separate features and target
    if "Survived" not in df.columns:
        raise ValueError("Training dataset must contain 'Survived' column.")

    X_train = df.drop(columns=["Survived"])
    y_train = df["Survived"].astype(int)

    # Select and initialize the model
    if args.model_type == "rf":
        print("Training Random Forest model...")
        model_instance = RandomForestTitanicModel()
    elif args.model_type == "xgboost":
        print("Training XGBoost model...")
        model_instance = XGBoostTitanicModel()
    else:
        print("Training Logistic Regression model...")
        model_instance = TitanicModel()
        
    trained_pipeline = model_instance.train(X_train, y_train)

    # Save fitted model pipeline
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trained_pipeline, f)

    print(f"Model successfully saved to: {output_path}")

if __name__ == "__main__":
    main()