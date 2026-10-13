import argparse
from pathlib import Path
import pickle
import pandas as pd

# Import the new class from your package
from my_package.model.model import TitanicModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Titanic survival baseline model")
    p.add_argument(
        "--input", required=True, help="Path to featurized training CSV file"
    )
    p.add_argument(
        "--output", required=True, help="Path to save the trained model pickle file"
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

    # Initialize the model class and train
    print("Training baseline model...")
    model_instance = TitanicModel()
    trained_pipeline = model_instance.train(X_train, y_train)

    # Save fitted model pipeline with pickle
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(trained_pipeline, f)

    print(f"Model successfully saved to: {output_path}")

if __name__ == "__main__":
    main()