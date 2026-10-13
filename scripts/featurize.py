import argparse
from pathlib import Path
import pandas as pd

# Import the new class from your package
from my_package.features.features_computer import TitanicFeaturesComputer

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract features for Titanic model")
    p.add_argument("--input", required=True, help="Path to preprocessed CSV file")
    p.add_argument("--output", required=True, help="Path to save featurized CSV file")
    return p

def main():
    args = build_parser().parse_args()

    # Load cleaned data
    df = pd.read_csv(args.input)

    # Initialize the class and extract features
    featurizer = TitanicFeaturesComputer()
    featured_df = featurizer.compute_features(df)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_path, index=False)
    print(f"Featurized dataset saved to: {output_path}")

if __name__ == "__main__":
    main()