import argparse
from pathlib import Path
import pandas as pd

# Import the new class from your package
from my_package.preprocessing.preprocessor import TitanicPreprocessor

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean raw Titanic dataset")
    p.add_argument("--input", required=True, help="Path to input raw CSV file")
    p.add_argument("--output", required=True, help="Path to save cleaned CSV file")
    return p

def main():
    args = build_parser().parse_args()

    # Load raw data
    raw_df = pd.read_csv(args.input)

    # Initialize the class and clean the data
    preprocessor = TitanicPreprocessor()
    cleaned_df = preprocessor.process(raw_df)

    # Save cleaned data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"Preprocessed data successfully saved to: {output_path}")

if __name__ == "__main__":
    main()