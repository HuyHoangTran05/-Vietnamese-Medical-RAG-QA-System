import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")


def main():
    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/raw")
        return

    for csv_file in csv_files:
        print("=" * 80)
        print(f"File: {csv_file}")

        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print()
        print(df.head(5).to_string())


if __name__ == "__main__":
    main()