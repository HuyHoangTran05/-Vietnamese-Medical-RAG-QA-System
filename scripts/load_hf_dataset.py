from datasets import load_dataset
from pathlib import Path


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    dataset_name = "hungnm/vietnamese-medical-qa"

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    print(dataset)

    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()

        output_path = RAW_DIR / f"{split_name}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Saved {split_name} to {output_path}")
        print(df.head())


if __name__ == "__main__":
    main()