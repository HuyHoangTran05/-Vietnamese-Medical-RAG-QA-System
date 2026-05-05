import pandas as pd
from pathlib import Path
import json
import re


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def fix_vietnamese_encoding(text: str) -> str:
    if not isinstance(text, str):
        return text

    try:
        return text.encode("latin1").decode("utf-8")
    except UnicodeError:
        try:
            return text.encode("cp1252").decode("utf-8")
        except UnicodeError:
            return text


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)
    text = fix_vietnamese_encoding(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_columns(df: pd.DataFrame):
    question_candidates = [
        "question",
        "questions",
        "câu hỏi",
        "query",
        "input",
        "instruction",
        "prompt",
    ]

    answer_candidates = [
        "answer",
        "answers",
        "câu trả lời",
        "response",
        "output",
        "completion",
    ]

    question_col = None
    answer_col = None

    for col in df.columns:
        if col.lower().strip() in question_candidates:
            question_col = col

    for col in df.columns:
        if col.lower().strip() in answer_candidates:
            answer_col = col

    return question_col, answer_col


def main():
    all_docs = []
    doc_id = 0

    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("No raw CSV files found in data/raw.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file}")

        df = pd.read_csv(csv_file, encoding="utf-8-sig")

        question_col, answer_col = detect_columns(df)

        print("Detected question column:", question_col)
        print("Detected answer column:", answer_col)

        if question_col is None or answer_col is None:
            print(f"Cannot detect columns in {csv_file}")
            print("Available columns:", df.columns.tolist())
            continue

        for _, row in df.iterrows():
            question = clean_text(row[question_col])
            answer = clean_text(row[answer_col])

            if not question and not answer:
                continue

            content = f"Câu hỏi: {question}\nTrả lời: {answer}"

            doc = {
                "doc_id": f"medical_{doc_id:06d}",
                "source": "hungnm/vietnamese-medical-qa",
                "split": csv_file.stem,
                "title": question[:120],
                "content": content,
                "question": question,
                "answer": answer,
            }

            all_docs.append(doc)
            doc_id += 1

    output_path = PROCESSED_DIR / "medical_corpus.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_docs)} documents to {output_path}")


if __name__ == "__main__":
    main()