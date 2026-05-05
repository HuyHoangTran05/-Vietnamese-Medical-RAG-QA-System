import json
from pathlib import Path
from typing import List, Dict


INPUT_PATH = Path("data/processed/medical_corpus.jsonl")
OUTPUT_PATH = Path("data/processed/medical_chunks.jsonl")


def load_jsonl(path: Path) -> List[Dict]:
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_PATH}")

    docs = load_jsonl(INPUT_PATH)

    all_chunks = []

    for idx, doc in enumerate(docs):
        question = doc.get("question", "").strip()
        answer = doc.get("answer", "").strip()

        text = f"Câu hỏi: {question}\nTrả lời: {answer}".strip()

        if not text:
            continue

        chunk = {
            "chunk_id": f"chunk_{idx:06d}",
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "split": doc.get("split"),
            "title": doc.get("title"),
            "chunk_type": "qa_pair",
            "question": question,
            "answer": answer,
            "text": text,
        }

        all_chunks.append(chunk)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Loaded {len(docs)} documents")
    print(f"Saved {len(all_chunks)} QA chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()