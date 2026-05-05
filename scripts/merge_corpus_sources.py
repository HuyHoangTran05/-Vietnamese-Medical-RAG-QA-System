import json
from pathlib import Path


INPUT_FILES = [
    Path("data/processed/medical_corpus.jsonl"),
    Path("data/processed/web_article_corpus.jsonl"),
    Path("data/processed/pdf_corpus.jsonl"),
]

OUTPUT_PATH = Path("data/processed/combined_medical_corpus.jsonl")


def load_jsonl(path: Path):
    docs = []

    if not path.exists():
        print(f"Skip missing file: {path}")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def normalize_doc(doc: dict, fallback_idx: int) -> dict:
    source_type = doc.get("source_type")

    if not source_type:
        if doc.get("question") and doc.get("answer"):
            source_type = "qa_pair"
        else:
            source_type = "unknown"

    content = doc.get("content", "")

    if not content and doc.get("question") and doc.get("answer"):
        content = f"Câu hỏi: {doc['question']}\nTrả lời: {doc['answer']}"

    return {
        "doc_id": doc.get("doc_id", f"doc_{fallback_idx:06d}"),
        "source": doc.get("source", "unknown"),
        "source_type": source_type,
        "trust_level": doc.get("trust_level", "medium"),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
        "file_path": doc.get("file_path", ""),
        "content": content,
        "question": doc.get("question", ""),
        "answer": doc.get("answer", ""),
        "split": doc.get("split", ""),
    }


def main():
    all_docs = []
    seen_ids = set()

    for input_file in INPUT_FILES:
        docs = load_jsonl(input_file)
        print(f"Loaded {len(docs)} docs from {input_file}")

        for doc in docs:
            normalized = normalize_doc(doc, len(all_docs))

            doc_id = normalized["doc_id"]
            if doc_id in seen_ids:
                continue

            if not normalized["content"]:
                continue

            seen_ids.add(doc_id)
            all_docs.append(normalized)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_docs)} combined docs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()