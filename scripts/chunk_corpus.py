import json
import re
from pathlib import Path
from typing import Dict, List


INPUT_PATH = Path("data/processed/combined_medical_corpus.jsonl")
OUTPUT_PATH = Path("data/processed/medical_chunks.jsonl")


def load_jsonl(path: Path) -> List[Dict]:
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    return docs


def split_by_paragraphs(text: str, max_chars: int = 900, overlap_chars: int = 150) -> List[str]:
    text = text.strip()

    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)

            if len(para) <= max_chars:
                current = para
            else:
                # fallback nếu 1 paragraph quá dài
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = end - overlap_chars
                current = ""

    if current:
        chunks.append(current)

    return chunks


def make_qa_chunk(doc: Dict, chunk_id: int) -> Dict:
    question = doc.get("question", "").strip()
    answer = doc.get("answer", "").strip()

    text = f"Câu hỏi: {question}\nTrả lời: {answer}".strip()

    return {
        "chunk_id": f"chunk_{chunk_id:06d}",
        "doc_id": doc.get("doc_id"),
        "source": doc.get("source"),
        "source_type": doc.get("source_type"),
        "trust_level": doc.get("trust_level"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "file_path": doc.get("file_path"),
        "chunk_index": 0,
        "chunk_type": "qa_pair",
        "question": question,
        "answer": answer,
        "text": text,
    }


def make_text_chunks(doc: Dict, start_chunk_id: int) -> List[Dict]:
    content = doc.get("content", "")
    text_chunks = split_by_paragraphs(content)

    chunks = []

    for idx, text in enumerate(text_chunks):
        chunk = {
            "chunk_id": f"chunk_{start_chunk_id + idx:06d}",
            "doc_id": doc.get("doc_id"),
            "source": doc.get("source"),
            "source_type": doc.get("source_type"),
            "trust_level": doc.get("trust_level"),
            "title": doc.get("title"),
            "url": doc.get("url"),
            "file_path": doc.get("file_path"),
            "chunk_index": idx,
            "chunk_type": "text_chunk",
            "question": "",
            "answer": "",
            "text": text,
        }

        chunks.append(chunk)

    return chunks


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. Please run scripts/merge_corpus_sources.py first."
        )

    docs = load_jsonl(INPUT_PATH)

    all_chunks = []
    chunk_id = 0

    for doc in docs:
        source_type = doc.get("source_type", "")

        if source_type == "qa_pair" and doc.get("question") and doc.get("answer"):
            chunk = make_qa_chunk(doc, chunk_id)

            if chunk["text"]:
                all_chunks.append(chunk)
                chunk_id += 1

        else:
            chunks = make_text_chunks(doc, chunk_id)

            all_chunks.extend(chunks)
            chunk_id += len(chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Loaded {len(docs)} documents")
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()