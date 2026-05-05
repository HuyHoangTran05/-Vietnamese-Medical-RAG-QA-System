import json
from pathlib import Path
from collections import Counter


CHUNKS_PATH = Path("data/processed/medical_chunks.jsonl")


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    chunks = load_jsonl(CHUNKS_PATH)

    print("Total chunks:", len(chunks))

    source_type_counter = Counter(c.get("source_type", "unknown") for c in chunks)
    print("\nChunks by source_type:")
    for k, v in source_type_counter.items():
        print(k, v)

    print("\nShortest text chunks:")
    text_chunks = [c for c in chunks if c.get("chunk_type") == "text_chunk"]
    text_chunks = sorted(text_chunks, key=lambda c: len(c.get("raw_text", c.get("text", ""))))

    for c in text_chunks[:10]:
        raw = c.get("raw_text", c.get("text", ""))
        print("=" * 100)
        print("chunk_id:", c.get("chunk_id"))
        print("source_type:", c.get("source_type"))
        print("title:", c.get("title"))
        print("length:", len(raw))
        print(raw[:500])


if __name__ == "__main__":
    main()