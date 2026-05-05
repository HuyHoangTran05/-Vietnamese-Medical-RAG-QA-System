import json
import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/medical_chunks.jsonl")

INDEX_DIR = Path("indexes/faiss")
INDEX_PATH = INDEX_DIR / "medical.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: Path):
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def setup_huggingface_auth():
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        login(token=hf_token)
        print("Hugging Face token loaded from .env")
    else:
        print("Warning: HF_TOKEN not found. You may be rate-limited.")


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_PATH}. Please run scripts/chunk_corpus.py first."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    setup_huggingface_auth()

    print(f"Loading chunks from {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Number of chunks: {len(chunks)}")

    if len(chunks) == 0:
        raise ValueError("No chunks found. Please check medical_chunks.jsonl")

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal}")

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()