import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = Path("indexes/faiss/medical.index")
METADATA_PATH = Path("indexes/faiss/metadata.pkl")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_retriever():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INDEX_PATH}. Please run scripts/build_faiss_index.py first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {METADATA_PATH}. Please run scripts/build_faiss_index.py first."
        )

    print(f"Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))

    print(f"Loading metadata from {METADATA_PATH}")
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    return model, index, metadata


def search(query: str, model, index, metadata, top_k: int = 5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = metadata[idx]

        results.append(
            {
                "score": float(score),
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "text": item.get("text"),
            }
        )

    return results


def print_results(query: str, results):
    print("\n" + "=" * 100)
    print(f"Query: {query}")
    print("=" * 100)

    for i, result in enumerate(results, start=1):
        print(f"\nTOP {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Doc ID: {result['doc_id']}")
        print("-" * 100)

        if result.get("question"):
            print("Câu hỏi trong corpus:")
            print(result["question"])
            print()

        if result.get("answer"):
            print("Câu trả lời trong corpus:")
            print(result["answer"])
        else:
            print(result.get("text", ""))


def main():
    model, index, metadata = load_retriever()

    while True:
        query = input("\nNhập câu hỏi y khoa, hoặc gõ 'exit' để thoát: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Đã thoát.")
            break

        if not query:
            print("Câu hỏi rỗng, vui lòng nhập lại.")
            continue

        results = search(
            query=query,
            model=model,
            index=index,
            metadata=metadata,
            top_k=5,
        )

        print_results(query, results)


if __name__ == "__main__":
    main()