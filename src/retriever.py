import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
from sentence_transformers import SentenceTransformer


class MedicalRetriever:
    def __init__(
        self,
        index_path: str = "indexes/faiss/medical.index",
        metadata_path: str = "indexes/faiss/metadata.pkl",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            item = self.metadata[idx]

            results.append(
                {
                    "score": float(score),
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "title": item.get("title"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "text": item.get("text"),
                    "source": item.get("source"),
                }
            )

        return results