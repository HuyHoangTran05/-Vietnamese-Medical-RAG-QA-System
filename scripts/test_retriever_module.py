from src.retriever import MedicalRetriever


def main():
    retriever = MedicalRetriever()

    while True:
        query = input("\nNhập câu hỏi y khoa, hoặc gõ 'exit' để thoát: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query:
            continue

        results = retriever.search(query, top_k=5)

        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print("=" * 100)

        for i, result in enumerate(results, start=1):
            print(f"\nTOP {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print("-" * 100)
            print("Câu hỏi trong corpus:")
            print(result.get("question", ""))
            print()
            print("Câu trả lời trong corpus:")
            print(result.get("answer", ""))


if __name__ == "__main__":
    main()