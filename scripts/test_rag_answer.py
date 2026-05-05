from src.rag_pipeline import MedicalRAGPipeline


def print_sources(retrieved_docs):
    print("\n" + "=" * 100)
    print("NGUỒN TRUY XUẤT")
    print("=" * 100)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\nNguồn {i}")
        print(f"Score: {doc.get('score', 0):.4f}")
        print(f"Chunk ID: {doc.get('chunk_id')}")
        print("-" * 100)
        print("Câu hỏi liên quan:")
        print(doc.get("question", ""))
        print()
        print("Câu trả lời liên quan:")
        print(doc.get("answer", ""))


def main():
    rag = MedicalRAGPipeline(top_k=3, use_llm=True)

    while True:
        question = input("\nNhập câu hỏi y khoa, hoặc gõ 'exit' để thoát: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Đã thoát.")
            break

        if not question:
            print("Câu hỏi rỗng, vui lòng nhập lại.")
            continue

        result = rag.answer(question)

        print("\n" + "=" * 100)
        print("CÂU TRẢ LỜI RAG")
        print("=" * 100)
        print(result["answer"])

        print_sources(result["retrieved_docs"])


if __name__ == "__main__":
    main()