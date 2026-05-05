from src.rag_pipeline import MedicalRAGPipeline


def print_sources(retrieved_docs):
    print("\n" + "=" * 100)
    print("NGUỒN TRUY XUẤT")
    print("=" * 100)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\nNguồn {i}")
        print(f"Score: {doc.get('score', 0):.4f}")
        print(f"Chunk ID: {doc.get('chunk_id')}")
        print(f"Source: {doc.get('source')}")
        print(f"Source type: {doc.get('source_type')}")
        print(f"Title: {doc.get('title')}")
        print(f"URL: {doc.get('url')}")
        print(f"File path: {doc.get('file_path')}")
        print("-" * 100)

        question = doc.get("question", "")
        answer = doc.get("answer", "")

        if question or answer:
            print("Câu hỏi liên quan:")
            print(question)
            print()
            print("Câu trả lời liên quan:")
            print(answer)
        else:
            print("Nội dung liên quan:")
            content = doc.get("raw_text") or doc.get("text") or ""
            print(content[:1200])


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