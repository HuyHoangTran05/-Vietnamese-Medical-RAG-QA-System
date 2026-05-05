from src.rag_pipeline import MedicalRAGPipeline


def main():
    rag = MedicalRAGPipeline(top_k=5)

    while True:
        question = input("\nNhập câu hỏi y khoa, hoặc gõ 'exit' để thoát: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        if not question:
            continue

        output = rag.build_prompt(question)

        print("\n" + "=" * 100)
        print("PROMPT ĐƯA VÀO LLM")
        print("=" * 100)
        print(output["prompt"])


if __name__ == "__main__":
    main()