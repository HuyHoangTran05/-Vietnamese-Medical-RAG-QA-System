from typing import List, Dict, Any

from src.retriever import MedicalRetriever
from src.prompts import MEDICAL_RAG_PROMPT
from src.llm_client import GeminiLLMClient


class MedicalRAGPipeline:
    def __init__(self, top_k: int = 3, use_llm: bool = True):
        self.retriever = MedicalRetriever()
        self.top_k = top_k
        self.use_llm = use_llm

        if self.use_llm:
            self.llm = GeminiLLMClient()
        else:
            self.llm = None

    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "Không tìm thấy ngữ cảnh đủ liên quan trong corpus hiện tại."

        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):
            score = doc.get("score", 0.0)
            source = doc.get("source", "")
            source_type = doc.get("source_type", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            file_path = doc.get("file_path", "")

            question = doc.get("question", "")
            answer = doc.get("answer", "")

            # QA pair thì dùng question + answer
            if question or answer:
                content = f"Câu hỏi liên quan: {question}\nCâu trả lời liên quan: {answer}".strip()
            else:
                # Web article / PDF thì dùng raw_text hoặc text
                content = doc.get("raw_text") or doc.get("text") or ""

            context = f"""
[Tài liệu {i}]
Điểm tương đồng: {score:.4f}
Nguồn: {source}
Loại nguồn: {source_type}
Tiêu đề: {title}
URL: {url}
File path: {file_path}

Nội dung:
{content}
""".strip()

            context_parts.append(context)

        return "\n\n".join(context_parts)

    def build_prompt(self, question: str) -> Dict[str, Any]:
        retrieved_docs = self.retriever.search(
            query=question,
            top_k=self.top_k,
        )

        context = self.build_context(retrieved_docs)

        prompt = MEDICAL_RAG_PROMPT.format(
            question=question,
            context=context,
        )

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "prompt": prompt,
        }

    def answer(self, question: str) -> Dict[str, Any]:
        rag_data = self.build_prompt(question)

        if not self.use_llm or self.llm is None:
            rag_data["answer"] = rag_data["prompt"]
            return rag_data

        final_answer = self.llm.generate(rag_data["prompt"])
        rag_data["answer"] = final_answer

        return rag_data