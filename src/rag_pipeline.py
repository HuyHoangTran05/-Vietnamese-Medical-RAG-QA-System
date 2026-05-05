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
        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):
            question = doc.get("question", "")
            answer = doc.get("answer", "")
            score = doc.get("score", 0.0)

            context = f"""
[Tài liệu {i}]
Điểm tương đồng: {score:.4f}
Câu hỏi liên quan: {question}
Câu trả lời liên quan: {answer}
""".strip()

            context_parts.append(context)

        return "\n\n".join(context_parts)

    def build_prompt(self, question: str) -> Dict[str, Any]:
        retrieved_docs = self.retriever.search(question, top_k=self.top_k)
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