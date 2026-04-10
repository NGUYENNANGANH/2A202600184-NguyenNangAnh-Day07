from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Lấy top-k chunks liên quan từ store
        results = self.store.search(question, top_k=top_k)

        # Bước 2: Ghép chunks thành context
        if not results:
            context = "Không tìm thấy tài liệu liên quan."
        else:
            context_parts = []
            for i, chunk in enumerate(results, 1):
                context_parts.append(f"[{i}] {chunk['content']}")
            context = "\n\n".join(context_parts)

        # Bước 3: Tạo prompt và gọi LLM
        prompt = (
            "Bạn là trợ lý thông minh. Hãy trả lời câu hỏi dựa trên các đoạn tài liệu dưới đây.\n"
            "Chỉ sử dụng thông tin từ tài liệu được cung cấp. "
            "Nếu không tìm thấy thông tin, hãy nói rõ.\n\n"
            f"=== TÀI LIỆU THAM KHẢO ===\n{context}\n\n"
            f"=== CÂU HỎI ===\n{question}\n\n"
            "=== CÂU TRẢ LỜI ==="
        )

        return self.llm_fn(prompt)
