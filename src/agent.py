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
        # TODO: store references to store and llm_fn
        # Nhiệm vụ: Lưu lại store (nơi chứa vector) và llm_fn (hàm gọi AI)
        # vào self để hàm answer() có thể dùng sau.
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # Nhiệm vụ:
        #   Bước 1 — Retrieve: Gọi store.search() để lấy top_k chunks liên quan nhất.
        #   Bước 2 — Build prompt: Ghép các chunks thành context, tạo prompt có cấu trúc
        #             rõ ràng cho LLM (context + câu hỏi).
        #   Bước 3 — Generate: Gọi llm_fn(prompt) để LLM sinh câu trả lời dựa trên context.

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
