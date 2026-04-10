from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        # Nhiệm vụ: Nhận 1 Document (text thô + metadata), gọi embedding_fn để biến
        # text thành vector số, sau đó đóng gói thành dict gồm: id, content, embedding, metadata.
        # Đây là bước "số hoá" tài liệu trước khi lưu vào store.
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata)
        metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        # Nhiệm vụ: Embed câu query thành vector, tính dot product giữa query vector
        # và từng record trong danh sách, sắp xếp theo score giảm dần,
        # trả về top_k kết quả tốt nhất (id, content, metadata, score).
        query_vec = self._embedding_fn(query)
        scored = [
            (rec, _dot(query_vec, rec["embedding"]))
            for rec in records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rec, score in scored[:top_k]:
            results.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score,
            })
        return results

    def add_documents(self, docs: list[Document]) -> None:
        # TODO: embed each doc and add to store
        # Nhiệm vụ: Duyệt qua từng Document trong danh sách, gọi _make_record để
        # chuyển thành record có vector, rồi lưu vào self._store (in-memory)
        # hoặc ChromaDB nếu có. Đây là bước "nạp dữ liệu" vào vector store.
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            if self._use_chroma and self._collection is not None:
                self._collection.add(
                    ids=[str(self._next_index)],
                    documents=[doc.content],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]],
                )
            else:
                self._store.append(record)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        # TODO: embed query, compute similarities, return top_k
        # Nhiệm vụ: Tìm top_k chunks giống nhất với câu query. Nếu dùng ChromaDB
        # thì gọi collection.query(), ngược lại gọi _search_records trên toàn bộ store.
        # Đây là bước "tra cứu" khi agent nhận câu hỏi từ user.
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_vec = self._embedding_fn(query)
            res = self._collection.query(query_embeddings=[query_vec], n_results=top_k)
            results = []
            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "id": res["ids"][0][i],
                    "content": doc,
                    "metadata": res["metadatas"][0][i],
                    "score": 1 - res["distances"][0][i],
                })
            return results
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        # TODO: return count
        # Nhiệm vụ: Trả về tổng số chunks đang được lưu trong store.
        # Dùng để kiểm tra store đã có dữ liệu chưa hoặc có bao nhiêu chunks.
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        # TODO: filter by metadata, then search among filtered chunks
        # Nhiệm vụ: Trước tiên lọc self._store theo metadata_filter (vd: category="python"),
        # sau đó chỉ tìm kiếm trong nhóm đã lọc đó thay vì toàn bộ store.
        # Giúp tăng độ chính xác khi biết trước muốn tìm trong loại tài liệu nào.
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)
        filtered = [
            rec for rec in self._store
            if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        # Nhiệm vụ: Xóa toàn bộ chunks có metadata["doc_id"] trùng với doc_id truyền vào.
        # Trả về True nếu có chunk bị xóa, False nếu không tìm thấy doc đó trong store.
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
