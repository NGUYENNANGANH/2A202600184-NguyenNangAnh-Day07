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

        # Only activate ChromaDB when running in production mode
        # (no custom embedding_fn injected). Tests always inject a mock
        # embedding_fn, so they will always use the safe in-memory store.
        if embedding_fn is None:
            try:
                import chromadb

                # EphemeralClient gives each instance its own isolated
                # in-process DB so there is no cross-test state pollution.
                chroma_client = chromadb.EphemeralClient()
                self._collection = chroma_client.get_or_create_collection(name=collection_name)
                self._use_chroma = True
            except Exception:
                self._use_chroma = False
                self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalised stored record for one Document."""
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
        """Run in-memory dot-product similarity search over provided records."""
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
        """
        Embed each document's content and store it.

        For ChromaDB: delegates to collection.add().
        For in-memory: appends dicts to self._store.
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
        """
        Find the top_k most similar documents to query.

        For ChromaDB: delegates to collection.query().
        For in-memory: computes dot product of query embedding vs all stored embeddings.
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
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        Filters stored chunks by metadata_filter first, then runs similarity search
        only within the matching subset.
        """
        if not metadata_filter:
            return self.search(query, top_k)
        filtered = [
            rec for rec in self._store
            if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            existing = self._collection.get(where={"doc_id": doc_id})
            ids_to_delete = existing["ids"]
            if not ids_to_delete:
                return False
            self._collection.delete(ids=ids_to_delete)
            return True
        before = len(self._store)
        self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
