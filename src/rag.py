"""RAG pipeline: embedding, indexing, retrieval, and context injection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import RAG_CONFIG


class RAGEncoder:
    """Wraps a sentence-transformers model for embedding text."""

    def __init__(self, model_name: str = RAG_CONFIG["embedding_model"]):
        self.model = SentenceTransformer(model_name)
        self.dimension = RAG_CONFIG["embedding_dim"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        return self.embed([text])[0]


class RAGStore:
    """FAISS vector store with metadata persistence."""

    def __init__(self, index_dir: str | Path = RAG_CONFIG["index_dir"]):
        self.index_dir = Path(index_dir)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load existing index and metadata from disk."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.jsonl"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.metadata = []
            if meta_path.exists():
                with open(meta_path) as f:
                    self.metadata = [json.loads(line) for line in f if line.strip()]
        else:
            self.index = None
            self.metadata = []

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

        with open(self.index_dir / "metadata.jsonl", "w") as f:
            for entry in self.metadata:
                f.write(json.dumps(entry) + "\n")

    def add(
        self,
        embeddings: np.ndarray,
        entries: list[dict],
    ) -> None:
        """Add embeddings with associated metadata to the store."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.index.add(embeddings)
        self.metadata.extend(entries)
        self._save()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = RAG_CONFIG["top_k"],
    ) -> list[dict]:
        """Retrieve the top-k most similar entries."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(dist)
            results.append(entry)

        return results

    @property
    def size(self) -> int:
        """Number of entries in the store."""
        return self.index.ntotal if self.index else 0


class RAGPipeline:
    """Full RAG pipeline: encode, index, retrieve, inject context."""

    def __init__(self):
        self.encoder = RAGEncoder()
        self.store = RAGStore()

    def index_output(
        self,
        text: str,
        task_name: str,
        adapter_name: str,
        extra: dict | None = None,
    ) -> None:
        """Embed a model output and store it with metadata."""
        embedding = self.encoder.embed_single(text)

        entry = {
            "text": text,
            "task_name": task_name,
            "adapter_name": adapter_name,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if extra:
            entry.update(extra)

        self.store.add(embedding.reshape(1, -1), [entry])

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_CONFIG["top_k"],
    ) -> list[dict]:
        """Retrieve relevant context for a query."""
        query_embedding = self.encoder.embed_single(query)
        return self.store.search(query_embedding, top_k)

    def inject_context(self, query: str, retrieved: list[dict]) -> str:
        """Build a prompt with retrieved context injected.

        Format:
            Context from previous tasks:
            [retrieved entries]

            Current query:
            {query}
        """
        if not retrieved:
            return query

        context_parts = []
        for i, entry in enumerate(retrieved, 1):
            task = entry.get("task_name", "unknown")
            text = entry.get("text", "")
            score = entry.get("score", 0.0)
            context_parts.append(f"[{i}] (task: {task}, relevance: {score:.2f}) {text}")

        context_block = "\n".join(context_parts)
        return (
            f"Context from previous tasks:\n{context_block}\n\n"
            f"Current query:\n{query}"
        )

    def query_with_context(self, query: str, top_k: int = RAG_CONFIG["top_k"]) -> str:
        """Retrieve context and inject it into the query in one step."""
        results = self.retrieve(query, top_k)
        return self.inject_context(query, results)

    def clear(self) -> None:
        """Reset the store (delete all entries)."""
        self.store.index = None
        self.store.metadata = []
        if self.store.index_dir.exists():
            for f in self.store.index_dir.iterdir():
                f.unlink()
