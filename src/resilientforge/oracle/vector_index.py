"""Semantic similarity index over normalized failure signatures.

`VectorIndex` is an abstract interface — "keep the storage layer behind
an interface so it can be swapped later" — so the backend can change
without touching callers. `ChromaVectorIndex` is the Phase 1
implementation, using chromadb in local/persistent mode — no external
service required.

Embedding function note: chromadb's *default* embedding function downloads
an onnx model from the network on first use. That breaks the "fast, no
network" unit-test tier and offline installs, so
`ChromaVectorIndex` defaults to `_HashingEmbeddingFunction`, a deterministic,
offline bag-of-words embedder — good enough for matching near-identical
normalized signatures, but weak on true semantic similarity. This is a
placeholder: swap in a real embedding model (local or hosted) behind this
same interface once tuning match quality against the failure-injection
suite calls for it.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _HashingEmbeddingFunction(EmbeddingFunction):
    """Deterministic, offline bag-of-words embedding (see module docstring)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    @staticmethod
    def name() -> str:
        return "resilientforge-hashing-v1"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _HashingEmbeddingFunction:
        return _HashingEmbeddingFunction(dim=config.get("dim", 256))

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}


@dataclass
class VectorMatch:
    id: str
    score: float  # similarity in [0, 1]-ish range; higher = more similar
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex(ABC):
    """Interface a vector store backend must implement."""

    @abstractmethod
    def add(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add or update (upsert) the embedding for `id`."""

    @abstractmethod
    def query(self, text: str, top_k: int = 5) -> list[VectorMatch]:
        """Return up to `top_k` matches ordered by descending similarity."""

    @abstractmethod
    def delete(self, id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> VectorIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ChromaVectorIndex(VectorIndex):
    def __init__(
        self,
        path: str | Path,
        collection_name: str = "failure_signatures",
        embedding_function: EmbeddingFunction | None = None,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function or _HashingEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        # chromadb rejects an empty-dict metadata entry outright, so only
        # pass metadatas at all when there's something non-empty to store.
        self._collection.upsert(
            ids=[id], documents=[text], metadatas=[metadata] if metadata else None
        )

    def query(self, text: str, top_k: int = 5) -> list[VectorMatch]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(query_texts=[text], n_results=min(top_k, count))
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        matches = []
        for match_id, distance, metadata in zip(ids, distances, metadatas):
            # cosine distance -> similarity; clamp for float noise around 0.
            similarity = max(0.0, min(1.0, 1.0 - distance))
            matches.append(VectorMatch(id=match_id, score=similarity, metadata=metadata or {}))
        return matches

    def delete(self, id: str) -> None:
        self._collection.delete(ids=[id])

    def close(self) -> None:
        pass
