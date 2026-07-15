"""An optional, genuinely-semantic embedder (Phase 5) — an alternative
to the default `_HashingEmbeddingFunction` (`oracle/vector_index.py`),
for callers willing to accept a much heavier dependency
(`sentence-transformers` + `torch`, roughly 1GB installed — measured
directly while building this, not estimated).

**Read this before reaching for it**: `tests/unit/test_embedder_quality.py`
runs the SAME realistic, labeled benchmark against both embedders, and
the honestly-reported result is a genuine surprise — on that benchmark,
this embedder did NOT outperform the free default. Hashing: recall 1.00,
precision ~0.55. This: recall 1.00, precision ~0.50 — slightly *worse*.
Its false positives include pairs that ARE semantically/topically
related ("card declined" vs "insufficient funds" — both genuinely about
payment failure) but need different fixes: general-purpose semantic
closeness isn't the same thing as "needs the same corrective action,"
which is what recipe matching actually needs. This module exists so the
option is available and easy to benchmark against your OWN real
signatures (a different domain or a different model could tell a
different story) — not because paying ~1GB for a fancier-sounding
technique is automatically the right call. Measure before switching;
don't take this docstring's own hopeful framing at construction time as
a claim of better results, because it measurably wasn't one here.

Needs ZERO changes to `wrap()`/`Oracle`/`core/engine.py` —
`ChromaVectorIndex`'s `embedding_function` parameter already makes this
fully pluggable today:

    from resilientforge.oracle import Oracle
    from resilientforge.oracle.vector_index import ChromaVectorIndex
    from resilientforge.oracle.semantic_embedding import SentenceTransformerEmbeddingFunction

    oracle = Oracle(
        "my_oracle_path",
        vector_index=ChromaVectorIndex(
            "my_oracle_path/vectors",
            embedding_function=SentenceTransformerEmbeddingFunction(),
        ),
    )
    wrapped = wrap(my_tool, oracle=oracle, reflect=reflect)

Lazy-imported (the `sentence_transformers` import is inside
`__init__`, not this module's top level) so nothing that transitively
imports `oracle/` — which is most of this package — hard-requires it.
Needs the `semantic` extra: `pip install resilientforge[semantic]`.
"""

from __future__ import annotations

from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """A real semantic embedder backed by a local `sentence-transformers`
    model — `all-MiniLM-L6-v2` (the standard lightweight default for
    short-text similarity tasks, ~80MB of model weights on top of the
    install itself) unless overridden. Downloads model weights from
    HuggingFace on first use, cached locally afterward — the one place
    in this codebase that needs network access, and only if you opt
    into this extra and this embedder specifically; nothing else in
    ResilientForge does."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self._model.encode(list(input), normalize_embeddings=True)
        return embeddings.tolist()

    @staticmethod
    def name() -> str:
        return "resilientforge-sentence-transformer-v1"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> SentenceTransformerEmbeddingFunction:
        return SentenceTransformerEmbeddingFunction(model_name=config.get("model_name", _DEFAULT_MODEL))

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.model_name}
