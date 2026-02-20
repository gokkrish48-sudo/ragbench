"""Document ingestion — loading, chunking, and embedding."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ragbench.utils.logging import get_logger, timer

log = get_logger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.text.encode()).hexdigest()[:12]


class DocumentIngestor:
    """Loads documents, chunks them, and computes embeddings."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model_name)
                log.info(f"Loaded embedding model: [bold]{self.embedding_model_name}[/]")
            except ImportError:
                log.warning("sentence-transformers not installed — embeddings disabled")
        return self._encoder

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        """Load documents from a file or directory."""
        path = Path(path)
        docs = []

        if path.is_dir():
            for f in sorted(path.glob("**/*")):
                if f.suffix in {".json", ".jsonl", ".txt", ".md", ".csv"}:
                    docs.extend(self._load_file(f))
        else:
            docs = self._load_file(path)

        log.info(f"Loaded [bold]{len(docs)}[/] documents from {path}")
        return docs

    def _load_file(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        elif path.suffix == ".jsonl":
            docs = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
            return docs
        elif path.suffix in {".txt", ".md"}:
            return [{"text": path.read_text(), "source": str(path)}]
        elif path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_dict("records")
        return []

    def chunk(self, documents: list[dict[str, Any]], text_field: str = "text") -> list[Chunk]:
        """Split documents into overlapping chunks."""
        chunks = []
        for doc_idx, doc in enumerate(documents):
            text = doc.get(text_field, "")
            if not text:
                continue

            meta = {k: v for k, v in doc.items() if k != text_field}
            meta["doc_idx"] = doc_idx

            words = text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)

            for i in range(0, len(words), step):
                chunk_words = words[i : i + self.chunk_size]
                if len(chunk_words) < 10:  # skip tiny trailing chunks
                    continue
                chunk_text = " ".join(chunk_words)
                chunk_id = f"doc{doc_idx}_chunk{i}"
                chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=meta))

        log.info(
            f"Chunked [bold]{len(documents)}[/] docs → [bold]{len(chunks)}[/] chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        """Compute embeddings for all chunks."""
        if self.encoder is None:
            log.warning("No encoder available — skipping embedding")
            return chunks

        texts = [c.text for c in chunks]
        with timer("Embedding", log):
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
                batch = texts[i : i + self.batch_size]
                embs = self.encoder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                for j, emb in enumerate(embs):
                    chunks[i + j].embedding = emb

        return chunks

    def ingest(
        self, path: str | Path, text_field: str = "text", compute_embeddings: bool = True
    ) -> list[Chunk]:
        """Full pipeline: load → chunk → embed."""
        docs = self.load(path)
        chunks = self.chunk(docs, text_field=text_field)
        if compute_embeddings:
            chunks = self.embed(chunks)
        return chunks
