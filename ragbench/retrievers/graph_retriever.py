"""Graph retriever — Neo4j knowledge graph with entity-linked retrieval."""

from __future__ import annotations

import time
from ragbench.ingest import Chunk
from ragbench.retrievers.base import BaseRetriever, RetrievalResult
from ragbench.utils.logging import get_logger

log = get_logger(__name__)


class GraphRetriever(BaseRetriever):
    """Neo4j-backed knowledge graph retriever with entity linking."""

    def __init__(
        self,
        top_k: int = 10,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        **kwargs,
    ):
        super().__init__(top_k=top_k)
        self.neo4j_uri = neo4j_uri
        self._driver = None
        self._chunks: list[Chunk] = []

        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            log.info(f"Neo4j connected: {neo4j_uri}")
        except Exception as e:
            log.warning(f"Neo4j not available: {e}. Graph retriever will use fallback.")

    @property
    def name(self) -> str:
        return "graph_neo4j"

    def index(self, chunks: list[Chunk]) -> None:
        """Build knowledge graph from chunks — extract entities and relations."""
        self._chunks = chunks

        if self._driver is None:
            log.warning("Neo4j not connected — storing chunks in memory only")
            return

        with self._driver.session() as session:
            # Clear existing
            session.run("MATCH (n:Chunk) DETACH DELETE n")

            # Insert chunks as nodes
            for chunk in chunks:
                session.run(
                    "CREATE (c:Chunk {id: $id, text: $text})",
                    id=chunk.id,
                    text=chunk.text[:500],
                )

            # Extract and link entities (simplified — production would use NER)
            for chunk in chunks:
                entities = self._extract_entities(chunk.text)
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        name=entity,
                        chunk_id=chunk.id,
                    )

        log.info(f"Graph index built: {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.top_k
        start = time.perf_counter()

        if self._driver is None:
            # Fallback: simple keyword matching
            query_lower = query.lower()
            scored = []
            for c in self._chunks:
                score = sum(1 for w in query_lower.split() if w in c.text.lower())
                scored.append((c, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [c for c, _ in scored[:k]]
            top_scores = [s for _, s in scored[:k]]
        else:
            # Graph traversal: find chunks connected via shared entities
            query_entities = self._extract_entities(query)
            with self._driver.session() as session:
                result = session.run(
                    """
                    UNWIND $entities AS entity_name
                    MATCH (e:Entity {name: entity_name})<-[:MENTIONS]-(c:Chunk)
                    RETURN c.id AS chunk_id, c.text AS text, count(e) AS score
                    ORDER BY score DESC
                    LIMIT $k
                    """,
                    entities=query_entities,
                    k=k,
                )
                records = list(result)

            chunk_map = {c.id: c for c in self._chunks}
            top_chunks = [chunk_map[r["chunk_id"]] for r in records if r["chunk_id"] in chunk_map]
            top_scores = [float(r["score"]) for r in records if r["chunk_id"] in chunk_map]

        latency = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            query=query, retrieved=top_chunks, scores=top_scores, latency_ms=latency
        )

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """Simple entity extraction — capitalize words as proxy for NER."""
        words = text.split()
        entities = []
        for w in words:
            cleaned = w.strip(".,!?;:\"'()[]")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                entities.append(cleaned.lower())
        return list(set(entities))[:20]

    def __del__(self):
        if self._driver:
            self._driver.close()
