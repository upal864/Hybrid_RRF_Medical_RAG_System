"""
pipeline/retriever.py

Hybrid Retrieval Pipeline:
  1. BM25 sparse retrieval → top-k candidates
  2. ChromaDB dense vector retrieval → top-k candidates
  3. Reciprocal Rank Fusion (RRF) → merged ranked list
  4. CrossEncoder reranker → best final results

This module exposes a single `HybridRetriever` class that can be imported
into the RAG query pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers.cross_encoder import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRetriever:
    """
    A retriever that fuses BM25 and dense vector results using
    Reciprocal Rank Fusion (RRF) and then reranks with a CrossEncoder.
    """

    def __init__(
        self,
        chunks_path: str = "corpus/all_chunks.json",
        chroma_dir: str = "chroma_db_bge",
        collection_name: str = "rag_corpus_bge",
        embed_model: str = "BAAI/bge-m3",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_retrieval: int = 20,   # candidates per retriever
        top_k_final: int = 5,        # final results after reranking
        rrf_k: int = 60,             # RRF constant (standard: 60)
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k

        # ── Load chunks ──────────────────────────────────────────────────────
        logging.info("Loading corpus chunks...")
        with open(chunks_path, encoding="utf-8") as f:
            self.all_chunks: List[Dict[str, Any]] = json.load(f)

        # ── BM25 setup ───────────────────────────────────────────────────────
        logging.info("Building BM25 index...")
        tokenized = [chunk["text"].lower().split() for chunk in self.all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        logging.info(f"  BM25 index built over {len(self.all_chunks)} chunks.")

        # ── Dense vector retriever ───────────────────────────────────────────
        logging.info("Loading embedding model (BAAI/bge-m3) on GPU (CUDA)...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        logging.info("  ChromaDB vector store loaded.")

        # ── CrossEncoder reranker ────────────────────────────────────────────
        logging.info(f"Loading CrossEncoder reranker on GPU: {reranker_model} ...")
        self.reranker = CrossEncoder(reranker_model, device="cuda")
        logging.info("  Reranker ready.")

    # ─────────────────────────────────────────────────────────────────────────
    def _bm25_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return top-k chunks by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # Sort indices by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            chunk = self.all_chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text":     chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "bm25_score": float(scores[idx]),
            })
        return results

    def _vector_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return top-k chunks by dense cosine similarity."""
        hits = self.vectorstore.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in hits:
            results.append({
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "text":     doc.page_content,
                "metadata": doc.metadata,
                "vector_score": float(score),
            })
        return results

    def _rrf_fuse(
        self,
        bm25_results:   List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion:
        RRF(d) = Σ 1 / (rrf_k + rank_in_list)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map:  Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            chunk_map[cid] = item

        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = item

        # Sort by fused RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in fused:
            entry = dict(chunk_map[cid])
            entry["rrf_score"] = round(score, 6)
            results.append(entry)
        return results

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Full pipeline:
        BM25 + Vector → RRF fusion → CrossEncoder reranker → top_k_final results
        """
        logging.info(f"Query: '{query}'")
        effective_top_k = top_k if top_k is not None else self.top_k_final

        # Step 1: Retrieve from both sources
        bm25_results   = self._bm25_retrieve(query, self.top_k_retrieval)
        vector_results = self._vector_retrieve(query, self.top_k_retrieval)
        logging.info(f"  BM25 candidates: {len(bm25_results)} | Vector candidates: {len(vector_results)}")

        # Step 2: Fuse via RRF
        fused = self._rrf_fuse(bm25_results, vector_results)
        logging.info(f"  After RRF fusion: {len(fused)} unique candidates")

        # Step 3: Rerank top fused candidates with CrossEncoder
        # Take enough candidates to give the reranker good signal
        candidates = fused[:max(effective_top_k * 4, 20)]
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)

        for item, score in zip(candidates, ce_scores):
            item["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        final = reranked[:effective_top_k]

        logging.info(f"  Final top-{effective_top_k} results ready.")
        return final


# ─── Module-level singleton + retrieve() wrapper ─────────────────────────────────
_retriever_instance: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Lazily create and cache the HybridRetriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        logging.info("Initializing HybridRetriever singleton...")
        _retriever_instance = HybridRetriever(
            chunks_path="corpus/all_chunks.json",
            chroma_dir="chroma_db_bge",
            top_k_retrieval=20,
            top_k_final=5,
        )
    return _retriever_instance


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Module-level convenience function.
    Import as: from pipeline.retriever import retrieve
    """
    return _get_retriever().retrieve(query, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test queries — one English, one Spanish, one WHO
    test_queries = [
        "What are the symptoms and treatment of heart failure?",
        "¿Cuáles son los síntomas del fallo cardíaco?",
        "WHO guidelines for COVID-19 clinical management",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"  QUERY: {query}")
        print(f"{'='*70}")
        results = retriever.retrieve(query)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] chunk_id    : {r['chunk_id']}")
            print(f"       source      : {r['metadata'].get('source', 'N/A')}")
            print(f"       language    : {r['metadata'].get('language', 'N/A')}")
            print(f"       rrf_score   : {r.get('rrf_score', 0):.5f}")
            print(f"       rerank_score: {r['rerank_score']:.4f}")
            print(f"       preview     : {r['text'][:200]}...")
