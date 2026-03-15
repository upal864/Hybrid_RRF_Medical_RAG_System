"""
api/main.py

FastAPI server for the Healthcare RAG System.
Endpoints: GET /, GET /health, POST /query, POST /feedback
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline.retriever import retrieve
from pipeline.generator import generate, expand_query
from pipeline.prompt_loader import get_version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Healthcare RAG API",
    description=(
        "Hybrid retrieval (BM25 + BAAI/bge-m3 dense vectors) with CrossEncoder reranking "
        "over MedlinePlus health topics and WHO clinical guidelines. "
        "Answers are citation-grounded using NVIDIA Nemotron-70B."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:       str  = Field(..., min_length=3, description="The user's medical question")
    top_k:          int  = Field(5, ge=1, le=20,   description="Number of results to return")
    expand_query:   bool = Field(False,             description="Use query expansion for broader recall")


class ChunkOut(BaseModel):
    chunk_number: int
    text:         str
    source:       str
    title:        str
    language:     str
    rerank_score: float


class QueryResponse(BaseModel):
    question:       str
    answer:         str
    valid:          bool
    refused:        bool
    sources:        list[str]
    chunks:         list[ChunkOut]
    prompt_version: str
    timestamp:      str


class FeedbackRequest(BaseModel):
    question: str
    answer:   str
    helpful:  bool
    comment:  str = ""


# ─── Startup event ────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    """Pre-warm the retriever so the first query is fast."""
    logging.info("🚀 Healthcare RAG API starting up...")
    logging.info(f"Prompt version: {get_version()}")
    # Trigger retriever singleton initialization (loads bge-m3 + BM25)
    retrieve("warmup query", top_k=1)
    logging.info("✅ Retriever warmed up and ready.")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {
        "status":         "ok",
        "message":        "Healthcare RAG API is running",
        "prompt_version": get_version(),
    }


@app.get("/health", tags=["Status"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_endpoint(req: QueryRequest):
    """
    Ask a medical question. Returns a citation-grounded answer
    sourced from MedlinePlus and WHO clinical guidelines.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # Optional query expansion for broader retrieval recall
        if req.expand_query:
            queries = expand_query(req.question)
            logging.info(f"Expanded queries: {queries}")
        else:
            queries = [req.question]

        # Retrieve from all query variants, deduplicate by chunk_id
        all_chunks = []
        seen_ids   = set()
        for q in queries:
            for c in retrieve(q, top_k=req.top_k):
                cid = c.get("chunk_id", c.get("text", "")[:50])
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_chunks.append(c)

        # Sort deduplicated chunks by rerank_score before sending to generator
        all_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        final_chunks = all_chunks[:req.top_k]  # cap at requested top_k

        # Generate answer with citation enforcement
        result = generate(req.question, final_chunks)

        # Build chunk output list only from cited chunks
        chunk_responses = []
        cited = result.get("cited_chunks", [])
        for i, c in enumerate(cited, start=1):
            meta = c.get("metadata", {})
            chunk_responses.append(ChunkOut(
                chunk_number = i,
                text         = c["text"],
                source       = meta.get("source", "unknown"),
                title        = meta.get("title", meta.get("filename", "N/A")),
                language     = meta.get("language", "en"),
                rerank_score = round(c.get("rerank_score", 0.0), 4),
            ))

        return QueryResponse(
            question       = req.question,
            answer         = result["answer"],
            valid          = result["valid"],
            refused        = result["refused"],
            sources        = result["sources"],
            chunks         = chunk_responses,
            prompt_version = result["prompt_version"],
            timestamp      = datetime.utcnow().isoformat(),
        )

    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", tags=["Feedback"])
def feedback(req: FeedbackRequest):
    """Log user feedback for offline evaluation and prompt tuning."""
    log_path = Path("eval/feedback_log.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question":  req.question,
        "answer":    req.answer,
        "helpful":   req.helpful,
        "comment":   req.comment,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info(f"Feedback logged: helpful={req.helpful}")
    return {"status": "recorded", "timestamp": entry["timestamp"]}
