"""
pipeline/generator.py

Generates grounded, cited answers using NVIDIA ChatNVIDIA LLM (openai/gpt-oss-120b).
Uses versioned prompts from prompts/prompts.yaml.
Enforces strict citation validation on every response.
"""

import re
import json
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path so pipeline.* imports always work
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from pipeline.prompt_loader import get_prompt, get_version

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── NVIDIA ChatNVIDIA client (gpt-oss-120b) ──────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    logging.warning("NVIDIA_API_KEY not found in environment variables. Generation will fail.")
LLM_MODEL      = "openai/gpt-oss-120b"

_client: ChatNVIDIA | None = None

def _get_client() -> ChatNVIDIA:
    """Lazily initialize the ChatNVIDIA singleton."""
    global _client
    if _client is None:
        _client = ChatNVIDIA(
            model=LLM_MODEL,
            api_key=NVIDIA_API_KEY,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
        )
    return _client


# ─── Context formatting ────────────────────────────────────────────────────────
def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered, structured context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta   = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        title  = meta.get("title", meta.get("filename", "N/A"))
        lang   = meta.get("language", "en")
        lang_tag = "(ES)" if lang == "es" else ""
        lines.append(
            f"[{i}] Source: {source} | Document: {title} {lang_tag}\n"
            f"{chunk['text'].strip()}"
        )
    return "\n\n---\n\n".join(lines)


# ─── Citation enforcement ──────────────────────────────────────────────────────
REFUSAL_PHRASE = "I cannot find sufficient information in the retrieved documents"

def enforce_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Validates that the generated answer:
    1. Contains inline citations [N]
    2. Only cites valid chunk numbers
    3. Has not refused without reason
    """
    # Case 1: Explicit refusal
    if REFUSAL_PHRASE.lower() in answer.lower():
        return {
            "valid":        False,
            "refused":      True,
            "answer":       answer.strip(),
            "cited_chunks": [],
            "sources":      [],
        }

    # Case 2: No citations at all — unchecked hallucination risk
    cited_nums = sorted(set(int(n) for n in re.findall(r'\[(\d+)\]', answer)))
    if not cited_nums:
        return {
            "valid":        False,
            "refused":      False,
            "answer":       (
                "⚠️ The model produced a response without citations. "
                "This answer cannot be verified against source documents. "
                "Please try rephrasing your question."
            ),
            "cited_chunks": [],
            "sources":      [],
        }

    # Case 3: Citations out of valid range
    valid_range = set(range(1, len(chunks) + 1))
    invalid = set(cited_nums) - valid_range
    if invalid:
        return {
            "valid":        False,
            "refused":      False,
            "answer":       (
                f"⚠️ The model cited chunk numbers {sorted(invalid)} which do not exist "
                f"(valid range: 1–{len(chunks)}). Retrieval error."
            ),
            "cited_chunks": [],
            "sources":      [],
        }

    # Case 4: All good — extract citation metadata
    cited_chunks = [chunks[n - 1] for n in cited_nums]
    seen = set()
    sources = []
    for c in cited_chunks:
        meta   = c.get("metadata", {})
        source = meta.get("source", "unknown")
        title  = meta.get("title", meta.get("filename", "N/A"))
        key = f"{source} | {title}"
        if key not in seen:
            seen.add(key)
            sources.append(key)

    return {
        "valid":        True,
        "refused":      False,
        "answer":       answer.strip(),
        "cited_chunks": cited_chunks,
        "sources":      sources,
    }


# ─── LLM call with retry ──────────────────────────────────────────────────────
def _call_llm(system: str, user: str, max_retries: int = 2) -> str:
    """Call ChatNVIDIA with retry on transient errors."""
    client = _get_client()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logging.warning(f"LLM call attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
    return ""


# ─── Query expansion ──────────────────────────────────────────────────────────
def expand_query(query: str) -> list[str]:
    """
    Generate 3 alternative phrasings of the query to improve BM25 + vector recall.
    Falls back to the original query on any error.
    """
    try:
        prompt = get_prompt("query_expansion", query=query)
        raw = _call_llm(prompt["system"], prompt["user"])
        # Parse JSON array — strip markdown fences if model added them
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        expansions = json.loads(cleaned)
        if isinstance(expansions, list) and all(isinstance(x, str) for x in expansions):
            logging.info(f"  Query expansions: {expansions}")
            return [query] + expansions[:3]
    except Exception as e:
        logging.warning(f"Query expansion failed: {e}. Using original query only.")
    return [query]


# ─── Master generate function ─────────────────────────────────────────────────
def generate(query: str, chunks: list[dict]) -> dict:
    """
    Full RAG generation pipeline:
    1. Format numbered context from retrieved chunks
    2. Load versioned prompt
    3. Call NVIDIA NIM LLM
    4. Enforce citation grounding
    5. Return structured result
    """
    if not chunks:
        return {
            "valid":          False,
            "refused":        True,
            "answer":         "No relevant documents were retrieved for this query.",
            "cited_chunks":   [],
            "sources":        [],
            "prompt_version": get_version(),
            "raw_answer":     "",
        }

    context = format_context(chunks)
    prompt  = get_prompt("rag_answer", context=context, question=query)

    logging.info(f"Calling NVIDIA NIM [{LLM_MODEL}] for query: '{query[:70]}'")
    raw_answer = _call_llm(prompt["system"], prompt["user"])
    logging.info(f"  LLM response length: {len(raw_answer)} chars")

    result = enforce_citations(raw_answer, chunks)
    result["prompt_version"] = prompt["version"]
    result["raw_answer"]     = raw_answer

    return result


# ─── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.retriever import retrieve

    test_queries = [
        "What are the symptoms of abdominal pain?",
        "WHO guidelines for diabetes management",
        "How is acute bronchitis treated?",
    ]

    for query in test_queries:
        print(f"\n{'═'*65}")
        print(f"QUERY: {query}")
        print('═'*65)
        chunks = retrieve(query)
        result = generate(query, chunks)
        print(f"Valid     : {result['valid']}")
        print(f"Refused   : {result['refused']}")
        print(f"Version   : {result['prompt_version']}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for s in result["sources"]:
            print(f"  • {s}")
