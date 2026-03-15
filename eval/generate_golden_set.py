"""
eval/generate_golden_set.py

Creates 11 diverse Golden Evaluation Q&A pairs from the corpus.
Each answer is directly extracted from the source chunk (no external knowledge).
Saves to eval/ragas_dataset.json for Ragas evaluation.
"""

import json
import random
import logging
import requests
import sys
import os
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO, format='%(message)s')

API_URL       = "http://localhost:8000/query"
OUTPUT_FILE   = _ROOT / "eval" / "ragas_dataset.json"
NUM_SAMPLES   = 11
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    logging.warning("NVIDIA_API_KEY not found in environment variables.")
LLM_MODEL     = "openai/gpt-oss-120b"

SYS_MSG = SystemMessage(content="""You are a medical evaluation dataset expert.
Given a chunk of text from a medical dataset, produce exactly one question-answer pair.

STRICT RULES:
- The QUESTION must be natural and specific enough to test a medical knowledge system.
- The ANSWER must be taken WORD-FOR-WORD or closely paraphrased from the provided text only.
- Do NOT use any external knowledge. Only use what is written in the text.
- Return ONLY valid JSON with this exact structure. No explanation. No markdown fences.

{"question": "...", "answer": "..."}""")


def _parse_json(raw: str) -> dict | None:
    """Try to parse JSON from LLM response, stripping code fences."""
    raw = raw.strip()
    # Remove markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Extract first {...} block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def generate_golden_dataset():
    chunks_path = _ROOT / "corpus" / "all_chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Stratified sampling from 3 buckets
    who_en  = [c for c in all_chunks if c["metadata"].get("source") == "who_guidelines"   and len(c["text"]) > 350]
    med_en  = [c for c in all_chunks if c["metadata"].get("source") == "medlineplus" and c["metadata"].get("language","en") == "en" and len(c["text"]) > 350]
    med_es  = [c for c in all_chunks if c["metadata"].get("source") == "medlineplus" and c["metadata"].get("language")         == "es" and len(c["text"]) > 350]

    # 5 WHO, 4 MedlinePlus English, 2 MedlinePlus Spanish
    pool = (
        random.sample(who_en, min(len(who_en), 5)) +
        random.sample(med_en, min(len(med_en), 4)) +
        random.sample(med_es, min(len(med_es), 2))
    )
    random.shuffle(pool)
    pool = pool[:NUM_SAMPLES]
    logging.info(f"Stratified sample: {len(pool)} chunks (target {NUM_SAMPLES})")

    llm = ChatNVIDIA(model=LLM_MODEL, api_key=NVIDIA_API_KEY, temperature=0.2)
    dataset = []

    for chunk in tqdm(pool, desc="Generating Golden Set"):
        text = chunk["text"]

        # Step 1: Generate Q&A from chunk
        try:
            resp = llm.invoke([SYS_MSG, HumanMessage(content=f"Text:\n{text}")])
            qa   = _parse_json(resp.content)
        except Exception as e:
            logging.warning(f"LLM failed: {e}")
            continue

        if not qa or not qa.get("question") or not qa.get("answer"):
            logging.warning("Skipping malformed LLM response")
            continue

        # Step 2: Hit the RAG endpoint with the generated question
        try:
            r = requests.post(API_URL, json={"question": qa["question"], "top_k": 3, "expand_query": False}, timeout=60)
            if r.status_code != 200:
                logging.warning(f"RAG API returned {r.status_code}")
                continue
            rag = r.json()
        except Exception as e:
            logging.warning(f"RAG API failed: {e}")
            continue

        rag_answer  = rag.get("answer", "")
        raw_chunks  = rag.get("chunks", [])
        # Keep contexts short (≤500 chars per chunk) to fit LLM judge token limits
        contexts    = [c.get("text", "") for c in raw_chunks]
        source_meta = chunk["metadata"]

        if not rag_answer or not contexts:
            logging.warning("Empty answer/contexts from RAG")
            continue

        dataset.append({
            "question":     qa["question"],
            "ground_truth": qa["answer"],
            "answer":       rag_answer,
            "contexts":     contexts,
            "source":       source_meta.get("source", "unknown"),
            "language":     source_meta.get("language", "en"),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logging.info(f"\n✅ Golden Set saved: {len(dataset)} samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_golden_dataset()
