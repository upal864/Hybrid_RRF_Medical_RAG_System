"""
eval/evaluate_rag.py

Evaluates the RAG pipeline using Ragas metrics:
  - faithfulness        (hallucination check — are answers grounded in context?)
  - answer_relevancy    (does the answer actually address the question?)

Uses eval/ragas_dataset.json → outputs eval/evaluation_results.csv
"""

import sys, json, logging, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATASET_FILE  = _ROOT / "eval" / "ragas_dataset.json"
OUTPUT_FILE   = _ROOT / "eval" / "evaluation_results_final.csv"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    logging.warning("NVIDIA_API_KEY not found in environment variables.")
LLM_MODEL     = "openai/gpt-oss-120b"
EMBED_MODEL   = "BAAI/bge-m3"


def run_evaluation():
    if not DATASET_FILE.exists():
        logging.error("Dataset not found. Run generate_golden_set.py first.")
        return

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logging.error("Dataset is empty.")
        return

    hf_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in data:
        if not all(k in item for k in ["question", "answer", "contexts", "ground_truth"]):
            continue
        if not item["answer"] or not item["contexts"]:
            continue
        hf_data["question"].append(item["question"])
        hf_data["answer"].append(item["answer"])
        hf_data["contexts"].append(item["contexts"])
        hf_data["ground_truth"].append(item["ground_truth"])

    if not hf_data["question"]:
        logging.error("No valid samples after filtering.")
        return

    dataset = Dataset.from_dict(hf_data)
    logging.info(f"Evaluating {len(dataset)} samples...")

    eval_llm = ChatNVIDIA(
        model=LLM_MODEL, 
        api_key=NVIDIA_API_KEY, 
        temperature=0.0,
        max_tokens=2048 # Using max_tokens instead of deprecated model_kwargs
    )
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    run_cfg = RunConfig(
        max_retries=5,
        max_wait=45,
        max_workers=1,   # Serial to completely avoid 429s
    )

    logging.info("Running Ragas metrics: faithfulness, answer_relevancy ...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    logging.info(f"\n=== RAGAS SCORES ===\n{result}")

    df = result.to_pandas()
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"✅ Results saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_evaluation()
