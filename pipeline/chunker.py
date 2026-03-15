import json
import logging
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def main():
    # ─── Initialize splitters ──────────────────────────────────────────────────
    # Standard splitting for dense PDFs
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Markdown specific splitting for MedlinePlus structure
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=200,
        length_function=count_tokens,
    )

    # ─── Load parsed files ─────────────────────────────────────────────────────
    docs_md = []
    docs_pdf = []
    
    md_path = Path("corpus/parsed_markdowns.json")
    if md_path.exists():
        with open(md_path, encoding="utf-8") as f:
            docs_md = json.load(f)
        logger.info(f"Loaded {len(docs_md)} markdown docs")
    else:
        logger.warning(f"File not found: {md_path}")

    pdf_path = Path("corpus/parsed_pdfs.json")
    if pdf_path.exists():
        with open(pdf_path, encoding="utf-8") as f:
            docs_pdf = json.load(f)
        logger.info(f"Loaded {len(docs_pdf)} PDF docs")
    else:
        logger.warning(f"File not found: {pdf_path}")

    # ─── Chunk Markdowns (language-aware) ──────────────────────────────────────
    md_chunks = []
    skipped_md = 0

    for doc_id, doc in enumerate(docs_md):
        # Handle cases where markdown parser extracted list/dict text by accident
        text = str(doc.get("text", ""))
        
        splits = md_splitter.split_text(text)
        
        for i, chunk_text in enumerate(splits):
            token_count = count_tokens(chunk_text)
            
            # Skip noise chunks (e.g., lone headers with no content, very short strings)
            if token_count < 30:
                skipped_md += 1
                continue
            
            md_chunks.append({
                "chunk_id":      f"md_{doc_id}_chunk{i}",
                "parent_doc_id": f"md_{doc_id}",
                "text":          chunk_text,
                "token_count":   token_count,
                "splitter":      "markdown_language_aware",
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks_in_doc": len(splits),
                }
            })

    logger.info(f"Markdown chunks created : {len(md_chunks)}")
    logger.info(f"   Skipped (too short)  : {skipped_md}")

    # ─── Chunk PDFs (standard recursive) ───────────────────────────────────────
    pdf_chunks = []
    skipped_pdf = 0

    for doc_id, doc in enumerate(docs_pdf):
        text = str(doc.get("text", ""))
        
        splits = pdf_splitter.split_text(text)
        
        for i, chunk_text in enumerate(splits):
            token_count = count_tokens(chunk_text)
            
            if token_count < 30:
                skipped_pdf += 1
                continue
            
            pdf_chunks.append({
                "chunk_id":      f"pdf_{doc_id}_chunk{i}",
                "parent_doc_id": f"pdf_{doc_id}",
                "text":          chunk_text,
                "token_count":   token_count,
                "splitter":      "recursive_standard",
                "metadata": {
                    **doc.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks_in_doc": len(splits),
                }
            })

    logger.info(f"PDF chunks created      : {len(pdf_chunks)}")
    logger.info(f"   Skipped (too short)  : {skipped_pdf}")

    # ─── Merge & Stats ─────────────────────────────────────────────────────────
    all_chunks = md_chunks + pdf_chunks

    if not all_chunks:
        logger.warning("No chunks generated. Check your source files.")
        return

    counts = [c["token_count"] for c in all_chunks]
    
    print(f"\n{'─'*45}")
    print(f"📦 TOTAL CHUNKS      : {len(all_chunks)}")
    print(f"📊 Average tokens    : {sum(counts)/len(counts):.1f}")
    print(f"📈 Max tokens        : {max(counts)}")
    print(f"📉 Min tokens        : {min(counts)}")
    print(f"  ├─ From markdowns  : {len(md_chunks)}")
    print(f"  └─ From PDFs       : {len(pdf_chunks)}")
    print(f"{'─'*45}")

    # ─── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("corpus/all_chunks.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved → {out_path}")
    except Exception as e:
        logger.error(f"Failed to save {out_path}: {e}")

    # ─── Quick sanity check ────────────────────────────────────────────────────
    if md_chunks and len(md_chunks) > 10:
        print("\n── SAMPLE MARKDOWN CHUNK ──")
        print(f"  chunk_id     : {md_chunks[10]['chunk_id']}")
        print(f"  parent_doc_id: {md_chunks[10]['parent_doc_id']}")
        print(f"  tokens       : {md_chunks[10]['token_count']}")
        print(f"  metadata.doc_type : {md_chunks[10]['metadata'].get('doc_type')}")
        print(f"  preview      : {md_chunks[10]['text'][:120]}...\n")

    if pdf_chunks:
        print("\n── SAMPLE PDF CHUNK ──")
        print(f"  chunk_id     : {pdf_chunks[0]['chunk_id']}")
        print(f"  parent_doc_id: {pdf_chunks[0]['parent_doc_id']}")
        print(f"  tokens       : {pdf_chunks[0]['token_count']}")
        print(f"  metadata.doc_type : {pdf_chunks[0]['metadata'].get('doc_type')}")
        print(f"  preview      : {pdf_chunks[0]['text'][:120]}...\n")

if __name__ == "__main__":
    main()
