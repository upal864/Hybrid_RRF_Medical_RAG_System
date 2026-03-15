import pdfplumber
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    try:
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 60:
                    pages.append(text.strip())
        
        full_text = "\n\n".join(pages)
        
        if len(full_text) < 300:
            logging.warning(f"Skipping (too short): {pdf_path.name}")
            return None
        
        return {
            "text": full_text,
            "metadata": {
                "source":     "who_guidelines",
                "filename":   pdf_path.name,
                "page_count": len(pages),
                "doc_type":   "clinical_guideline",
            }
        }
    except Exception as e:
        logging.error(f"Error on {pdf_path.name}: {e}")
        return None

if __name__ == "__main__":
    docs_pdf: List[Dict[str, Any]] = []
    pdf_dir = Path("corpus/pdfs")
    
    if not pdf_dir.exists():
        logging.error(f"Directory not found: {pdf_dir}")
        exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    logging.info(f"Starting to parse {len(pdf_files)} PDF files in {pdf_dir}")

    for pdf_file in sorted(pdf_files):
        logging.info(f"Parsing {pdf_file.name}...")
        doc = parse_pdf(pdf_file)
        if doc:
            docs_pdf.append(doc)

    logging.info(f"✅ Parsed {len(docs_pdf)} WHO PDFs")

    output_file = Path("corpus/parsed_pdfs.json")
    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(docs_pdf, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved parsed PDFs to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
