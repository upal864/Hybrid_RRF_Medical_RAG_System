import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_markdowns(folder: str) -> List[Dict[str, Any]]:
    docs = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        logging.error(f"Directory not found: {folder}")
        return docs

    md_files = sorted(folder_path.glob("*.md"))
    logging.info(f"Found {len(md_files)} markdown files to process.")
    
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
            
            if len(text) < 100:  # skip near-empty files
                logging.debug(f"Skipping {md_file.name} (too short)")
                continue
            
            # Derive a clean title from filename: "Abdominal_Pain.md" -> "Abdominal Pain"
            title = md_file.stem.replace("_", " ")
            
            # Better heuristic for language detection based on common Spanish words in the text
            spanish_keywords = [" tratamiento ", " síntomas ", " causas ", " prevención ", " médicos ", " salud ", " enfermedad "]
            is_spanish = any(keyword in text.lower() for keyword in spanish_keywords)
            
            docs.append({
                "text": f"# {title}\n\n{text}",
                "metadata": {
                    "source":    "medlineplus",
                    "filename":  md_file.name,
                    "title":     title,
                    "doc_type":  "health_topic",
                    "language":  "es" if is_spanish else "en",
                }
            })
        except Exception as e:
            logging.error(f"Failed to process {md_file.name}: {e}")
    
    logging.info(f"✅ Successfully parsed {len(docs)} markdown files")
    return docs

if __name__ == "__main__":
    docs_md = parse_markdowns("corpus/markdowns")
    
    output_file = Path("corpus/parsed_markdowns.json")
    try:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(docs_md, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved parsed markdowns to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")
