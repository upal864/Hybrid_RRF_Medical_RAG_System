import json
import logging
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    chunks_path = Path("corpus/all_chunks.json")
    if not chunks_path.exists():
        logging.error(f"Missing {chunks_path}")
        return

    with open(chunks_path, encoding='utf-8') as f:
        chunks_data = json.load(f)

    logging.info(f"Loaded {len(chunks_data)} chunks for embedding.")

    # Convert to LangChain Documents
    documents = []
    for chunk in chunks_data:
        metadata = chunk.get("metadata", {}).copy()
        metadata["chunk_id"] = chunk["chunk_id"]
        metadata["parent_doc_id"] = chunk["parent_doc_id"]
        
        # Chroma requires metadata values to be int, float, str, or bool
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)
                
        doc = Document(page_content=chunk["text"], metadata=clean_metadata)
        documents.append(doc)

    # Clean existing Chroma DB to make a fresh one
    persist_directory = "chroma_db_bge"
    if Path(persist_directory).exists():
        logging.info(f"Deleting existing database at {persist_directory}...")
        shutil.rmtree(persist_directory)

    # Initialize Embeddings (bge-m3)
    logging.info("Initializing BAAI/bge-m3 embedding model on CPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logging.info(f"Initializing fresh Chroma DB in {persist_directory}...")
    vectorstore = Chroma(
        collection_name="rag_corpus_bge",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    batch_size = 300  # Local models can be heavy, smaller batch is safer for RAM
    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(documents))
        batch = documents[start_idx:end_idx]
        
        logging.info(f"Processing batch {i+1}/{total_batches} (Chunks {start_idx} to {end_idx-1})...")
        
        try:
            vectorstore.add_documents(batch)
            logging.info(f"✅ Successfully embedded and added batch {i+1} to Chroma.")
        except Exception as e:
            logging.error(f"❌ Error adding batch {i+1}: {e}")
            raise e

    logging.info(f"🎉 All {len(documents)} chunks have been successfully embedded locally using BAAI/bge-m3 and stored in {persist_directory}!")

if __name__ == "__main__":
    main()
