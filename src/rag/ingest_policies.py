# src/rag/ingest_policies.py
import os
import json
from pathlib import Path
from uuid import uuid4
from time import time

# LangChain imports (canonical)
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# sentence-transformers model (local MiniLM)
from sentence_transformers import SentenceTransformer

# ---------------- paths ----------------
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # src/rag -> src -> project root
POLICIES_DIR = PROJECT_ROOT / "Policy"
LOGS_DIR = CURRENT_DIR / "logs"
DATA_DIR = CURRENT_DIR / "data"
CHUNK_MANIFEST = LOGS_DIR / "chunks_manifest.jsonl"
FAISS_DIR = DATA_DIR / "faiss_index"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# ---------------- chunking ----------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class MiniLMEmbeddings(Embeddings):
    """
    Minimal embeddings adapter for LangChain/FAISS:
    - embed_documents(list[str]) -> List[List[float]]
    - embed_query(str) -> List[float]
    - __call__(str) -> List[float]  (so older LangChain/FAISS code that calls the object works)
    """
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # sentence-transformers returns numpy arrays; convert to lists
        arr = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return [a.tolist() for a in arr]

    def embed_query(self, text):
        arr = self.model.encode([text], batch_size=1, show_progress_bar=False)
        return arr[0].tolist()

    # Make the adapter callable for single-text embedding (some LangChain FAISS versions call embedding_fn(text))
    def __call__(self, text):
        # If a list is passed, treat as documents; if a single string, treat as query
        if isinstance(text, (list, tuple)):
            return self.embed_documents(list(text))
        return self.embed_query(text)


# instantiate adapter
embeddings = MiniLMEmbeddings()

# ---------------- ingestion params ----------------
EMBED_BATCH = 32       # batch size for processing chunks
ID_PREFIX = "policy_chunk_"

def build_or_load_faiss():
    """Load existing FAISS index if present, else return None."""
    try:
        # FAISS.load_local expects an embeddings object with embed_documents/embed_query
        store = FAISS.load_local(str(FAISS_DIR), embeddings)
        print(f"Loaded existing FAISS index from {FAISS_DIR}")
        return store
    except Exception:
        print("No existing FAISS index found. A new one will be created incrementally.")
        return None

def ingest_all_policies():
    print(f"Starting ingestion. Policies dir: {POLICIES_DIR}")
    txt_files = sorted(POLICIES_DIR.glob("*.txt"))
    if not txt_files:
        print("No policy files found. Nothing to do.")
        return

    faiss_store = build_or_load_faiss()
    manifest_entries = []

    for file_idx, file_path in enumerate(txt_files, start=1):
        doc_id = file_path.name
        print(f"\nProcessing file {file_idx}/{len(txt_files)}: {doc_id}")

        loader = TextLoader(str(file_path), encoding="utf-8")
        try:
            docs = loader.load()
        except Exception as e:
            print(f"  ❗ Error loading {doc_id}: {e}")
            continue

        chunks = splitter.split_documents(docs)
        texts = []
        metadatas = []
        ids = []

        for idx, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if not text:
                continue
            texts.append(text)
            metadatas.append({"doc_id": doc_id, "chunk_id": idx, "source_path": str(file_path)})
            ids.append(ID_PREFIX + str(uuid4()))
            manifest_entries.append({"doc_id": doc_id, "chunk_id": idx, "preview": text[:120] + "..."})

        if not texts:
            print("  (no chunks extracted)")
            continue

        # Embed in small batches and upsert incrementally
        n = len(texts)
        print(f"  Embedding {n} chunks in batches of {EMBED_BATCH}...")
        start_time = time()
        for i in range(0, n, EMBED_BATCH):
            batch_texts = texts[i : i + EMBED_BATCH]
            batch_meta = metadatas[i : i + EMBED_BATCH]
            batch_ids = ids[i : i + EMBED_BATCH]

            if faiss_store is None:
                # Create index from first batch
                print(f"    Creating FAISS index from first batch of {len(batch_texts)} texts...")
                faiss_store = FAISS.from_texts(
                    texts=batch_texts,
                    embedding=embeddings,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )
                print("    FAISS index created.")
            else:
                # Upsert new texts
                faiss_store.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)

            processed = min(i + EMBED_BATCH, n)
            elapsed = time() - start_time
            print(f"    Processed {processed}/{n} chunks for {doc_id} (elapsed {elapsed:.1f}s)")

        # Save index after processing this file
        try:
            faiss_store.save_local(str(FAISS_DIR)) #type: ignore
            print(f"  Saved FAISS index to {FAISS_DIR}")
        except Exception as e:
            print(f"  ❗ Error saving FAISS index: {e}")

    # Write manifest file
    with open(CHUNK_MANIFEST, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    print("\nIngestion complete. Manifest written to:", CHUNK_MANIFEST)


if __name__ == "__main__":
    ingest_all_policies()
