"""
scripts/build_kb.py
----------------------
Builds the local knowledge base queried by core/subagents.RAGRetrieverAgent
(Phase 4). Run standalone: `python scripts/build_kb.py` whenever spec/ files
or reference docs change.

Chunks: spec/*.yaml (platform policies), any files under kb_sources/ (e.g.
brand guidelines, past winning listings, category-specific rules).
"""

import os
import glob
import logging

import yaml
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("build_kb")

SPEC_DIR = os.getenv("SPEC_DIR", "spec")
KB_SOURCES_DIR = os.getenv("KB_SOURCES_DIR", "kb_sources")
CHROMA_PATH = os.getenv("CHROMA_PATH", ".chroma_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def load_sources() -> list:
    docs = []

    for path in glob.glob(os.path.join(SPEC_DIR, "*.yaml")):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        text = yaml.dump(data)
        docs.append({"source": os.path.basename(path), "text": text})

    if os.path.isdir(KB_SOURCES_DIR):
        for path in glob.glob(os.path.join(KB_SOURCES_DIR, "**/*.*"), recursive=True):
            if path.lower().endswith((".txt", ".md")):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append({"source": os.path.basename(path), "text": f.read()})

    return docs


class KBClient:
    """Thin wrapper handed to core/manager.py as `kb_client`."""

    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("listing_kb")
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def query(self, text: str, top_k: int = 8) -> list:
        embedding = self.embedder.encode([text]).tolist()
        result = self.collection.query(query_embeddings=embedding, n_results=top_k)
        docs = result.get("documents", [[]])
        return docs[0] if docs else []


def build():
    logging.basicConfig(level=logging.INFO)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("listing_kb")
    embedder = SentenceTransformer(EMBED_MODEL)

    docs = load_sources()
    logger.info("Loaded %s source documents", len(docs))

    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})

    if not texts:
        logger.warning("No text found to index. Check SPEC_DIR / KB_SOURCES_DIR.")
        return

    embeddings = embedder.encode(texts).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    logger.info("Indexed %s chunks into %s", len(texts), CHROMA_PATH)


if __name__ == "__main__":
    build()