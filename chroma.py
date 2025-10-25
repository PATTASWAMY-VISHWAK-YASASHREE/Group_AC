import os
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

class ChromaClientWrapper:
  """Wrapper around a Chroma CloudClient for simple upsert and search operations.

  Expects environment variables or explicit args:
    - CHROMA_API_KEY
    - CHROMA_TENANT
    - CHROMA_DB
  """

  def __init__(self, collection_name: str = "tickets", api_key: str | None = None, tenant: str | None = None, database: str | None = None):
    api_key = api_key or os.getenv("CHROMA_API_KEY")
    tenant = tenant or os.getenv("CHROMA_TENANT")
    database = database or os.getenv("CHROMA_DB")

    if not api_key:
      raise ValueError("CHROMA_API_KEY is required in environment or constructor")

    # Create CloudClient
    self.client = chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)

    # get or create collection
    try:
      self.collection = self.client.get_collection(collection_name)
    except Exception:
      self.collection = self.client.create_collection(collection_name)

  def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
    """Add or update documents in the collection."""
    return self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

  def search_by_embedding(self, query_embedding: List[float], n_results: int = 3) -> Dict[str, Any]:
    """Query the collection using a vector and return Chroma result format.

    Returns a dict with keys: 'ids', 'documents', 'metadatas', 'distances'
    Compatible with format expected by rag_chain.py
    """
    result = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    # Chroma returns nested lists; unpack and structure for downstream use
    if not result or not result.get("documents"):
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    # Extract first result set (query returns list of lists)
    docs_list = result.get("documents", [[]])[0] if result.get("documents") else []
    metas_list = result.get("metadatas", [[]])[0] if result.get("metadatas") else []
    dists_list = result.get("distances", [[]])[0] if result.get("distances") else []

    # Build IDs list from metadata ticket_id
    ids_list = []
    for i, meta in enumerate(metas_list):
        tid = meta.get("ticket_id", f"unknown_{i}") if meta else f"unknown_{i}"
        ids_list.append(tid)

    return {
        "ids": ids_list,
        "documents": docs_list,
        "metadatas": metas_list,
        "distances": dists_list
    }

if __name__ == "__main__":
  print("Chroma client wrapper loaded. Set CHROMA_API_KEY env var to use.")