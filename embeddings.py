import json
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings using a sentence-transformers model.

    Methods
    - embed_text(text) -> List[float]
    - embed_texts(list_of_texts) -> List[List[float]]
    - load_tickets(file_path) -> List[dict]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """Return a single embedding as a Python list."""
        emb = self.model.encode(text)
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        embs = self.model.encode(texts)
        return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

    @staticmethod
    def load_tickets(file_path: str = "tickets.json") -> List[dict]:
        """Load tickets JSON file and return list of ticket dicts."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tickets", [])


if __name__ == "__main__":
    # Load JSON data
    tickets = EmbeddingGenerator.load_tickets("tickets.json")

    # Extract the texts
    texts = [item["customer_query"] for item in tickets]

    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(texts)

    # Print embeddings
    for i, emb in enumerate(embeddings):
        print(f"ID: {tickets[i]['ticket_id']}, Text: {tickets[i]['customer_query']}, Embedding vector size: {emb[:5]}...")  # print first 5 values for brevity
