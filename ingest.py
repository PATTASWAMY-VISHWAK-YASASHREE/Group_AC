import os
import argparse
from typing import List
from dotenv import load_dotenv

from embeddings import EmbeddingGenerator
from chroma import ChromaClientWrapper

# Load env vars from .env file
load_dotenv()


def build_document(ticket: dict) -> str:
    """Return only the customer query for embedding."""
    return ticket.get("customer_query", "").strip()


def ingest(tickets_file: str, dry_run: bool = False, collection_name: str = "tickets"):
    # load tickets
    tickets = EmbeddingGenerator.load_tickets(tickets_file)
    if not tickets:
        print("No tickets found in", tickets_file)
        return

    embedder = EmbeddingGenerator()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []

    texts_for_embedding: List[str] = []

    for t in tickets:
        tid = t.get("ticket_id") or t.get("id")
        ids.append(tid)
        doc = build_document(t)
        documents.append(doc)
        # Save useful metadata including resolution for reference during search
        meta = {
            k: t.get(k) for k in ("ticket_id", "category", "status", "priority", "created_date")
        }
        # Also add resolution to metadata so it's available after search
        if t.get("resolution"):
            meta["resolution"] = t["resolution"]
        metadatas.append(meta)
        # Embed only the customer query
        texts_for_embedding.append(doc)

    print(f"Generated {len(ids)} document entries. Generating embeddings...")
    embeddings = embedder.embed_texts(texts_for_embedding)

    if dry_run:
        print("Dry run enabled — not upserting to Chroma. Sample output:")
        for i in range(min(3, len(ids))):
            print(f"ID={ids[i]}, meta={metadatas[i]}, doc_preview={documents[i][:120]}...")
        return

    # create chroma wrapper and upsert
    chroma = ChromaClientWrapper(collection_name=collection_name)

    print("Upserting embeddings to Chroma collection:", collection_name)
    resp = chroma.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    print("Upsert response:", resp)


def main():
    parser = argparse.ArgumentParser(description="Ingest tickets.json into Chroma with embeddings")
    parser.add_argument("--tickets", default="tickets.json", help="Path to tickets.json")
    parser.add_argument("--dry-run", action="store_true", help="Generate embeddings but don't upsert to Chroma")
    parser.add_argument("--collection", default="tickets", help="Chroma collection name")
    args = parser.parse_args()

    ingest(tickets_file=args.tickets, dry_run=args.dry_run, collection_name=args.collection)


if __name__ == "__main__":
    main()
