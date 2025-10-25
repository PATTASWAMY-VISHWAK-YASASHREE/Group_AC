#!/usr/bin/env python
"""
Delete the old Chroma collection and re-ingest with corrected embeddings.
This ensures we only embed customer_query, not the full resolution.
"""
import os
from dotenv import load_dotenv
from chroma import ChromaClientWrapper

load_dotenv()

# Delete old collection
try:
    client = ChromaClientWrapper()
    # Get the raw client to delete collection
    client.client.delete_collection(name="tickets")
    print("✓ Old collection 'tickets' deleted")
except Exception as e:
    print(f"Note: Could not delete collection (may not exist): {e}")

print("\nNow run: python ingest.py --tickets tickets.json")
print("This will ingest with customer_query embeddings only.")
