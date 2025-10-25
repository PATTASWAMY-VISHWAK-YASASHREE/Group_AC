import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings import EmbeddingGenerator
from chroma import ChromaClientWrapper
from llm import LLMClient


class ChatRequest(BaseModel):
	query: str
	top_k: int = 3


class ChatResponse(BaseModel):
	answer: str
	sources: List[str]


app = FastAPI(title="Telecom AI Support Agent")


def build_prompt(user_query: str, contexts: List[dict]) -> str:
	"""Create a simple RAG prompt for the LLM using ticket contexts."""
	ctx_lines = []
	for c in contexts:
		# include ticket id and snippet
		tid = c.get("id") or c.get("metadata", {}).get("ticket_id")
		doc = c.get("document") or c.get("metadata", {}).get("customer_query")
		if tid and doc:
			ctx_lines.append(f"[{tid}] {doc}")
	context_text = "\n".join(ctx_lines)

	prompt = (
		"You are a helpful telecom support assistant. Use simple English and short sentences so a customer can understand. "
		"When you reference past tickets, include the ticket id in square brackets.\n\n"
		"Context from past tickets:\n" + (context_text or "(no context found)") + "\n\n"
		f"User question: {user_query}\n\n"
		"Answer concisely. If you suggest steps, keep them actionable and short."
	)
	return prompt


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
	# initialize components; in production these would be created once on startup
	try:
		embedder = EmbeddingGenerator()
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Embedding model error: {e}")

	try:
		chroma_client = ChromaClientWrapper()
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Chroma client error: {e}")

	try:
		llm = LLMClient()
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"LLM client error: {e}")

	# embed query
	q_emb = embedder.embed_text(req.query)

	# retrieve top-k contexts
	hits = chroma_client.search_by_embedding(q_emb, n_results=req.top_k)
	print("Retrieved hits:", hits)
	prompt = build_prompt(req.query, hits)

	answer = llm.generate(prompt)

	sources = [h.get("id") for h in hits]

	return ChatResponse(answer=answer, sources=sources)


if __name__ == "__main__":
	import uvicorn

	port = int(os.environ.get("PORT", 8000))
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

