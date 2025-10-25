import os
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embeddings import EmbeddingGenerator
from chroma import ChromaClientWrapper
from rag_chain import TelecomRAGChain

load_dotenv()

# Initialize components once at startup (not per-request for performance)
try:
    embedding_gen = EmbeddingGenerator()
    chroma_client = ChromaClientWrapper()
    rag_chain = TelecomRAGChain()
except Exception as e:
    raise RuntimeError(f"Failed to initialize components: {e}")


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    needs_escalation: bool


app = FastAPI(
    title="Telecom Support AI Agent",
    description="RAG-powered telecom support using LangChain + Chroma + Gemini",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Telecom Support AI Agent",
        "components": ["embedding", "chroma", "rag_chain", "llm"]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint using LangChain RAG orchestration.

    Flow:
    1. Embed user query
    2. Retrieve similar tickets from Chroma
    3. Generate response using LangChain RAG chain
    4. Check if escalation needed
    5. Return answer + metadata

    Args:
        req: ChatRequest with query and optional top_k (default 3)

    Returns:
        ChatResponse with answer, source tickets, and escalation flag
    """
    try:
        # Validate input
        if not req.query or len(req.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query too short. Minimum 3 characters required."
            )

        # Step 1: Embed query
        q_emb = embedding_gen.embed_text(req.query)

        # Step 2: Search Chroma for similar tickets
        search_results = chroma_client.search_by_embedding(
            q_emb,
            n_results=req.top_k
        )

        if not search_results or not search_results.get("documents"):
            return ChatResponse(
                answer="I don't have similar cases in my database. Please contact our support team for assistance.",
                sources=[],
                needs_escalation=True
            )

        # Step 3-4: Run LangChain RAG pipeline
        rag_result = rag_chain.run(req.query, search_results)

        return ChatResponse(
            answer=rag_result["answer"],
            sources=rag_result["source_tickets"],
            needs_escalation=rag_result["needs_escalation"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/chat/debug")
def chat_debug(req: ChatRequest):
    """
    Debug endpoint that returns full RAG pipeline details.
    Useful for understanding how the system retrieved and answered.
    """
    try:
        q_emb = embedding_gen.embed_text(req.query)
        search_results = chroma_client.search_by_embedding(q_emb, n_results=req.top_k)
        rag_result = rag_chain.run(req.query, search_results)

        return {
            "query": req.query,
            "retrieved_count": len(search_results.get("documents", [])),
            "retrieved_cases": search_results.get("ids", []),
            "answer": rag_result["answer"],
            "needs_escalation": rag_result["needs_escalation"],
            "confidence": rag_result["confidence"],
            "formatted_context": rag_chain.format_context(search_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
