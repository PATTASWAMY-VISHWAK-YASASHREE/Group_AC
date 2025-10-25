import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

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

# In-memory session storage: {session_id: {"messages": [...], "created": datetime, "updated": datetime}}
SESSIONS: Dict[str, Dict] = {}


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    needs_escalation: bool


class SessionChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 3


class SessionChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[str]
    needs_escalation: bool
    conversation_history: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]


class SessionInfoResponse(BaseModel):
    session_id: str
    created_at: str
    last_updated: str
    message_count: int


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
        "components": ["embedding", "chroma", "rag_chain", "llm"],
        "sessions_active": len(SESSIONS)
    }


# ============ Helper Functions for Session Management ============

def create_session(session_id: str) -> None:
    """Create a new chat session."""
    SESSIONS[session_id] = {
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat()
    }


def add_to_session(session_id: str, role: str, content: str) -> None:
    """Add a message to a session's conversation history."""
    if session_id not in SESSIONS:
        create_session(session_id)
    
    SESSIONS[session_id]["messages"].append({"role": role, "content": content})
    SESSIONS[session_id]["last_updated"] = datetime.utcnow().isoformat()


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history for a session."""
    if session_id not in SESSIONS:
        return []
    return SESSIONS[session_id]["messages"]


def format_history_for_context(history: List[Dict[str, str]]) -> str:
    """Format conversation history as context for the RAG chain."""
    if not history:
        return ""
    
    history_text = "CONVERSATION HISTORY:\n"
    for msg in history[-6:]:  # Keep last 6 messages for context (3 turns)
        role = "Customer" if msg["role"] == "user" else "Support Agent"
        history_text += f"{role}: {msg['content']}\n"
    
    return history_text + "\n---\n\n"


# ============ Stateless Chat Endpoint (no session) ============

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint using LangChain RAG orchestration (stateless).

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


# ============ Session-based Chat Endpoint (with history) ============

@app.post("/session/chat", response_model=SessionChatResponse)
def session_chat(req: SessionChatRequest):
    """
    Chat endpoint with conversation history (stateful).
    
    The LLM can see and reference previous messages in the conversation.
    Session history is stored in-memory and will be cleared on app restart.
    
    Args:
        req: SessionChatRequest with session_id, query, and optional top_k
    
    Returns:
        SessionChatResponse with answer, sources, escalation flag, and full conversation history
    """
    try:
        # Validate input
        if not req.query or len(req.query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query too short (min 3 chars)")
        
        if not req.session_id or len(req.session_id.strip()) < 1:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        # Create session if it doesn't exist
        if req.session_id not in SESSIONS:
            create_session(req.session_id)
        
        # Step 1: Embed query
        q_emb = embedding_gen.embed_text(req.query)
        
        # Step 2: Search Chroma
        search_results = chroma_client.search_by_embedding(q_emb, n_results=req.top_k)
        
        if not search_results or not search_results.get("documents"):
            answer = "I don't have similar cases in my database. Please contact our support team."
            add_to_session(req.session_id, "user", req.query)
            add_to_session(req.session_id, "assistant", answer)
            return SessionChatResponse(
                session_id=req.session_id,
                answer=answer,
                sources=[],
                needs_escalation=True,
                conversation_history=get_session_history(req.session_id)
            )
        
        # Step 3-4: Run RAG pipeline
        rag_result = rag_chain.run(req.query, search_results)
        
        # Step 5: Add to session history
        add_to_session(req.session_id, "user", req.query)
        add_to_session(req.session_id, "assistant", rag_result["answer"])
        
        return SessionChatResponse(
            session_id=req.session_id,
            answer=rag_result["answer"],
            sources=rag_result["source_tickets"],
            needs_escalation=rag_result["needs_escalation"],
            conversation_history=get_session_history(req.session_id)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/session/{session_id}")
def get_session_info(session_id: str) -> SessionInfoResponse:
    """Get information about a session (message count, timestamps)."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    return SessionInfoResponse(
        session_id=session_id,
        created_at=session["created_at"],
        last_updated=session["last_updated"],
        message_count=len(session["messages"])
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear (delete) a session and all its history."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    del SESSIONS[session_id]
    return {"message": f"Session {session_id} cleared"}


@app.get("/sessions")
def list_sessions():
    """List all active sessions with summary info."""
    sessions_info = []
    for sid, session in SESSIONS.items():
        sessions_info.append({
            "session_id": sid,
            "created_at": session["created_at"],
            "message_count": len(session["messages"])
        })
    return {"sessions": sessions_info, "total": len(sessions_info)}


# ============ Debug Endpoint ============

@app.post("/chat/debug")
def debug_chat(req: ChatRequest):
    """
    Debug endpoint for testing RAG pipeline step-by-step.
    
    Returns intermediate results: retrieved documents, formatted context, and LLM response.
    """
    try:
        q_emb = embedding_gen.embed_text(req.query)
        search_results = chroma_client.search_by_embedding(q_emb, n_results=req.top_k)
        rag_result = rag_chain.run(req.query, search_results)

        return {
            "query": req.query,
            "retrieved_count": len(search_results.get("documents", [])),
            "retrieved_case_ids": search_results.get("ids", []),
            "answer": rag_result["answer"],
            "needs_escalation": rag_result["needs_escalation"],
            "source_tickets": rag_result["source_tickets"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
