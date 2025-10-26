import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from embeddings import EmbeddingGenerator
from chroma import ChromaClientWrapper
from rag_chain import TelecomRAGChain
from userdata_manager import get_user_manager, get_user_by_phone, format_user_context, extract_phone_number

load_dotenv()

# Initialize components once at startup (not per-request for performance)
try:
    embedding_gen = EmbeddingGenerator()
    chroma_client = ChromaClientWrapper()
    rag_chain = TelecomRAGChain()
    user_manager = get_user_manager()
except Exception as e:
    raise RuntimeError(f"Failed to initialize components: {e}")

# In-memory session storage: {session_id: {"messages": [...], "created": datetime, "updated": datetime, "type": "chat|voice", "caller": None}}
SESSIONS: Dict[str, Dict] = {}


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    phone_number: Optional[str] = None  # Optional phone number for user context


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    needs_escalation: bool


class SessionChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 3
    phone_number: Optional[str] = None  # Optional phone number for user context


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

# ============ CORS Configuration ============
# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
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

def create_session(session_id: str, session_type: str = "chat", caller: Optional[str] = None) -> None:
    """Create a new chat or voice session.
    
    Args:
        session_id: Unique identifier for the session
        session_type: "chat" for text, "voice" for phone calls
        caller: Phone number for voice sessions
    """
    SESSIONS[session_id] = {
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "type": session_type,
        "caller": caller
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


def get_ai_answer_via_rag(question: str, phone_number: Optional[str] = None, session_id: Optional[str] = None, top_k: int = 3) -> Dict:
    """
    Get AI answer using the RAG chain with optional user data.
    
    Args:
        question: Customer's question
        phone_number: Optional phone number to load user data
        session_id: Optional session ID to include conversation history
        top_k: Number of similar tickets to retrieve
    
    Returns:
        Dict with "answer", "sources", and "needs_escalation"
    """
    try:
        # Load user data if phone number is provided
        user_context = ""
        if phone_number:
            user_data = get_user_by_phone(phone_number)
            if user_data:
                user_context = format_user_context(user_data)
                print(f"✓ Loaded user context for {phone_number}")
        
        # Embed query
        q_emb = embedding_gen.embed_text(question)
        
        # Search Chroma
        search_results = chroma_client.search_by_embedding(q_emb, n_results=top_k)
        
        if not search_results or not search_results.get("documents"):
            return {
                "answer": "I don't have similar cases in my database. Please contact our support team for assistance.",
                "sources": [],
                "needs_escalation": True
            }
        
        # Run RAG pipeline with user context
        rag_result = rag_chain.run(question, search_results, user_context=user_context)
        
        return rag_result
        
    except Exception as e:
        print(f"❌ RAG Chain Error: {e}")
        return {
            "answer": "I'm having trouble connecting to my knowledge base right now. Let me transfer you to a human agent who can help.",
            "sources": [],
            "needs_escalation": True
        }


# ============ Stateless Chat Endpoint (no session) ============

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint using LangChain RAG orchestration (stateless).

    Flow:
    1. Extract phone number from request (optional)
    2. Load user data if phone number provided
    3. Embed user query
    4. Retrieve similar tickets from Chroma
    5. Generate response using LangChain RAG chain with user + case context
    6. Check if escalation needed
    7. Return answer + metadata

    Args:
        req: ChatRequest with query, optional top_k (default 3), and optional phone_number

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

        # Extract phone number if provided
        phone_number = None
        if req.phone_number:
            phone_number = extract_phone_number(req.phone_number)
            if phone_number:
                print(f"📱 Using phone number: {phone_number}")

        # Step 1-2: Load user data if phone number available
        user_context = ""
        if phone_number:
            user_data = get_user_by_phone(phone_number)
            if user_data:
                user_context = format_user_context(user_data)

        # Step 3: Embed query
        q_emb = embedding_gen.embed_text(req.query)

        # Step 4: Search Chroma for similar tickets
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

        # Step 5-6: Run LangChain RAG pipeline with user context
        rag_result = rag_chain.run(req.query, search_results, user_context=user_context)

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
    User data is loaded from phone number if provided.
    Session history is stored in-memory and will be cleared on app restart.
    
    Args:
        req: SessionChatRequest with session_id, query, optional top_k, and optional phone_number
    
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
        
        # Extract phone number if provided
        phone_number = None
        if req.phone_number:
            phone_number = extract_phone_number(req.phone_number)
            if phone_number:
                print(f"📱 Using phone number for session: {phone_number}")
        
        # Load user data if phone number available
        user_context = ""
        if phone_number:
            user_data = get_user_by_phone(phone_number)
            if user_data:
                user_context = format_user_context(user_data)
        
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
        
        # Step 3-4: Run RAG pipeline with user context
        rag_result = rag_chain.run(req.query, search_results, user_context=user_context)
        
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
    
    Returns intermediate results: user data, retrieved documents, formatted context, and LLM response.
    """
    try:
        # Extract phone number if provided
        phone_number = None
        user_context = ""
        if req.phone_number:
            phone_number = extract_phone_number(req.phone_number)
            if phone_number:
                user_data = get_user_by_phone(phone_number)
                if user_data:
                    user_context = format_user_context(user_data)
        
        q_emb = embedding_gen.embed_text(req.query)
        search_results = chroma_client.search_by_embedding(q_emb, n_results=req.top_k)
        rag_result = rag_chain.run(req.query, search_results, user_context=user_context)

        return {
            "query": req.query,
            "phone_number": phone_number,
            "user_context_included": bool(user_context),
            "retrieved_count": len(search_results.get("documents", [])),
            "retrieved_case_ids": search_results.get("ids", []),
            "answer": rag_result["answer"],
            "needs_escalation": rag_result["needs_escalation"],
            "source_tickets": rag_result["source_tickets"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ VOICE CALL ENDPOINTS (Twilio Integration) ============

@app.post("/voice/incoming")
async def incoming_call(request: Request):
    """
    Handle incoming phone calls - personalized greeting with user name from userdata.
    Integrates with Twilio to capture speech input.
    """
    try:
        form_data = await request.form()
        caller = form_data.get('From', 'Unknown')
        call_sid = form_data.get('CallSid', 'unknown')
        
        print(f"\n📞 Incoming call from: {caller} (Call SID: {call_sid})")
        
        # Extract phone number and load user data for personalized greeting
        phone_number = extract_phone_number(caller)
        user_name = "there"
        user_context_loaded = False
        
        if phone_number:
            try:
                user_data = get_user_by_phone(phone_number)
                if user_data and "name" in user_data:
                    user_name = user_data["name"].split()[0]  # Get first name only
                    user_context_loaded = True
                    print(f"   ✓ User identified: {user_name}")
            except Exception as e:
                print(f"   ⚠️ Could not load user data: {e}")
        
        # Create voice session with user info
        create_session(call_sid, session_type="voice", caller=caller)
        SESSIONS[call_sid]["user_name"] = user_name
        SESSIONS[call_sid]["phone_number"] = phone_number
        
        # Personalized greeting
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Hello {user_name}! Welcome to TelecomCare AI Assistant.
        I can help you with any telecom issue.
        Please tell me your problem or question.
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        method="POST"
        language="en-IN" 
        speechTimeout="auto"
        timeout="8"
        hints="wifi, mobile data, internet, bill, payment, router, network, not working, slow, problem, recharge, plan, sim card">
        <Say voice="Polly.Aditi" language="en-IN">
            I'm listening. Please speak now.
        </Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">
        I didn't hear anything. Please call back when you're ready. Goodbye!
    </Say>
    <Hangup/>
</Response>"""
        
        return Response(content=twiml, media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in incoming_call: {e}")
        return Response(
            content="""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Sorry, something went wrong. Please try again later.
    </Say>
    <Hangup/>
</Response>""",
            media_type="application/xml"
        )


@app.post("/voice/process")
async def process_speech(
    SpeechResult: str = Form(None),
    Confidence: float = Form(0.0),
    From: str = Form(None),
    CallSid: str = Form(None)
):
    """
    Process speech input using the RAG chain with user context.
    After response, ask user YES/NO to continue or end call.
    """
    try:
        print(f"\n🎤 Speech received from {From}")
        print(f"   CallSid: {CallSid}")
        print(f"   Transcription: {SpeechResult}")
        print(f"   Confidence: {Confidence * 100:.1f}%")
        
        # Get user name from session
        user_name = "there"
        phone_number = None
        if CallSid in SESSIONS:
            user_name = SESSIONS[CallSid].get("user_name", "there")
            phone_number = SESSIONS[CallSid].get("phone_number")
        
        # Check if speech was understood
        if not SpeechResult or Confidence < 0.5:
            print("   ⚠️ Low confidence, asking user to repeat")
            
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Sorry {user_name}, I couldn't understand that clearly. Could you please repeat your question more slowly?
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        method="POST"
        language="en-IN" 
        speechTimeout="auto"
        timeout="8"
        hints="wifi, mobile data, internet, bill, payment, router, network, not working, slow, problem, recharge, plan, sim card">
        <Say voice="Polly.Aditi" language="en-IN">
            Please speak your question again.
        </Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">
        I didn't hear anything. Please call back when you're ready. Goodbye!
    </Say>
    <Hangup/>
</Response>"""
            
            return Response(content=twiml, media_type="application/xml")
        
        # Extract phone number if not in session
        if not phone_number:
            phone_number = extract_phone_number(From) if From else None
        
        if phone_number:
            print(f"   📱 Using phone: {phone_number}")
        
        # Get AI-powered answer using RAG chain with user data
        print(f"   🌟 Getting RAG chain response...")
        rag_result = get_ai_answer_via_rag(SpeechResult, phone_number=phone_number, session_id=CallSid)
        answer = rag_result["answer"]
        needs_escalation = rag_result["needs_escalation"]
        
        print(f"   💬 AI Answer: {answer[:100]}...")
        print(f"   🚨 Escalation needed: {needs_escalation}")
        
        # Store in session
        if CallSid in SESSIONS:
            add_to_session(CallSid, "user", SpeechResult)
            add_to_session(CallSid, "assistant", answer)
        
        # Escape special characters for XML
        answer_safe = answer.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
        
        # Ask for another question (no YES/NO required, just wait for next question or timeout)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        {answer_safe}
    </Say>
    <Pause length="1"/>
    <Say voice="Polly.Aditi" language="en-IN">
        {user_name}, do you have another question?
    </Say>
    <Gather 
        input="speech" 
        action="/voice/process" 
        method="POST"
        language="en-IN" 
        speechTimeout="auto"
        timeout="8"
        hints="wifi, mobile data, internet, bill, payment, router, network, not working, slow, problem, recharge, plan, sim card">
        <Say voice="Polly.Aditi" language="en-IN">
            I'm listening.
        </Say>
    </Gather>
    <Say voice="Polly.Aditi" language="en-IN">
        I didn't hear another question. Thank you for calling TelecomCare, {user_name}. Have a wonderful day!
    </Say>
    <Hangup/>
</Response>"""
        
        return Response(content=twiml, media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in process_speech: {e}")
        return Response(
            content="""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Sorry, something went wrong. Please try again later.
    </Say>
    <Hangup/>
</Response>""",
            media_type="application/xml"
        )


@app.post("/voice/followup")
async def followup(
    SpeechResult: str = Form(None),
    From: str = Form(None),
    CallSid: str = Form(None)
):
    """
    Legacy endpoint - maintained for backward compatibility.
    New flow: /voice/process directly handles questions without YES/NO confirmation.
    If user responds with a question, it's processed.
    If timeout occurs with no response, call ends gracefully.
    """
    try:
        user_name = "there"
        if CallSid in SESSIONS:
            user_name = SESSIONS[CallSid].get("user_name", "there")
        
        # If no response or unclear, end call
        print(f"\n🔄 Follow-up from {From}: no response or end of call")
        print(f"   CallSid: {CallSid}")
        
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Thank you for calling TelecomCare, {user_name}. Have a wonderful day!
    </Say>
    <Hangup/>
</Response>"""
        
        return Response(content=twiml, media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in followup: {e}")
        return Response(
            content="""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi" language="en-IN">
        Thank you for calling TelecomCare. Goodbye!
    </Say>
    <Hangup/>
</Response>""",
            media_type="application/xml"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    
    print("\n" + "="*60)
    print("🤖 Telecom Support AI Agent (Chat + Voice)")
    print("="*60)
    print(f"\n📍 Server starting at: http://localhost:{port}")
    print("🌟 AI Model: Gemini 2.5 Flash + RAG Chain")
    print("\n📋 CHAT ENDPOINTS:")
    print(f"   GET  /health            - Health check")
    print(f"   POST /chat              - Stateless chat")
    print(f"   POST /session/chat      - Session-based chat")
    print(f"   GET  /session/{{id}}      - Get session info")
    print(f"   DELETE /session/{{id}}    - Clear session")
    print(f"   GET  /sessions          - List all sessions")
    print(f"   POST /chat/debug        - Debug RAG pipeline")
    print("\n📞 VOICE CALL ENDPOINTS (Twilio):")
    print(f"   POST /voice/incoming    - Handle incoming calls")
    print(f"   POST /voice/process     - Process speech with RAG")
    print(f"   POST /voice/followup    - Handle follow-up questions")
    print("\n⚠️  Prerequisites:")
    print("   1. GEMINI_API_KEY in .env")
    print("   2. TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN in .env")
    print("   3. ngrok running for Twilio webhooks")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
