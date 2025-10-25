# Architecture - Voice Call Integration

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TELECOM SUPPORT AI AGENT                     │
│                      (main.py - Integrated)                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼──────┐  ┌────▼─────┐  ┌────▼──────┐
         │  WEB CHAT   │  │  VOICE   │  │ API DEBUG │
         │ (React)     │  │ (Twilio) │  │           │
         └──────┬──────┘  └────┬─────┘  └────┬──────┘
                │              │              │
         ┌──────▼──────────────▼──────────────▼──────┐
         │         FASTAPI Main Application         │
         │              (main.py)                   │
         └──────┬──────────────┬──────────────┬─────┘
                │              │              │
        ┌───────▼──┐    ┌──────▼──────┐  ┌───▼──────┐
        │   CHAT   │    │   VOICE     │  │ SESSION  │
        │ENDPOINTS │    │ ENDPOINTS   │  │ MANAGER  │
        └───────┬──┘    └──────┬──────┘  └───┬──────┘
                │              │              │
         ┌──────▼──────────────▼──────────────▼──────┐
         │         SHARED AI LOGIC LAYER             │
         │    (get_ai_answer_via_rag function)      │
         │                                           │
         │  1. Embed query → EmbeddingGenerator      │
         │  2. Search → ChromaClientWrapper          │
         │  3. Generate → RAG Chain                  │
         │  4. Return → Answer + Sources             │
         └──────┬──────────────────────────────────┬─┘
                │                                  │
        ┌───────▼────────┐              ┌─────────▼────────┐
        │ EXTERNAL APIS  │              │ DATA STORAGE     │
        ├────────────────┤              ├──────────────────┤
        │ • Gemini LLM   │              │ • Chroma DB      │
        │ • Twilio       │              │ • Session Store  │
        │ • Embeddings   │              │ • tickets.json   │
        └────────────────┘              └──────────────────┘
```

---

## Request Flow - Chat Path

```
Web User → POST /chat → ChatRequest
    ↓
Validate input (3+ chars)
    ↓
Call get_ai_answer_via_rag(query)
    ├─ Embed query
    ├─ Search Chroma (top_k results)
    ├─ Run RAG chain
    └─ Return answer + sources + escalation
    ↓
Return ChatResponse (JSON)
```

---

## Request Flow - Voice Path

```
Phone Call → Twilio → POST /voice/incoming
    ↓
Create voice session (CallSid)
    ↓
Return TwiML → Greet + Listen
    ↓
User speaks question (speech-to-text)
    ↓
POST /voice/process (Twilio callback)
    ├─ Validate confidence (> 0.4)
    ├─ Call get_ai_answer_via_rag(transcription)
    │   ├─ Embed query
    │   ├─ Search Chroma
    │   ├─ Run RAG chain
    │   └─ Return answer
    ├─ Store in session history
    └─ Return TwiML → Speak answer + Ask follow-up
    ↓
User responds yes/no
    ↓
POST /voice/followup
    ├─ Parse yes/no intent
    ├─ If YES → Loop to /voice/process
    └─ If NO → Hang up
```

---

## Session Management

```
┌─────────────────────────────────────────┐
│         SESSIONS Dictionary             │
├─────────────────────────────────────────┤
│ {                                       │
│   "session_123": {                      │
│     "messages": [                       │
│       {                                 │
│         "role": "user",                 │
│         "content": "wifi problem"       │
│       },                                │
│       {                                 │
│         "role": "assistant",            │
│         "content": "Try restarting..." │
│       }                                 │
│     ],                                  │
│     "created_at": "2024-10-25...",      │
│     "last_updated": "2024-10-25...",    │
│     "type": "chat",                     │
│     "caller": null                      │
│   },                                    │
│   "call_789": {                         │
│     "messages": [...],                  │
│     "created_at": "...",                │
│     "last_updated": "...",              │
│     "type": "voice",                    │
│     "caller": "+91-9876543210"          │
│   }                                     │
│ }                                       │
└─────────────────────────────────────────┘
```

---

## Shared RAG Logic (Core Brain)

```
get_ai_answer_via_rag(question, session_id=None, top_k=3)
│
├─1. EMBEDDING
│   embedding_gen.embed_text(question)
│   └─ Returns: vector embedding
│
├─2. RETRIEVAL
│   chroma_client.search_by_embedding(embedding, n_results=top_k)
│   └─ Returns: {documents, ids, distances, metadatas}
│
├─3. GENERATION
│   rag_chain.run(question, search_results)
│   ├─ Format context from documents
│   ├─ Create RAG prompt
│   ├─ Call Gemini LLM
│   ├─ Parse escalation need
│   └─ Returns: {answer, source_tickets, needs_escalation}
│
└─ RETURN: Dict with all response metadata
```

---

## Twilio Integration Points

```
Twilio Cloud
    │
    ├─ Phone Call Received
    │
    ├─ Webhook POST to /voice/incoming
    │   │
    │   └─ Response: TwiML (Greet + Gather speech)
    │
    ├─ User speaks
    │
    ├─ Twilio Speech-to-Text
    │
    ├─ Webhook POST to /voice/process
    │   │
    │   ├─ SpeechResult: transcribed text
    │   ├─ Confidence: 0.0-1.0
    │   └─ Response: TwiML (Speak + Gather yes/no)
    │
    ├─ User says yes/no
    │
    └─ Webhook POST to /voice/followup
        │
        ├─ If YES → POST /voice/process again (loop)
        └─ If NO → Response: TwiML (Hangup)
```

---

## Data Flow for Single Voice Call

```
┌─────────────────────────────────────────┐
│  Phone Call from +91-9876543210         │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  /voice/incoming    │
        │ Create session #789 │
        └──────────┬──────────┘
                   │
        ┌──────────▼─────────────┐
        │ User: "wifi not work"  │
        └──────────┬─────────────┘
                   │
        ┌──────────▼──────────────────┐
        │  /voice/process              │
        │  1. Confidence: 0.92 ✓       │
        │  2. Embed question           │
        │  3. Search Chroma (3 docs)   │
        │  4. RAG chain response       │
        │  5. Store in session         │
        └──────────┬──────────────────┘
                   │
     ┌─────────────▼──────────────┐
     │ AI: "Restart your router"  │
     │ Ask: "More questions?"      │
     └─────────────┬──────────────┘
                   │
        ┌──────────▼─────────────┐
        │  /voice/followup        │
        │ User: "No"              │
        └──────────┬─────────────┘
                   │
        ┌──────────▼──────────────┐
        │ Hangup gracefully       │
        │ Session #789 cleared    │
        └─────────────────────────┘
```

---

## Component Interactions

```
┌─────────────────────────────────────────────────────────┐
│                    main.py                              │
│  ┌────────────────────────────────────────────────┐    │
│  │ FastAPI App with Voice + Chat Endpoints       │    │
│  │                                                │    │
│  │  • /chat, /session/chat (text)                │    │
│  │  • /voice/incoming, /process, /followup      │    │
│  │  • Session management layer                  │    │
│  └────────┬─────────────────────────────────┬──┘    │
└───────────┼─────────────────────────────────┼────────┘
            │                                 │
  ┌─────────▼────────┐          ┌────────────▼──────┐
  │ EmbeddingGenerator│          │ ChromaClientWrapper│
  │                  │          │                   │
  │ • sentence-trans │          │ • Chroma DB       │
  │ • Embed text     │          │ • Vector search   │
  │ • Normalize      │          │ • Retrieve docs   │
  └──────────────────┘          └─────────┬──────────┘
                                          │
                    ┌─────────────────────▼──────────┐
                    │    RAG Chain                   │
                    │ (rag_chain.py)                 │
                    │                                │
                    │ • LangChain orchestration      │
                    │ • Prompt templates            │
                    │ • Gemini LLM call             │
                    │ • Escalation detection        │
                    │ • Output parsing              │
                    └────────┬──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Gemini 2.5 API │
                    │ (LLM inference) │
                    └─────────────────┘
```

---

## Session Storage Strategy

### In-Memory (Current)
- ✅ Fast
- ✅ No DB overhead
- ❌ Lost on restart
- ❌ Not scalable

### For Production, Consider:
- PostgreSQL + SQLAlchemy
- Redis for session cache
- DynamoDB for AWS deployments

---

## Error Handling Strategy

```
┌─────────────────────────────────────┐
│  Request to /voice/process          │
└──────────────┬──────────────────────┘
               │
        ┌──────▼──────┐
        │ Try Block   │
        └──┬──────┬───┘
           │      │
       ✓   │      │ ✗
           │      └──┐
        ┌──▼──┐  ┌───▼──────────┐
        │ OK  │  │ Exception    │
        │ ✓   │  │ Caught!      │
        └─────┘  │              │
                 │ Return error │
                 │ TwiML to     │
                 │ caller       │
                 │              │
                 │ Log error    │
                 │ for debug    │
                 └──────────────┘
```

---

## Confidence Scoring (Speech Recognition)

```
Twilio Returns Confidence (0.0 - 1.0)

┌─────────────────────────────────────┐
│ 0.0────────────────0.4────────1.0   │
│ │                    │         │     │
│ LOW              THRESHOLD    HIGH  │
│ │                    │         │     │
│ REJECT           ACCEPT      PERFECT│
│ "Ask again"    "Process"     "✓"   │
└─────────────────────────────────────┘

Default Threshold: 0.4
- Below: Ask user to repeat
- Above: Process with RAG
```

---

## Call States (Voice Session Lifecycle)

```
┌─────────────┐
│   INCOMING  │ (Twilio webhook)
└──────┬──────┘
       │
       ▼
┌──────────────┐
│   LISTENING  │ (Gathering speech)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ PROCESSING   │ (RAG chain)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ RESPONDING   │ (TTS response)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ FOLLOW-UP    │ (Ask for more)
└──────┬───────┘
       │
    YES│ NO
       │  │
    ┌──▼┬▼──┐
    │   X   │
    └───▼───┘
        │
        ▼
   ┌─────────┐
   │ HANGUP  │ (End call)
   └─────────┘
```

