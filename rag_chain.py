import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class TelecomRAGChain:
    """
    Orchestrates RAG pipeline using LangChain for telecom support queries.
    - Retrieves relevant tickets from Chroma
    - Generates response using Gemini LLM via LangChain
    - Handles escalation logic
    - Uses prompt templates for consistency
    """

    def __init__(self):
        """Initialize LLM and prompt templates."""
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("GENAI_API_KEY environment variable is required")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.3
        )

        # RAG Prompt Template for generating support responses
        self.rag_prompt = PromptTemplate(
            template="""You are a helpful telecom support assistant. Answer the customer's question in simple English.

CONTEXT FROM PAST SUPPORT CASES:
{context}

CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Answer in simple, easy-to-understand English (8th grade reading level)
2. Be concise (2-3 sentences maximum)
3. Include ticket IDs from similar cases (e.g., "As seen in ticket T001...")
4. If unsure, suggest escalation to a human agent
5. Do NOT make up information

ANSWER:""",
            input_variables=["context", "question"]
        )

        # Escalation Detection Prompt
        self.escalation_prompt = PromptTemplate(
            template="""You are an escalation detection system. Analyze if this support case needs immediate escalation to a human agent.

CUSTOMER QUESTION: {question}

AI RESPONSE: {response}

ESCALATION TRIGGERS (respond YES if ANY apply):
- Billing/payment issues or disputes
- Account security concerns (SIM cloning, fraud, stolen device)
- Angry/frustrated/angry customer tone
- Complex technical issues requiring technician
- Low confidence response (words like "might", "probably", "unclear", "I'm not sure")

Respond with ONLY: "YES" or "NO" (no explanation)""",
            input_variables=["question", "response"]
        )

        self.output_parser = StrOutputParser()

    def format_context(self, search_results):
        """
        Format Chroma search results into readable context for the LLM.

        Args:
            search_results: Dict with 'documents', 'ids', 'distances', 'metadatas' from Chroma

        Returns:
            str: Formatted context for RAG prompt
        """
        if not search_results or not search_results.get("documents"):
            return "No similar cases found in database."

        context_parts = []
        documents = search_results.get("documents", [])
        ids = search_results.get("ids", [])
        metadatas = search_results.get("metadatas", [])
        distances = search_results.get("distances", [])

        for i, doc in enumerate(documents):
            ticket_id = ids[i] if i < len(ids) else f"Case_{i}"
            distance = distances[i] if i < len(distances) else 0
            relevance_score = 1 - distance  # Convert distance to similarity

            # Include resolution from metadata if available
            resolution = ""
            if i < len(metadatas) and metadatas[i].get("resolution"):
                resolution = f"\nResolution: {metadatas[i]['resolution']}"

            context_parts.append(
                f"CASE {ticket_id} (relevance: {relevance_score:.1%}):\n"
                f"Issue: {doc}{resolution}"
            )

        return "\n\n".join(context_parts)

    def generate_response(self, question, context):
        """
        Generate RAG response using LangChain chain.

        Args:
            question (str): Customer query
            context (str): Formatted search results from Chroma

        Returns:
            dict: {'response': str, 'context_used': str}
        """
        # Build RAG chain using LangChain
        rag_chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | self.output_parser
        )

        # Generate response
        response = rag_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "response": response.strip(),
            "context_used": context
        }

    def check_escalation(self, question, response):
        """
        Detect if query needs escalation using LangChain.

        Args:
            question (str): Original customer query
            response (str): AI-generated response

        Returns:
            bool: True if escalation needed
        """
        escalation_chain = (
            {
                "question": RunnablePassthrough(),
                "response": RunnablePassthrough()
            }
            | self.escalation_prompt
            | self.llm
            | self.output_parser
        )

        decision = escalation_chain.invoke({
            "question": question,
            "response": response
        }).strip().upper()

        return "YES" in decision

    def run(self, question, search_results):
        """
        Full RAG pipeline: retrieve -> format -> generate -> escalate check.

        Args:
            question (str): Customer query
            search_results (dict): Chroma search results with documents, ids, metadatas

        Returns:
            dict: {
                'answer': str,
                'source_tickets': list,
                'needs_escalation': bool,
                'confidence': str
            }
        """
        # Step 1: Format retrieved context
        context = self.format_context(search_results)

        # Step 2: Generate RAG response using LangChain
        result = self.generate_response(question, context)

        # Step 3: Check if escalation needed
        needs_escalation = self.check_escalation(question, result["response"])

        return {
            "answer": result["response"],
            "source_tickets": search_results.get("ids", []),
            "needs_escalation": needs_escalation,
            "confidence": "low" if needs_escalation else "high"
        }


if __name__ == "__main__":
    # Simple test
    chain = TelecomRAGChain()
    print("✓ TelecomRAGChain initialized successfully")
    print(f"  Temperature: {chain.llm.temperature}")
