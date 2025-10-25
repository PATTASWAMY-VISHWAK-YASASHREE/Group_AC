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
            template="""You are a helpful telecom support assistant. Answer the customer's question clearly and concisely.

{user_context}{case_context}
CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Use the customer account information to provide personalized answers
2. Answer in simple, easy-to-understand English (8th grade reading level)
3. Be concise (1-3 sentences maximum)
4. Focus on actionable steps the customer can take RIGHT NOW
5. Do NOT mention ticket IDs or case numbers to the customer
6. If unsure, simply say "I'm not sure about this - please contact our support team"

ANSWER:""",
            input_variables=["user_context", "case_context", "question"]
        )

        # Escalation Detection Prompt
        self.escalation_prompt = PromptTemplate(
            template="""You are an escalation detection system. Analyze if this support case MUST be escalated to a human agent.

CUSTOMER QUESTION: {question}

AI RESPONSE: {response}

ESCALATION TRIGGERS (respond YES ONLY if serious):
- Account security issue (SIM cloning, fraud, stolen device, unauthorized access)
- Billing dispute or fraud (unauthorized charges, double billing that needs refund)
- Angry/frustrated/hostile customer tone (harsh language, threats, complaints)
- Complex technical issue that requires technician visit or hardware replacement
- Account closure or major account changes (closing account, ownership transfer)
- Issue with AI response (contains "might", "probably", "unclear", "I'm not sure", "I don't know")

Respond with ONLY: "YES" or "NO" (no explanation)""",
            input_variables=["question", "response"]
        )

        self.output_parser = StrOutputParser()

    def format_case_context(self, search_results):
        """
        Format Chroma search results into readable context for the LLM.

        Args:
            search_results: Dict with 'documents', 'ids', 'distances', 'metadatas' from Chroma

        Returns:
            str: Formatted context for RAG prompt
        """
        if not search_results or not search_results.get("documents"):
            return "SIMILAR SOLUTIONS FROM PAST CASES:\nNo similar cases found in database."

        context_parts = ["SIMILAR SOLUTIONS FROM PAST CASES:"]
        documents = search_results.get("documents", [])
        metadatas = search_results.get("metadatas", [])

        for i, doc in enumerate(documents):
            # Include resolution from metadata if available
            resolution = ""
            if i < len(metadatas) and metadatas[i].get("resolution"):
                resolution = f"\n{metadatas[i]['resolution']}"

            # Format without ticket IDs for cleaner context
            context_parts.append(f"Solution {i+1}:\n{doc}{resolution}")

        return "\n\n".join(context_parts)

    def generate_response(self, question, user_context, case_context):
        """
        Generate RAG response using LangChain chain.

        Args:
            question (str): Customer query
            user_context (str): Formatted user account information
            case_context (str): Formatted search results from Chroma

        Returns:
            dict: {'response': str, 'user_context': str, 'case_context': str}
        """
        # Build RAG chain using LangChain
        rag_chain = (
            {
                "user_context": RunnablePassthrough(),
                "case_context": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | self.output_parser
        )

        # Generate response
        response = rag_chain.invoke({
            "user_context": user_context,
            "case_context": case_context,
            "question": question
        })

        return {
            "response": response.strip(),
            "user_context": user_context,
            "case_context": case_context
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

    def run(self, question, search_results, user_context=""):
        """
        Full RAG pipeline: retrieve -> format -> generate -> escalate check.

        Args:
            question (str): Customer query
            search_results (dict): Chroma search results with documents, ids, metadatas
            user_context (str): Optional formatted user account information

        Returns:
            dict: {
                'answer': str,
                'source_tickets': list,
                'needs_escalation': bool,
                'confidence': str
            }
        """
        # Step 1: Format retrieved case context
        case_context = self.format_case_context(search_results)

        # Step 2: Generate RAG response using LangChain with user + case context
        result = self.generate_response(question, user_context, case_context)

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
