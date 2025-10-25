import os
from typing import Optional
from dotenv import load_dotenv

try:
    from google import genai
except Exception:
    genai = None

# Load env vars from .env file
load_dotenv()


class LLMClient:
    """Simple wrapper for Google GenAI (Gemini) model calls.

    Uses GENAI_API_KEY environment variable by default.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        api_key = api_key or os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("GENAI_API_KEY environment variable is required")
        if genai is None:
            raise RuntimeError("google.genai SDK not available. Install google-genai or adjust imports.")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=prompt)
        # SDK may return .text or nested structures; try common accessors
        if hasattr(resp, "text") and resp.text:
            return resp.text
        # fallback to string representation
        try:
            return str(resp)
        except Exception:
            return ""


if __name__ == "__main__":
    print("LLM client wrapper loaded. Set GENAI_API_KEY env var to use.")
from google import genai

client = genai.Client(api_key="AIzaSyBEr4igZrrbH8CGT8RI3L1n0KeCyK_Em6M")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)