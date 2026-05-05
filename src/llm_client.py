import os
from typing import Optional

from dotenv import load_dotenv
from google import genai


class GeminiLLMClient:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY. "
                "Please add it to your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text or ""