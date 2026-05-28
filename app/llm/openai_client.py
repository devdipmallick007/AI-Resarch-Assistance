from __future__ import annotations

import os

from dotenv import load_dotenv


class OpenAIClient:
    def __init__(self) -> None:
        load_dotenv()
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self._gemini_client = None
        if self.provider == "gemini" and self.gemini_api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception:
                self._gemini_client = None
        elif self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "gemini":
            return self._gemini_chat(system_prompt, user_prompt)
        if not self._client:
            return ""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "gemini":
            return []
        if not self._client:
            return []
        try:
            response = self._client.embeddings.create(model=self.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            return []

    def _gemini_chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self._gemini_client:
            return ""
        try:
            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return getattr(response, "text", "") or ""
        except Exception:
            return ""
