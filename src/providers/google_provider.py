import asyncio
import os
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GoogleProvider:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def complete(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        try:
            contents = self._messages_to_contents(messages)
            output_tokens = max(max_tokens, 2048) if "2.5-pro" in self._model else max_tokens
            config = types.GenerateContentConfig(
                max_output_tokens=output_tokens,
                temperature=temperature,
                response_modalities=["TEXT"],
            )
            resp = self._get_client().models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            return (resp.text or "").strip()
        except Exception:
            logger.exception("Google complete failed")
            raise

    async def complete_async(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        return await asyncio.to_thread(self.complete, messages, max_tokens, temperature)

    def _messages_to_contents(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}\n\n")
            else:
                parts.append(f"{content}\n\n")
        return "".join(parts).strip()

    def max_context_tokens(self) -> int:
        return 1_000_000
