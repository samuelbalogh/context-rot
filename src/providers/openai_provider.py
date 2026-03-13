import os
import logging
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5.1"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self._api_key)
        return self._async_client

    _NO_TEMPERATURE_MODELS = ("gpt-5-mini", "o3-mini")

    def _complete_kwargs(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        kwargs = {"model": self._model, "messages": messages}
        if not any(m in self._model for m in self._NO_TEMPERATURE_MODELS):
            kwargs["temperature"] = temperature
        if any(m in self._model for m in self._NO_TEMPERATURE_MODELS) and max_tokens < 8192:
            max_tokens = 8192
        if "gpt-5" in self._model:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def complete(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        try:
            resp = self._get_client().chat.completions.create(**self._complete_kwargs(messages, max_tokens, temperature))
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("OpenAI complete failed")
            raise

    async def complete_async(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        try:
            resp = await self._get_async_client().chat.completions.create(**self._complete_kwargs(messages, max_tokens, temperature))
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("OpenAI complete failed")
            raise

    def max_context_tokens(self) -> int:
        if "gpt-5-mini" in self._model:
            return 400_000
        return 128_000
