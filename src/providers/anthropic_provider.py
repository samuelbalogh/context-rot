import asyncio
import os
import logging
from anthropic import Anthropic, AsyncAnthropic
from anthropic import RateLimitError

logger = logging.getLogger(__name__)
RATE_LIMIT_BACKOFF_SEC = 75
RATE_LIMIT_MAX_RETRIES = 3


class AnthropicProvider:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-20250514"):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._client: Anthropic | None = None
        self._async_client: AsyncAnthropic | None = None

    def _get_client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _get_async_client(self) -> AsyncAnthropic:
        if self._async_client is None:
            self._async_client = AsyncAnthropic(api_key=self._api_key)
        return self._async_client

    def _complete_kwargs(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        api_messages = [m for m in messages if m.get("role") != "system"]
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "temperature": temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        return kwargs

    def complete(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        try:
            resp = self._get_client().messages.create(**self._complete_kwargs(messages, max_tokens, temperature))
            text = resp.content[0].text if resp.content else ""
            return text.strip()
        except Exception:
            logger.exception("Anthropic complete failed")
            raise

    async def complete_async(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        kwargs = self._complete_kwargs(messages, max_tokens, temperature)
        for attempt in range(RATE_LIMIT_MAX_RETRIES):
            try:
                resp = await self._get_async_client().messages.create(**kwargs)
                text = resp.content[0].text if resp.content else ""
                return text.strip()
            except RateLimitError as e:
                if attempt < RATE_LIMIT_MAX_RETRIES - 1:
                    logger.warning("Anthropic 429, waiting %ds before retry (%d/%d)", RATE_LIMIT_BACKOFF_SEC, attempt + 1, RATE_LIMIT_MAX_RETRIES)
                    await asyncio.sleep(RATE_LIMIT_BACKOFF_SEC)
                else:
                    logger.exception("Anthropic complete failed after %d retries", RATE_LIMIT_MAX_RETRIES)
                    raise

    def max_context_tokens(self) -> int:
        return 200_000
