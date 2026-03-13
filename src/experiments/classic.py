import logging
from src.haystack.builder import build_haystack
from src.providers.base import ProviderProtocol

logger = logging.getLogger(__name__)


def _build_messages(question: str, haystack: str) -> list[dict]:
    return [
        {"role": "system", "content": "Answer the question using only the provided context. If the answer is not in the context, say so."},
        {"role": "user", "content": f"Context:\n\n{haystack}\n\nQuestion: {question}"},
    ]


def run_classic_niah(
    provider: ProviderProtocol,
    variant: str,
    question: str,
    needle: str,
    context_length: int,
    needle_position: float,
    chunks_override: list[tuple[str, str]] | None = None,
    fake_needles: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> str:
    haystack = build_haystack(variant, context_length, needle, needle_position, chunks_override, fake_needles, exclude_prefixes)
    return provider.complete(_build_messages(question, haystack), max_tokens=256, temperature=0.0)


async def run_classic_niah_async(
    provider: ProviderProtocol,
    variant: str,
    question: str,
    needle: str,
    context_length: int,
    needle_position: float,
    chunks_override: list[tuple[str, str]] | None = None,
    fake_needles: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> str:
    haystack = build_haystack(variant, context_length, needle, needle_position, chunks_override, fake_needles, exclude_prefixes)
    return await provider.complete_async(_build_messages(question, haystack), max_tokens=256, temperature=0.0)
