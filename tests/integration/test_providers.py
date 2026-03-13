import os
import pytest
from src.config import load_env
from src.providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from src.experiments.classic import run_classic_niah_async
from src.haystack.builder import build_haystack
from src.haystack.pg_essays import load_cached as load_pg
from src.haystack.arxiv import load_cached as load_arxiv

pytestmark = pytest.mark.integration

load_env()


def _has_keys(*keys):
    return all(os.environ.get(k) for k in keys)


@pytest.mark.skipif(not _has_keys("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_openai_provider_integration():
    pg = load_pg()
    arxiv = load_arxiv()
    if not pg or not arxiv:
        pytest.skip("Haystack data not fetched - run make fetch-haystacks")
    chunks = pg[:2] if pg else arxiv[:2]
    if not chunks:
        pytest.skip("No haystack chunks")
    provider = OpenAIProvider(model="gpt-4o")
    out = await run_classic_niah_async(
        provider, "pg", "What is 2+2?", "4", 200, 0.5, chunks_override=chunks
    )
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.skipif(not _has_keys("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
async def test_anthropic_provider_integration():
    pg = load_pg()
    arxiv = load_arxiv()
    if not pg or not arxiv:
        pytest.skip("Haystack data not fetched - run make fetch-haystacks")
    chunks = pg[:2] if pg else arxiv[:2]
    if not chunks:
        pytest.skip("No haystack chunks")
    provider = AnthropicProvider(model="claude-sonnet-4-20250514")
    out = await run_classic_niah_async(
        provider, "pg", "What is 2+2?", "4", 200, 0.5, chunks_override=chunks
    )
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.skipif(not (_has_keys("GOOGLE_API_KEY") or _has_keys("GEMINI_API_KEY")), reason="GOOGLE_API_KEY or GEMINI_API_KEY not set")
@pytest.mark.asyncio
async def test_google_provider_integration():
    pg = load_pg()
    arxiv = load_arxiv()
    if not pg or not arxiv:
        pytest.skip("Haystack data not fetched - run make fetch-haystacks")
    chunks = pg[:2] if pg else arxiv[:2]
    if not chunks:
        pytest.skip("No haystack chunks")
    provider = GoogleProvider(model="gemini-2.5-flash")
    out = await run_classic_niah_async(
        provider, "pg", "What is 2+2?", "4", 200, 0.5, chunks_override=chunks
    )
    assert isinstance(out, str)
    assert len(out) > 0
