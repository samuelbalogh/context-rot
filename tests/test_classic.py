import pytest
from src.experiments.classic import run_classic_niah, run_classic_niah_async


def test_run_classic_niah_mock_provider(sample_haystack_chunks):
    class MockProvider:
        def complete(self, messages, max_tokens=256, temperature=0.0):
            return "A"
        async def complete_async(self, messages, max_tokens=256, temperature=0.0):
            return "A"
        def max_context_tokens(self):
            return 128000

    out = run_classic_niah(
        MockProvider(),
        "pg",
        "Q?",
        "A",
        200,
        0.5,
        chunks_override=sample_haystack_chunks,
    )
    assert out == "A"


@pytest.mark.asyncio
async def test_run_classic_niah_async_mock_provider(sample_haystack_chunks):
    class MockProvider:
        def complete(self, messages, max_tokens=256, temperature=0.0):
            return "A"
        async def complete_async(self, messages, max_tokens=256, temperature=0.0):
            return "A"
        def max_context_tokens(self):
            return 128000

    out = await run_classic_niah_async(
        MockProvider(),
        "pg",
        "Q?",
        "A",
        200,
        0.5,
        chunks_override=sample_haystack_chunks,
    )
    assert out == "A"
