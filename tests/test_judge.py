from types import SimpleNamespace

import pytest

import src.evaluation.judge as judge


def test_judge_prompt_contains_inputs():
    prompt = judge._judge_prompt("Q?", "A", "model said A")

    assert "Question: Q?" in prompt
    assert "Expected (must appear exactly): A" in prompt
    assert "Output: model said A" in prompt


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("correct", "correct"),
        ("incorrect", "incorrect"),
        ("abstained", "abstained"),
        ("  INCORRECT  ", "incorrect"),
    ],
)
def test_parse_verdict(verdict, expected):
    assert judge._parse_verdict(verdict) == expected


def test_judge_correctness_uses_gpt5_max_completion_tokens(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))]
            )

    monkeypatch.setattr(judge, "OpenAI", FakeOpenAI)

    result = judge.judge_correctness("Q?", "A", "A", "gpt-5.4")

    assert result == "correct"
    assert calls["max_completion_tokens"] == 16
    assert "max_tokens" not in calls


def test_judge_correctness_uses_max_tokens_for_non_gpt5(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="abstained"))]
            )

    monkeypatch.setattr(judge, "OpenAI", FakeOpenAI)

    result = judge.judge_correctness("Q?", "A", "B", "gpt-4o")

    assert result == "abstained"
    assert calls["max_tokens"] == 16
    assert "max_completion_tokens" not in calls


@pytest.mark.asyncio
async def test_judge_correctness_async_uses_max_tokens_for_non_gpt5(monkeypatch):
    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        async def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="incorrect"))]
            )

    monkeypatch.setattr(judge, "AsyncOpenAI", FakeAsyncOpenAI)

    result = await judge.judge_correctness_async("Q?", "A", "B", "gpt-4o")

    assert result == "incorrect"
    assert calls["max_tokens"] == 16
    assert "max_completion_tokens" not in calls


@pytest.mark.asyncio
async def test_judge_correctness_async_uses_gpt5_max_completion_tokens(monkeypatch):
    calls = {}

    class FakeAsyncOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        async def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="correct"))]
            )

    monkeypatch.setattr(judge, "AsyncOpenAI", FakeAsyncOpenAI)

    result = await judge.judge_correctness_async("Q?", "A", "A", "gpt-5.4")

    assert result == "correct"
    assert calls["max_completion_tokens"] == 16
    assert "max_tokens" not in calls
