from types import SimpleNamespace

import pytest

import src.providers.anthropic_provider as anthropic_provider
import src.providers.google_provider as google_provider
import src.providers.openai_provider as openai_provider


def _openai_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _anthropic_response(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_openai_get_client_caches_instance(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, api_key=None):
            created.append(api_key)

    provider = openai_provider.OpenAIProvider(api_key="key")
    monkeypatch.setattr(openai_provider, "OpenAI", FakeOpenAI)

    first = provider._get_client()
    second = provider._get_client()

    assert first is second
    assert created == ["key"]


def test_openai_get_async_client_caches_instance(monkeypatch):
    created = []

    class FakeAsyncOpenAI:
        def __init__(self, api_key=None):
            created.append(api_key)

    provider = openai_provider.OpenAIProvider(api_key="key")
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", FakeAsyncOpenAI)

    first = provider._get_async_client()
    second = provider._get_async_client()

    assert first is second
    assert created == ["key"]


def test_openai_complete_kwargs_switches_by_model():
    provider = openai_provider.OpenAIProvider(model="gpt-5.4")
    kwargs = provider._complete_kwargs([{"role": "user", "content": "hi"}], 77, 0.2)
    assert kwargs["max_completion_tokens"] == 77
    assert "max_tokens" not in kwargs

    provider = openai_provider.OpenAIProvider(model="gpt-4o")
    kwargs = provider._complete_kwargs([{"role": "user", "content": "hi"}], 55, 0.1)
    assert kwargs["max_tokens"] == 55
    assert "max_completion_tokens" not in kwargs


def test_openai_complete_strips_response(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.update(kwargs)
            return _openai_response("  hello  ")

    provider = openai_provider.OpenAIProvider(model="gpt-4o")
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.complete([{"role": "user", "content": "hi"}], max_tokens=12, temperature=0.3)

    assert result == "hello"
    assert calls["model"] == "gpt-4o"
    assert calls["max_tokens"] == 12


@pytest.mark.asyncio
async def test_openai_complete_async_strips_response(monkeypatch):
    calls = {}

    class FakeAsyncClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        async def create(self, **kwargs):
            calls.update(kwargs)
            return _openai_response("  async hello  ")

    provider = openai_provider.OpenAIProvider(model="gpt-5.4")
    monkeypatch.setattr(provider, "_get_async_client", lambda: FakeAsyncClient())

    result = await provider.complete_async([{"role": "user", "content": "hi"}], max_tokens=13, temperature=0.4)

    assert result == "async hello"
    assert calls["max_completion_tokens"] == 13


def test_openai_complete_logs_and_raises(monkeypatch):
    provider = openai_provider.OpenAIProvider()
    errors = []

    class Boom(Exception):
        pass

    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(Boom("x"))))))
    monkeypatch.setattr(openai_provider.logger, "exception", lambda msg: errors.append(msg))

    with pytest.raises(Boom):
        provider.complete([{"role": "user", "content": "hi"}])

    assert errors == ["OpenAI complete failed"]


def test_anthropic_get_client_caches_instance(monkeypatch):
    created = []

    class FakeAnthropic:
        def __init__(self, api_key=None):
            created.append(api_key)

    provider = anthropic_provider.AnthropicProvider(api_key="key")
    monkeypatch.setattr(anthropic_provider, "Anthropic", FakeAnthropic)

    first = provider._get_client()
    second = provider._get_client()

    assert first is second
    assert created == ["key"]


def test_anthropic_get_async_client_caches_instance(monkeypatch):
    created = []

    class FakeAsyncAnthropic:
        def __init__(self, api_key=None):
            created.append(api_key)

    provider = anthropic_provider.AnthropicProvider(api_key="key")
    monkeypatch.setattr(anthropic_provider, "AsyncAnthropic", FakeAsyncAnthropic)

    first = provider._get_async_client()
    second = provider._get_async_client()

    assert first is second
    assert created == ["key"]


def test_anthropic_complete_kwargs_merges_system_messages():
    provider = anthropic_provider.AnthropicProvider(model="claude")
    kwargs = provider._complete_kwargs(
        [
            {"role": "system", "content": "rule 1"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "rule 2"},
        ],
        33,
        0.2,
    )

    assert kwargs["model"] == "claude"
    assert kwargs["max_tokens"] == 33
    assert kwargs["temperature"] == 0.2
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["system"] == "rule 1\n\nrule 2"


def test_anthropic_complete_without_system(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            calls.update(kwargs)
            return _anthropic_response("  hi  ")

    provider = anthropic_provider.AnthropicProvider(model="claude")
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.complete([{"role": "user", "content": "hello"}], max_tokens=9)

    assert result == "hi"
    assert "system" not in calls


def test_anthropic_complete_logs_and_raises(monkeypatch):
    provider = anthropic_provider.AnthropicProvider()
    errors = []

    class Boom(Exception):
        pass

    monkeypatch.setattr(provider, "_get_client", lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(Boom("x")))))
    monkeypatch.setattr(anthropic_provider.logger, "exception", lambda msg: errors.append(msg))

    with pytest.raises(Boom):
        provider.complete([{"role": "user", "content": "hi"}])

    assert errors == ["Anthropic complete failed"]


@pytest.mark.asyncio
async def test_anthropic_complete_async_retries_then_succeeds(monkeypatch):
    class FakeRateLimitError(Exception):
        pass

    calls = {"count": 0}
    sleeps = []
    warnings = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeAsyncClient:
        def __init__(self):
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise FakeRateLimitError("slow down")
            return _anthropic_response("  done  ")

    provider = anthropic_provider.AnthropicProvider()
    monkeypatch.setattr(anthropic_provider, "RateLimitError", FakeRateLimitError)
    monkeypatch.setattr(provider, "_get_async_client", lambda: FakeAsyncClient())
    monkeypatch.setattr(anthropic_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(anthropic_provider.logger, "warning", lambda *args: warnings.append(args))

    result = await provider.complete_async([{"role": "user", "content": "hi"}])

    assert result == "done"
    assert calls["count"] == 3
    assert sleeps == [anthropic_provider.RATE_LIMIT_BACKOFF_SEC, anthropic_provider.RATE_LIMIT_BACKOFF_SEC]
    assert len(warnings) == 2


@pytest.mark.asyncio
async def test_anthropic_complete_async_raises_after_retries(monkeypatch):
    class FakeRateLimitError(Exception):
        pass

    sleeps = []
    errors = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeAsyncClient:
        def __init__(self):
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **kwargs):
            raise FakeRateLimitError("still limited")

    provider = anthropic_provider.AnthropicProvider()
    monkeypatch.setattr(anthropic_provider, "RateLimitError", FakeRateLimitError)
    monkeypatch.setattr(provider, "_get_async_client", lambda: FakeAsyncClient())
    monkeypatch.setattr(anthropic_provider.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(anthropic_provider.logger, "exception", lambda *args: errors.append(args))

    with pytest.raises(FakeRateLimitError):
        await provider.complete_async([{"role": "user", "content": "hi"}])

    assert sleeps == [
        anthropic_provider.RATE_LIMIT_BACKOFF_SEC,
        anthropic_provider.RATE_LIMIT_BACKOFF_SEC,
    ]
    assert len(errors) == 1


def test_google_get_client_caches_instance(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, api_key=None):
            created.append(api_key)

    provider = google_provider.GoogleProvider(api_key="key")
    monkeypatch.setattr(google_provider.genai, "Client", FakeClient)

    first = provider._get_client()
    second = provider._get_client()

    assert first is second
    assert created == ["key"]


def test_google_messages_to_contents_formats_roles():
    provider = google_provider.GoogleProvider()

    result = provider._messages_to_contents(
        [
            {"role": "system", "content": "rule"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "previous"},
        ]
    )

    assert result == "[System]: rule\n\nhello\n\nprevious"


def test_google_complete_uses_25_pro_min_output_tokens(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return SimpleNamespace(text="  result  ")

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    provider = google_provider.GoogleProvider(model="gemini-2.5-pro")
    monkeypatch.setattr(google_provider.types, "GenerateContentConfig", FakeConfig)
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())

    result = provider.complete(
        [{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}],
        max_tokens=100,
        temperature=0.7,
    )

    assert result == "result"
    assert captured["config_kwargs"]["max_output_tokens"] == 2048
    assert captured["generate_kwargs"]["contents"] == "[System]: rule\n\nhi"


def test_google_complete_logs_and_raises(monkeypatch):
    provider = google_provider.GoogleProvider()
    errors = []

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    class Boom(Exception):
        pass

    class FakeModels:
        def generate_content(self, **kwargs):
            raise Boom("x")

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(google_provider.types, "GenerateContentConfig", FakeConfig)
    monkeypatch.setattr(provider, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(google_provider.logger, "exception", lambda msg: errors.append(msg))

    with pytest.raises(Boom):
        provider.complete([{"role": "user", "content": "hi"}])

    assert errors == ["Google complete failed"]


@pytest.mark.asyncio
async def test_google_complete_async_uses_to_thread(monkeypatch):
    provider = google_provider.GoogleProvider()
    calls = {}

    async def fake_to_thread(func, *args):
        calls["func"] = func
        calls["args"] = args
        return "thread result"

    monkeypatch.setattr(google_provider.asyncio, "to_thread", fake_to_thread)

    result = await provider.complete_async([{"role": "user", "content": "hi"}], 21, 0.6)

    assert result == "thread result"
    assert calls["func"] == provider.complete
    assert calls["args"] == ([{"role": "user", "content": "hi"}], 21, 0.6)


def test_provider_max_context_tokens():
    assert openai_provider.OpenAIProvider().max_context_tokens() == 128_000
    assert anthropic_provider.AnthropicProvider().max_context_tokens() == 200_000
    assert google_provider.GoogleProvider().max_context_tokens() == 1_000_000
