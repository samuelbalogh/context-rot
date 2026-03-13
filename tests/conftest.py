import pytest


@pytest.fixture
def sample_config(tmp_path):
    config = {
        "models": {"openai": "gpt-4o", "anthropic": "claude-sonnet", "google": "gemini"},
        "needle_positions": [0, 0.5, 1.0],
        "context_lengths": [1000, 4000],
        "needles": {
            "pg": {"question": "Q?", "answer": "A"},
            "arxiv": {"question": "Q2?", "answer": "A2"},
        },
    }
    path = tmp_path / "config.yaml"
    import yaml
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def sample_haystack_chunks():
    return [("a", "x " * 100), ("b", "y " * 100)]
