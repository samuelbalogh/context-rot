import pytest
import src.haystack.builder as builder
from src.haystack.builder import build_haystack, count_tokens


def test_count_tokens():
    assert count_tokens("hello") > 0
    assert count_tokens("") == 0
    assert count_tokens("x " * 50) > 10


def test_build_haystack_with_override_contains_needle(sample_haystack_chunks):
    needle = "THE_NEEDLE"
    result = build_haystack(
        "pg", 500, needle, 0.5, chunks_override=sample_haystack_chunks
    )
    assert needle in result


def test_build_haystack_position_start(sample_haystack_chunks):
    needle = "NEEDLE"
    result = build_haystack("pg", 300, needle, 0.0, chunks_override=sample_haystack_chunks)
    parts = result.split(needle)
    assert len(parts) == 2
    assert parts[0].strip() == "" or len(parts[0]) < 50


def test_build_haystack_position_end(sample_haystack_chunks):
    needle = "NEEDLE"
    result = build_haystack("pg", 300, needle, 1.0, chunks_override=sample_haystack_chunks)
    parts = result.split(needle)
    assert len(parts) == 2
    assert parts[1].strip() == "" or len(parts[1]) < 50


def test_build_haystack_approx_token_count(sample_haystack_chunks):
    needle = "X"
    target = 200
    result = build_haystack("pg", target, needle, 0.5, chunks_override=sample_haystack_chunks)
    tok_count = count_tokens(result)
    assert target - 20 <= tok_count <= target + 50


def test_build_haystack_empty_chunks_raises():
    with pytest.raises(ValueError, match="No haystack data"):
        build_haystack("pg", 100, "x", 0.5, chunks_override=[])

def test_build_haystack_with_fake_needles(sample_haystack_chunks):
    needle = "REAL"
    fakes = ["FAKE1", "FAKE2"]
    result = build_haystack(
        "pg", 400, needle, 0.5, chunks_override=sample_haystack_chunks, fake_needles=fakes
    )
    assert needle in result
    assert "FAKE1" in result
    assert "FAKE2" in result


def test_build_haystack_with_four_fake_needles(sample_haystack_chunks):
    needle = "REAL"
    fakes = ["FAKE1", "FAKE2", "FAKE3", "FAKE4"]
    result = build_haystack(
        "pg", 600, needle, 0.5, chunks_override=sample_haystack_chunks, fake_needles=fakes
    )
    assert needle in result
    for f in fakes:
        assert f in result


def test_build_haystack_code_variant_with_override(sample_haystack_chunks):
    needle = "def foo(): pass"
    result = build_haystack(
        "code", 500, needle, 0.5, chunks_override=sample_haystack_chunks
    )
    assert needle in result


def test_get_chunks_for_variant_uses_fallback_fetch(monkeypatch):
    monkeypatch.setattr(builder, "load_pg", lambda: [])
    monkeypatch.setattr(builder, "fetch_pg", lambda: [("a", "pg chunk")])

    assert builder._get_chunks_for_variant("pg") == ["pg chunk"]


def test_get_chunks_for_variant_unknown_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        builder._get_chunks_for_variant("nope")

