from types import SimpleNamespace

import src.haystack.pg_essays as pg
from src.haystack.pg_essays import _extract_links, _extract_text


def test_extract_links_relative():
    html = '<a href="greatwork.html">Link</a><a href="arc.html">Arc</a>'
    links = _extract_links(html)
    assert "https://paulgraham.com/greatwork.html" in links
    assert "https://paulgraham.com/arc.html" in links


def test_extract_links_full_url():
    html = '<a href="https://paulgraham.com/essay.html">Essay</a>'
    links = _extract_links(html)
    assert "https://paulgraham.com/essay.html" in links


def test_extract_links_skips_index():
    html = '<a href="index.html">Index</a><a href="articles.html">Articles</a><a href="other.html">Other</a>'
    links = _extract_links(html)
    assert "https://paulgraham.com/other.html" in links
    assert "https://paulgraham.com/index.html" not in links
    assert "https://paulgraham.com/articles.html" not in links


def test_extract_text_strips_tags():
    html = "<p>Hello <b>world</b></p>"
    assert "Hello" in _extract_text(html)
    assert "world" in _extract_text(html)
    assert "<" not in _extract_text(html)


def test_extract_text_removes_script():
    html = "<p>Content</p><script>alert(1)</script><p>More</p>"
    result = _extract_text(html)
    assert "Content" in result
    assert "alert" not in result


def test_fetch_essay_upgrades_http(monkeypatch):
    called = {}

    def fake_get(url, follow_redirects=True, timeout=30):
        called["url"] = url
        return SimpleNamespace(text="<p>Essay</p>", raise_for_status=lambda: None)

    monkeypatch.setattr(pg.httpx, "get", fake_get)

    result = pg.fetch_essay("http://www.paulgraham.com/test.html")

    assert called["url"] == "https://paulgraham.com/test.html"
    assert result == "Essay"


def test_fetch_all_uses_cache_and_fetches_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pg, "DATA_DIR", tmp_path)

    def fake_get(url, timeout=30):
        return SimpleNamespace(
            text='<a href="cached.html">Cached</a><a href="fresh.html">Fresh</a>',
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(pg.httpx, "get", fake_get)
    monkeypatch.setattr(pg, "fetch_essay", lambda url: f"text for {url}")
    (tmp_path / "cached.txt").write_text("cached body")

    essays = pg.fetch_all()

    assert essays == [
        ("cached", "cached body"),
        ("fresh", "text for https://paulgraham.com/fresh.html"),
    ]
    assert (tmp_path / "fresh.txt").read_text() == "text for https://paulgraham.com/fresh.html"


def test_load_cached_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "DATA_DIR", tmp_path / "missing")

    assert pg.load_cached() == []
