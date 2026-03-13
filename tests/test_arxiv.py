from types import SimpleNamespace

import src.haystack.arxiv as arxiv
from src.haystack.arxiv import _parse_arxiv_response, _load_cached


def test_parse_arxiv_response():
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2401.00001</id>
        <title>Test Paper</title>
        <summary>Abstract text here.</summary>
    </entry>
    </feed>"""
    papers = _parse_arxiv_response(xml)
    assert len(papers) == 1
    assert papers[0][0] == "2401.00001"
    assert papers[0][1] == "Test Paper"
    assert papers[0][2] == "Abstract text here."


def test_parse_arxiv_empty():
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    assert _parse_arxiv_response(xml) == []


def test_load_cached(tmp_path):
    content = "[pid1]\ntitle1\n\nabstract1\n\n---PAPER---\n\n[pid2]\ntitle2\n\nabstract2"
    cache = tmp_path / "papers.txt"
    cache.write_text(content)
    papers = _load_cached(cache)
    assert len(papers) == 2
    assert papers[0] == ("pid1", "title1\n\nabstract1")
    assert papers[1] == ("pid2", "title2\n\nabstract2")


def test_fetch_category_calls_api(monkeypatch):
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2401.00001</id>
        <title>Test Paper</title>
        <summary>Abstract text here.</summary>
    </entry>
    </feed>"""
    called = {}

    def fake_get(url, params=None, timeout=60):
        called["url"] = url
        called["params"] = params
        return SimpleNamespace(text=xml, raise_for_status=lambda: None)

    monkeypatch.setattr(arxiv.httpx, "get", fake_get)

    papers = arxiv.fetch_category("cs.IR", 7)

    assert called["url"] == arxiv.BASE_URL
    assert called["params"]["search_query"] == "cat:cs.IR"
    assert called["params"]["max_results"] == 7
    assert papers[0][0] == "2401.00001"


def test_fetch_all_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(arxiv, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        arxiv,
        "fetch_category",
        lambda category, max_results: [("p1", "Title", f"Summary {category}")],
    )

    papers = arxiv.fetch_all()

    assert papers == [("p1", "Title\n\nSummary cs.IR"), ("p1", "Title\n\nSummary cs.CL")]
    assert (tmp_path / "papers.txt").exists()


def test_load_cached_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "DATA_DIR", tmp_path / "missing")

    assert arxiv.load_cached() == []
