from pathlib import Path

import src.haystack.oss_code as oss_code
from src.haystack.oss_code import load_cached, _collect_py_files, _fetch_repo, fetch_all


def test_collect_py_files_skips_tests(tmp_path):
    (tmp_path / "foo.py").write_text("x")
    (tmp_path / "test_bar.py").write_text("x")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"")
    found = _collect_py_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "foo.py"


def test_load_cached_reads_all_repos(monkeypatch, tmp_path):
    monkeypatch.setattr(oss_code, "DATA_DIR", tmp_path)
    monkeypatch.setattr(oss_code, "REPOS", [("requests", "x"), ("flask", "y")])
    (tmp_path / "requests").mkdir()
    (tmp_path / "requests" / "api.py").write_text("print('requests')")
    (tmp_path / "flask").mkdir()
    (tmp_path / "flask" / "app.py").write_text("print('flask')")

    chunks = load_cached()

    assert chunks == [
        ("flask/app.py", "print('flask')"),
        ("requests/api.py", "print('requests')"),
    ]


def test_fetch_repo_clones_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(oss_code, "DATA_DIR", tmp_path)

    calls = []

    def fake_run(cmd, check=False, capture_output=False, timeout=None, cwd=None):
        calls.append((cmd, cwd))
        repo_dir = tmp_path / "demo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "main.py").write_text("print('demo')")

    monkeypatch.setattr(oss_code.subprocess, "run", fake_run)

    chunks = _fetch_repo("demo", "https://example.com/demo.git")

    assert calls[0][0][:2] == ["git", "clone"]
    assert chunks == [("main.py", "print('demo')")]


def test_fetch_repo_pulls_when_repo_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(oss_code, "DATA_DIR", tmp_path)
    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("print('demo')")
    calls = []

    def fake_run(cmd, check=False, capture_output=False, timeout=None, cwd=None):
        calls.append((cmd, cwd))

    monkeypatch.setattr(oss_code.subprocess, "run", fake_run)

    chunks = _fetch_repo("demo", "https://example.com/demo.git")

    assert calls == [(["git", "pull"], repo_dir)]
    assert chunks == [("main.py", "print('demo')")]


def test_fetch_all_prefixes_repo_names(monkeypatch, tmp_path):
    monkeypatch.setattr(oss_code, "DATA_DIR", tmp_path)
    monkeypatch.setattr(oss_code, "REPOS", [("requests", "x"), ("django", "y")])

    def fake_fetch_repo(name, url):
        return [("a.py", f"from {name}")]

    monkeypatch.setattr(oss_code, "_fetch_repo", fake_fetch_repo)

    chunks = fetch_all()

    assert chunks == [
        ("requests/a.py", "from requests"),
        ("django/a.py", "from django"),
    ]
