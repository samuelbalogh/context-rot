from pathlib import Path

import src.config as config
from src.config import load_config


def test_load_config():
    cfg = load_config()
    assert "models" in cfg
    assert "needle_positions" in cfg
    assert "context_lengths" in cfg
    assert "needles" in cfg


def test_load_config_has_expected_structure():
    cfg = load_config()
    assert isinstance(cfg["needle_positions"], list)
    assert isinstance(cfg["context_lengths"], list)
    assert "pg" in cfg["needles"]
    assert "arxiv" in cfg["needles"]
    assert "code" in cfg["needles"]
    pg = cfg["needles"]["pg"]
    assert "pairs" in pg or ("question" in pg and "answer" in pg)
    if "pairs" in pg:
        assert len(pg["pairs"]) >= 1
        assert "question" in pg["pairs"][0] and "answer" in pg["pairs"][0]


def test_load_env_uses_override_and_explicit_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("CONTEXT_ROT_ENV", "/tmp/override.env")
    monkeypatch.setattr(config, "load_dotenv", lambda path: calls.append(Path(path)))

    config.load_env(tmp_path / "explicit.env")

    assert calls == [Path("/tmp/override.env"), tmp_path / "explicit.env"]


def test_load_env_uses_project_env_when_present(monkeypatch, tmp_path):
    calls = []
    project_root = tmp_path / "project"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    (project_root / ".env").write_text("X=1")
    monkeypatch.delenv("CONTEXT_ROT_ENV", raising=False)
    monkeypatch.setattr(config, "__file__", str(src_dir / "config.py"))
    monkeypatch.setattr(config, "load_dotenv", lambda path: calls.append(Path(path)))

    config.load_env()

    assert calls == [project_root / ".env"]
