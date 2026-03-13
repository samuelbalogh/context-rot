from pathlib import Path

import pytest

from src.haystack.oss_code import load_perturbed


def test_load_perturbed_empty_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("src.haystack.oss_code.DATA_DIR", tmp_path)
    assert load_perturbed() == []


def test_load_perturbed_returns_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr("src.haystack.oss_code.DATA_DIR", tmp_path)
    perturbed_dir = tmp_path / "django_perturbed"
    perturbed_dir.mkdir()
    cache_base = perturbed_dir / "django" / "core" / "cache" / "backends"
    cache_base.mkdir(parents=True)
    (cache_base / "base.py").write_text(
        'nibbler_ttl = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 719))\nclass BenderCache: pass'
    )
    chunks = load_perturbed()
    assert len(chunks) >= 1
    paths = [p for p, _ in chunks]
    assert any("django_perturbed" in p for p in paths)
    content = dict(chunks).get("django_perturbed/django/core/cache/backends/base.py", "")
    assert "BenderCache" in content
    assert "719" in content


def test_perturb_script_creates_valid_output(monkeypatch, tmp_path):
    import scripts.perturb_django as pd

    source = tmp_path / "django"
    source.mkdir()
    (source / "django").mkdir(parents=True)
    cache_base = source / "django" / "core" / "cache" / "backends"
    cache_base.mkdir(parents=True)
    (cache_base / "base.py").write_text(
        'class BaseCache:\n    def __init__(self, params):\n        timeout = params.get("timeout", params.get("TIMEOUT", 300))'
    )
    monkeypatch.setattr(pd, "SOURCE", source)
    monkeypatch.setattr(pd, "TARGET", tmp_path / "django_perturbed")
    pd.run()
    base_py = tmp_path / "django_perturbed" / "django" / "core" / "cache" / "backends" / "base.py"
    assert base_py.exists()
    text = base_py.read_text()
    assert "BenderCache" in text
    assert "nibbler_ttl = fry_opts.get" in text
    assert "719" in text
    assert "BaseCache" not in text
    assert 'params.get("timeout"' not in text


def test_build_haystack_code_perturbed(monkeypatch, tmp_path):
    import src.haystack.builder as builder

    monkeypatch.setattr("src.haystack.oss_code.DATA_DIR", tmp_path)
    perturbed_dir = tmp_path / "django_perturbed"
    perturbed_dir.mkdir()
    cache_base = perturbed_dir / "django" / "core" / "cache" / "backends"
    cache_base.mkdir(parents=True)
    (cache_base / "base.py").write_text('nibbler_ttl = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 719))\n' * 5)
    needle = 'nibbler_ttl = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 719))'
    result = builder.build_haystack("code_perturbed", 500, needle, 0.5)
    assert needle in result


def test_build_haystack_code_perturbed_missing_raises(monkeypatch, tmp_path):
    import src.haystack.builder as builder

    monkeypatch.setattr("src.haystack.oss_code.DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="django_perturbed"):
        builder.build_haystack("code_perturbed", 500, "needle", 0.5)
