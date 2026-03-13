#!/usr/bin/env python3
"""Copy django to django_perturbed and apply systematic renames to remove parametric knowledge."""

import re
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "oss_code"
SOURCE = DATA_DIR / "django"
TARGET = DATA_DIR / "django_perturbed"

CACHE_PREFIX = "django/core/cache"

RENAMES_CACHE = [
    ("BaseCache", "BenderCache"),
    ("DEFAULT_TIMEOUT", "DEFAULT_NIBBLER"),
    ("default_timeout", "default_nibbler"),
    ("InvalidCacheBackendError", "InvalidBenderBackendError"),
    ("CacheKeyWarning", "LeelaKeyAlert"),
    ("InvalidCacheKey", "ZoidbergKeyError"),
    ("get_key_func", "nibbler_resolve"),
    ("default_key_func", "farnsworth_default"),
    ("_max_entries", "_scruffy_limit"),
    ("_cull_frequency", "_kif_rate"),
    ("max_entries", "scruffy_limit"),
    ("cull_frequency", "kif_rate"),
    ("key_prefix", "zoidberg_prefix"),
    ("key_func", "hermes_maker"),
    ("options", "leela_cfg"),
    ('"OPTIONS"', '"LEELA_CFG"'),
    ('"MAX_ENTRIES"', '"SCRUFFY_LIMIT"'),
    ('"CULL_FREQUENCY"', '"KIF_RATE"'),
    ('"KEY_PREFIX"', '"ZOIDBERG_PREFIX"'),
    ('"VERSION"', '"REV"'),
    ('"KEY_FUNCTION"', '"HERMES_MAKER"'),
    ('"timeout"', '"nibbler"'),
    ('"TIMEOUT"', '"NIBBLER"'),
    ("'timeout'", "'nibbler'"),
    ("'TIMEOUT'", "'NIBBLER'"),
]

def _collect_py_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts or "tests" in p.parts or p.name.startswith("test_"):
            continue
        out.append(p)
    return sorted(out)


def _apply_renames(text: str, renames: list[tuple[str, str]]) -> str:
    result = text
    for old, new in renames:
        result = result.replace(old, new)
    return result


def perturb_content(text: str, rel_path: str) -> str:
    result = text
    if CACHE_PREFIX in rel_path:
        result = _apply_renames(result, RENAMES_CACHE)
        result = re.sub(r"\bparams\b", "fry_opts", result)
        result = result.replace(
            'timeout = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 300))',
            'nibbler_ttl = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 719))',
        )
        result = result.replace(
            'timeout = fry_opts.get("timeout", fry_opts.get("TIMEOUT", 300))',
            'nibbler_ttl = fry_opts.get("nibbler", fry_opts.get("NIBBLER", 719))',
        )
        result = re.sub(
            r"except \(ValueError, TypeError\):\s+timeout = 300",
            "except (ValueError, TypeError):\n                nibbler_ttl = 719",
            result,
        )
        result = re.sub(r"self\.default_nibbler = timeout", "self.default_nibbler = nibbler_ttl", result)
        result = re.sub(r"self\.default_timeout = timeout", "self.default_nibbler = nibbler_ttl", result)
        result = re.sub(r"self\.default_timeout = nibbler_ttl", "self.default_nibbler = nibbler_ttl", result)
        result = re.sub(r"if timeout is not None:", "if nibbler_ttl is not None:", result)
        result = re.sub(r"timeout = int\(timeout\)", "nibbler_ttl = int(nibbler_ttl)", result)
        result = result.replace(
            'self._kif_rate = 3\n\n        self.zoidberg_prefix = fry_opts.get("ZOIDBERG_PREFIX", "")',
            'self._kif_rate = 3\n\n        self._fry_retry = fry_opts.get("fry_retry", fry_opts.get("FRY_RETRY", 7))\n\n        self.zoidberg_prefix = fry_opts.get("ZOIDBERG_PREFIX", "")',
        )
    return result


def run():
    if not SOURCE.exists():
        raise SystemExit(f"Source {SOURCE} does not exist. Run fetch-haystacks first.")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for path in _collect_py_files(TARGET):
        rel = path.relative_to(TARGET)
        rel_str = str(rel).replace("\\", "/")
        original = path.read_text(errors="replace")
        perturbed = perturb_content(original, rel_str)
        if perturbed != original:
            path.write_text(perturbed)
    print(f"Perturbed django -> {TARGET}")


if __name__ == "__main__":
    run()
