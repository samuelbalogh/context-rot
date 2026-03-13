import logging
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "oss_code"
REPOS = [
    ("requests", "https://github.com/requests/requests.git"),
    ("flask", "https://github.com/pallets/flask.git"),
    ("django", "https://github.com/django/django.git"),
]
PERTURBED_REPOS = ["django_perturbed"]

logger = logging.getLogger(__name__)


def _collect_py_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts or "tests" in p.parts or p.name.startswith("test_"):
            continue
        out.append(p)
    return sorted(out)


def _fetch_repo(name: str, url: str) -> list[tuple[str, str]]:
    repo_dir = DATA_DIR / name
    if repo_dir.exists():
        try:
            subprocess.run(["git", "pull"], cwd=repo_dir, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("git pull timeout for %s", name)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo_dir)], check=True, capture_output=True, timeout=120)
    chunks = []
    for p in _collect_py_files(repo_dir):
        rel = p.relative_to(repo_dir)
        text = p.read_text(errors="replace")
        chunks.append((str(rel), text))
    return chunks


def fetch_all() -> list[tuple[str, str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for name, url in REPOS:
        try:
            chunks = _fetch_repo(name, url)
            for fp, content in chunks:
                all_chunks.append((f"{name}/{fp}", content))
            logger.info("Fetched %s: %d files", name, len(chunks))
        except Exception as e:
            logger.warning("Failed %s: %s", name, e)
    return all_chunks


def load_cached() -> list[tuple[str, str]]:
    if not DATA_DIR.exists():
        return []
    all_chunks = []
    for name, _ in REPOS:
        repo_dir = DATA_DIR / name
        if not repo_dir.exists():
            continue
        for p in _collect_py_files(repo_dir):
            rel = p.relative_to(repo_dir)
            try:
                text = p.read_text(errors="replace")
                all_chunks.append((f"{name}/{rel}", text))
            except Exception as e:
                logger.warning("Skip %s: %s", p, e)
    return sorted(all_chunks, key=lambda x: x[0])


def load_perturbed() -> list[tuple[str, str]]:
    if not DATA_DIR.exists():
        return []
    all_chunks = []
    for name in PERTURBED_REPOS:
        repo_dir = DATA_DIR / name
        if not repo_dir.exists():
            continue
        for p in _collect_py_files(repo_dir):
            rel = p.relative_to(repo_dir)
            try:
                text = p.read_text(errors="replace")
                all_chunks.append((f"{name}/{rel}", text))
            except Exception as e:
                logger.warning("Skip %s: %s", p, e)
    return sorted(all_chunks, key=lambda x: x[0])
