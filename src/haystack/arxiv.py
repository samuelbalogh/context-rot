import logging
from pathlib import Path
import httpx

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "arxiv"
BASE_URL = "https://export.arxiv.org/api/query"

logger = logging.getLogger(__name__)


def _parse_arxiv_response(xml: str) -> list[tuple[str, str, str]]:
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        paper_id = ""
        if id_el is not None and id_el.text:
            paper_id = id_el.text.split("/")[-1]
        if title and summary:
            papers.append((paper_id, title, summary))
    return papers


def fetch_category(category: str = "cs.IR", max_results: int = 50) -> list[tuple[str, str, str]]:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = httpx.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return _parse_arxiv_response(resp.text)


def fetch_all() -> list[tuple[str, str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "papers.txt"
    if cache_path.exists():
        return _load_cached(cache_path)
    papers = []
    for cat in ["cs.IR", "cs.CL"]:
        for pid, title, summary in fetch_category(cat, 25):
            text = f"{title}\n\n{summary}"
            papers.append((pid, text))
    content = "\n\n---PAPER---\n\n".join(f"[{pid}]\n{t}" for pid, t in papers)
    cache_path.write_text(content)
    return papers


def _load_cached(path: Path) -> list[tuple[str, str]]:
    content = path.read_text()
    papers = []
    for block in content.split("---PAPER---"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        pid = lines[0].strip("[]") if lines else ""
        text = lines[1] if len(lines) > 1 else block
        papers.append((pid, text.strip()))
    return papers


def load_cached() -> list[tuple[str, str]]:
    cache_path = DATA_DIR / "papers.txt"
    if not cache_path.exists():
        return []
    return _load_cached(cache_path)
