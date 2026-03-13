import re
import logging
from pathlib import Path
import httpx

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pg_essays"
INDEX_URL = "https://paulgraham.com/articles.html"

logger = logging.getLogger(__name__)


def _extract_links(html: str) -> list[str]:
    base = "https://paulgraham.com"
    full = re.findall(r'href="(https?://(?:www\.)?paulgraham\.com/[^"]+\.html)"', html)
    rel1 = re.findall(r'href="(/[a-zA-Z0-9_.-]+\.html)"', html)
    rel2 = re.findall(r'href="([a-zA-Z0-9_.-]+\.html)"', html)
    links = list(dict.fromkeys(full))
    for p in rel1:
        links.append(base + p)
    for p in rel2:
        if p not in ("index.html", "articles.html", "books.html"):
            links.append(f"{base}/{p}")
    return list(dict.fromkeys(links))


def _extract_text(html: str) -> str:
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_essay(url: str) -> str:
    if url.startswith("http://"):
        url = url.replace("http://www.", "https://", 1).replace("http://", "https://", 1)
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    return _extract_text(resp.text)


def fetch_all() -> list[tuple[str, str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(INDEX_URL, timeout=30)
    resp.raise_for_status()
    links = _extract_links(resp.text)
    essays = []
    for url in links:
        slug = Path(url).stem
        cache_path = DATA_DIR / f"{slug}.txt"
        if cache_path.exists():
            essays.append((slug, cache_path.read_text()))
            continue
        try:
            text = fetch_essay(url)
            cache_path.write_text(text)
            essays.append((slug, text))
            logger.info("Fetched %s", slug)
        except Exception as e:
            logger.warning("Failed %s: %s", url, e)
    return essays


def load_cached() -> list[tuple[str, str]]:
    if not DATA_DIR.exists():
        return []
    essays = []
    for p in sorted(DATA_DIR.glob("*.txt")):
        essays.append((p.stem, p.read_text()))
    return essays
