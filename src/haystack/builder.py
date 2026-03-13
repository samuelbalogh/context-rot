import tiktoken
import logging
from src.haystack.pg_essays import load_cached as load_pg
from src.haystack.pg_essays import fetch_all as fetch_pg
from src.haystack.arxiv import load_cached as load_arxiv
from src.haystack.arxiv import fetch_all as fetch_arxiv
from src.haystack.oss_code import load_cached as load_oss_code
from src.haystack.oss_code import load_perturbed as load_oss_perturbed
from src.haystack.oss_code import fetch_all as fetch_oss_code

logger = logging.getLogger(__name__)
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _get_chunks_for_variant(variant: str, exclude_prefixes: list[str] | None = None) -> list[str]:
    if variant == "pg":
        data = load_pg() or fetch_pg()
    elif variant == "arxiv":
        data = load_arxiv() or fetch_arxiv()
    elif variant == "code":
        data = load_oss_code() or fetch_oss_code()
    elif variant == "code_perturbed":
        data = load_oss_perturbed()
        if not data:
            raise ValueError("code_perturbed requires django_perturbed. Run: uv run python scripts/perturb_django.py")
    else:
        raise ValueError(f"Unknown variant {variant}")
    if exclude_prefixes:
        data = [(p, t) for p, t in data if not any(p.startswith(prefix) for prefix in exclude_prefixes)]
    return [t for _, t in data]


def build_haystack(
    variant: str,
    target_tokens: int,
    needle: str,
    needle_position: float,
    chunks_override: list[tuple[str, str]] | None = None,
    fake_needles: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> str:
    if chunks_override is not None:
        chunks = [t for _, t in chunks_override]
    else:
        chunks = _get_chunks_for_variant(variant, exclude_prefixes)
    if not chunks:
        raise ValueError(f"No haystack data for variant {variant}")
    full = "\n\n".join(chunks)
    tokens = count_tokens(full)
    if tokens < target_tokens:
        repeat = (target_tokens // tokens) + 1
        full = "\n\n".join([full] * repeat)
    encoded = ENCODING.encode(full)
    if not fake_needles or len(fake_needles) < 2:
        needle_tokens = count_tokens(needle)
        hay_len = target_tokens - needle_tokens
        insert_pos = int(hay_len * needle_position)
        before_tok = encoded[:insert_pos]
        after_tok = encoded[insert_pos : insert_pos + (hay_len - insert_pos)]
        before = ENCODING.decode(before_tok)
        after = ENCODING.decode(after_tok)
        return f"{before}\n\n{needle}\n\n{after}"
    fakes = list(fake_needles)
    n_slots = len(fakes) + 1
    if needle_position == 0.0:
        needle_slot = 0
    elif needle_position == 1.0:
        needle_slot = n_slots - 1
    else:
        needle_slot = n_slots // 2
    slots = []
    fake_idx = 0
    for i in range(n_slots):
        slots.append(needle if i == needle_slot else fakes[fake_idx] if fake_idx < len(fakes) else needle)
        if i != needle_slot:
            fake_idx += 1
    total_needle_tokens = sum(count_tokens(s) for s in slots)
    hay_len = target_tokens - total_needle_tokens
    n_segments = n_slots - 1
    seg_len = hay_len // n_segments
    parts = []
    for i in range(n_slots):
        parts.append(slots[i])
        if i < n_segments:
            start = i * seg_len
            end = (i + 1) * seg_len if i < n_segments - 1 else hay_len
            hay_tok = encoded[start:end]
            parts.append(ENCODING.decode(hay_tok))
    return "\n\n".join(parts)
