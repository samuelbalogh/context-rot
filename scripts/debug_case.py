#!/usr/bin/env python3
"""Debug a single case: verify needle in context and raw LLM response."""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, load_env
from src.haystack.builder import build_haystack, count_tokens
from src.providers import OpenAIProvider
from src.experiments.classic import _build_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=64000)
    args = parser.parse_args()

    load_env()
    config = load_config()
    needles = config.get("needles", {}).get("code", {}).get("pairs", [])
    pair = next((p for p in needles if p.get("id") == "django_reverse_viewname_format"), None)
    if not pair:
        print("django_reverse_viewname_format not found in config")
        return 1

    question = pair["question"]
    needle_text = pair["needle"]
    answer = pair["answer"]
    fake_needles = pair.get("fake_needles", [])
    ctx_len = args.context_length
    pos = 0.5

    haystack = build_haystack("code", ctx_len, needle_text, pos, fake_needles=fake_needles)
    needle_stripped = needle_text.strip()
    needle_in_context = needle_stripped in haystack
    print("=== Needle in context ===")
    print(f"Needle: {repr(needle_stripped)}")
    print(f"Needle present: {needle_in_context}")
    if needle_in_context:
        idx = haystack.find(needle_stripped)
        print(f"Position: char {idx}, snippet: ...{haystack[max(0,idx-50):idx+len(needle_stripped)+50]}...")
    else:
        similar = [s for s in haystack.split("\n\n") if "viewname" in s or "split" in s][:3]
        print("Relevant snippets:", similar[:200] if similar else "none")

    print("\n=== Haystack stats ===")
    print(f"Tokens: {count_tokens(haystack)}, target: {ctx_len}")

    messages = _build_messages(question, haystack)
    provider = OpenAIProvider(model=config.get("models", {}).get("openai", "gpt-5-mini"))

    print("\n=== Raw API call ===")
    kwargs = provider._complete_kwargs(messages, max_tokens=256, temperature=0.0)
    print(f"Model: {kwargs['model']}, max_completion_tokens: {kwargs.get('max_completion_tokens')}")

    async def run():
        client = provider._get_async_client()
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        raw_content = choice.message.content
        parsed = (raw_content or "").strip()
        print("\n=== Raw response ===")
        print(f"finish_reason: {choice.finish_reason}")
        print(f"message: {choice.message}")
        print(f"raw content repr: {repr(raw_content)}")
        print(f"raw content length: {len(raw_content) if raw_content else 0}")
        if hasattr(resp, "model_dump"):
            dump = resp.model_dump()
            print(f"full response keys: {list(dump.keys())}")
            if "choices" in dump and dump["choices"]:
                c = dump["choices"][0]
                print(f"choice keys: {list(c.keys())}")
                if "message" in c:
                    print(f"message keys: {list(c['message'].keys())}")
        print("\n=== Parsed (after .strip()) ===")
        print(f"repr: {repr(parsed)}")
        print(f"length: {len(parsed)}")
        if parsed:
            print(f"content: {parsed[:500]}{'...' if len(parsed) > 500 else ''}")
        return parsed

    result = asyncio.run(run())
    print("\n=== Expected vs Model Response ===")
    print(f"Expected: {answer}")
    print(f"Model:    {result}")
    print("\n=== Summary ===")
    print(f"Context length: {ctx_len}")
    print(f"Needle in context: {needle_in_context}")
    print(f"Model returned empty: {len(result) == 0}")
    return 0


if __name__ == "__main__":
    exit(main())
