#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, load_env
from src.haystack.builder import build_haystack
from src.providers import OpenAIProvider, AnthropicProvider, GoogleProvider
from src.experiments.classic import run_classic_niah_async
from src.evaluation.judge import judge_correctness_async
from src.output.json_writer import write_json
from src.output.csv_writer import write_csv
from src.output.html_report import write_html_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path(__file__).parent / "results"


def get_provider(name: str, model: str):
    if name == "openai":
        return OpenAIProvider(model=model)
    if name == "anthropic":
        return AnthropicProvider(model=model)
    if name == "google":
        return GoogleProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")


async def _run_one(
    provider,
    model_key: str,
    variant: str,
    pair_id: str,
    question: str,
    needle_text: str,
    answer_for_judge: str,
    fake_needles: list,
    ctx_len: int,
    pos: float,
    judge_model: str,
    run_id: str,
    dry_run: bool,
    results: list,
    results_lock: asyncio.Lock,
    existing_path: Path,
    exclude_prefixes: list[str] | None = None,
):
    if dry_run:
        r = {"model": model_key, "variant": variant, "needle_pair": pair_id, "question": question, "expected_answer": answer_for_judge, "context_length": ctx_len, "position": pos, "correct": "correct", "model_output": "[dry-run]"}
        async with results_lock:
            results.append(r)
            write_json(results, existing_path)
        return r
    try:
        output = await run_classic_niah_async(provider, variant, question, needle_text, ctx_len, pos, fake_needles=fake_needles, exclude_prefixes=exclude_prefixes)
        correct = await judge_correctness_async(question, answer_for_judge, output, judge_model)
        r = {"run_id": run_id, "model": model_key, "variant": variant, "needle_pair": pair_id, "question": question, "expected_answer": answer_for_judge, "context_length": ctx_len, "position": pos, "correct": correct, "model_output": output}
    except Exception as e:
        logger.exception("Failed %s %s %s len=%s pos=%s: %s", model_key, variant, pair_id, ctx_len, pos, e)
        r = {"run_id": run_id, "model": model_key, "variant": variant, "needle_pair": pair_id, "question": question, "expected_answer": answer_for_judge, "context_length": ctx_len, "position": pos, "correct": "error", "model_output": str(e)}
    async with results_lock:
        results.append(r)
        write_json(results, existing_path)
    logger.info("%s %s %s len=%s pos=%s -> %s", model_key, variant, pair_id, ctx_len, pos, r.get("correct", "?"))
    return r


async def _run_provider(
    provider_name: str,
    model: str,
    work_items: list[tuple],
    judge_model: str,
    run_id: str,
    dry_run: bool,
    global_semaphore: asyncio.Semaphore | None,
    workers_per_provider: int,
    results: list,
    results_lock: asyncio.Lock,
    existing_path: Path,
) -> list:
    provider = get_provider(provider_name, model)
    model_key = f"{provider_name}:{model}"
    max_ctx = max(w[5] for w in work_items)
    workers = 1 if provider_name == "anthropic" and max_ctx > 30_000 else workers_per_provider
    provider_semaphore = asyncio.Semaphore(workers) if workers > 0 else None

    async def run_with_semaphores(variant, pair_id, question, needle_text, answer_for_judge, fake_needles, ctx_len, pos, exclude_prefixes):
        async def do():
            return await _run_one(provider, model_key, variant, pair_id, question, needle_text, answer_for_judge, fake_needles, ctx_len, pos, judge_model, run_id, dry_run, results, results_lock, existing_path, exclude_prefixes)
        if provider_semaphore and global_semaphore:
            async with provider_semaphore:
                async with global_semaphore:
                    return await do()
        elif provider_semaphore:
            async with provider_semaphore:
                return await do()
        elif global_semaphore:
            async with global_semaphore:
                return await do()
        return await do()

    tasks = [run_with_semaphores(v, pid, q, nt, aj, fn, cl, p, ex) for v, pid, q, nt, aj, fn, cl, p, ex in work_items]
    return await asyncio.gather(*tasks)


def _run_dir_suffix(providers_config: list) -> str:
    safe = lambda s: str(s).replace(":", "_").replace(".", "_")
    return "_".join(f"{p}_{safe(m)}" for p, m in providers_config)


def _latest_run_dir(results_dir: Path) -> Path | None:
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    return max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--workers-per-provider", type=int, default=3)
    parser.add_argument("--context-lengths", type=str, default=None, help="Comma-separated token counts, e.g. 6000")
    parser.add_argument("--needle-positions", type=str, default=None, help="Comma-separated positions 0-1, e.g. 0.5")
    parser.add_argument("--variant", type=str, default="pg,arxiv", help="Comma-separated variants: pg, arxiv, code. Default: pg,arxiv")
    parser.add_argument("--needle-pairs", type=str, default=None, help="Comma-separated pair IDs to run (e.g. flask_url_for_context,django_reverse_viewname_format)")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--trials", type=int, default=None, help="Runs per condition (model, variant, pair, length, position). Aggregated in chart.")
    args = parser.parse_args()
    load_env()
    config = load_config()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_models = config.get("models", {})
    if args.models:
        model_specs = {}
        for part in args.models.split(","):
            part = part.strip()
            if ":" in part:
                provider, model = part.split(":", 1)
                model_specs[provider.strip()] = model.strip()
            else:
                model_specs[part] = base_models.get(part, part)
    else:
        model_specs = {
            "openai": base_models.get("openai", "gpt-5.1"),
            "google": base_models.get("google", "gemini-2.5-pro"),
        }
    providers_config = [(name, model_specs[name]) for name in ["openai", "anthropic", "google"] if name in model_specs]
    if args.report_only:
        if (args.results_dir / "results.json").exists():
            run_dir = args.results_dir
        else:
            run_dir = _latest_run_dir(args.results_dir)
        if not run_dir or not (run_dir / "results.json").exists():
            logger.error("No results file found in %s", args.results_dir)
            return 1
        results = json.loads((run_dir / "results.json").read_text())
        write_html_report(results, run_dir / "report.html")
        write_csv(results, run_dir / "results.csv")
        logger.info("Report written to %s", run_dir / "report.html")
        return 0
    run_dir = args.results_dir / f"{run_id}_{_run_dir_suffix(providers_config)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    needle_positions = config.get("needle_positions", [0, 0.5, 1.0])
    if args.needle_positions:
        needle_positions = [float(x.strip()) for x in args.needle_positions.split(",")]
    context_lengths = config.get("context_lengths", [1000, 4000, 16000, 32000, 64000, 128000])
    if args.context_lengths:
        context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    needles = config.get("needles", {})
    variants_filter = [v.strip() for v in args.variant.split(",") if v.strip()]
    needles = {k: v for k, v in needles.items() if k in variants_filter}
    judge_model = config.get("judge_model", "gpt-5.4")
    trials = args.trials if args.trials is not None else config.get("trials", 1)
    results = []
    max_calls = args.max_calls or 9999
    global_semaphore = asyncio.Semaphore(max_calls) if args.max_calls else None
    workers_per_provider = args.workers_per_provider

    pairs_filter = [p.strip() for p in (args.needle_pairs or "").split(",") if p.strip()]

    def _pairs_from_cfg(cfg):
        if "pairs" in cfg:
            out = []
            for i, p in enumerate(cfg["pairs"]):
                pid = p.get("id", str(i))
                if pairs_filter and pid not in pairs_filter:
                    continue
                q = p.get("question", "")
                a = p.get("answer", "")
                needle_text = p.get("needle") or a
                exclude = p.get("haystack_exclude")
                if exclude and not isinstance(exclude, list):
                    exclude = [exclude]
                out.append((pid, q, needle_text, a, p.get("fake_needles", []), exclude))
            return out
        q, a = cfg.get("question", ""), cfg.get("answer", "")
        needle_text = cfg.get("needle") or a
        exclude = cfg.get("haystack_exclude")
        if exclude and not isinstance(exclude, list):
            exclude = [exclude]
        return [("0", q, needle_text, a, cfg.get("fake_needles", []), exclude)] if q and a else []

    all_work = []
    for provider_name, model in providers_config:
        model_key = f"{provider_name}:{model}"
        for variant, needle_cfg in needles.items():
            for pair_id, question, needle_text, answer_for_judge, fake_needles, exclude_prefixes in _pairs_from_cfg(needle_cfg):
                if not question or not needle_text:
                    continue
                for ctx_len in context_lengths:
                    for pos in needle_positions:
                        for _ in range(trials):
                            all_work.append((provider_name, model, variant, pair_id, question, needle_text, answer_for_judge, fake_needles, ctx_len, pos, exclude_prefixes))
                        if len(all_work) >= max_calls:
                            break
                    if len(all_work) >= max_calls:
                        break
                if len(all_work) >= max_calls:
                    break
            if len(all_work) >= max_calls:
                break
        if len(all_work) >= max_calls:
            break
    work_per_provider = {}
    for item in all_work:
        provider_name, model = item[0], item[1]
        key = (provider_name, model)
        if key not in work_per_provider:
            work_per_provider[key] = []
        work_per_provider[key].append((item[2], item[3], item[4], item[5], item[6], item[7], item[8], item[9], item[10]))
    work_per_provider = [(k[0], k[1], v) for k, v in work_per_provider.items()]
    results_lock = asyncio.Lock()

    seen = set()
    for provider_name, model, work_list in work_per_provider:
        for variant, pair_id, question, needle_text, answer_for_judge, fake_needles, ctx_len, pos, exclude_prefixes in work_list:
            key = (variant, needle_text, ctx_len, pos, tuple(exclude_prefixes or []))
            if key in seen:
                continue
            seen.add(key)
            haystack = build_haystack(variant, ctx_len, needle_text, pos, fake_needles=fake_needles, exclude_prefixes=exclude_prefixes)
            needle_stripped = needle_text.strip()
            if needle_stripped not in haystack:
                logger.error("Needle not in context: variant=%s pair=%s len=%s pos=%s", variant, pair_id, ctx_len, pos)
                logger.error("Needle: %s", repr(needle_stripped[:100]))
                return 1
            if answer_for_judge not in haystack:
                logger.warning("Expected answer %r not in context (needle may contain it): variant=%s pair=%s", answer_for_judge, variant, pair_id)
    logger.info("Verified needle in context for %d unique cases", len(seen))
    if trials > 1:
        logger.info("Running %d trials per condition in parallel (%d total work items)", trials, len(all_work))

    async def run_all():
        provider_tasks = [
            _run_provider(name, model, work, judge_model, run_id, args.dry_run, global_semaphore, workers_per_provider, results, results_lock, results_path)
            for name, model, work in work_per_provider
            if work
        ]
        await asyncio.gather(*provider_tasks)

    asyncio.run(run_all())
    write_json(results, results_path)
    if trials > 1:
        meta = {"trials": trials, "run_id": run_id}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    write_csv(results, run_dir / "results.csv")
    write_html_report(results, run_dir / "report.html")
    logger.info("Done. %d results. Report: %s", len(results), run_dir / "report.html")
    return 0


if __name__ == "__main__":
    exit(main())
