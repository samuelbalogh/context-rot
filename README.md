# Context Rot Needle-in-Haystack Experiment

Replicates [Chroma's Context Rot](https://research.trychroma.com/context-rot) NIAH methodology. Classic needle-in-haystack with Paul Graham essays and arXiv papers as haystacks.

## Architecture

```
config.yaml + .env  -->  Haystack (PG + arXiv)  -->  Classic NIAH  -->  Providers (OpenAI, Anthropic, Google)
                                                                              |
                                                                              v
                                                                    LLM Judge (GPT-4o)
                                                                              |
                                                                              v
                                                                    JSON / CSV / HTML
```

- **Haystacks**: PG essays from paulgraham.com, arXiv abstracts from cs.IR/cs.CL, OSS code (requests, flask)
- **Perturbed codebase** (`code_perturbed`): Django cache module copied and systematically renamed (`scripts/perturb_django.py`) so the model cannot rely on parametric knowledge. Identifiers like `BaseCache`, `max_entries`, `timeout` become `BenderCache`, `scruffy_limit`, `nibbler`; default values (e.g. 300, 3) are perturbed to 719, 417, 13. This setup reproduces the **lost-in-the-middle** effect: accuracy is high at short context (e.g. 100% at 200 tokens) and drops sharply at longer lengths (e.g. 6.7% at 64K).
- **Experiment**: Insert needle at 0%, 50%, 100% of context; vary context length (1K–128K)
- **Judge**: GPT-4o evaluates model output vs expected needle
- **Output**: `results/` with JSON, CSV, and HTML report (Chart.js)

## Setup

```bash
make install
```

Set API keys in `.env` or parent `all/.env`:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Override env path: `CONTEXT_ROT_ENV=/path/to/.env`

```bash
make fetch-haystacks   # Download PG essays + arXiv papers (cached in data/)
```

## Run

```bash
make run          # Full experiment (3 models, pg+arxiv variants, 6 lengths, 3 positions)
make run-quick    # 1 model, 6 calls
make run-code     # Code variant only (requests/flask haystack, coding needles)
make perturb-django       # Build django_perturbed (required before run-code-perturbed)
make run-code-perturbed   # Perturbed variant (BenderCache needles, lost-in-the-middle)
make run-code-perturbed-sweep  # Full sweep: bender_cache_unguided, all context lengths
make report       # Regenerate HTML from existing results
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Log without API calls |
| `--max-calls N` | Limit API calls |
| `--workers-per-provider N` | Max concurrent requests per provider (default: 3). Anthropic auto-uses 1 when context > 30K to avoid 429. |
| `--models openai:gpt-5.1,anthropic:claude-sonnet-4-6` | Filter providers/models |
| `--variant pg,arxiv,code` | Variants to run (default: pg,arxiv). Use `code` for codebase haystack. |
| `--report-only` | Generate HTML from `results_latest.json` |
| `--results-dir PATH` | Output directory |

## Config (`config.yaml`)

| Key | Description |
|-----|-------------|
| `models` | Model IDs per provider (openai, anthropic, google) |
| `judge_model` | Model for correctness evaluation |
| `needle_positions` | [0, 0.5, 1.0] = start, middle, end |
| `context_lengths` | Token counts, e.g. [1000, 4000, 16000, 32000, 64000, 128000] |
| `needles` | Per-variant `question` and `answer` (the needle) |

## Rate Limits

Anthropic enforces 30K input tokens/min on some tiers. For context > 30K tokens, the runner uses 1 worker for Anthropic and retries 429s with 75s backoff. To skip Anthropic: `--models openai,google`.

## Testing

```bash
make test              # Unit tests (no API calls)
make test-integration  # Integration tests (require API keys, real API calls)
```

Unit tests cover config, haystack builder, PG/arXiv parsing, metrics, output writers, and classic NIAH (with mock provider). Integration tests verify each provider's API usage with a minimal prompt.

## Project Structure

```
context-rot/
├── run.py              # CLI entry
├── config.yaml
├── src/
│   ├── config.py
│   ├── providers/      # OpenAI, Anthropic, Google
│   ├── haystack/       # PG essays, arXiv, builder
│   ├── experiments/    # classic NIAH
│   ├── evaluation/     # judge, metrics
│   └── output/        # JSON, CSV, HTML
├── data/               # Cached haystacks (gitignored)
├── results/            # JSON, CSV, report.html
└── tests/
```

## References

- [Chroma Context Rot](https://research.trychroma.com/context-rot)
- [Chroma context-rot GitHub](https://github.com/chroma-core/context-rot)
- [Greg Kamradt NIAH](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
