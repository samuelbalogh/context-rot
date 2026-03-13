.PHONY: install test test-integration fetch-haystacks perturb-django run run-quick run-code run-code-smoke run-code-large run-code-64k-128k run-code-smoke-64k-128k run-code-350k run-code-cheap run-code-perturbed run-code-perturbed-sweep debug-django-reverse report

MODELS ?= openai,google
VARIANT ?= code
MAX_CALLS ?=
ARGS ?=
RESULTS_DIR ?= results

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing -m "not integration"

test-integration:
	uv run pytest tests/integration/ -v -m integration

fetch-haystacks:
	uv run python scripts/fetch_haystacks.py

perturb-django:
	uv run python scripts/perturb_django.py

run:
	uv run python run.py --results-dir $(RESULTS_DIR) $(ARGS)

run-quick:
	uv run python run.py --models openai --max-calls 6

run-code:
	uv run python run.py --results-dir $(RESULTS_DIR) --variant $(VARIANT) --models $(MODELS) $(if $(MAX_CALLS),--max-calls $(MAX_CALLS)) $(if $(TRIALS),--trials $(TRIALS)) $(if $(NEEDLE_PAIRS),--needle-pairs $(NEEDLE_PAIRS)) $(if $(CONTEXT_LENGTHS),--context-lengths $(CONTEXT_LENGTHS)) $(if $(NEEDLE_POSITIONS),--needle-positions $(NEEDLE_POSITIONS)) $(ARGS)

run-code-smoke: NEEDLE_PAIRS = flask_url_for_context,django_reverse_viewname_format
run-code-smoke: CONTEXT_LENGTHS = 4000
run-code-smoke: NEEDLE_POSITIONS = 0.5
run-code-smoke: run-code report

run-code-large: MAX_CALLS ?= 100
run-code-large: run-code report

run-code-64k-128k: NEEDLE_PAIRS = flask_url_for_context,django_reverse_viewname_format
run-code-64k-128k: CONTEXT_LENGTHS = 64000,128000
run-code-64k-128k: NEEDLE_POSITIONS = 0.5
run-code-64k-128k: run-code report

run-code-smoke-64k-128k: NEEDLE_PAIRS = flask_url_for_context,django_reverse_viewname_format
run-code-smoke-64k-128k: CONTEXT_LENGTHS = 64000,128000
run-code-smoke-64k-128k: NEEDLE_POSITIONS = $(shell python -c "import random; print(round(random.uniform(0.4, 0.7), 2))")
run-code-smoke-64k-128k: run-code report

run-code-350k: MODELS = openai
run-code-350k: NEEDLE_PAIRS = flask_url_for_context,django_reverse_viewname_format
run-code-350k: CONTEXT_LENGTHS = 260000
run-code-350k: NEEDLE_POSITIONS = $(shell python -c "import random; print(round(random.uniform(0.4, 0.7), 2))")
run-code-350k: run-code report

run-code-cheap: NEEDLE_PAIRS = requests_get_method,flask_url_for_context,django_reverse_viewname_format,django_reverse_viewname_code_style,django_cache_default_timeout
run-code-cheap: CONTEXT_LENGTHS = 200, 300, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000
run-code-cheap: NEEDLE_POSITIONS = 0.25,0.5,0.75
run-code-cheap: run-code report

run-code-perturbed: VARIANT = code_perturbed
run-code-perturbed: NEEDLE_PAIRS = bender_cache_default_nibbler
run-code-perturbed: run-code report

run-code-perturbed-sweep: VARIANT = code_perturbed
run-code-perturbed-sweep: NEEDLE_PAIRS = bender_cache_unguided
run-code-perturbed-sweep: CONTEXT_LENGTHS = 200,300,500,1000,2000,4000,8000,16000,32000,64000,128000
run-code-perturbed-sweep: NEEDLE_POSITIONS = 0.5
run-code-perturbed-sweep: TRIALS = 10
run-code-perturbed-sweep: MODELS = openai
run-code-perturbed-sweep: run-code report

debug-django-reverse:
	uv run python scripts/debug_case.py $(if $(CONTEXT_LENGTH),--context-length $(CONTEXT_LENGTH))

report:
	uv run python run.py --report-only --results-dir $(RESULTS_DIR)
