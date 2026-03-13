import os
import logging
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


def _judge_prompt(question: str, expected: str, model_output: str) -> str:
    return f"""Does the model output convey the same answer as the expected answer? Accept semantically equivalent answers: same meaning, rephrasing, or case differences (e.g. "GET" = "get" = "the get HTTP method"). Only mark incorrect if the answer is wrong, contradicts the expected, or says the information is not in the context.

Question: {question}
Expected answer: {expected}
Model output: {model_output}

Answer with exactly one word: correct, incorrect, or abstained."""


def _parse_verdict(verdict: str) -> str:
    v = verdict.strip().lower()
    if "incorrect" in v:
        return "incorrect"
    if "correct" in v:
        return "correct"
    return "abstained"


def judge_correctness(
    question: str,
    expected: str,
    model_output: str,
    judge_model: str = "gpt-5.4",
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    kwargs = {"model": judge_model, "messages": [{"role": "user", "content": _judge_prompt(question, expected, model_output)}], "temperature": 0.0}
    if "gpt-5" in judge_model:
        kwargs["max_completion_tokens"] = 16
    else:
        kwargs["max_tokens"] = 16
    resp = client.chat.completions.create(**kwargs)
    return _parse_verdict(resp.choices[0].message.content or "")


async def judge_correctness_async(
    question: str,
    expected: str,
    model_output: str,
    judge_model: str = "gpt-5.4",
) -> str:
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    kwargs = {"model": judge_model, "messages": [{"role": "user", "content": _judge_prompt(question, expected, model_output)}], "temperature": 0.0}
    if "gpt-5" in judge_model:
        kwargs["max_completion_tokens"] = 16
    else:
        kwargs["max_tokens"] = 16
    resp = await client.chat.completions.create(**kwargs)
    return _parse_verdict(resp.choices[0].message.content or "")
