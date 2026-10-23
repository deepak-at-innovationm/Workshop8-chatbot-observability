"""Deterministic evaluation checks and an optional relevance judge.

Tool selection is scored from the chatbot's recorded tool invocation, not from
whether the final sentence sounds like it contains a time.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.chatbot import PUBLIC_ERROR_MESSAGE, message_text

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "examples.json"
JUDGE_PROMPT = """You judge whether an assistant answer is relevant to the question.

Question:
{question}

Criteria:
{criteria}

Answer:
{answer}

Reply with JSON only, no markdown:
{{"relevant": true or false, "reason": "one short sentence"}}
Relevance means the answer addresses the question. It is not a proof that every detail is correct.
"""


def load_examples(path: Path | None = None) -> list[dict]:
    dataset_path = path or DATASET_PATH
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Evaluation dataset at {dataset_path} must be a non-empty list")
    return payload


def answer_contains_today(answer: str, timezone_name: str = "Asia/Kolkata") -> bool:
    """True when the answer cites today's date in the server timezone."""
    today = datetime.now(ZoneInfo(timezone_name))
    if today.date().isoformat() in answer:
        return True
    month = today.strftime("%B")
    has_day = re.search(rf"\b{today.day}\b", answer) is not None
    return month in answer and str(today.year) in answer and has_day


def score_tool_selection(output: dict, reference: dict) -> dict:
    expected = bool(reference.get("expect_tool"))
    actual = bool(output.get("tool_used"))
    name_ok = output.get("tool_name") == "get_current_datetime" if expected else True
    passed = actual == expected and name_ok
    return {
        "key": "tool_selection",
        "score": 1 if passed else 0,
        "comment": (
            f"expected tool={expected}, actual tool_used={actual}, tool_name={output.get('tool_name')}"
        ),
    }


def score_error_handling(output: dict, reference: dict) -> dict:
    message = output.get("error") or ""
    leaked = any(token in message for token in ("Traceback", "API_KEY", "Simulated model failure"))
    if reference.get("expect_error"):
        passed = message == PUBLIC_ERROR_MESSAGE and not leaked
        comment = "Safe public error." if passed else f"Unexpected error text: {message}"
    else:
        passed = message == "" and not leaked
        comment = "No error expected." if passed else f"Unexpected error: {message}"
    return {"key": "error_handling", "score": 1 if passed else 0, "comment": comment}


def score_response_present(output: dict, reference: dict) -> dict:
    if reference.get("expect_error"):
        return {"key": "response_present", "score": None, "comment": "Skipped for the error example."}
    answer = (output.get("answer") or "").strip()
    return {
        "key": "response_present",
        "score": 1 if answer else 0,
        "comment": "Answer is non-empty." if answer else "The answer was empty.",
    }


def score_datetime_grounding(output: dict, reference: dict, timezone_name: str = "Asia/Kolkata") -> dict:
    if reference.get("category") != "datetime":
        return {"key": "datetime_grounding", "score": None, "comment": "Skipped; not a datetime example."}
    if not output.get("tool_used"):
        return {"key": "datetime_grounding", "score": 0, "comment": "The datetime tool was not invoked."}
    ok = answer_contains_today(output.get("answer") or "", timezone_name)
    return {
        "key": "datetime_grounding",
        "score": 1 if ok else 0,
        "comment": "Answer includes today's date." if ok else "Answer did not include today's date from the tool.",
    }


def deterministic_scores(output: dict, reference: dict, timezone_name: str = "Asia/Kolkata") -> list[dict]:
    return [
        score_tool_selection(output, reference),
        score_error_handling(output, reference),
        score_response_present(output, reference),
        score_datetime_grounding(output, reference, timezone_name),
    ]


def judge_relevance(model, question: str, answer: str, criteria: str) -> dict:
    """Ask the configured chat model whether an open answer addresses the question."""
    from langchain_core.messages import HumanMessage

    response = model.invoke(
        [
            HumanMessage(
                content=JUDGE_PROMPT.format(question=question, criteria=criteria, answer=answer)
            )
        ]
    )
    text = message_text(response)
    parsed = _parse_judge_json(text)
    if parsed is None:
        return {"key": "relevance", "score": 0, "comment": "The judge did not return JSON."}
    relevant = bool(parsed.get("relevant"))
    reason = str(parsed.get("reason") or "").strip()
    return {
        "key": "relevance",
        "score": 1 if relevant else 0,
        "comment": reason or "LLM relevance judgment. This does not prove factual correctness.",
    }


def _parse_judge_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = cleaned.removesuffix("```").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None
