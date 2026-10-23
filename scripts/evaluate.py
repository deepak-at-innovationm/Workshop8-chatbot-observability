"""Run the chatbot on the evaluation dataset and record the result.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --no-upload
    python scripts/evaluate.py --no-llm-judge
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chatbot import ChatService, ChatbotError  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.evaluation import (  # noqa: E402
    deterministic_scores,
    judge_relevance,
    load_examples,
)
from app.tracing import apply_runtime_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MLOps chatbot demo.")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Score locally and do not create a LangSmith experiment.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip the relevance judge. Tool selection and error checks still run.",
    )
    return parser.parse_args()


def target_fn(service: ChatService):
    def target(inputs: dict) -> dict:
        try:
            result = service.answer(
                inputs.get("question") or "",
                session_id=str(uuid.uuid4()),
                demo_fault=inputs.get("demo_fault"),
            )
        except ChatbotError as exc:
            return {
                "answer": "",
                "tool_used": False,
                "tool_name": None,
                "error": exc.public_message,
            }
        return {
            "answer": result.answer,
            "tool_used": result.tool_used,
            "tool_name": result.tool_name,
            "error": None,
        }

    return target


def _print_summary(rows: list[dict]) -> int:
    failures = 0
    print("\nLocal evaluation summary")
    print("-" * 72)
    for row in rows:
        question = row["question"]
        parts = []
        for score in row["scores"]:
            if score["score"] is None:
                continue
            mark = "pass" if score["score"] == 1 else "fail"
            if score["score"] != 1:
                failures += 1
            label = f"{score['key']}={mark}"
            if score["score"] != 1 and score.get("comment"):
                label = f"{label}: {score['comment']}"
            parts.append(label)
        print(f"- {question}")
        print("  " + ", ".join(parts))
    total = sum(1 for row in rows for score in row["scores"] if score["score"] is not None)
    passed = total - failures
    print("-" * 72)
    print(f"Deterministic and judge checks passed: {passed}/{total}")
    print("Tool selection uses the recorded tool invocation, not a string match on the answer.")
    print("A relevance score is a model judgment. It does not prove the answer is correct.")
    return failures


def _run_local(service: ChatService, examples: list[dict], use_judge: bool) -> list[dict]:
    call = target_fn(service)
    rows = []
    judge_model = service._require_model() if use_judge else None
    for example in examples:
        output = call(example["inputs"])
        scores = deterministic_scores(output, example["outputs"], service.settings.app_timezone)
        if use_judge and example["outputs"].get("category") == "general":
            try:
                scores.append(
                    judge_relevance(
                        judge_model,
                        example["inputs"]["question"],
                        output.get("answer") or "",
                        example["outputs"].get("criteria") or "",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one judge failure should not hide the other scores
                scores.append({"key": "relevance", "score": 0, "comment": f"Judge failed: {exc.__class__.__name__}"})
        rows.append({"question": example["inputs"]["question"], "scores": scores})
    return rows


def _langsmith_evaluators(timezone_name: str, judge_model, use_judge: bool):
    def tool_selection(run, example):
        from app.evaluation import score_tool_selection

        return score_tool_selection(run.outputs or {}, example.outputs or {})

    def error_handling(run, example):
        from app.evaluation import score_error_handling

        return score_error_handling(run.outputs or {}, example.outputs or {})

    def response_present(run, example):
        from app.evaluation import score_response_present

        return score_response_present(run.outputs or {}, example.outputs or {})

    def datetime_grounding(run, example):
        from app.evaluation import score_datetime_grounding

        return score_datetime_grounding(run.outputs or {}, example.outputs or {}, timezone_name)

    evaluators = [tool_selection, error_handling, response_present, datetime_grounding]
    if use_judge and judge_model is not None:

        def relevance(run, example):
            reference = example.outputs or {}
            if reference.get("category") != "general":
                return {"key": "relevance", "score": None, "comment": "Skipped; not a general-knowledge example."}
            try:
                return judge_relevance(
                    judge_model,
                    (example.inputs or {}).get("question", ""),
                    (run.outputs or {}).get("answer") or "",
                    reference.get("criteria") or "",
                )
            except Exception as exc:  # noqa: BLE001
                return {"key": "relevance", "score": 0, "comment": f"Judge failed: {exc.__class__.__name__}"}

        evaluators.append(relevance)
    return evaluators


def example_create_payload(example: dict, *, field_names: set[str]) -> dict:
    """Build one dataset row for the installed LangSmith client.

    LangSmith 0.8 stores expected labels in ``outputs``. Later clients renamed
    that field to ``reference_outputs`` and ignore the old name on create.
    Sending the wrong key uploads the question with empty labels.
    """
    payload = {
        "inputs": example["inputs"],
        "metadata": example.get("metadata") or {},
    }
    if "reference_outputs" in field_names:
        payload["reference_outputs"] = example["outputs"]
    else:
        payload["outputs"] = example["outputs"]
    return payload


def example_identity(inputs: dict | None, outputs: dict | None) -> tuple[str, str]:
    """Stable comparison key for inputs and expected labels."""
    return (
        json.dumps(inputs or {}, sort_keys=True),
        json.dumps(outputs or {}, sort_keys=True),
    )


def dataset_examples_match(remote_rows: list[tuple[str, str]], local_rows: list[tuple[str, str]]) -> bool:
    return sorted(remote_rows) == sorted(local_rows)


def _sync_dataset(client, dataset_name: str, examples: list[dict]):
    from langsmith.schemas import ExampleCreate
    from langsmith.utils import LangSmithNotFoundError

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name,
            description="General chatbot demo: everyday questions, datetime tool use, and one error case.",
        )
    remote = list(client.list_examples(dataset_id=dataset.id))
    remote_rows = [example_identity(example.inputs, example.outputs) for example in remote]
    local_rows = [example_identity(example["inputs"], example["outputs"]) for example in examples]
    if not dataset_examples_match(remote_rows, local_rows):
        if remote:
            client.delete_examples([example.id for example in remote])
        field_names = set(ExampleCreate.model_fields)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[example_create_payload(example, field_names=field_names) for example in examples],
        )
    return dataset


def main() -> int:
    args = parse_args()
    settings = get_settings()
    apply_runtime_env(settings)
    if not settings.openai_api_key:
        print("OPENAI_API_KEY is not set. Add it to .env before running evaluation.")
        print("The offline checks are covered by: pytest")
        return 2

    examples = load_examples()
    service = ChatService(settings)
    use_judge = not args.no_llm_judge
    upload = not args.no_upload and bool(settings.langsmith_api_key)

    if not upload:
        if not args.no_upload:
            print("LANGSMITH_API_KEY is not set. Printing a local summary only.")
        rows = _run_local(service, examples, use_judge)
        failures = _print_summary(rows)
        return 1 if failures else 0

    from langsmith import Client, evaluate

    client = Client()
    dataset_name = "mlops-chatbot-demo-eval"
    _sync_dataset(client, dataset_name, examples)
    results = evaluate(
        target_fn(service),
        data=dataset_name,
        evaluators=_langsmith_evaluators(settings.app_timezone, service._require_model(), use_judge),
        experiment_prefix="mlops-chatbot",
        description="Compare tool use, safe errors, and answer presence for the MLOps chatbot demo.",
        max_concurrency=0,
        client=client,
        metadata={
            "application": "MLOps Chatbot Demo",
            "model_provider": "openai",
            "model_name": settings.llm_model,
        },
    )
    rows = []
    for row in results:
        scores = []
        evaluation_results = row["evaluation_results"]
        result_items = (
            evaluation_results["results"]
            if isinstance(evaluation_results, dict)
            else evaluation_results.results
        )
        for item in result_items:
            key = item["key"] if isinstance(item, dict) else item.key
            score = item["score"] if isinstance(item, dict) else item.score
            comment = item["comment"] if isinstance(item, dict) else item.comment
            scores.append({"key": key, "score": score, "comment": comment})
        question = (row["example"].inputs or {}).get("question", "")
        rows.append({"question": question, "scores": scores})
    failures = _print_summary(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
