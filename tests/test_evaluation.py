from langchain_core.messages import AIMessage

from app.chatbot import PUBLIC_ERROR_MESSAGE
from app.evaluation import (
    deterministic_scores,
    judge_relevance,
    load_examples,
    score_tool_selection,
)


class JudgeModel:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, messages, config=None):
        del messages, config
        self.calls += 1
        return AIMessage(content=self.content)


def test_dataset_covers_the_demo_cases():
    examples = load_examples()
    assert 5 <= len(examples) <= 10
    categories = [example["outputs"]["category"] for example in examples]
    assert categories.count("general") >= 2
    assert categories.count("datetime") >= 2
    assert "error" in categories
    assert any(example["outputs"]["expect_tool"] for example in examples)
    assert any(not example["outputs"]["expect_tool"] and example["outputs"]["category"] == "general" for example in examples)


def test_tool_selection_uses_the_invocation_flag_not_the_wording():
    sounded_like_a_clock = score_tool_selection(
        {"answer": "It is 10:00 in Asia/Kolkata.", "tool_used": False, "tool_name": None, "error": None},
        {"expect_tool": True, "category": "datetime"},
    )
    actually_called = score_tool_selection(
        {"answer": "Here is the clock.", "tool_used": True, "tool_name": "get_current_datetime", "error": None},
        {"expect_tool": True, "category": "datetime"},
    )
    assert sounded_like_a_clock["score"] == 0
    assert actually_called["score"] == 1


def test_general_question_fails_when_the_tool_was_used():
    score = score_tool_selection(
        {"answer": "MLOps is useful.", "tool_used": True, "tool_name": "get_current_datetime", "error": None},
        {"expect_tool": False, "category": "general"},
    )
    assert score["score"] == 0


def test_error_example_requires_the_safe_message():
    scores = {
        item["key"]: item
        for item in deterministic_scores(
            {"answer": "", "tool_used": False, "tool_name": None, "error": PUBLIC_ERROR_MESSAGE},
            {"expect_tool": False, "expect_error": True, "category": "error"},
        )
    }
    assert scores["error_handling"]["score"] == 1
    assert scores["tool_selection"]["score"] == 1

    leaked = score_tool_selection(
        {"answer": "", "tool_used": False, "tool_name": None, "error": "Traceback: secret"},
        {"expect_tool": False},
    )
    assert leaked["score"] == 1  # tool selection does not look at the error text
    from app.evaluation import score_error_handling

    assert score_error_handling(
        {"error": "Traceback: secret API_KEY"},
        {"expect_error": True},
    )["score"] == 0


def test_relevance_judge_parses_model_json():
    model = JudgeModel('{"relevant": true, "reason": "The answer describes MLOps."}')
    score = judge_relevance(model, "What is MLOps?", "It is the practice of operating ML systems.", "Mentions operations.")
    assert score["score"] == 1
    assert model.calls == 1

    negative = JudgeModel("```json\n{\"relevant\": false, \"reason\": \"Off topic.\"}\n```")
    assert judge_relevance(negative, "What is MLOps?", "The time is noon.", "Mentions operations.")["score"] == 0
