import json
from datetime import datetime
from zoneinfo import ZoneInfo

from app.datetime_tool import build_datetime_tool


def test_tool_name_and_schema():
    tool = build_datetime_tool("Asia/Kolkata")
    assert tool.name == "get_current_datetime"
    schema = tool.args_schema.model_json_schema()
    assert "unused" not in schema.get("required", [])


def test_tool_returns_live_clock():
    tool = build_datetime_tool("Asia/Kolkata")
    before = datetime.now(ZoneInfo("Asia/Kolkata"))
    payload = json.loads(tool.invoke({}))
    after = datetime.now(ZoneInfo("Asia/Kolkata"))

    assert payload["timezone"] == "Asia/Kolkata"
    assert payload["date"] == before.date().isoformat()
    parsed = datetime.fromisoformat(payload["iso8601"])
    assert before.replace(microsecond=0) <= parsed <= after
    assert payload["time"] == parsed.strftime("%H:%M:%S")
    assert parsed.tzinfo is not None
