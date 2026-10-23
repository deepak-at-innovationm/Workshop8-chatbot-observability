"""The demo's single LangChain tool: the server clock."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GetCurrentDateTimeInput(BaseModel):
    """Arguments for get_current_datetime. No field is required."""

    unused: str = Field(
        default="",
        description="Leave this empty. The tool reads the server clock and ignores this value.",
    )


def build_datetime_tool(timezone_name: str = "Asia/Kolkata"):
    """Build the datetime tool bound to a configured IANA timezone."""

    @tool(args_schema=GetCurrentDateTimeInput)
    def get_current_datetime(unused: str = "") -> str:
        """Return the current date and time in the server timezone.

        Call this tool only when the user asks for the current date, the current
        time, or today's date. Do not call it for general knowledge questions.
        """
        del unused  # The schema field exists so the model can call the tool with no meaningful input.
        try:
            zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Unknown timezone {timezone_name!r}. Set APP_TIMEZONE to an IANA name such as Asia/Kolkata."
            ) from exc
        now = datetime.now(zone)
        return json.dumps(
            {
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "timezone": timezone_name,
                "iso8601": now.isoformat(),
            }
        )

    return get_current_datetime
