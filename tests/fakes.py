"""Shared fake chat model for unit tests."""

from langsmith.run_helpers import get_current_run_tree


class ScriptedChatModel:
    """Returns queued replies and records every invoke. `bind_tools` shares the queues."""

    def __init__(self, replies, invocations=None):
        self.replies = replies
        self.invocations = invocations if invocations is not None else []
        self.tools = None

    def bind_tools(self, tools, **kwargs):
        del kwargs
        bound = ScriptedChatModel(self.replies, self.invocations)
        bound.tools = list(tools)
        return bound

    def invoke(self, messages, config=None):
        self.invocations.append(
            {
                "messages": messages,
                "config": config,
                "tools": self.tools,
                "run": get_current_run_tree(),
            }
        )
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
