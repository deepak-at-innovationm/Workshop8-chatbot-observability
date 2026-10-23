"""In-memory chat history.

History lives in the server process. It is isolated by session id and discarded
when the process stops. This is intentional for a local demo.
"""

from threading import Lock


class SessionStore:
    """Bounded message history keyed by session id."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 2:
            raise ValueError("history_max_messages must be at least 2")
        self.max_messages = max_messages
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> list[dict[str, str]]:
        """Return a copy of one session's messages. Other sessions are not included."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Store one completed turn and drop the oldest messages past the limit."""
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.append({"role": "user", "content": user_message[:2000]})
            messages.append({"role": "assistant", "content": assistant_message[:2000]})
            overflow = len(messages) - self.max_messages
            if overflow > 0:
                del messages[:overflow]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
