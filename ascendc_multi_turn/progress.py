from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from typing import Any, TextIO


class ProgressTask:
    def __init__(self, reporter: "ProgressReporter", label: str):
        self.reporter = reporter
        self.label = label
        self.started_at = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reporter.emit(f"{label} · started")
        if self.reporter.enabled and self.reporter.heartbeat_interval > 0:
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.reporter.heartbeat_interval):
            elapsed = int(time.monotonic() - self.started_at)
            self.reporter.emit(f"{self.label} · still running ({elapsed}s)")

    def finish(self, *, status: str = "completed", detail: str = "") -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        elapsed = time.monotonic() - self.started_at
        self.reporter.record_event(
            {
                "label": self.label,
                "status": status,
                "detail": detail,
                "elapsed_seconds": elapsed,
            }
        )
        suffix = f" · {detail}" if detail else ""
        self.reporter.emit(f"{self.label} · {status} ({elapsed:.1f}s){suffix}")


class ProgressReporter:
    """Thread-safe, line-oriented progress output intended for stderr."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        stream: TextIO | None = None,
        heartbeat_interval: float = 15.0,
    ):
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_interval = heartbeat_interval
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def record_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(dict(event))

    def event_cursor(self) -> int:
        with self._lock:
            return len(self._events)

    def events_since(self, cursor: int) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._events[cursor:]]

    def emit(self, message: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        with self._lock:
            print(f"[{timestamp}] {message}", file=self.stream, flush=True)

    def start(self, label: str) -> ProgressTask:
        return ProgressTask(self, label)


def token_detail(usage: dict[str, object]) -> str:
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return f"tokens={total}"
    return "tokens=unknown"
