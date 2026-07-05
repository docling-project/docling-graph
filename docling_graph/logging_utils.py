"""Unified logging for the docling-graph pipeline.

Every pipeline component logs through one handler with one format:

    HH:MM:SS LEVEL    [Component] message

- ``get_component_logger("DenseExtraction")`` returns a logger whose records
  render with the ``[DenseExtraction]`` prefix.
- ``batch_tag(i, n)`` builds the standard ``[Batch 14/50]`` context prefix for
  batch-scoped warnings (truncations, splits, retries).
- ``ProgressTracker`` emits granular progress lines (index, %, rate, elapsed,
  ETA) for long batch jobs. It logs plain lines instead of a live bar, so
  progress interleaves cleanly with warnings from any worker thread.
- ``configure_logging()`` installs the handler once (idempotent); third-party
  records pass through the same formatter, tagged with their package name.

Output is colorized with ANSI codes only when the target stream is an
interactive terminal (and ``NO_COLOR`` is unset), so redirected output and
log files stay plain text. ``FORCE_COLOR`` overrides the TTY check.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from typing import Any, MutableMapping, TextIO

PACKAGE_LOGGER_NAME = "docling_graph"

_DATE_FORMAT = "%H:%M:%S"

# ANSI styles. The timestamp stays uncolored; level/tag/message are styled per
# severity: INFO keeps a neutral message with a bright tag so component names
# are scannable, WARNING is yellow throughout, ERROR/CRITICAL red throughout.
_RESET = "\x1b[0m"
_CYAN = "\x1b[36m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_BOLD_RED = "\x1b[1;31m"
_BRIGHT_BLUE = "\x1b[94m"
_PURPLE = "\x1b[35m"

# (level_color, tag_color, message_color); None = terminal default.
_LEVEL_STYLES: dict[int, tuple[str, str, str | None]] = {
    logging.DEBUG: (_CYAN, _CYAN, None),
    logging.INFO: (_GREEN, _BRIGHT_BLUE, None),
    logging.WARNING: (_YELLOW, _YELLOW, _YELLOW),
    logging.ERROR: (_RED, _RED, _RED),
    logging.CRITICAL: (_BOLD_RED, _BOLD_RED, _BOLD_RED),
}

# Metrics highlighted inside otherwise-neutral (INFO/DEBUG) messages:
# `key=value` values, `14/50` counts, percentages, and MM:SS durations.
_METRIC_RE = re.compile(
    r"(?<==)[^\s,()]+"  # value after key=
    r"|\b\d+/\d+\b"  # 14/50 progress counts
    r"|\b\d+(?:\.\d+)?%"  # percentages
    r"|\b\d+:\d{2}(?::\d{2})?\b"  # 00:16 / 1:02:03 durations
)


def _supports_color(stream: TextIO | None) -> bool:
    """ANSI colors only for interactive terminals; never for files/pipes.

    Honors the ``NO_COLOR`` (https://no-color.org) and ``FORCE_COLOR``
    conventions, and treats ``TERM=dumb`` as color-incapable.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return stream is not None and stream.isatty()
    except (AttributeError, ValueError):
        return False


class _PipelineFormatter(logging.Formatter):
    """Uniform ``HH:MM:SS LEVEL [Component] message`` line, optionally colored.

    The line is assembled manually (instead of a format string) so the level
    column is padded *before* color codes are added — ANSI escapes would
    otherwise break the ``%-8s`` alignment.
    """

    def __init__(self, *, color: bool) -> None:
        super().__init__(datefmt=_DATE_FORMAT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = f"{record.levelname:<8}"
        tag = f"[{getattr(record, 'component', record.name)}]"
        message = record.getMessage()

        if self._color:
            level_color, tag_color, msg_color = _LEVEL_STYLES.get(
                record.levelno, _LEVEL_STYLES[logging.INFO]
            )
            level = f"{level_color}{level}{_RESET}"
            tag = f"{tag_color}{tag}{_RESET}"
            if msg_color:
                message = f"{msg_color}{message}{_RESET}"
            else:
                # Neutral message: highlight metrics (counts, %, durations,
                # key=value) so progress lines scan quickly.
                message = _METRIC_RE.sub(lambda m: f"{_PURPLE}{m.group(0)}{_RESET}", message)

        line = f"{timestamp} {level} {tag} {message}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            line = f"{line}\n{record.exc_text}"
        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"
        return line


class _ComponentDefaulter(logging.Filter):
    """Guarantee every record has a ``component`` for the formatter.

    Records from :func:`get_component_logger` carry their explicit component;
    everything else (plain module loggers, third-party libraries) falls back
    to a name derived from the logger path.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "component"):
            name = record.name or "root"
            if name.startswith(PACKAGE_LOGGER_NAME):
                record.component = name.rsplit(".", 1)[-1]
            else:
                record.component = name.split(".", 1)[0]
        return True


def configure_logging(*, verbose: bool = False) -> None:
    """Install the uniform pipeline handler on the root logger (idempotent).

    Rooting the handler means third-party warnings (docling, transformers,
    httpx, ...) share the exact same format as pipeline output. The
    ``docling_graph`` tree emits INFO+ (DEBUG+ when ``verbose``); everything
    else stays at WARNING+ so progress lines are never drowned out. Colors are
    enabled only when the stream is an interactive terminal.
    """
    root = logging.getLogger()
    handler = next((h for h in root.handlers if getattr(h, "_docling_graph_handler", False)), None)
    if handler is None:
        stream = sys.stderr
        handler = logging.StreamHandler(stream)
        handler._docling_graph_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(_PipelineFormatter(color=_supports_color(stream)))
        handler.addFilter(_ComponentDefaulter())
        root.addHandler(handler)
    if root.level in (logging.NOTSET,) or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    logging.getLogger(PACKAGE_LOGGER_NAME).setLevel(logging.DEBUG if verbose else logging.INFO)


def ensure_logging() -> None:
    """Configure logging only when the host application has not.

    Called by pipeline entry points so programmatic runs still produce
    readable progress output without clobbering an existing logging setup.
    """
    if not logging.getLogger().handlers:
        configure_logging()


class _ComponentAdapter(logging.LoggerAdapter):
    """LoggerAdapter that stamps every record with a fixed component name.

    Per-call ``extra`` still wins on key collisions, so a helper can override
    the component for a single message (e.g. contract-specific prefixes).
    """

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        merged = dict(self.extra or {})
        merged.update(kwargs.get("extra") or {})
        kwargs["extra"] = merged
        return msg, kwargs


def get_component_logger(component: str, name: str | None = None) -> logging.LoggerAdapter:
    """Logger whose records render as ``[component]`` in the uniform format.

    ``name`` positions the logger in the hierarchy (pass ``__name__`` so
    per-module level control and ``caplog`` filtering keep working); it
    defaults to ``docling_graph.<component>``.
    """
    base = logging.getLogger(name or f"{PACKAGE_LOGGER_NAME}.{component}")
    return _ComponentAdapter(base, {"component": component})


def batch_tag(index: int, total: int) -> str:
    """Standard ``[Batch i/total]`` prefix (1-based) for batch-scoped messages."""
    return f"[Batch {index + 1}/{total}]"


def _pluralize(unit: str, count: int) -> str:
    if count == 1:
        return unit
    return unit + ("es" if unit.endswith(("s", "sh", "ch", "x", "z")) else "s")


def _fmt_duration(seconds: float) -> str:
    """``MM:SS`` under an hour, ``H:MM:SS`` above."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressTracker:
    """Granular, thread-safe progress logging for long batch jobs.

    Emits plain log lines so progress interleaves cleanly with warnings from
    worker threads::

        Phase 1 (skeleton): 14/50 batches (28%) | 2.31 batches/s | elapsed 00:06 | ETA 00:16

    Updates are rate-limited to ``log_every_seconds``, except the first and
    final updates which always log.
    """

    def __init__(
        self,
        logger: logging.Logger | logging.LoggerAdapter,
        *,
        total: int,
        label: str,
        unit: str = "batch",
        log_every_seconds: float = 5.0,
    ) -> None:
        self._logger = logger
        self._total = max(0, int(total))
        self._label = label
        self._unit = unit
        self._interval = log_every_seconds
        self._done = 0
        self._started = time.perf_counter()
        self._last_emit = 0.0
        self._lock = threading.Lock()

    def advance(self, count: int = 1, note: str | None = None) -> None:
        """Mark ``count`` more units complete; log when due (always on first/last)."""
        now = time.perf_counter()
        with self._lock:
            self._done += count
            done = self._done
            due = (
                self._last_emit == 0.0
                or done >= self._total
                or (now - self._last_emit) >= self._interval
            )
            if due:
                self._last_emit = now
        if due:
            self._emit(done, now, note)

    def _emit(self, done: int, now: float, note: str | None) -> None:
        elapsed = now - self._started
        rate = done / elapsed if elapsed > 0 else 0.0
        units = _pluralize(self._unit, self._total)
        if self._total:
            head = (
                f"{self._label}: {done}/{self._total} {units} ({100.0 * done / self._total:.0f}%)"
            )
        else:
            head = f"{self._label}: {done} {units}"
        parts = [head, f"{rate:.2f} {self._unit}/s", f"elapsed {_fmt_duration(elapsed)}"]
        if rate > 0 and self._total and done < self._total:
            parts.append(f"ETA {_fmt_duration((self._total - done) / rate)}")
        if note:
            parts.append(note)
        self._logger.info(" | ".join(parts))

    def finish(self, note: str | None = None) -> None:
        """Log the completion summary (total processed, wall time, average rate)."""
        elapsed = time.perf_counter() - self._started
        with self._lock:
            done = self._done
        rate = done / elapsed if elapsed > 0 else 0.0
        units = _pluralize(self._unit, done)
        msg = (
            f"{self._label}: completed {done}/{self._total} {units} "
            f"in {_fmt_duration(elapsed)} ({rate:.2f} {self._unit}/s)"
        )
        if note:
            msg += f" | {note}"
        self._logger.info(msg)
