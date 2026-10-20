"""structlog JSON setup; get_logger(component) binds the block name."""

import logging
import sys
from pathlib import Path

import structlog
from structlog.typing import FilteringBoundLogger


class _TeeLogger:
    """structlog logger target: writes each rendered line to stdout and appends to a file."""

    def __init__(self, log_file: str | None) -> None:
        self._path = Path(log_file) if log_file else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def msg(self, line: str) -> None:
        print(line, file=sys.stdout, flush=True)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    log = debug = info = warn = warning = failure = err = error = critical = exception = fatal = msg


def configure_logging(level: str, log_file: str | None) -> None:
    """One JSON object per line to stdout and, if given, appended to log_file."""
    numeric = logging.getLevelNamesMapping()[level.upper()]
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=lambda *_args: _TeeLogger(log_file),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> FilteringBoundLogger:
    """A logger with `component` pre-bound; component must equal the calling file's stem."""
    # Lazy proxy: config is resolved at the first log call, so a module-level
    # `log = get_logger("x")` created before configure_logging still emits JSON.
    logger: FilteringBoundLogger = structlog.get_logger(component=component)
    return logger


def bind_request(request_id: str) -> None:
    """Attach request_id to every subsequent line in this context."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request() -> None:
    structlog.contextvars.clear_contextvars()
