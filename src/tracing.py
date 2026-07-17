import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import logger

TRACE_FILE = "traces.jsonl"


def _write_trace(record: dict) -> None:
    try:
        with open(TRACE_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write trace record: %s", exc)


@contextmanager
def trace_node(node_name: str, turn_id: str):
    """Time a node's execution and append one JSON line to traces.jsonl.

    Usage:
        with trace_node("router", turn_id) as trace:
            ...
            trace["route"] = route

    The yielded dict can be mutated by the caller to fill in `route`,
    `fallback_triggered`, `degraded`, and `tokens`. If the wrapped code
    raises, `error` is populated and the exception still propagates —
    tracing is observational, never a source of failure for the turn.
    """
    start = time.perf_counter()
    record = {
        "timestamp": None,
        "turn_id": turn_id,
        "node": node_name,
        "latency_ms": None,
        "route": None,
        "fallback_triggered": False,
        "degraded": False,
        "error": None,
        "tokens": None,
    }
    try:
        yield record
    except Exception as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        _write_trace(record)


def usage_from_message(message) -> dict | None:
    """Extract {input, output, total} token counts from an AIMessage, if present."""
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    return {
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
        "total": usage.get("total_tokens"),
    }
