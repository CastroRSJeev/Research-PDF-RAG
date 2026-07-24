from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
import config


def _write(record: dict) -> None:
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_query(query: str, chunks_used: int, latency_ms: float | None = None) -> None:
    _write({
        "event": "query",
        "query_hash": _hash(query),
        "query_preview": query[:100],
        "chunks_used": chunks_used,
        "latency_ms": latency_ms,
    })


def log_guardrail(query: str, event_type: str, detail: str) -> None:
    _write({
        "event": "guardrail",
        "event_type": event_type,
        "query_hash": _hash(query),
        "query_preview": query[:100],
        "detail": detail,
    })


def log_pii(source: str, pii_type: str, count: int) -> None:
    _write({"event": "pii_detection", "source": source, "pii_type": pii_type, "count": count})


def log_ingestion(source: str, chunks_created: int, chunks_new: int) -> None:
    _write({
        "event": "ingestion",
        "source": source,
        "chunks_created": chunks_created,
        "chunks_new": chunks_new,
    })
