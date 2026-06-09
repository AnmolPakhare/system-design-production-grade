"""Structured JSON logging.

Emitting logs as JSON lines makes them ingestible by log aggregators
(Cloud Logging, ELK, Loki) and lets us attach a per-request ``request_id`` so a
single query can be traced end to end.
"""
import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any structured extras passed via logger.info(..., extra={...}).
        for key, value in getattr(record, "__dict__", {}).items():
            if key in ("request_id", "latency_ms", "route", "stage", "n_sources",
                       "model", "grounded"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
