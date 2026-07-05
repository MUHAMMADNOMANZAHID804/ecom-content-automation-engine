"""
scripts/audit_logger.py
-------------------------
Structured JSONL audit trail for every pipeline phase call. Referenced by
core/manager.py at the start/end of every phase, and by subagents.py internally.
"""

import os
import json
import time
import logging
import uuid
from typing import Any, Dict

logger = logging.getLogger("audit_logger")

LOG_DIR = os.getenv("AUDIT_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class AuditLogger:
    def __init__(self, run_id: str = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = os.path.join(LOG_DIR, f"run_{self.run_id}.jsonl")

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {
            "run_id": self.run_id,
            "event": event,
            "ts": time.time(),
            "payload": payload,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to write audit log: %s", e)
        logger.info("[%s] %s: %s", self.run_id, event,
                    json.dumps(payload, default=str)[:300])

    def read_run(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]