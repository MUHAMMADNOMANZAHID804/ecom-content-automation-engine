"""
tools/file_mcp.py
--------------------
File-system MCP tool. Every read/write the pipeline needs (Jungle Scout CSV,
uploaded reports, generated PDFs) goes through here so paths are validated
against an allow-listed root — prevents path traversal from user-supplied
filenames reaching arbitrary disk locations.
"""

import os
import logging
from typing import List

logger = logging.getLogger("file_mcp")

ALLOWED_ROOTS = [
    os.getenv("UPLOADS_DIR", "uploads"),
    os.getenv("REPORTS_DIR", "generated_reports"),
    os.getenv("TMP_DIR", "/tmp"),
]


class FileMCPError(Exception):
    pass


class FileMCP:
    def __init__(self, allowed_roots: List[str] = None):
        self.allowed_roots = [os.path.abspath(r) for r in (allowed_roots or ALLOWED_ROOTS)]
        for root in self.allowed_roots:
            os.makedirs(root, exist_ok=True)

    def _validate(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        if not any(abs_path.startswith(root) for root in self.allowed_roots):
            raise FileMCPError(
                f"Path '{path}' is outside allow-listed directories: {self.allowed_roots}"
            )
        return abs_path

    def read_bytes(self, path: str) -> bytes:
        abs_path = self._validate(path)
        with open(abs_path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> str:
        abs_path = self._validate(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(data)
        logger.info("Wrote %s bytes to %s", len(data), abs_path)
        return abs_path

    def list_dir(self, root: str) -> List[str]:
        abs_root = self._validate(root)
        return [os.path.join(abs_root, f) for f in os.listdir(abs_root)]

    def delete(self, path: str) -> None:
        abs_path = self._validate(path)
        if os.path.exists(abs_path):
            os.remove(abs_path)
            logger.info("Deleted %s", abs_path)