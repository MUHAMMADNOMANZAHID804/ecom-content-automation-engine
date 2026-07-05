"""
tools/sandboxed_executor.py
------------------------------
Runs deterministic helper scripts (e.g. CSV parsing edge cases, one-off data
cleanup) in an isolated subprocess with a timeout and no ambient environment
leakage. This is the "Harness" referenced in your Podcast 1 notes: AI-written
code never runs directly in the main process.

NOT used for the Groq LLM calls themselves — only for executing generated or
maintained utility scripts (e.g. scripts/build_kb.py could be invoked this way
from a UI 'Rebuild KB' button instead of importing it directly).
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("sandboxed_executor")

DEFAULT_TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 20000


class SandboxExecutionError(RuntimeError):
    pass


class SandboxedExecutor:
    def __init__(self, timeout_s: int = DEFAULT_TIMEOUT_S,
                 allowed_env: Optional[Dict[str, str]] = None):
        self.timeout_s = timeout_s
        # Only pass through an explicit allow-list of env vars (e.g. GROQ_API_KEY),
        # never the full parent environment, to limit blast radius.
        self.allowed_env = allowed_env or {
            k: v for k, v in os.environ.items()
            if k in {"GROQ_API_KEY", "PATH", "PYTHONPATH"}
        }

    def run_python(self, script_path: str, args: List[str] = None) -> str:
        return self._run(["python3", script_path, *(args or [])])

    def run_bash(self, command: List[str]) -> str:
        return self._run(command)

    def _run(self, cmd: List[str]) -> str:
        logger.info("Sandboxed exec: %s (timeout=%ss)", cmd, self.timeout_s)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=self.allowed_env,
                cwd=os.getcwd(),
            )
        except subprocess.TimeoutExpired as e:
            raise SandboxExecutionError(f"Execution timed out after {self.timeout_s}s") from e

        if result.returncode != 0:
            raise SandboxExecutionError(
                f"Command exited {result.returncode}. stderr:\n{result.stderr[-2000:]}"
            )
        return result.stdout[:MAX_OUTPUT_CHARS]