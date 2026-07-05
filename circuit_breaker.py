"""
scripts/circuit_breaker.py
----------------------------
Simple circuit breaker around external (Groq API) calls made from
core/manager.py. Prevents hammering a failing endpoint and gives the manager
a clean failure signal instead of a raw exception storm.

States: CLOSED (normal) -> OPEN (blocking calls) -> HALF_OPEN (probe) -> CLOSED.
"""

import time
import logging
from typing import Any, Callable

logger = logging.getLogger("circuit_breaker")


class CircuitBreakerOpenError(RuntimeError):
    pass


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 4, reset_timeout_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._failures = 0
        self._state = "CLOSED"
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        if self._state == "OPEN" and (time.time() - self._opened_at) >= self.reset_timeout_s:
            self._state = "HALF_OPEN"
        return self._state

    def call(self, fn: Callable[[], Any]) -> Any:
        current_state = self.state
        if current_state == "OPEN":
            raise CircuitBreakerOpenError(
                "Circuit breaker is OPEN — too many recent failures. "
                f"Retry after {self.reset_timeout_s}s."
            )
        try:
            result = fn()
        except Exception:
            self._failures += 1
            logger.warning("Circuit breaker recorded failure %s/%s",
                            self._failures, self.failure_threshold)
            if self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.time()
                logger.error("Circuit breaker OPEN — blocking further calls for %ss",
                             self.reset_timeout_s)
            raise
        else:
            if current_state == "HALF_OPEN":
                logger.info("Circuit breaker probe succeeded — closing circuit.")
            self._failures = 0
            self._state = "CLOSED"
            return result