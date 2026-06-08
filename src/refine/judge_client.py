"""Async dual token-bucket rate limiter (QPS + TPM).

Resurrected verbatim from the deleted ``src/labeling/ratelimit.py``
(commit ``a22ff72^``). Used by :mod:`src.refine.judge` to fan out
~500 OpenAI-compatible chat completions per Phase 3 turn while
respecting the endpoint's per-second and per-minute budgets.

Two independent limits enforce simultaneously:
- QPS  (calls per second)    — deducts 1 per ``acquire`` call
- TPM  (tokens per minute)   — deducts ``estimated_tokens`` per call

Both buckets refill continuously at their respective rates.
``acquire`` sleeps (via ``asyncio.sleep``) until both budgets allow
the call to proceed. An ``asyncio.Lock`` serialises bucket state
across concurrent coroutines.

Typical usage::

    limiter = AsyncRateLimiter(qps_limit=8.0, tpm_limit=200_000.0)
    estimated = int(len(chunk_content) / 1.5 * 1.3) + 400
    await limiter.acquire(estimated)
    response = await client.chat.completions.create(...)
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Dual token-bucket rate limiter for async code."""

    def __init__(self, *, qps_limit: float, tpm_limit: float) -> None:
        self._qps_limit = qps_limit
        self._tpm_limit = tpm_limit
        self._tps_rate = tpm_limit / 60.0

        self._qps_bucket = qps_limit
        self._tpm_bucket = tpm_limit
        self._last_refill = time.monotonic()

        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        self._qps_bucket = min(
            self._qps_limit,
            self._qps_bucket + self._qps_limit * elapsed,
        )
        self._tpm_bucket = min(
            self._tpm_limit,
            self._tpm_bucket + self._tps_rate * elapsed,
        )

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until both QPS and TPM buckets have capacity.

        Deducts 1 from the QPS bucket and ``estimated_tokens`` from
        the TPM bucket. Releases the lock while sleeping so other
        coroutines can check the buckets in the interim.
        """
        while True:
            async with self._lock:
                self._refill()

                qps_wait = 0.0
                if self._qps_bucket < 1.0:
                    qps_wait = (1.0 - self._qps_bucket) / self._qps_limit

                tpm_wait = 0.0
                if self._tpm_bucket < estimated_tokens:
                    deficit = estimated_tokens - self._tpm_bucket
                    tpm_wait = deficit / self._tps_rate

                wait = max(qps_wait, tpm_wait)
                if wait <= 0:
                    self._qps_bucket -= 1.0
                    self._tpm_bucket -= estimated_tokens
                    return

            await asyncio.sleep(wait)

    @property
    def stats(self) -> dict[str, float]:
        """Return current bucket fill levels for observability."""
        return {
            "qps_bucket": self._qps_bucket,
            "tpm_bucket": self._tpm_bucket,
            "qps_limit": self._qps_limit,
            "tpm_limit": self._tpm_limit,
        }
