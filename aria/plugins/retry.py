"""
ARIA LLM retry policy
---------------------
Retries failed LLM calls with exponential backoff and gives every failure a clear,
vendor-neutral error code.

Standard library only. The vendor SDKs are never imported: errors are classified by
looking at status codes, ``retry_after`` values and exception class names, so this module
works with the real SDKs, with fakes in tests and with any custom plugin.

    policy = RetryPolicy()                       # 3 retries, 0.5s base, 30s cap
    text = await call_with_retry(lambda: client.call(), policy)
    print(policy.stats())                        # counts, per-code failures, last error

Rules:
    * RATE_LIMIT, SERVER, TIMEOUT, TRANSPORT and EMPTY are retried.
    * AUTH, BAD_REQUEST and UNKNOWN are never retried.
    * ``asyncio.CancelledError`` is never swallowed or retried, also not while waiting.
    * Every attempt is recorded (bounded list) and counted (unbounded counters), so a
      failure is never hidden from the audit trail or the health report.
"""

from __future__ import annotations

import asyncio
import enum
import errno as _errno
import inspect
import logging
import math
import random
import socket
import ssl
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

__all__ = [
    "AttemptRecord",
    "EmptyResponseError",
    "LLMError",
    "LLMErrorCode",
    "RetryPolicy",
    "call_with_retry",
    "classify",
    "get_retry_after",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─── Error codes ──────────────────────────────────────────────────────────────

class LLMErrorCode(str, enum.Enum):
    """Vendor-neutral reason why an LLM call failed."""

    RATE_LIMIT  = "rate_limit"
    SERVER      = "server"
    TIMEOUT     = "timeout"
    TRANSPORT   = "transport"
    EMPTY       = "empty"
    AUTH        = "auth"
    BAD_REQUEST = "bad_request"
    UNKNOWN     = "unknown"

    @property
    def retryable(self) -> bool:
        """True if a call that failed with this code is worth trying again."""
        return self in _RETRYABLE


_RETRYABLE = frozenset({
    LLMErrorCode.RATE_LIMIT,
    LLMErrorCode.SERVER,
    LLMErrorCode.TIMEOUT,
    LLMErrorCode.TRANSPORT,
    LLMErrorCode.EMPTY,
})


class EmptyResponseError(Exception):
    """Raised by a plugin when the provider answered but gave no usable text or content."""


class LLMError(Exception):
    """
    An LLM call failed for good: retries ran out, or the error is not retryable.

    Attributes:
        code:      the LLMErrorCode of the last failure
        attempts:  how many calls were made in total
        cause:     the original exception (also chained as ``__cause__``)
        history:   the attempt records of this one call, oldest first

    When the original exception was a normal Exception subclass (for example
    ``ConnectionError``), the raised error is also an instance of that class. Code that
    already catches the original type keeps working.
    """

    def __init__(
        self,
        message: str,
        *,
        code: LLMErrorCode = LLMErrorCode.UNKNOWN,
        attempts: int = 1,
        cause: Optional[BaseException] = None,
        history: tuple[AttemptRecord, ...] = (),
    ):
        super().__init__(message)
        self.message  = message
        self.code     = code
        self.attempts = attempts
        self.cause    = cause
        self.history  = history

    def __str__(self) -> str:
        return self.message


_COMPAT_CLASSES: dict[type, type] = {}


def _compat_class(cause_type: type) -> type:
    """A subclass of both LLMError and the original exception type (cached)."""
    cls = _COMPAT_CLASSES.get(cause_type)
    if cls is None:
        cls = type("LLMError", (LLMError, cause_type), {"__module__": __name__})
        _COMPAT_CLASSES[cause_type] = cls
    return cls


def _build_error(
    message: str,
    code: LLMErrorCode,
    attempts: int,
    cause: BaseException,
    history: tuple[AttemptRecord, ...],
) -> LLMError:
    cause_type = type(cause)
    if isinstance(cause, Exception) and not isinstance(cause, LLMError):
        try:
            err = _compat_class(cause_type)(
                message, code=code, attempts=attempts, cause=cause, history=history,
            )
            if isinstance(err, LLMError):
                return err
        except Exception:  # odd exception layouts cannot be combined; fall back
            _COMPAT_CLASSES.pop(cause_type, None)
    return LLMError(message, code=code, attempts=attempts, cause=cause, history=history)


# ─── Classification ───────────────────────────────────────────────────────────

# Ordered: the first rule with a matching fragment wins. "timeout" must come before
# "connection" because APITimeoutError is a subclass of APIConnectionError in the SDKs.
_NAME_RULES: tuple[tuple[LLMErrorCode, tuple[str, ...]], ...] = (
    (LLMErrorCode.RATE_LIMIT,  ("ratelimit", "toomanyrequests")),
    (LLMErrorCode.AUTH,        ("authentication", "permissiondenied", "unauthorized",
                                "forbidden")),
    (LLMErrorCode.BAD_REQUEST, ("badrequest", "invalidrequest", "unprocessable")),
    (LLMErrorCode.SERVER,      ("internalserver", "servererror", "serviceunavailable",
                                "overloaded", "badgateway", "gatewaytimeout")),
    (LLMErrorCode.TIMEOUT,     ("timeout", "timedout")),
    (LLMErrorCode.TRANSPORT,   ("connection", "connecterror", "networkerror",
                                "remoteprotocol", "readerror", "writeerror",
                                "closedresource")),
)

# Names that must match exactly: as a fragment "notfounderror" would also catch the builtin
# FileNotFoundError, which has nothing to do with the provider.
_EXACT_NAMES: dict[str, LLMErrorCode] = {
    "notfounderror": LLMErrorCode.BAD_REQUEST,
}

_NETWORK_ERRNOS = frozenset(
    getattr(_errno, name) for name in (
        "ECONNRESET", "ECONNREFUSED", "ECONNABORTED", "EPIPE", "EHOSTUNREACH",
        "EHOSTDOWN", "ENETUNREACH", "ENETDOWN", "ENETRESET", "EADDRNOTAVAIL",
    ) if hasattr(_errno, name)
)


def _status_to_code(status: int) -> Optional[LLMErrorCode]:
    if status == 429:
        return LLMErrorCode.RATE_LIMIT
    if status == 408:
        return LLMErrorCode.TIMEOUT
    if status in (401, 403):
        return LLMErrorCode.AUTH
    if 500 <= status <= 599:  # includes 529 (overloaded)
        return LLMErrorCode.SERVER
    if 400 <= status <= 499:
        return LLMErrorCode.BAD_REQUEST
    return None


def _as_status(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 100 <= value <= 599 else None


def _status_of(exc: BaseException) -> Optional[int]:
    """Find an HTTP status on the error or on its ``response``."""
    for attr in ("status_code", "status", "http_status"):
        status = _as_status(getattr(exc, attr, None))
        if status is not None:
            return status
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            status = _as_status(getattr(response, attr, None))
            if status is not None:
                return status
    return None


def classify(exc: BaseException) -> LLMErrorCode:
    """
    Map any exception to an LLMErrorCode without importing a vendor SDK.

    Order: our own errors, then an HTTP status, then exception class names (the whole
    class hierarchy, most specific class first), then builtin timeout and network errors.
    """
    if isinstance(exc, LLMError):
        return exc.code
    if isinstance(exc, EmptyResponseError):
        return LLMErrorCode.EMPTY

    status = _status_of(exc)
    if status is not None:
        code = _status_to_code(status)
        if code is not None:
            return code

    for klass in type(exc).__mro__:
        name = klass.__name__.lower()
        if name in _EXACT_NAMES:
            return _EXACT_NAMES[name]
        for code, fragments in _NAME_RULES:
            if any(fragment in name for fragment in fragments):
                return code

    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, socket.timeout)):
        return LLMErrorCode.TIMEOUT
    if isinstance(exc, (ConnectionError, socket.gaierror, socket.herror, ssl.SSLError)):
        return LLMErrorCode.TRANSPORT
    if isinstance(exc, OSError):
        if exc.errno == _errno.ETIMEDOUT:
            return LLMErrorCode.TIMEOUT
        if exc.errno in _NETWORK_ERRNOS:
            return LLMErrorCode.TRANSPORT
    return LLMErrorCode.UNKNOWN


def _to_seconds(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return None
    return seconds


def _header(headers: Any, name: str) -> Any:
    """Case-insensitive header lookup that works on dicts and httpx-style objects."""
    if headers is None:
        return None
    try:
        value = headers.get(name)
        if value is not None:
            return value
        for key, val in headers.items():
            if str(key).lower() == name:
                return val
    except Exception:
        return None
    return None


def get_retry_after(exc: BaseException) -> Optional[float]:
    """
    Seconds the server asked us to wait, or None.

    Looks at a ``retry_after`` attribute, then the ``retry-after-ms`` and ``retry-after``
    headers on ``response.headers`` (or ``headers``). Only numeric values are understood;
    an HTTP date or a negative number is ignored.
    """
    seconds = _to_seconds(getattr(exc, "retry_after", None))
    if seconds is not None:
        return seconds
    response = getattr(exc, "response", None)
    for headers in (getattr(response, "headers", None), getattr(exc, "headers", None)):
        millis = _to_seconds(_header(headers, "retry-after-ms"))
        if millis is not None:
            return millis / 1000.0
        seconds = _to_seconds(_header(headers, "retry-after"))
        if seconds is not None:
            return seconds
    return None


# ─── Records ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttemptRecord:
    """
    One attempt (or the end of one call).

    outcome is one of:
        "success"        the call worked
        "retry"          failed, will wait ``delay`` seconds and try again
        "exhausted"      failed and no retries are left (final)
        "not_retryable"  failed with an error that is never retried (final)
        "cancelled"      the caller cancelled while waiting to retry (final)
    """

    operation:   str
    attempt:     int
    outcome:     str
    code:        Optional[LLMErrorCode]
    delay:       float
    error:       str
    timestamp:   float
    retry_after: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["code"] = self.code.value if self.code is not None else None
        return data


# ─── Policy ───────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Exponential backoff with a cap and jitter, plus the record of what happened.

    The wait before retry number n is ``min(max_delay, base_delay * multiplier ** (n-1))``
    reduced by a random share of up to ``jitter`` (0.25 means 0 to 25 percent shorter). The
    jitter only shortens the wait, so it can never go above ``max_delay``. When the error
    carries a retry-after value, that value is used instead, up to ``max_delay``.

    ``sleep``, ``clock`` and ``rng`` are injectable so tests never really wait.
    ``history`` keeps the last ``max_history`` attempt records; the counters in ``stats()``
    are not bounded.
    """

    max_retries: int   = 3
    base_delay:  float = 0.5
    max_delay:   float = 30.0
    multiplier:  float = 2.0
    jitter:      float = 0.25
    max_history: int   = 200
    sleep: Callable[[float], Awaitable[Any]] = field(default=asyncio.sleep, repr=False)
    clock: Callable[[], float]               = field(default=time.time, repr=False)
    rng:   Callable[[], float]               = field(default=random.random, repr=False)

    history: deque[AttemptRecord] = field(init=False, repr=False, compare=False)
    _counters: dict[str, int] = field(init=False, repr=False, compare=False)
    _by_code: dict[str, int] = field(init=False, repr=False, compare=False)
    _last_error: Optional[dict[str, Any]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("base_delay and max_delay must be 0 or more")
        if self.multiplier < 1:
            raise ValueError("multiplier must be 1 or more")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")
        if self.max_history < 1:
            raise ValueError("max_history must be 1 or more")
        self.history = deque(maxlen=self.max_history)
        self.reset()

    # ── timing ──

    def backoff(self, retry_number: int) -> float:
        """The wait before retry ``retry_number`` (1 is the first retry), without jitter."""
        try:
            raw = self.base_delay * (self.multiplier ** max(0, retry_number - 1))
        except OverflowError:
            raw = self.max_delay
        return min(self.max_delay, raw)

    def delay_for(self, retry_number: int, retry_after: Optional[float] = None) -> float:
        """The wait before retry ``retry_number``: retry-after if given, else jittered backoff."""
        if retry_after is not None:
            return min(self.max_delay, max(0.0, retry_after))
        delay = self.backoff(retry_number) * (1.0 - self.jitter * self.rng())
        return min(self.max_delay, max(0.0, delay))

    # ── bookkeeping ──

    def reset(self) -> None:
        """Forget all records and counters."""
        self.history.clear()
        self._counters = {
            "calls": 0, "succeeded": 0, "failed": 0, "cancelled": 0,
            "attempts": 0, "retries": 0,
        }
        self._by_code = {}
        self._last_error = None

    def _record(
        self,
        operation: str,
        attempt: int,
        outcome: str,
        code: Optional[LLMErrorCode],
        delay: float,
        error: str,
        retry_after: Optional[float] = None,
    ) -> AttemptRecord:
        record = AttemptRecord(
            operation=operation, attempt=attempt, outcome=outcome, code=code,
            delay=delay, error=error, timestamp=self.clock(), retry_after=retry_after,
        )
        self.history.append(record)
        if outcome != "cancelled":
            self._counters["attempts"] += 1
        if outcome == "retry":
            self._counters["retries"] += 1
        if code is not None and outcome != "cancelled":
            self._by_code[code.value] = self._by_code.get(code.value, 0) + 1
            self._last_error = record.to_dict()
        if outcome == "success":
            self._counters["succeeded"] += 1
        elif outcome in ("exhausted", "not_retryable"):
            self._counters["failed"] += 1
        elif outcome == "cancelled":
            self._counters["cancelled"] += 1
        return record

    def stats(self) -> dict[str, Any]:
        """Counters for the health report. Safe to log: no prompts or responses inside."""
        return {
            **self._counters,
            "failures_by_code": dict(self._by_code),
            "last_error": dict(self._last_error) if self._last_error else None,
        }

    def describe(self, recent: int = 5) -> dict[str, Any]:
        """Settings, counters and the most recent attempt records, as plain data."""
        records = list(self.history)[-recent:] if recent > 0 else []
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "multiplier": self.multiplier,
            "jitter": self.jitter,
            "retryable_codes": sorted(code.value for code in _RETRYABLE),
            "stats": self.stats(),
            "recent_attempts": [r.to_dict() for r in records],
        }


# ─── The retry loop ───────────────────────────────────────────────────────────

_MAX_ERROR_CHARS = 500


def _error_text(exc: BaseException) -> str:
    """Short error text for records. A long message is cut and says by how much."""
    text = str(exc).strip()
    if len(text) > _MAX_ERROR_CHARS:
        text = f"{text[:_MAX_ERROR_CHARS]}... [{len(text) - _MAX_ERROR_CHARS} more characters omitted]"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


async def _notify(
    on_attempt: Optional[Callable[[AttemptRecord], Any]], record: AttemptRecord
) -> None:
    if on_attempt is None:
        return
    try:
        result = on_attempt(record)
        if inspect.isawaitable(result):
            await result
    except Exception:  # a broken hook must not break (or hide) the LLM call
        logger.warning("on_attempt hook failed", exc_info=True)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: Optional[RetryPolicy] = None,
    *,
    on_attempt: Optional[Callable[[AttemptRecord], Any]] = None,
    operation: str = "llm",
) -> T:
    """
    Call ``fn()`` until it succeeds, retrying retryable failures with backoff.

    ``fn`` takes no arguments and returns an awaitable (use a lambda or a closure).
    ``on_attempt`` is called with each AttemptRecord (it may be a plain function or a
    coroutine function). Returns the result of the first successful call.

    Raises:
        LLMError: the error is not retryable, or the retries ran out. It carries ``code``,
            ``attempts`` and ``cause`` and is chained to the original exception.
        asyncio.CancelledError: always passed through, also while waiting between attempts.
    """
    pol = policy if policy is not None else RetryPolicy()
    pol._counters["calls"] += 1
    own: list[AttemptRecord] = []  # this call only; pol.history is bounded and shared
    attempt = 0

    while True:
        attempt += 1
        try:
            result = await fn()
        except asyncio.CancelledError:
            pol._record(operation, attempt, "cancelled", None, 0.0,
                        "cancelled during the call")
            raise
        except Exception as exc:
            code = classify(exc)
            text = _error_text(exc)
            retry_after = get_retry_after(exc)

            if not code.retryable or attempt > pol.max_retries:
                outcome = "exhausted" if code.retryable else "not_retryable"
                record = pol._record(operation, attempt, outcome, code, 0.0, text, retry_after)
                own.append(record)
                await _notify(on_attempt, record)
                if code.retryable:
                    message = (f"LLM call failed after {attempt} attempt(s) "
                               f"[{code.value}]: {text}")
                else:
                    message = (f"LLM call failed with an error of kind [{code.value}] that is "
                               f"not retried (attempt {attempt}): {text}")
                raise _build_error(message, code, attempt, exc, tuple(own)) from exc

            delay = pol.delay_for(attempt, retry_after)
            record = pol._record(operation, attempt, "retry", code, delay, text, retry_after)
            own.append(record)
            await _notify(on_attempt, record)
            try:
                await pol.sleep(delay)
            except asyncio.CancelledError:
                pol._record(operation, attempt, "cancelled", code, delay,
                            "cancelled while waiting to retry", retry_after)
                raise
            continue

        record = pol._record(operation, attempt, "success", None, 0.0, "")
        await _notify(on_attempt, record)
        return result
