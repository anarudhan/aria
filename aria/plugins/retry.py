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
import re
import socket
import ssl
import time
import unicodedata
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
    "sanitize_error_text",
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
        code:       the LLMErrorCode of the last failure
        attempts:   how many calls were made in total
        cause:      the original exception (also chained as ``__cause__``). It is the
                    untouched original, so it holds the ORIGINAL, unredacted text.
        cause_type: the class name of the cause, as text
        cause_text: the cause's message, redacted and cut (see sanitize_error_text)
        history:    the attempt records of this one call, oldest first

    It is a plain exception (not a subclass of the original one), so catch ``LLMError`` and
    look at ``.code`` (and ``.cause`` if you need the original) to decide what to do.

    Pickling keeps ``code``, ``attempts``, the message, ``cause_type``, ``cause_text`` and
    ``history``, and DROPS ``cause`` (it becomes None). Real SDK errors often cannot be
    unpickled, and pickling them would copy their unredacted text. ``copy.copy`` and
    ``copy.deepcopy`` keep the same ``cause`` object.
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
        self.cause_type = "" if cause is None else _type_name(cause)
        self.cause_text = "" if cause is None else sanitize_error_text(_safe_str(cause))

    def __str__(self) -> str:
        return self.message

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild_llm_error,
            (self.message, self.code.value, self.attempts, self.cause_type,
             self.cause_text, tuple(self.history)),
        )

    def __copy__(self) -> "LLMError":
        clone = LLMError(self.message, code=self.code, attempts=self.attempts,
                         cause=self.cause, history=tuple(self.history))
        clone.cause_type, clone.cause_text = self.cause_type, self.cause_text
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> "LLMError":
        return self.__copy__()


def _rebuild_llm_error(
    message: str,
    code_value: str,
    attempts: int,
    cause_type: str,
    cause_text: str,
    history: tuple[AttemptRecord, ...],
) -> LLMError:
    err = LLMError(message, code=LLMErrorCode(code_value), attempts=attempts,
                   cause=None, history=history)
    err.cause_type, err.cause_text = cause_type, cause_text
    return err


def _build_error(
    message: str,
    code: LLMErrorCode,
    attempts: int,
    cause: BaseException,
    history: tuple[AttemptRecord, ...],
) -> LLMError:
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
                                "closedresource", "proxyerror")),
)

# Names that must match exactly: as a fragment "notfounderror" would also catch the builtin
# FileNotFoundError, which has nothing to do with the provider.
_EXACT_NAMES: dict[str, LLMErrorCode] = {
    "notfounderror": LLMErrorCode.BAD_REQUEST,
}

# Plain programming errors are never classified by their class NAME (a "TimeoutConfigError"
# that is a ValueError is a bug in the caller, not a provider timeout). A mapped HTTP status
# on them is still honoured, because the status check runs first.
_PROGRAMMING_ERRORS = (
    ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError, LookupError,
)

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
    if status == 409:  # conflict: both vendor SDKs retry it; SERVER is the closest code
        return LLMErrorCode.SERVER
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


def _safe_getattr(obj: Any, name: str) -> Any:
    """getattr that never raises: a property that blows up simply reads as None."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _status_of(exc: BaseException) -> Optional[int]:
    """Find an HTTP status on the error or on its ``response``."""
    for attr in ("status_code", "status", "http_status"):
        status = _as_status(_safe_getattr(exc, attr))
        if status is not None:
            return status
    response = _safe_getattr(exc, "response")
    if response is not None:
        for attr in ("status_code", "status"):
            status = _as_status(_safe_getattr(response, attr))
            if status is not None:
                return status
    return None


def classify(exc: BaseException) -> LLMErrorCode:
    """
    Map any exception to an LLMErrorCode without importing a vendor SDK. Never raises.

    Order: our own errors, then an HTTP status, then exception class names (the whole
    class hierarchy, most specific class first; skipped for plain programming errors such
    as ValueError), then builtin timeout, TLS and network
    errors.
    """
    try:
        return _classify(exc)
    except Exception:
        return LLMErrorCode.UNKNOWN


def _classify(exc: BaseException) -> LLMErrorCode:
    if isinstance(exc, LLMError):
        return exc.code
    if isinstance(exc, EmptyResponseError):
        return LLMErrorCode.EMPTY

    status = _status_of(exc)
    if status is not None:
        code = _status_to_code(status)
        if code is not None:
            return code

    if not isinstance(exc, _PROGRAMMING_ERRORS):
        for klass in type(exc).__mro__:
            name = klass.__name__.lower()
            if name in _EXACT_NAMES:
                return _EXACT_NAMES[name]
            for code, fragments in _NAME_RULES:
                if any(fragment in name for fragment in fragments):
                    return code

    if isinstance(exc, ssl.SSLCertVerificationError):
        return LLMErrorCode.AUTH   # a bad certificate will not fix itself
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
    except (TypeError, ValueError, OverflowError):
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
    seconds = _to_seconds(_safe_getattr(exc, "retry_after"))
    if seconds is not None:
        return seconds
    response = _safe_getattr(exc, "response")
    for headers in (_safe_getattr(response, "headers"), _safe_getattr(exc, "headers")):
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
        "cancelled"      the caller cancelled, during the call or the wait (final)
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


MAX_RETRIES_LIMIT = 100
MAX_HISTORY_LIMIT = 100_000


def _check_int(name: str, value: Any, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum or value > maximum:
        shown = repr(value) if abs(value) < 10 ** 12 else "a very large number"
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {shown}")


def _check_number(
    name: str, value: Any, *, minimum: float, maximum: Optional[float] = None
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:  # an int too big for a float
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum} or more" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}, got {value!r}")


# ─── Policy ───────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Exponential backoff with a cap and jitter, plus the record of what happened.

    The wait before retry number n is ``min(max_delay, base_delay * multiplier ** (n-1))``
    reduced by a random share of up to ``jitter`` (0.25 means 0 to 25 percent shorter). The
    jitter only shortens the wait, so it can never go above ``max_delay``. When the error
    carries a retry-after value, that value is used instead, up to ``max_delay``.

    ``sleep``, ``clock`` and ``rng`` are injectable so tests never really wait. Leave
    ``sleep`` as None to use ``asyncio.sleep``, looked up at the moment of waiting (so
    monkeypatching ``asyncio.sleep`` works).
    ``history`` keeps the last ``max_history`` attempt records; the counters in ``stats()``
    are not bounded.
    """

    max_retries: int   = 3
    base_delay:  float = 0.5
    max_delay:   float = 30.0
    multiplier:  float = 2.0
    jitter:      float = 0.25
    max_history: int   = 200
    sleep: Optional[Callable[[float], Awaitable[Any]]] = field(default=None, repr=False)
    clock: Callable[[], float]               = field(default=time.time, repr=False)
    rng:   Callable[[], float]               = field(default=random.random, repr=False)
    # clock and rng may also be passed as None, which means the default.

    history: deque[AttemptRecord] = field(init=False, repr=False, compare=False)
    _counters: dict[str, int] = field(init=False, repr=False, compare=False)
    _by_code: dict[str, int] = field(init=False, repr=False, compare=False)
    _last_error: Optional[dict[str, Any]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check_int("max_retries", self.max_retries, minimum=0, maximum=MAX_RETRIES_LIMIT)
        _check_number("base_delay", self.base_delay, minimum=0)
        _check_number("max_delay", self.max_delay, minimum=0)
        _check_number("multiplier", self.multiplier, minimum=1)
        _check_number("jitter", self.jitter, minimum=0, maximum=1)
        _check_int("max_history", self.max_history, minimum=1, maximum=MAX_HISTORY_LIMIT)
        if self.sleep is not None and not callable(self.sleep):
            raise TypeError(f"sleep must be callable or None, got {self.sleep!r}")
        for name, default in (("clock", time.time), ("rng", random.random)):
            value = getattr(self, name)
            if value is None:
                setattr(self, name, default)
            elif not callable(value):
                raise TypeError(f"{name} must be callable or None, got {value!r}")
        self.history = deque(maxlen=self.max_history)
        self.reset()

    async def _sleep(self, seconds: float) -> None:
        """Wait ``seconds`` with the injected sleep, or asyncio.sleep resolved right now."""
        sleeper = self.sleep if self.sleep is not None else asyncio.sleep
        await sleeper(seconds)

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
        *,
        extra: bool = False,
    ) -> AttemptRecord:
        # extra=True marks a follow-up record for an attempt that was already counted
        # (for example "the wait itself failed"), so counters are not incremented twice.
        record = AttemptRecord(
            operation=operation, attempt=attempt, outcome=outcome, code=code,
            delay=delay, error=error, timestamp=self.clock(), retry_after=retry_after,
        )
        self.history.append(record)
        if outcome != "cancelled" and not extra:
            self._counters["attempts"] += 1
        if outcome == "retry":
            self._counters["retries"] += 1
        if code is not None and outcome != "cancelled":
            if not extra:
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
        """
        Counters for the health report.

        ``last_error`` holds the redacted, truncated error text (see ``sanitize_error_text``).
        Prompts and responses are never stored by this module, but error text written by the
        provider can quote parts of a request, so treat it as potentially sensitive.
        """
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


# ─── Error text safety ────────────────────────────────────────────────────────

MAX_ERROR_TEXT = 500

_REDACTED = "[REDACTED]"

# Building blocks. Every repetition is bounded or applies to a character class that cannot
# overlap with the next part, and no pattern starts with an open-ended prefix, so matching
# stays linear in the input size (see the 100k-character timing test).
_S = r"(?:[-_]|%5[Ff]|%2[Dd])?"                      # optional "_" or "-" inside a label
_SEP = r"""(?:\\?["'])?\s{0,5}(?:[:=]|%3[AaDd])\s{0,5}"""   # [closing quote] : or = [spaces]
_QUOTED = r"""\\?"[^"\\]{0,1000}\\?"|\\?'[^'\\]{0,1000}\\?'"""   # also JSON-escaped \"..\"
_BARE_CHARS = r"""[^\s,;'"&}\])\\]"""
_BARE = rf"""\\?["']?{_BARE_CHARS}+"""                # also an unterminated opening quote
_NOT_DONE = r"(?!\s*\[REDACTED\])"                      # never re-redact our own marker
_VALUE = rf"{_NOT_DONE}(?:{_QUOTED}|{_BARE})"


def _label(words: str) -> "re.Pattern[str]":
    return re.compile(rf"(?i)((?:{words}){_SEP})" + _VALUE)


# Applied in order after NFKC normalisation. Each keeps the label and hides only the value.
_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    # scheme://user:password@host  -> hide the password (proxy and connection errors print these)
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{0,30}://[^\s/:@?#]{1,200}:)[^\s/@?#]{1,500}(?=@)"),
     r"\1" + _REDACTED),
    # Authorization: Digest username="x", nonce="y", response="z"  -> the whole header value
    (re.compile(r"(?i)(authorization" + _SEP + r")digest\b[^\n\r]{0,2000}"),
     r"\1" + _REDACTED),
    # Authorization: <any scheme> <value>  (one or two words, so the value is always hidden)
    (re.compile(
        rf"(?i)(authorization{_SEP}){_NOT_DONE}"
        rf"(?:{_QUOTED}|{_BARE}(?:[ \t]{{1,10}}(?!\s*\[REDACTED\]){_BARE_CHARS}+)?)"),
     r"\1" + _REDACTED),
    # Cookie: / Set-Cookie: the rest of the header line (or the quoted value)
    (re.compile(rf"(?i)(cookie{_SEP}){_NOT_DONE}(?:{_QUOTED}|[^\n\r]{{1,2000}})"),
     r"\1" + _REDACTED),
    # password=, passwd=, secret=, token=, api_key=, access_key=, openai-organization= ...
    # (in plain text, JSON, escaped JSON, URL-encoded and with quoted values that hold spaces)
    (_label(rf"pass(?:word|wd|phrase)|secret|token|api{_S}key|access{_S}key|private{_S}key"
            rf"|openai{_S}organization"), r"\1" + _REDACTED),
    # a bare key=value, such as a URL query ?key=abc (not "monkey=")
    (re.compile(rf"(?i)((?<![A-Za-z0-9])key(?:=|%3[Dd])){_VALUE}"), r"\1" + _REDACTED),
    # Bearer tokens anywhere
    (re.compile(r"(?i)\bbearer\s{1,100}[A-Za-z0-9._~+/=\-]+"), "Bearer " + _REDACTED),
    # sk-... keys, also with dots (sk-ant-api03-..., sk-proj-..., sk-abc.def.ghi)
    (re.compile(r"\bsk-[A-Za-z0-9_.\-]{8,}"), "sk-" + _REDACTED),
    # JSON web tokens
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]*"),
     _REDACTED),
    # AWS access key ids
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _REDACTED),
    # Google API keys
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), _REDACTED),
    # GitHub tokens
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"), _REDACTED),
    # Slack tokens
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{8,}"), _REDACTED),
)


def sanitize_error_text(text: str, limit: int = MAX_ERROR_TEXT) -> str:
    """
    Make error text safer to store and print: hide common secrets, then cut it to ``limit``
    characters (an ellipsis and the original length are added when it is cut).

    BEST EFFORT, NOT A GUARANTEE. The text is first normalised with NFKC (so full-width
    look-alike letters become plain ones), then these shapes are hidden:

    * values after ``password``, ``passwd``, ``passphrase``, ``secret`` (so also
      ``client_secret``), ``token`` (``access_token``, ``x-auth-token``), ``api_key``
      (``x-api-key``), ``access_key`` (``aws_secret_access_key``), ``private_key``,
      ``openai-organization`` and a bare ``key=``, written with ``=`` or ``:``, in plain
      text, JSON (also backslash-escaped) and URL-encoded (``api%5Fkey=``, ``key%3D``);
      quoted values may contain spaces
    * the password in ``scheme://user:password@host`` URLs
    * ``Authorization`` / ``Proxy-Authorization`` values with any scheme (Basic, Bearer,
      Token, Digest ...); for Digest the whole header value
    * ``Cookie`` / ``Set-Cookie`` header values
    * ``Bearer`` tokens, ``sk-`` keys (also with dots), JSON web tokens, AWS key ids
      (``AKIA``/``ASIA``), Google keys (``AIza``), GitHub tokens (``ghp_`` ``gho_``
      ``ghu_`` ``ghs_`` ``ghr_`` ``github_pat_``), Slack tokens (``xox[abprs]-``)

    Anything else (other secret formats, secrets with no label, personal data, a secret
    split by markup) is NOT detected. Redaction runs before cutting, and the marker
    reports the length of the text as it was given.
    """
    original_length = len(text)
    text = unicodedata.normalize("NFKC", text)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > limit:
        text = f"{text[:limit]}... [truncated, original length {original_length} characters]"
    return text


# ─── The retry loop ───────────────────────────────────────────────────────────

def _type_name(exc: BaseException) -> str:
    try:
        name = type(exc).__name__
        return name if isinstance(name, str) and name else "Exception"
    except Exception:
        return "Exception"


def _safe_str(exc: BaseException) -> str:
    try:
        return str(exc).strip()
    except Exception:
        return "<error text unavailable>"


def _error_text(exc: BaseException) -> str:
    """Redacted and truncated ``ExceptionType: message`` for records and messages."""
    text = _safe_str(exc)
    name = _type_name(exc)
    return sanitize_error_text(f"{name}: {text}" if text else name)


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
            ``attempts`` and ``cause`` and is chained to the original exception. If the
            wait between attempts itself fails (a broken injected ``sleep``), the error
            says so and is chained to that failure.
        asyncio.CancelledError: always passed through, whether the cancel arrives while
            the call is running or while waiting between attempts. It is recorded as a
            "cancelled" outcome first.
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
                        "cancelled while the call was running")
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
                    message = (f"LLM call failed with a {code.value} error that is not "
                               f"retried (attempt {attempt}): {text}")
                raise _build_error(message, code, attempt, exc, tuple(own)) from exc

            delay = pol.delay_for(attempt, retry_after)
            record = pol._record(operation, attempt, "retry", code, delay, text, retry_after)
            own.append(record)
            await _notify(on_attempt, record)
            try:
                await pol._sleep(delay)
            except asyncio.CancelledError:
                pol._record(operation, attempt, "cancelled", code, delay,
                            "cancelled while waiting to retry", retry_after)
                raise
            except Exception as sleep_exc:
                wait_text = _error_text(sleep_exc)
                record = pol._record(
                    operation, attempt, "exhausted", code, delay,
                    f"retry wait failed: {wait_text}", retry_after, extra=True)
                own.append(record)
                await _notify(on_attempt, record)
                message = (f"LLM call failed after {attempt} attempt(s) [{code.value}]: "
                           f"{text}; the wait before the next retry also failed: {wait_text}")
                raise _build_error(message, code, attempt, exc, tuple(own)) from sleep_exc
            continue

        record = pol._record(operation, attempt, "success", None, 0.0, "")
        await _notify(on_attempt, record)
        return result
