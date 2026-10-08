"""
ARIA secret redaction
---------------------
Helpers that hide secrets before text or tool arguments are stored, logged or audited.

* ``sanitize_error_text(text)``  hide common secret shapes in a piece of text and cut it
* ``redact_args(value)``         a JSON-safe, redacted COPY of tool arguments for audit records
* ``ArgsDigester``               a keyed (HMAC) digest of arguments, so weak secrets cannot be
                                 guessed from a stored digest
* ``default_digester()``         the shared per-process digester

Standard library only. This module imports nothing from the rest of ARIA, so any layer
(plugins, permissions, tools, the pipeline) can use it without creating an import cycle.

BEST EFFORT, NOT A GUARANTEE. Redaction knows a list of secret shapes and key names. A
secret in any other shape goes through unchanged. See ``docs/redaction.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import unicodedata
from itertools import islice
from typing import Any, Optional

__all__ = [
    "MAX_ERROR_TEXT",
    "REDACTED",
    "ArgsDigester",
    "default_digester",
    "redact_args",
    "sanitize_error_text",
]

# ─── Error text safety (moved here from aria/plugins/retry.py, unchanged) ─────

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


REDACTED = _REDACTED   # the public name of the marker


# ─── Redacting tool arguments ─────────────────────────────────────────────────

_MAX_DEPTH_LIMIT = 64                 # hard cap on ``max_depth`` (keeps recursion small)
_MAX_NODES = 10_000                   # values visited per redact_args call, in total
_MAX_TOTAL_CHARS = 1_000_000          # characters scanned per redact_args call, in total
_MAX_SCAN_CHARS = 200_000             # a single string longer than this is not scanned at all
_MAX_INT_BITS = 256                   # bigger ints are summarised (json cannot print huge ints)
_MAX_NAME = 80                        # longest type name printed in "<TypeName>"

# Words that make a key a secret when they are a whole word of the key (after splitting on
# "_", "-", spaces, dots and camelCase boundaries). A trailing "s" is allowed (``tokens``).
_SECRET_WORDS = frozenset({
    "token", "auth", "oauth", "cookie", "session", "pwd", "pass", "passcode", "jwt",
    "credential", "bearer",
})

# Fragments that make a key a secret when they appear anywhere in the key with the separators
# removed (``clientSecret``, ``aws_secret_access_key``, ``X-Api-Key``, ``dbpassword``).
_SECRET_FRAGMENTS = (
    "password", "passwd", "passphrase", "secret", "credential", "authorization",
    "apikey", "accesskey", "privatekey", "signingkey", "encryptionkey", "masterkey",
    "sshkey", "authkey", "sessionkey", "clientkey",
)

_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = "0123456789"
_FULL = slice(None)


def _is_secret_key(name: str) -> bool:
    """True if a dict key (``name``) says that its value is a secret."""
    if len(name) > 1000:                               # keep the work bounded
        name = name[:500] + "_" + name[-500:]
    name = unicodedata.normalize("NFKC", name)
    name = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", name))
    name = _NOT_ALNUM.sub("_", name.lower())
    words = [w for w in (part.rstrip(_DIGITS) for part in name.split("_")) if w]
    if not words:
        return False
    flat = "".join(words)
    if any(fragment in flat for fragment in _SECRET_FRAGMENTS):
        return True
    for word in words:
        if word in _SECRET_WORDS or (word.endswith("s") and word[:-1] in _SECRET_WORDS):
            return True
    # anything ending in "_key" (a bare "key" alone is not treated as a secret)
    return len(words) > 1 and words[-1] == "key"


class _Budget:
    """Work left for one ``redact_args`` call, so a hostile input cannot make it slow."""

    __slots__ = ("nodes", "chars")

    def __init__(self) -> None:
        self.nodes = _MAX_NODES
        self.chars = _MAX_TOTAL_CHARS


def _type_label(value: Any) -> str:
    """The class name of ``value`` for markers. Uses ``type()``, never the object itself."""
    try:
        name = type(value).__name__
        if isinstance(name, str) and name:
            return str.__getitem__(name, slice(0, _MAX_NAME))
    except Exception:
        pass
    return "object"


def _clean_str(text: str, limit: int, budget: _Budget) -> str:
    """Redact and cut one string leaf (or key). ``text`` must be an exact ``str``."""
    size = len(text)
    if size > _MAX_SCAN_CHARS or size > budget.chars:
        return f"<string, {size} characters, not scanned>"
    budget.chars -= size
    return sanitize_error_text(text, limit)


def _scalar(value: Any) -> tuple[bool, Any]:
    """
    Convert a non-container value into a JSON scalar WITHOUT calling any method of the
    object itself (no ``__str__``, ``__repr__``, ``__int__``, ``__len__``, ...).
    Returns ``(True, scalar)``; a ``str`` is returned unscanned (the caller redacts it).
    Returns ``(False, None)`` for anything that is not a scalar.
    """
    if value is None:
        return True, None
    kind = type(value)
    if kind is bool:
        return True, value
    if issubclass(kind, str):
        return True, str.__getitem__(value, _FULL)
    if issubclass(kind, int):
        number = int.__index__(value)
        if number.bit_length() > _MAX_INT_BITS:
            return True, f"<int, {number.bit_length()} bits>"
        return True, number
    if issubclass(kind, float):
        real = float.__float__(value)
        if math.isfinite(real):
            return True, real
        return True, "nan" if math.isnan(real) else ("inf" if real > 0 else "-inf")
    if issubclass(kind, bytes):
        return True, f"<bytes {bytes.__len__(value)}>"
    if issubclass(kind, bytearray):
        return True, f"<bytes {bytearray.__len__(value)}>"
    return False, None


def _key_text(key: Any, limit: int, budget: _Budget) -> tuple[str, bool]:
    """The output name for a dict key, and whether the key names a secret."""
    ok, raw = _scalar(key)
    if ok:
        text = raw if isinstance(raw, str) else str(raw)
    else:
        text = f"<{_type_label(key)}>"
    return _clean_str(text, limit, budget), _is_secret_key(text)


def _unique(name: str, taken: dict[str, Any], counts: dict[str, int]) -> str:
    """``name``, or ``name#2``, ``name#3`` ... when two keys give the same text."""
    if name not in taken:
        return name
    n = counts.get(name, 1)
    while True:
        n += 1
        candidate = f"{name}#{n}"
        if candidate not in taken:
            counts[name] = n
            return candidate


def _redact(
    value: Any,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
    path: set[int],
    budget: _Budget,
) -> Any:
    if depth > max_depth:
        return "<max depth>"
    if budget.nodes <= 0:
        return "<budget exceeded>"
    budget.nodes -= 1

    kind = type(value)
    if issubclass(kind, str):
        return _clean_str(str.__getitem__(value, _FULL), max_string, budget)
    ok, scalar = _scalar(value)
    if ok:
        return scalar                      # None, bool, int, float or a marker string

    is_dict = issubclass(kind, dict)
    if not (is_dict or issubclass(kind, (list, tuple, set, frozenset))):
        return f"<{_type_label(value)}>"

    marker = id(value)
    if marker in path:
        return "<cycle>"
    path.add(marker)
    try:
        if is_dict:
            total = dict.__len__(value)
            result: dict[str, Any] = {}
            counts: dict[str, int] = {}
            for key, item in islice(dict.items(value), max_items):
                name, secret = _key_text(key, max_string, budget)
                name = _unique(name, result, counts)
                if secret:
                    result[name] = REDACTED         # the value is never looked at
                else:
                    result[name] = _redact(item, depth + 1, max_depth, max_items,
                                           max_string, path, budget)
            if total > max_items:
                result[_unique("<omitted>", result, counts)] = f"<{total - max_items} more items omitted>"
            return result

        if issubclass(kind, list):
            total, items = list.__len__(value), list.__iter__(value)
        elif issubclass(kind, tuple):
            total, items = tuple.__len__(value), tuple.__iter__(value)
        elif issubclass(kind, frozenset):
            total, items = frozenset.__len__(value), frozenset.__iter__(value)
        else:
            total, items = set.__len__(value), set.__iter__(value)
        out = [_redact(item, depth + 1, max_depth, max_items, max_string, path, budget)
               for item in islice(items, max_items)]
        if total > max_items:
            out.append(f"<{total - max_items} more items omitted>")
        return out
    except Exception:
        # for example "dictionary changed size during iteration" from another thread
        return f"<{_type_label(value)}, unreadable>"
    finally:
        path.discard(marker)


def _bounded(value: Any, default: int, low: int, high: int) -> int:
    if type(value) is bool or not isinstance(value, int):
        return default
    return max(low, min(high, int.__index__(value)))


def redact_args(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 100,
    max_string: int = 200,
) -> Any:
    """
    Return a JSON-safe, redacted COPY of tool arguments, for audit records.

    The input is never changed and never printed. The result contains only ``dict`` (with
    ``str`` keys), ``list``, ``str``, ``int``, finite ``float``, ``bool`` and ``None``.

    * A dict value is replaced by ``"[REDACTED]"`` when its KEY names a secret, whatever the
      type of the value. Keys are compared after Unicode NFKC, lower-casing and splitting on
      ``-``, ``_``, spaces, dots and camelCase, so ``apiKey``, ``X-Api-Key`` and
      ``api_key`` are the same. Secret keys: anything with ``password``, ``passwd``,
      ``passphrase``, ``secret``, ``credential(s)``, ``authorization``, ``apikey``,
      ``accesskey``, ``privatekey`` (and some other ``...key`` names), or the whole words
      ``token``, ``auth``, ``oauth``, ``cookie``, ``session``, ``bearer``, ``pwd``,
      ``pass``, ``passcode``, ``jwt`` (a trailing ``s`` is allowed), or ending in ``_key``
      (``_secret`` and ``_token`` are covered by the words above).
    * Strings go through ``sanitize_error_text`` with ``limit=max_string``.
    * ``int``, finite ``float``, ``bool`` and ``None`` are kept. ``nan`` and ``inf`` become
      the strings ``"nan"``, ``"inf"`` and ``"-inf"``. Ints over 256 bits become a marker.
    * ``bytes`` become ``"<bytes N>"``. Any other object becomes ``"<TypeName>"``; its
      ``__repr__``, ``__str__`` and attributes are never used, so it cannot leak or raise.
    * ``tuple``, ``set`` and ``frozenset`` become lists (a set keeps its iteration order).
    * Below ``max_depth`` levels the value becomes ``"<max depth>"``; a container with more
      than ``max_items`` items is cut and ends with ``"<N more items omitted>"``; a
      reference to a container that is already being copied becomes ``"<cycle>"``.
    * Dict keys that are not strings are turned into text without calling the key's own
      methods, and are redacted like strings.

    Bounds that cannot be turned off: ``max_depth`` is at most 64, one call looks at no more
    than 10,000 values and 1,000,000 characters in total, and a single string over 200,000
    characters is replaced by a marker without being scanned. Past those, values become
    ``"<budget exceeded>"``.

    It never raises for any input (``KeyboardInterrupt`` and ``SystemExit`` excepted).
    BEST EFFORT: a secret under a harmless key name and in an unknown shape is NOT detected.
    """
    try:
        return _redact(
            value,
            0,
            _bounded(max_depth, 6, 0, _MAX_DEPTH_LIMIT),
            _bounded(max_items, 100, 0, _MAX_NODES),
            _bounded(max_string, 200, 0, _MAX_SCAN_CHARS),
            set(),
            _Budget(),
        )
    except Exception:
        return "<redaction failed>"


# ─── Keyed digest of arguments ────────────────────────────────────────────────

class ArgsDigester:
    """
    HMAC-SHA256 digest of tool arguments, with a random key that lives only in this object.

    A plain hash of arguments can be guessed offline when the secret is weak (a short
    password, a PIN): the attacker hashes candidates and compares. Without the key the same
    guess cannot be checked against an HMAC. The key is 32 random bytes made when the object
    is created, is never printed, and the object cannot be pickled or copied. Two digesters
    give different digests for the same input; one digester always gives the same.

    The input is serialised to canonical JSON first (sorted keys, no whitespace, ASCII only),
    so equal arguments give equal digests whatever the key order. Tuples and lists are the
    same in JSON. Values that JSON cannot hold (bytes, objects, sets, nan, inf, non-string
    keys, cycles, nesting too deep) raise ``TypeError`` and are never turned into text.
    """

    __slots__ = ("_key",)

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def digest(self, obj: Any) -> str:
        """The 64 hex character HMAC-SHA256 of the canonical JSON form of ``obj``."""
        text = _canonical_json(obj)
        return hmac.new(self._key, text.encode("ascii"), hashlib.sha256).hexdigest()

    def __repr__(self) -> str:
        return "<ArgsDigester>"

    def __reduce__(self) -> Any:
        raise TypeError("ArgsDigester holds a secret key and cannot be pickled or copied")


def _check_json_keys(obj: Any) -> None:
    """Reject non-string dict keys (json would silently turn 1 and "1" into the same key)."""
    seen: set[int] = set()
    stack = [obj]
    while stack:
        item = stack.pop()
        kind = type(item)
        if issubclass(kind, dict):
            if id(item) in seen:
                continue
            seen.add(id(item))
            for key, child in dict.items(item):
                if not issubclass(type(key), str):
                    raise TypeError(f"dict key of type {_type_label(key)} is not a string")
                stack.append(child)
        elif issubclass(kind, (list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            stack.extend(list.__iter__(item) if issubclass(kind, list) else tuple.__iter__(item))


def _canonical_json(obj: Any) -> str:
    try:
        _check_json_keys(obj)
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except TypeError:
        raise
    except (ValueError, RecursionError, OverflowError) as exc:
        # circular reference, nan/inf, too deep, int too large to print
        raise TypeError(f"arguments cannot be serialised to canonical JSON: {exc}") from None
    except Exception:
        # an object whose own methods raise (the message could hold anything, so drop it)
        raise TypeError("arguments cannot be serialised to canonical JSON") from None


_default_digester: Optional[ArgsDigester] = None
_default_lock = threading.Lock()


def default_digester() -> ArgsDigester:
    """The shared per-process ``ArgsDigester``, created on first use."""
    global _default_digester
    if _default_digester is None:
        with _default_lock:
            if _default_digester is None:
                _default_digester = ArgsDigester()
    return _default_digester
