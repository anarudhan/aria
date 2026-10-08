"""
ARIA secret redaction
---------------------
Helpers that hide secrets before text or tool arguments are stored, logged or audited.

* ``sanitize_error_text(text)``  hide common secret shapes in a piece of text and cut it
* ``redact_args(value)``         a JSON-safe, redacted COPY of tool arguments for audit records
* ``ArgsDigester``               a keyed (HMAC) digest of arguments, so weak secrets cannot be
                                 guessed from a stored digest
* ``default_digester()``         the shared per-process digester

Standard library only. This file itself imports nothing from ARIA, but importing
``aria.security.redact`` still runs ``aria/__init__.py`` and ``aria/security/__init__.py``
(patcher, scanner). It is free of import cycles today only because ``aria/security/patcher.py``
imports the permission engine under ``TYPE_CHECKING`` only. That rule must stay: nothing in
``aria/security/`` may import ``aria.permissions.engine`` at run time.

BEST EFFORT, NOT A GUARANTEE. Redaction knows a list of secret shapes and key names. A
secret in any other shape goes through unchanged. Work is bounded by explicit budgets
(see the limits in ``redact_args``, ``sanitize_error_text`` and ``ArgsDigester``); the
timings were measured with the tests in ``tests/test_redact.py``, not proven. See
``docs/redaction.md``.
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
from array import array
from contextvars import ContextVar
from functools import partial
from itertools import compress, islice, repeat
from typing import Any, Optional

__all__ = [
    "MAX_ERROR_TEXT",
    "REDACTED",
    "ArgsDigester",
    "default_digester",
    "redact_args",
    "sanitize_error_text",
]

# ─── Error text safety (moved here from aria/plugins/retry.py; the patterns have since
# been extended on purpose, see docs/redaction.md) ─────────────────────────────

MAX_ERROR_TEXT = 500

_REDACTED = "[REDACTED]"

# Building blocks. Every repetition is bounded or applies to a character class that cannot
# overlap with the next part, and no pattern starts with an open-ended prefix, so matching
# stays linear in the input size (see the 100k-character timing test).
_S = r"(?:[-_]|%5[Ff]|%2[Dd])?"                      # optional "_" or "-" inside a label
_SEP = r"""(?:\\?["'])?\s{0,5}(?:[:=]|%3[AaDd])\s{0,5}"""   # [closing quote] : or = [spaces]
# Tried in this order: JSON-escaped \"..\" (a whole JSON document inside a string), then an
# ordinary quoted value that may hold backslash escapes ("ab\"cd ef"), then the loose forms
# that tolerate a missing or extra backslash. The inner repetitions are bounded and their two
# branches cannot match the same character, so matching stays linear.
_QUOTED = (
    r"""\\"[^"\\]{0,1000}\\"|\\'[^'\\]{0,1000}\\'"""
    r"""|"(?:[^"\\]|\\.){0,1000}"|'(?:[^'\\]|\\.){0,1000}'"""
    r"""|\\?"[^"\\]{0,1000}\\?"|\\?'[^'\\]{0,1000}\\?'"""
)
_BARE_CHARS = r"""[^\s,;'"&}\])\\]"""
# Our own marker counts as ONE ordinary value character, so a value that starts with, or
# contains, "[REDACTED]" is hidden whole and replaced by one marker. That makes redacting
# twice change nothing ("password=[REDACTED]" is replaced by itself) and gives a forged
# marker no special power: "password=[REDACTED]@x" is hidden exactly like "password=y@x".
_NO_BRACKET = r"""[^\s,;'"&}\])\\\[]"""                   # a value character that is not "["
_BRACKET = r"(?:\[REDACTED\]|\[)"                          # the marker, or a lone "["
_UNITS_OPT = rf"{_NO_BRACKET}*(?:{_BRACKET}{_NO_BRACKET}*)*"   # value characters, maybe none
_UNITS = rf"(?={_BARE_CHARS}){_UNITS_OPT}"                    # at least one (linear: runs
#                                                            # of plain characters, then "[")
_BARE = rf"""\\?["']?{_UNITS}"""                       # also an unterminated opening quote
_QUOTED_FULL = rf"(?:{_QUOTED}){_UNITS_OPT}"          # a quoted value and what sticks to it
_VALUE = rf"(?:{_QUOTED_FULL}|{_BARE})"
# After "Authorization:" a value that is only markers is done: it is what a two-word value
# ("Basic abc extra") leaves in front of the third word, and the third word is not hidden.
_ONLY_MARKERS = rf"(?:\[REDACTED\])+(?!{_BARE_CHARS})"


def _label(words: str) -> "re.Pattern[str]":
    return re.compile(rf"(?i)((?:{words}){_SEP})" + _VALUE)


# Applied in order after NFKC normalisation. Each keeps the label and hides only the value.
_SECRET_PATTERNS: tuple[tuple["re.Pattern[str]", str], ...] = (
    # PEM private keys: from the BEGIN line to the END line (or to the end of the text)
    (re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY(?: BLOCK)?-----.*?"
                r"(?:-----END [A-Z ]{0,40}PRIVATE KEY(?: BLOCK)?-----|\Z)", re.DOTALL),
     _REDACTED),
    # scheme://user:password@host  -> hide the password (proxy and connection errors print these)
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]{0,30}://[^\s/:@?#]{0,200}:)[^\s/@?#]{1,500}(?=@)"),
     r"\1" + _REDACTED),
    # Authorization: Digest username="x", nonce="y", response="z"  -> the whole header value
    (re.compile(r"(?i)(authorization" + _SEP + r")digest\b[^\n\r]{0,2000}"),
     r"\1" + _REDACTED),
    # Authorization: <any scheme> <value>  (one or two words, so the value is always hidden)
    (re.compile(
        rf"(?i)(authorization{_SEP})"
        rf"(?:{_ONLY_MARKERS}|{_QUOTED_FULL}|{_BARE}(?:[ \t]{{1,10}}{_UNITS})?)"),
     r"\1" + _REDACTED),
    # Cookie: / Set-Cookie: the rest of the header line, or only the quoted value (or our marker)
    # when it is a whole JSON-style value (followed by "," "}" "]" or the end of the line)
    (re.compile(rf"(?i)(cookie{_SEP})"
                rf"(?:(?:{_QUOTED}|(?:\[REDACTED\])+)(?=[ \t]*(?:[,}}\]]|[\n\r]|\Z))"
                rf"|(?:{_QUOTED})[^\n\r]{{0,2000}}|[^\n\r]{{1,2000}})"),
     r"\1" + _REDACTED),
    # password=, passwd=, secret=, token=, api_key=, access_key=, openai-organization= ...
    # (in plain text, JSON, escaped JSON, URL-encoded and with quoted values that hold spaces)
    (_label(rf"pass(?:word|wd|phrase)|secret|token|api{_S}key|access{_S}key|private{_S}key"
            rf"|openai{_S}organization|signature"), r"\1" + _REDACTED),
    # pwd=, pass=, DB_PASS=, sig=  (only as a whole word, so "bypass=" and "design=" stay)
    (re.compile(rf"(?i)((?<![A-Za-z0-9])(?:pwd|pass|sig){_SEP}){_VALUE}"), r"\1" + _REDACTED),
    # a bare key=value, such as a URL query ?key=abc (not "monkey=")
    (re.compile(rf"(?i)((?<![A-Za-z0-9])key(?:=|%3[Dd])){_VALUE}"), r"\1" + _REDACTED),
    # Bearer tokens anywhere
    (re.compile(r"(?i)\bbearer\s{1,100}(?:\[REDACTED\]|[A-Za-z0-9._~+/=\-])+"),
     "Bearer " + _REDACTED),
    # sk-... keys, also with dots (sk-ant-api03-..., sk-proj-..., sk-abc.def.ghi)
    (re.compile(r"\bsk-(?:\[REDACTED\]|[A-Za-z0-9_.\-]){8,}"), "sk-" + _REDACTED),
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


# Longest text sanitize_error_text will scan, counted AFTER NFKC normalisation (one character can
# expand to 18: U+FDFA). A longer text is replaced by a marker, never passed through unscanned.
_MAX_SANITIZE_CHARS = 1_000_000
_NFKC_CHUNK = 10_000
_MAX_MARK_RUN = 32          # combining marks allowed after a slice boundary (see below)


def _normalise_bounded(text: str, cap: int) -> Optional[str]:
    """
    NFKC-normalise ``text`` in slices and give up (return None) as soon as the result would
    be longer than ``cap``, so a text that expands cannot make the work large. A slice never
    ends in front of a combining mark, but a run of more than 32 combining marks at a slice
    boundary also gives up (returns None): nobody writes text like that, and following the
    run to its end would make the work unbounded. Plain ASCII needs no normalisation.
    """
    if text.isascii():
        return text if len(text) <= cap else None
    parts = []
    total = 0
    start = 0
    size = len(text)
    while start < size:
        end = min(start + _NFKC_CHUNK, size)
        marks = 0
        while end < size and unicodedata.combining(text[end]):
            end += 1
            marks += 1
            if marks > _MAX_MARK_RUN:
                return None
        piece = unicodedata.normalize("NFKC", text[start:end])
        total += len(piece)
        if total > cap:
            return None
        parts.append(piece)
        start = end
    return "".join(parts)


# ─── The shadow: how split and look-alike labels are read ────────────────────
#
# The secret patterns only know plain spellings. To also read ``pa<zero width space>ssword=x``
# or ``p<Cyrillic a>ssword=x`` the patterns are run on a "shadow" of the text: invisible,
# control and combining characters removed, Cyrillic and Greek look-alike letters folded to
# Latin, look-alikes of ":" and "=" folded to those. Every shadow character remembers where it
# came from, so what the patterns find is hidden in the ORIGINAL text, and only there: the
# rest of the text is returned exactly as it was.

_EXTRA_INVISIBLE = frozenset(map(chr, (
    0x115F, 0x1160, 0x3164, 0xFFA0,   # Hangul fillers
    0x2800,                           # Braille blank
    0x0640,                           # Arabic tatweel
    0x02BC,                           # modifier letter apostrophe
)))
# marks, format, private use, unassigned and surrogate characters
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Mn", "Mc", "Me", "Co", "Cn", "Cs"})
_KEPT_CONTROLS = frozenset("\t\n\x0b\x0c\r\x85")      # control characters that are white space
_SEPARATOR_LOOKALIKES = {
    chr(0x2236): ":", chr(0xA789): ":",              # ratio, modifier letter colon
    chr(0x2254): "=", chr(0x2255): "=", chr(0xA78A): "=",   # colon equals, equals colon, modifier equals
}
_MAX_PASSES = 3                 # sweeps of all patterns over one shadow


def _is_invisible(char: str) -> bool:
    """True for a character that is dropped from the shadow."""
    category = unicodedata.category(char)
    if category == "Cc":
        return char not in _KEPT_CONTROLS
    return category in _INVISIBLE_CATEGORIES or char in _EXTRA_INVISIBLE


def _fold_char(char: str) -> str:
    """The shadow of one character of an NFKC text: "" (dropped) or exactly one character."""
    if char in _SEPARATOR_LOOKALIKES:
        return _SEPARATOR_LOOKALIKES[char]
    if _is_invisible(char):
        return ""
    plain = "".join(
        part for part in unicodedata.normalize("NFKD", char) if not _is_invisible(part)
    ).translate(_CONFUSABLES)
    return plain if len(plain) == 1 and plain.isascii() else char


_NFKD = partial(unicodedata.normalize, "NFKD")
_ASCII_CONTROLS = frozenset(map(chr, [*range(9), *range(14, 32), 127]))   # dropped from the shadow


class _FoldContext:
    """
    What is known about the characters seen so far in ONE call: the characters, the ones that
    are dropped from the shadow, and the table of the ones that change. A pure memo: it lives
    for one ``redact_args`` call (or one ``sanitize_error_text`` call), is never shared between
    calls, and is emptied when it holds more than ``_MAX_CONTEXT_CHARS`` characters.
    """

    __slots__ = ("known", "hidden", "changed", "table")

    def __init__(self) -> None:
        self.known: set[str] = set()
        self.hidden: set[str] = set()
        self.changed: set[str] = set()
        self.table: dict[int, str] = {}

    def clear(self) -> None:
        self.known.clear()
        self.hidden.clear()
        self.changed.clear()
        self.table.clear()


_MAX_CONTEXT_CHARS = 50_000
_HIDE_CATEGORIES = _INVISIBLE_CATEGORIES | {"Cc"}
_SKIP_CATEGORIES = _HIDE_CATEGORIES | {"Lo"}
_FOLD_CONTEXT: "ContextVar[Optional[_FoldContext]]" = ContextVar("aria_redact_fold", default=None)


def _fold_table(chars: "set[str]") -> _FoldContext:
    """
    How the shadow of a text made of the characters ``chars`` differs from the text: the set
    of characters that are dropped, the set that change and a ``str.translate`` table for them
    (the three also hold characters of earlier texts of the same call, which does no harm).

    It is built from DISTINCT characters, once per call, so the work is the number of distinct
    characters and does not depend on what other calls saw. Dropped characters are sorted out
    in bulk by category. A character that decomposes goes through ``_fold_char`` only when
    its decomposition holds an ASCII character or a look-alike; a Hangul syllable, an Indic
    composite or a precomposed non-Latin letter cannot change and is skipped in bulk.
    """
    context = _FOLD_CONTEXT.get()
    if context is None:
        context = _FoldContext()
    fresh = chars - context.known
    if not fresh:
        return context
    if len(context.known) + len(fresh) > _MAX_CONTEXT_CHARS:
        context.clear()
        fresh = set(chars)
    others = [char for char in fresh if char > "\x7f"]
    categories = list(map(unicodedata.category, others))
    hidden = {char for char, category in zip(others, categories) if category in _HIDE_CATEGORIES}
    hidden.difference_update(_KEPT_CONTROLS)               # white space among the controls
    hidden.update(fresh & _EXTRA_INVISIBLE)
    hidden.update(fresh & _ASCII_CONTROLS)
    # Other letters ("Lo": Hangul, CJK, kana, Indic and so on) are never hidden and, once the text
    # is NFKC-normalised, never change; checked for every code point in tests/test_redact.py.
    rest = [
        char for char, category in zip(others, categories)
        if category not in _SKIP_CATEGORIES and char not in hidden
    ]
    decomposed = list(map(_NFKD, rest))
    candidates: list[str] = []
    if _MAY_CHANGE.search("".join(decomposed)) is not None:    # one look at all of them first
        candidates = [
            char for char, plain in zip(rest, decomposed)
            if plain != char and _MAY_CHANGE.search(plain) is not None
        ]
    candidates.extend(fresh & _FOLD_SPECIAL)
    for char in candidates:
        if char not in hidden:
            folded = _fold_char(char)
            if folded != char:
                context.table[ord(char)] = folded
                context.changed.add(char)
    if context.known:
        context.hidden |= hidden
        context.known |= fresh
    else:                                           # the first text of a call: no copying
        context.hidden = hidden
        context.known = fresh
    return context


_ASCII_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_MARKER_RUN = re.compile(rb"\x01+")


def _build_shadow(text: str) -> Optional[tuple[str, "array[int]"]]:
    """
    The shadow of an NFKC-normalised ``text`` and, for each shadow character, the position of
    the text character it came from. Returns None when the shadow is the text itself.
    """
    if text.isascii() and _ASCII_CONTROL.search(text) is None:
        return None
    chars = set(text)
    context = _fold_table(chars)
    hidden, table = context.hidden, context.table
    drops = not chars.isdisjoint(hidden)
    if not drops and chars.isdisjoint(context.changed):
        return None
    size = len(text)
    if drops:
        kept = [char not in hidden for char in text]
        shadow = "".join(compress(text, kept))
        origin = array("I", compress(range(size), kept))
    else:                                                  # nothing dropped: one for one
        shadow = text
        origin = array("I", range(size))
    return shadow.translate(table), origin


def _apply_patterns(text: str) -> str:
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _sweep(
    shadow: str, first: "array[int]", last_of: "array[int]", marked: bytearray
) -> tuple[str, "array[int]", "array[int]", bytearray, bool]:
    """One pass of all patterns over ``shadow``, keeping the three tables in step."""
    changed = False
    for pattern, replacement in _SECRET_PATTERNS:
        keeps_label = replacement.startswith("\\1")
        marker = replacement[2:] if keeps_label else replacement
        matches = [
            match for match in pattern.finditer(shadow)
            if shadow[match.end(1) if keeps_label else match.start():match.end()] != marker
        ]                                   # a value that already is the marker is no change
        if not matches:
            continue
        changed = True
        size = len(marker)
        new_shadow: list[str] = []
        new_first: "array[int]" = array("I")
        new_last: "array[int]" = array("I")
        new_marked = bytearray()
        add_text, add_first = new_shadow.append, new_first.extend
        add_last, add_marked = new_last.extend, new_marked.extend
        ones = b"\x01" * size
        done = 0
        for match in matches:
            start = match.end(1) if keeps_label else match.start()
            end = match.end()
            add_text(shadow[done:start])
            add_first(first[done:start])
            add_last(last_of[done:start])
            add_marked(marked[done:start])
            add_text(marker)
            add_first(repeat(first[start], size))
            add_last(repeat(last_of[end - 1], size))
            add_marked(ones)
            done = end
        add_text(shadow[done:])
        add_first(first[done:])
        add_last(last_of[done:])
        add_marked(marked[done:])
        shadow, first, last_of, marked = "".join(new_shadow), new_first, new_last, new_marked
    return shadow, first, last_of, marked, changed


def _redact_mapped(text: str, shadow: str, first: "array[int]") -> str:
    """
    Run the patterns on ``shadow`` and hide what they find in ``text``.

    Each pattern is applied exactly as in ``_apply_patterns`` (same order, same replacements),
    but alongside the shadow three tables are kept for each character: the first and the end
    of the span of ``text`` it stands for, and whether it belongs to a replacement. At the end
    every replacement is cut into ``text`` over the span it stands for. Characters between
    the replacements, invisible ones included, are not touched. The sweep over all patterns is
    repeated (at most 3 sweeps) until a sweep hides nothing more.
    """
    last_of = array("I", map((1).__add__, first))
    marked = bytearray(len(shadow))
    for _ in range(_MAX_PASSES):
        shadow, first, last_of, marked, changed = _sweep(shadow, first, last_of, marked)
        if not changed:
            break
    pieces = []
    cursor = 0
    for run in _MARKER_RUN.finditer(marked):
        begin, finish = run.span()
        left = max(first[begin], cursor)
        pieces.append(text[cursor:left])
        pieces.append(shadow[begin:finish])
        cursor = max(last_of[finish - 1], left)
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_text(text: str) -> str:
    """Hide secrets in an NFKC-normalised text, reading split and look-alike labels too."""
    built = _build_shadow(text)
    if built is None:                        # nothing to fold: one plain pass, as always
        return _apply_patterns(text)
    return _redact_mapped(text, built[0], built[1])


def sanitize_error_text(text: str, limit: int = MAX_ERROR_TEXT) -> str:
    """
    Make error text safer to store and print: hide common secrets, then cut it to ``limit``
    characters (an ellipsis and the original length are added when it is cut).

    BEST EFFORT, NOT A GUARANTEE. The text is first normalised with NFKC (so full-width
    look-alike letters become plain ones), then these shapes are hidden:

    * values after ``password``, ``passwd``, ``passphrase``, ``secret`` (so also
      ``client_secret``), ``token`` (``access_token``, ``x-auth-token``), ``api_key``
      (``x-api-key``), ``access_key`` (``aws_secret_access_key``), ``private_key``,
      ``openai-organization``, ``signature`` (``x-amz-signature``), the whole words ``pwd``,
      ``pass`` (``DB_PASS``) and ``sig``, and a bare ``key=``, written with ``=`` or ``:``, in
      plain text, JSON (also backslash-escaped) and URL-encoded (``api%5Fkey=``, ``key%3D``);
      quoted values may contain spaces and backslash escapes
    * the password in ``scheme://user:password@host`` URLs (the user may be empty)
    * ``Authorization`` / ``Proxy-Authorization`` values with any scheme (Basic, Bearer,
      Token, Digest ...); for Digest the whole header value
    * ``Cookie`` / ``Set-Cookie`` header values
    * PEM private keys (``-----BEGIN ... PRIVATE KEY-----`` up to the END line or the end)
    * ``Bearer`` tokens, ``sk-`` keys (also with dots), JSON web tokens, AWS key ids
      (``AKIA``/``ASIA``), Google keys (``AIza``), GitHub tokens (``ghp_`` ``gho_``
      ``ghu_`` ``ghs_`` ``ghr_`` ``github_pat_``), Slack tokens (``xox[abprs]-``)

    Anything else (other secret formats, secrets with no label, personal data, a secret
    split by markup) is NOT detected. Redaction runs before cutting, and the marker
    reports the length of the text as it was given.

    A text of more than 1,000,000 characters after NFKC normalisation is not scanned: it is
    replaced by ``<text, N characters, not scanned>`` so that nothing leaks and the work
    stays bounded. The cap is applied to the text before any redaction, so redaction making a
    text longer can never trigger it. The same marker is returned for a text with a run of more than 32
    combining marks at a slice boundary (see ``_normalise_bounded``).

    SPLIT AND LOOK-ALIKE LABELS. The patterns are also run on a "shadow" of the text:
    invisible, control, combining and other mark characters removed, Cyrillic and Greek
    look-alike letters folded to Latin, look-alikes of ``:`` and ``=`` folded to those. So
    ``pa<zero width space>ssword=x`` and ``p<Cyrillic a>ssword=x`` are hidden too. What the
    patterns find in the shadow is hidden in the original text, and only there: the rest of
    the text is returned as it was (invisible characters and accents included), except that
    the hidden value and the characters inside it are replaced. The run is repeated (at most
    3 passes) until a pass hides nothing more. See ``docs/redaction.md`` for the characters.
    """
    original_length = len(text)
    normalised = _normalise_bounded(text, _MAX_SANITIZE_CHARS)
    if normalised is None:
        return f"<text, {original_length} characters, not scanned>"
    text = _redact_text(normalised)
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
_MAX_KEY_CHARS = 1000                 # a longer key is treated as a secret (fail closed)
_MAX_NAME_CHARS = 256                 # longest "name"/"first item" text taken as a secret name

# Words that make a key a secret when they are a whole word of the key (after splitting on
# non-letters and camelCase boundaries). A trailing "s" is allowed (``tokens``).
_SECRET_WORDS = frozenset({
    "token", "auth", "oauth", "cookie", "session", "pwd", "pass", "passcode", "jwt",
    "credential", "bearer", "signature", "sig", "otp", "totp", "pin", "seed", "mnemonic",
    "privkey",
})

# Fragments that make a key a secret when they appear anywhere in the key with the separators
# removed (``clientSecret``, ``accesstoken``, ``X-Api-Key``, ``dbpassword``, ``csrftoken``).
_SECRET_FRAGMENTS = (
    "password", "passwd", "passphrase", "secret", "credential", "authorization",
    "apikey", "accesskey", "privatekey", "signingkey", "encryptionkey", "masterkey",
    "sshkey", "authkey", "sessionkey", "clientkey", "token", "pwd", "cookie", "jwt",
    "bearer", "signature", "mnemonic", "privkey", "privatepem", "keydata", "keymaterial",
    "pincode", "sessionid",
)

# Letters of other alphabets that look like Latin ones (Cyrillic and Greek), folded to Latin
# before a key is classified. A small explicit list, not a complete confusables table.
_CONFUSABLES = str.maketrans(
    "аеорсухіјѕԁԛԝһ"
    "кмтнвӏı"
    "АВЕКМНОРСТХУІЈ"
    "ЅԚԜ"
    "αεικνορτυχγω"
    "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ",
    "aeopcyxijsdqwh" "kmthbli"
    "ABEKMHOPCTXYIJ" "SQW"
    "aeikvoptuxyw"
    "ABEZHIKMNOPTYX",
)

# characters whose shadow differs although they do not decompose (look-alikes, separators)
_FOLD_SPECIAL = frozenset(chr(code) for code in _CONFUSABLES) | frozenset(_SEPARATOR_LOOKALIKES)
# a decomposition that can lead to a different shadow: it holds an ASCII character or a look-alike
_MAY_CHANGE = re.compile("[\x00-\x7f" + re.escape("".join(sorted(_FOLD_SPECIAL))) + "]")

_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")
_SURROGATES = re.compile(r"[\ud800-\udfff]")
_DIGITS = "0123456789"
_FULL = slice(None)


def _is_secret_key(name: str) -> bool:
    """True if a dict key (``name``) says that its value is a secret."""
    if len(name) > _MAX_KEY_CHARS:                     # cannot classify cheaply: fail closed
        return True
    name = unicodedata.normalize("NFKD", name)           # also splits "o" + accent apart
    name = "".join(
        c for c in name if unicodedata.category(c) != "Cc" and not _is_invisible(c))
    name = name.translate(_CONFUSABLES)
    name = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", name))
    name = _NOT_ALNUM.sub("_", name.lower())
    words = [w for w in (part.rstrip(_DIGITS) for part in name.split("_")) if w]
    if not words:
        return False
    flat = "".join(words)
    # "tokenizer" and "tokenize" are about text, not secrets
    if "token" in flat.replace("tokeniz", ""):
        return True
    if any(fragment in flat for fragment in _SECRET_FRAGMENTS if fragment != "token"):
        return True
    if flat.endswith("auth"):                          # xauth, proxyauth (not "author")
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


_TYPE_NAME = type.__dict__["__name__"]   # reads the real name, never a metaclass property


def _type_label(value: Any) -> str:
    """The class name of ``value`` for markers. Runs no code of the object or its class."""
    try:
        name = _TYPE_NAME.__get__(type(value))
        if type(name) is str and name:
            return sanitize_error_text(name[:_MAX_NAME], 10 * _MAX_NAME)
    except Exception:
        pass
    return "object"


def _clean_str(text: str, limit: int, budget: _Budget) -> str:
    """Redact and cut one string leaf (or key). ``text`` must be an exact ``str``."""
    size = len(text)
    if size > 2 * _MAX_SCAN_CHARS or size > budget.chars:
        return f"<string, {size} characters, not scanned>"
    budget.chars -= size                                   # charged even if we give up below
    if not text.isascii() and _SURROGATES.search(text):
        text = _SURROGATES.sub("\ufffd", text)            # lone surrogates cannot be UTF-8
    norm = _normalise_bounded(text, min(_MAX_SCAN_CHARS, budget.chars + size))
    if norm is None:
        return f"<string, {size} characters, not scanned>"
    budget.chars -= max(0, len(norm) - size)               # NFKC may have made it longer
    return sanitize_error_text(norm, limit)


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


def _classify(text: str, budget: _Budget) -> bool:
    """``_is_secret_key`` with a work charge; out of budget means secret (fail closed)."""
    size = len(text)
    if size > _MAX_KEY_CHARS:
        return True
    budget.chars -= size
    if budget.chars < 0:
        return True
    return _is_secret_key(text)


def _secret_name(text: str, budget: _Budget) -> bool:
    """True if ``text`` (the value of a "name" entry, the first item of a pair) is a secret name."""
    if len(text) > _MAX_NAME_CHARS:
        return False
    return _classify(text, budget)


def _key_info(key: Any, limit: int, budget: _Budget) -> tuple[str, bool, Optional[str]]:
    """
    For one dict key: the output name, whether it names a secret, and its text if it is a
    string. Keys that are not strings, numbers, None or bytes are treated as secret.
    """
    kind = type(key)
    if issubclass(kind, str):
        raw = str.__getitem__(key, _FULL)
        return _clean_str(raw, limit, budget), _classify(raw, budget), raw
    if issubclass(kind, (bytes, bytearray)):
        size = bytes.__len__(key) if issubclass(kind, bytes) else bytearray.__len__(key)
        if size > _MAX_KEY_CHARS:
            return f"<bytes {size}>", True, None
        decoded = (bytes.decode(key, "latin-1") if issubclass(kind, bytes)
                   else bytearray.decode(key, "latin-1"))
        return f"<bytes {size}>", _classify(decoded, budget), None
    ok, raw_scalar = _scalar(key)
    if ok:
        return _clean_str(str(raw_scalar), limit, budget), False, None
    return f"<{_type_label(key)}>", True, None


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


_NAME_ENTRY_KEYS = frozenset({"name", "key"})


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
            entries = list(islice(dict.items(value), max_items))
            infos = [_key_info(key, max_string, budget) for key, _ in entries]
            # {"name": "DB_PASSWORD", "value": "x"}: the "name" entry says "value" is a secret
            name_says_secret = False
            for (_, item), (_, _, raw) in zip(entries, infos):
                if raw is not None and len(raw) <= 4 and raw.lower() in _NAME_ENTRY_KEYS:
                    text_ok, text = _scalar(item)
                    if text_ok and isinstance(text, str) and _secret_name(text, budget):
                        name_says_secret = True
            result: dict[str, Any] = {}
            counts: dict[str, int] = {}
            for (_, item), (name, secret, raw) in zip(entries, infos):
                name = _unique(name, result, counts)
                if secret or (name_says_secret and raw is not None and raw.lower() == "value"):
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
        snapshot = list(islice(items, max_items))
        out = []
        for index, item in enumerate(snapshot):
            if index == 1 and total == 2 and issubclass(kind, (list, tuple)):
                # ("Authorization", "Basic abc"): a secret name as first item hides the second
                first_ok, first = _scalar(snapshot[0])
                if first_ok and isinstance(first, str) and _secret_name(first, budget):
                    out.append(REDACTED)            # the second item is never looked at
                    continue
            out.append(_redact(item, depth + 1, max_depth, max_items, max_string, path, budget))
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
    ``str`` keys), ``list``, ``str``, ``int``, finite ``float``, ``bool`` and ``None``, and
    every string in it can be written as UTF-8.

    * A dict value is replaced by ``"[REDACTED]"`` when its KEY names a secret, whatever the
      type of the value. Keys are compared after Unicode NFKC, after dropping invisible,
      control and combining characters, after folding common Cyrillic and Greek look-alike
      letters, lower-casing and splitting on non-letters and camelCase, so ``apiKey``,
      ``X-Api-Key`` and ``api_key`` are the same. See ``docs/redaction.md`` for the list.
      A key longer than 1000 characters counts as secret.
    * A key that is not a string, number, bool, None or bytes (a tuple, an object ...) counts
      as secret. A bytes key is read as Latin-1 and classified like a string.
    * ``{"name": "DB_PASSWORD", "value": "x"}``: when a ``name`` or ``key`` entry holds a secret
      name, the ``value`` entry is hidden. ``("Authorization", "Basic abc")``: a list or tuple of
      exactly two items whose first item is a secret name has its second item hidden.
    * Strings go through ``sanitize_error_text`` with ``limit=max_string``. A lone surrogate
      becomes U+FFFD.
    * ``int``, finite ``float``, ``bool`` and ``None`` are kept. ``nan`` and ``inf`` become
      the strings ``"nan"``, ``"inf"`` and ``"-inf"``. Ints over 256 bits become a marker.
    * ``bytes`` become ``"<bytes N>"``. Any other object becomes ``"<TypeName>"``; its
      ``__repr__``, ``__str__``, attributes and its class's ``__name__`` property are never
      used, so it cannot leak or raise.
    * ``tuple``, ``set`` and ``frozenset`` become lists (a set keeps its iteration order).
    * Below ``max_depth`` levels the value becomes ``"<max depth>"``; a container with more
      than ``max_items`` items is cut and ends with ``"<N more items omitted>"``; a
      reference to a container that is already being copied becomes ``"<cycle>"``.

    Bounds that cannot be turned off: ``max_depth`` is at most 64; one call looks at no more
    than 10,000 values; the characters it scans (counted after NFKC normalisation, which can
    turn one character into up to 18) add up to no more than 1,000,000, and one string over
    200,000 characters is replaced by a marker without being scanned. Past those, values
    become ``"<string, N characters, not scanned>"`` or ``"<budget exceeded>"``. Timing was
    measured, not proven: see ``tests/test_redact.py``.

    It never raises for any input (``KeyboardInterrupt`` and ``SystemExit`` excepted).
    BEST EFFORT: a secret under a harmless key name and in an unknown shape is NOT detected.
    """
    token = _FOLD_CONTEXT.set(_FoldContext())          # the memo of this call only
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
    finally:
        _FOLD_CONTEXT.reset(token)


# ─── Keyed digest of arguments ────────────────────────────────────────────────

_DIGEST_MAX_NODES = 200_000          # values visited, counting every visit of a shared value
_DIGEST_MAX_CHARS = 16_000_000       # characters of text and keys, counting every visit
_DIGEST_MAX_DEPTH = 100              # nesting levels
_DIGEST_MAX_INT_BITS = 4096          # bigger ints cannot be printed by json
_PURPOSE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{0,31}\Z")
_TOO_BIG = "arguments are too large, too deep or too shared to digest"
_NOT_JSON = "arguments hold a value that canonical JSON cannot represent"


def _snapshot(obj: Any) -> Any:
    """
    Build a fresh copy of ``obj`` out of exact ``dict``, ``list``, ``str``, ``int``, ``float``,
    ``bool`` and ``None`` (a tuple becomes a list), or raise ``TypeError``.

    Only the exact built-in types are accepted, so no method of a user class can run and the
    thing that is serialised is the thing that was inspected. Every visit is counted, shared
    values included, so ``x = [x, x]`` repeated 60 times stops at the budget instead of
    doubling the work at every level. The message never contains a value.
    """
    holder: list[Any] = [None]
    stack: list[tuple[Any, Any, Any, int]] = [(obj, holder, 0, 0)]
    nodes = 0
    chars = 0
    while stack:
        src, parent, slot, depth = stack.pop()
        nodes += 1
        if nodes > _DIGEST_MAX_NODES or depth > _DIGEST_MAX_DEPTH:
            raise TypeError(_TOO_BIG)
        kind = type(src)
        if src is None or kind is bool:
            parent[slot] = src
        elif kind is int:
            bits = src.bit_length()
            if bits > _DIGEST_MAX_INT_BITS:
                raise TypeError(_NOT_JSON)
            chars += bits // 3 + 1
            parent[slot] = src
        elif kind is float:
            if not math.isfinite(src):
                raise TypeError(_NOT_JSON)
            chars += 24
            parent[slot] = src
        elif kind is str:
            chars += len(src)
            if chars > _DIGEST_MAX_CHARS:
                raise TypeError(_TOO_BIG)
            if not src.isascii() and _SURROGATES.search(src):
                raise TypeError(_NOT_JSON)
            parent[slot] = src
        elif kind is list or kind is tuple:
            if nodes + len(stack) + len(src) > _DIGEST_MAX_NODES:     # before copying
                raise TypeError(_TOO_BIG)
            items = list(src)
            if nodes + len(stack) + len(items) > _DIGEST_MAX_NODES:
                raise TypeError(_TOO_BIG)
            copy_: list[Any] = [None] * len(items)
            parent[slot] = copy_
            for index, child in enumerate(items):
                stack.append((child, copy_, index, depth + 1))
        elif kind is dict:
            if nodes + len(stack) + len(src) > _DIGEST_MAX_NODES:     # before copying
                raise TypeError(_TOO_BIG)
            pairs = list(src.items())
            if nodes + len(stack) + len(pairs) > _DIGEST_MAX_NODES:
                raise TypeError(_TOO_BIG)
            copy_dict: dict[str, Any] = {}
            parent[slot] = copy_dict
            for name, child in pairs:
                if type(name) is not str:
                    raise TypeError(_NOT_JSON)
                chars += len(name)
                if chars > _DIGEST_MAX_CHARS:
                    raise TypeError(_TOO_BIG)
                if not name.isascii() and _SURROGATES.search(name):
                    raise TypeError(_NOT_JSON)
                copy_dict[name] = None
                stack.append((child, copy_dict, name, depth + 1))
        else:
            raise TypeError(_NOT_JSON)
        if chars > _DIGEST_MAX_CHARS:
            raise TypeError(_TOO_BIG)
    return holder[0]


def _canonical_json(obj: Any) -> str:
    safe = _snapshot(obj)
    try:
        return json.dumps(
            safe,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except Exception:
        raise TypeError(_NOT_JSON) from None


def _check_purpose(purpose: Any) -> bytes:
    if type(purpose) is not str:
        raise TypeError("purpose must be a str")
    if _PURPOSE.match(purpose) is None:
        raise ValueError("purpose must be a short ASCII identifier (letters, digits, _ . -)")
    return purpose.encode("ascii")


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
    same in JSON. Only the exact types ``dict`` (with ``str`` keys), ``list``, ``tuple``,
    ``str``, ``int``, ``float``, ``bool`` and ``None`` are accepted; no method of any other
    class is ever called. Anything else (bytes, sets, objects, subclasses of the types above,
    nan, inf, non-string keys, strings with lone surrogates, huge ints) raises ``TypeError``,
    and so does input over the limits (200,000 values and 16,000,000 characters, every visit
    counted, 100 levels deep). The error message never contains a value.

    ``purpose`` separates uses: the same arguments digested as ``"args"`` and as ``"result"``
    give different digests. It is added to the HMAC input with a 4 byte length in front, so
    no purpose and payload can be mistaken for another.
    """

    __slots__ = ("_key",)

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def digest(self, obj: Any, purpose: str = "args") -> str:
        """The 64 hex character HMAC-SHA256 of ``purpose`` and the canonical JSON of ``obj``."""
        label = _check_purpose(purpose)
        payload = _canonical_json(obj).encode("ascii")
        message = len(label).to_bytes(4, "big") + label + payload
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def matches(self, obj: Any, digest: Any, purpose: str = "args") -> bool:
        """
        True if ``digest`` is the digest of ``obj`` for ``purpose`` (constant-time compare).
        Returns False, and never raises, for a malformed digest, a bad purpose or an ``obj``
        that ``digest`` would reject.
        """
        try:
            if type(digest) is not str or len(digest) != 64 or not digest.isascii():
                return False
            expected = self.digest(obj, purpose)
            return hmac.compare_digest(expected.encode("ascii"), digest.lower().encode("ascii"))
        except Exception:
            return False

    def __repr__(self) -> str:
        return "<ArgsDigester>"

    def __reduce__(self) -> Any:
        raise TypeError("ArgsDigester holds a secret key and cannot be pickled or copied")


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
