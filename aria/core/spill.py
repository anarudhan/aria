"""
ARIA Spill Store
----------------
Honest handling of large agent and tool output (issue #40).

Before this module, ARIA cut long text with blind slices such as
``str(result)[:300]`` and told nobody. The model saw a sentence that simply
stopped, with no hint that anything was missing and no way to get the rest.

``SpillStore.preview_or_spill(text, max_chars)`` replaces that:

* text that fits is returned unchanged, and nothing is reported as omitted;
* text that does not fit is saved in full to a private file, and the caller
  gets a preview: the head, the tail, and a marker that says exactly how many
  characters were left out, gives an opaque locator, and shows how to read more;
* if the file cannot be saved for any reason, the FULL text stays inline and a
  warning is recorded. Information is never dropped.

Units: every offset and length in this module counts CHARACTERS (Python
``str`` code points), not bytes, so multi-byte text is never split in the
middle of a character.

Security notes:

* The folder is created with mode 0700 and each spill file with mode 0600,
  created exclusively (O_CREAT | O_EXCL, never following symlinks).
* Locators are random 128-bit hex ids, never file paths. ``read`` accepts only
  ids this store instance issued, validates them against a strict pattern, and
  checks that the resolved path is inside the store folder.

Standard library only.
"""

from __future__ import annotations

import os
import re
import secrets
import stat as _stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "READ_TOOL_NAME",
    "SpillError",
    "SpillNotFound",
    "SpillPage",
    "SpillResult",
    "SpillStore",
    "get_default_store",
    "reset_default_store",
    "set_default_store",
]

# Name of the tool a model calls to read the rest of a spilled item.
READ_TOOL_NAME = "read_spilled"

# Locator format: 32 lowercase hex characters (128 random bits). Nothing else
# is ever accepted by read(), so "/", "..", NUL, spaces and so on cannot get in.
_LOCATOR_RE = re.compile(r"[0-9a-f]{32}")
_FILE_SUFFIX = ".spill"

DEFAULT_RETENTION_SECONDS = 24 * 60 * 60   # one day
DEFAULT_READ_CHARS = 2000                  # default page size for read()
DEFAULT_MAX_READ_CHARS = 20_000            # largest page one read() returns
_MAX_WARNINGS = 100                        # warnings list is bounded
_ID_RETRIES = 5                            # tries if a random id ever collides


class SpillError(ValueError):
    """The locator or arguments given to the store are not acceptable."""


class SpillNotFound(SpillError):
    """The locator is well formed but was not issued by this store, or the
    spilled file is gone (for example removed by retention cleanup)."""


@dataclass(frozen=True)
class SpillResult:
    """Outcome of ``preview_or_spill``.

    ``text`` is what should be shown to the model. ``str(result)`` returns it.
    """

    text: str                       # the text to show (full text, or the preview)
    total_chars: int                # length of the original text
    omitted_chars: int              # 0 unless the text was shortened
    spilled: bool                   # True when the full text was saved to disk
    locator: Optional[str] = None   # opaque id, set only when spilled
    warning: Optional[str] = None   # set when saving failed and text stayed inline

    @property
    def truncated(self) -> bool:
        """True when ``text`` is a shortened preview of the original."""
        return self.omitted_chars > 0

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class SpillPage:
    """One slice read back from a spilled item (offsets are characters)."""

    text: str
    offset: int
    end: int            # offset of the first character NOT in this page
    total_chars: int

    @property
    def has_more(self) -> bool:
        return self.end < self.total_chars


class SpillStore:
    """Private on-disk store for output that is too large to show in full."""

    def __init__(
        self,
        base_dir: Union[str, "os.PathLike[str]"],
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        max_read_chars: int = DEFAULT_MAX_READ_CHARS,
    ) -> None:
        if retention_seconds < 0:
            raise ValueError("retention_seconds must be >= 0")
        if max_read_chars < 1:
            raise ValueError("max_read_chars must be >= 1")
        self.base_dir = Path(base_dir)
        self.retention_seconds = float(retention_seconds)
        self.max_read_chars = int(max_read_chars)
        self.warnings: list[str] = []          # newest last, bounded
        self.warnings_total = 0                # includes dropped old ones
        self.spills_total = 0                  # items saved by this instance
        self.omitted_total = 0                 # characters omitted from previews
        self._issued: set[str] = set()
        self._lock = threading.Lock()
        self._ready = False
        self._base_resolved = self.base_dir
        self._ensure_dir()

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def usable(self) -> bool:
        """False when the folder could not be created or is not safe to use."""
        return self._ready or self._ensure_dir()

    def preview_or_spill(self, text: str, max_chars: int) -> SpillResult:
        """Return ``text`` unchanged if it fits in ``max_chars``, else a preview.

        The preview shows about two thirds of ``max_chars`` from the start and
        the rest from the end of the text, then a marker. The marker is extra,
        so the result is longer than ``max_chars`` by the marker's length.
        """
        if max_chars < 0:
            raise ValueError("max_chars must be >= 0")
        total = len(text)
        if total <= max_chars:
            return SpillResult(text=text, total_chars=total, omitted_chars=0, spilled=False)

        head_n = (max_chars * 2 + 2) // 3
        tail_n = max_chars - head_n
        omitted = total - head_n - tail_n

        try:
            locator = self._save(text)
        except Exception as exc:  # disk full, permissions, anything: never drop text
            warning = (
                f"spill failed ({type(exc).__name__}: {exc}); "
                f"kept all {total} characters inline"
            )
            self._warn(warning)
            return SpillResult(
                text=text, total_chars=total, omitted_chars=0, spilled=False, warning=warning
            )

        head = text[:head_n]
        tail = text[total - tail_n:] if tail_n else ""
        marker = (
            f"[... {omitted} characters omitted. This output was SHORTENED: "
            f"showing the first {head_n} and last {tail_n} of {total} characters. "
            f"The full text is saved under spill id {locator}. To read the omitted part, "
            f'call {READ_TOOL_NAME}(locator="{locator}", offset={head_n}, length=2000); '
            f"offset and length count characters. ...]"
        )
        with self._lock:
            self.spills_total += 1
            self.omitted_total += omitted
        return SpillResult(
            text=f"{head}\n{marker}\n{tail}",
            total_chars=total,
            omitted_chars=omitted,
            spilled=True,
            locator=locator,
        )

    def read_page(
        self, locator: str, offset: int = 0, length: int = DEFAULT_READ_CHARS
    ) -> SpillPage:
        """Read ``length`` characters starting at character ``offset``.

        ``length`` is clamped to ``max_read_chars``; check ``page.end`` and
        ``page.has_more`` to see what was returned. Raises ``SpillError`` for a
        bad locator or bad numbers and ``SpillNotFound`` for an unknown id.
        """
        if isinstance(offset, bool) or isinstance(length, bool):
            raise SpillError("offset and length must be integers")
        if not isinstance(offset, int) or not isinstance(length, int):
            raise SpillError("offset and length must be integers")
        if offset < 0:
            raise SpillError("offset must be >= 0")
        if length < 1:
            raise SpillError("length must be >= 1")
        length = min(length, self.max_read_chars)

        path = self._path_for(locator)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        except FileNotFoundError:
            raise SpillNotFound("spilled output not found (it may have expired)") from None
        except OSError as exc:
            raise SpillNotFound(f"spilled output cannot be opened ({type(exc).__name__})") from None
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read()
        full = raw.decode("utf-8", errors="surrogatepass")
        total = len(full)
        end = min(offset + length, total)
        return SpillPage(text=full[offset:end], offset=offset, end=max(end, offset), total_chars=total)

    def read(self, locator: str, offset: int = 0, length: int = DEFAULT_READ_CHARS) -> str:
        """Return the slice ``[offset, offset + length)`` of a spilled item.

        Offsets and lengths count characters. See ``read_page`` for details.
        """
        return self.read_page(locator, offset, length).text

    def cleanup(self, max_age_seconds: Optional[float] = None) -> int:
        """Delete spill files older than ``max_age_seconds`` (default: the
        store's ``retention_seconds``). Returns how many files were removed."""
        age = self.retention_seconds if max_age_seconds is None else float(max_age_seconds)
        if age < 0:
            raise ValueError("max_age_seconds must be >= 0")
        cutoff = time.time() - age
        removed = 0
        for path, st in self._spill_files():
            if st.st_mtime <= cutoff and self._remove(path):
                removed += 1
        return removed

    def clear_all(self) -> int:
        """Delete every spill file in the folder. Returns how many were removed."""
        removed = 0
        for path, _ in self._spill_files():
            if self._remove(path):
                removed += 1
        return removed

    def describe(self) -> dict[str, Any]:
        """Counts and sizes for the health report."""
        files = self._spill_files()
        return {
            "base_dir": str(self.base_dir),
            "usable": self._ready,
            "files": len(files),
            "bytes": sum(st.st_size for _, st in files),
            "spills_this_session": self.spills_total,
            "characters_omitted_from_previews": self.omitted_total,
            "retention_seconds": self.retention_seconds,
            "warnings": len(self.warnings),
            "warnings_total": self.warnings_total,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _warn(self, message: str) -> None:
        with self._lock:
            self.warnings.append(message)
            self.warnings_total += 1
            if len(self.warnings) > _MAX_WARNINGS:
                del self.warnings[: len(self.warnings) - _MAX_WARNINGS]

    def _ensure_dir(self) -> bool:
        """Create the folder privately, or check an existing one is safe."""
        try:
            try:
                os.mkdir(self.base_dir, 0o700)
            except FileExistsError:
                pass
            except FileNotFoundError:
                os.makedirs(self.base_dir.parent, exist_ok=True)
                os.mkdir(self.base_dir, 0o700)
            st = os.lstat(self.base_dir)
            if not _stat.S_ISDIR(st.st_mode):
                raise OSError(f"{self.base_dir} is not a real directory")
            getuid = getattr(os, "geteuid", None)
            if getuid is not None and st.st_uid != getuid():
                raise OSError(f"{self.base_dir} is owned by another user")
            if os.name == "posix" and _stat.S_IMODE(st.st_mode) != 0o700:
                os.chmod(self.base_dir, 0o700)
            self._base_resolved = self.base_dir.resolve()
            self._ready = True
            return True
        except Exception as exc:
            self._ready = False
            self._warn(f"spill folder unusable ({type(exc).__name__}: {exc})")
            return False

    def _save(self, text: str) -> str:
        """Write text to a new exclusive 0600 file and return its locator."""
        if not self._ready and not self._ensure_dir():
            raise OSError("spill folder is not usable")
        data = text.encode("utf-8", errors="surrogatepass")
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        )
        for _ in range(_ID_RETRIES):
            locator = secrets.token_hex(16)
            path = self.base_dir / (locator + _FILE_SUFFIX)
            try:
                fd = os.open(path, flags, 0o600)
            except FileExistsError:
                continue  # never overwrite; pick another random id
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fd = -1  # fdopen owns it now
                    fh.write(data)
            except BaseException:
                if fd != -1:
                    os.close(fd)
                self._remove(path)  # no half-written file left behind
                raise
            with self._lock:
                self._issued.add(locator)
            return locator
        raise OSError("could not create a unique spill file")

    def _path_for(self, locator: Any) -> Path:
        """Validate a locator and return its file path, inside the folder."""
        if not isinstance(locator, str) or not _LOCATOR_RE.fullmatch(locator):
            raise SpillError("invalid spill locator")
        with self._lock:
            issued = locator in self._issued
        if not issued:
            raise SpillNotFound("unknown spill locator")
        if not self._ready and not self._ensure_dir():
            raise SpillNotFound("spill folder is not usable")
        path = self.base_dir / (locator + _FILE_SUFFIX)
        # Belt and braces: the pattern already rules out separators and "..",
        # but also confirm where the path really points (symlinks included).
        resolved = path.resolve()
        if resolved.parent != self._base_resolved:
            raise SpillError("invalid spill locator")
        return path

    def _spill_files(self) -> list[tuple[Path, os.stat_result]]:
        out: list[tuple[Path, os.stat_result]] = []
        try:
            names = os.listdir(self.base_dir)
        except OSError:
            return out
        for name in names:
            if not name.endswith(_FILE_SUFFIX) or not _LOCATOR_RE.fullmatch(name[: -len(_FILE_SUFFIX)]):
                continue  # never touch files we did not create
            path = self.base_dir / name
            try:
                st = os.lstat(path)
            except OSError:
                continue
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            out.append((path, st))
        return out

    def _remove(self, path: Path) -> bool:
        try:
            os.unlink(path)
        except OSError:
            return False
        with self._lock:
            self._issued.discard(path.name[: -len(_FILE_SUFFIX)])
        return True


# ─── Default store ────────────────────────────────────────────────────────────

_default_store: Optional[SpillStore] = None
_default_lock = threading.Lock()


def _default_dir() -> Path:
    """ARIA_SPILL_DIR if set, else a per-user folder in the system temp area."""
    env = os.environ.get("ARIA_SPILL_DIR")
    if env:
        return Path(env)
    getuid = getattr(os, "getuid", None)
    suffix = str(getuid()) if getuid is not None else "user"
    return Path(tempfile.gettempdir()) / f"aria-spill-{suffix}"


def get_default_store() -> SpillStore:
    """Process-wide store used by agents that were not given one.

    Location: ``$ARIA_SPILL_DIR`` or ``<system temp>/aria-spill-<uid>``, never
    inside the repository. If that folder is not safe to use (for example it is
    owned by someone else), a fresh private temporary folder is used instead.
    Old files are swept once, when the store is first created.
    """
    global _default_store
    with _default_lock:
        if _default_store is None:
            store = SpillStore(_default_dir())
            if not store.usable:
                failed = store.warnings
                store = SpillStore(tempfile.mkdtemp(prefix="aria-spill-"))
                store.warnings[:0] = failed
                store.warnings_total += len(failed)
            store.cleanup()
            _default_store = store
        return _default_store


def set_default_store(store: Optional[SpillStore]) -> None:
    """Replace the process-wide default store (``None`` resets it)."""
    global _default_store
    with _default_lock:
        _default_store = store


def reset_default_store() -> None:
    """Forget the default store so the next ``get_default_store`` makes a new one."""
    set_default_store(None)
