"""
ARIA Permission Scope Matching
------------------------------
Decides whether a granted scope covers a requested scope.

A scope is a string that names the resource a permission applies to.
ARIA uses three kinds, and each has its own rule:

  "*"         Everything. Only a grant can use it. A specific grant never
              covers a request for "*".

  Path        A file system path such as /home/user/project or ~/notes.
              ~ is expanded and .. segments and symlinks are resolved first.
              A grant for a folder covers that folder and everything inside
              it, matched on whole path parts only. /a/b covers /a/b/c but
              never /a/bc.

  URL         Anything containing "://", such as https://example.com/api.
              Scheme, host and port must be equal. The path is matched on
              whole segments after dot segments are removed. A request that
              carries embedded credentials (user:pass@host) is never covered
              unless the grant carries the same ones. URLs that different
              software reads differently are refused: backslashes, control
              characters or spaces, ";" in the path, empty path segments, and
              the encoded forms %2f, %5c, %25, %00, %c0, %c1 and %u in the path, and
              non-ASCII characters in the path (percent-encode them).

  Anything    Ids and names, for example a vulnerability id or a capability
  else        id. These match only when equal.

Grants are stored in canonical form. A path grant becomes an absolute real
path when it is created (see canonical_scope), and relative path grants such as
"." or "./src" are refused because they would mean something different every
time the working folder changes. At check time a path grant must still resolve
to itself, so replacing a granted folder with a symlink cannot widen the grant.

Every failure to parse or resolve a scope means "not covered". The rule is
fail closed.

Known limit: a path is checked at one moment and used at a later one. A
symlink could be swapped in between. Code that acts on a path after a
permission check should act on the resolved path (see resolve_path) and keep
the gap as short as it can.
"""

from __future__ import annotations

import os
import posixpath
import re
from typing import Optional
from urllib.parse import unquote, urlsplit

WILDCARD = "*"

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443, "ftp": 21}
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_url(scope: str) -> bool:
    return "://" in scope


def _is_path(scope: str) -> bool:
    return (
        scope.startswith(("/", "~", "./", "../", "\\\\"))
        or scope in (".", "..")
        or bool(_WINDOWS_DRIVE.match(scope))
    )


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _check_grant_url(url: str) -> None:
    if not _url_is_unambiguous(url):
        raise ValueError(f"URL scope {url!r} is ambiguous (see aria/permissions/scope.py)")
    parts = urlsplit(url)
    if not parts.hostname:
        raise ValueError(f"URL scope {url!r} has no host")
    if parts.query or parts.fragment or "?" in url or "#" in url:
        raise ValueError("URL scopes cannot contain a query or a fragment")
    if any(seg in (".", "..") for seg in unquote(parts.path).split("/")):
        raise ValueError("URL scopes cannot contain dot segments")


def is_relative_path(scope: str) -> bool:
    """True for path scopes that depend on the working folder, such as ./src."""
    return scope in (".", "..") or scope.startswith(("./", "../", ".\\", "..\\"))


def is_path_scope(scope: str) -> bool:
    return isinstance(scope, str) and _is_path(scope)


def canonical_scope(scope: str) -> str:
    """
    The form in which a scope is stored in a grant. Call it when a grant is
    created and nowhere else: resolving again later, for example when a saved
    grant is loaded, would let a swapped symlink rewrite the grant.

    Path scopes become absolute real paths (~ expanded, .. and symlinks
    resolved). URL scopes are checked and kept as written. Other scopes are
    returned unchanged. Raises ValueError for anything that has no stable
    meaning: control characters, relative or unresolvable paths, and URLs
    that are ambiguous or carry a query, a fragment or dot segments.
    """
    if not isinstance(scope, str):
        raise ValueError("Scope must be a string")
    if _CONTROL_CHARS.search(scope):
        raise ValueError("Scope contains control characters")
    if _is_path(scope):
        if is_relative_path(scope):
            raise ValueError(
                f"Relative path scope {scope!r} is not allowed. "
                "Use an absolute path (or ~/...) so the grant means the same thing later."
            )
        resolved = resolve_path(scope)
        if resolved is None:
            raise ValueError(f"Path scope {scope!r} cannot be resolved to an absolute path")
        return resolved
    if _is_url(scope):
        _check_grant_url(scope)
    return scope


def grant_is_stable(scope: str) -> bool:
    """
    A stored path grant must still resolve to itself. If a granted folder was
    later replaced by a symlink, it no longer does, and the grant stops
    matching instead of silently following the link.
    """
    if not isinstance(scope, str):
        return False
    if not _is_path(scope):
        return True
    return resolve_path(scope) == os.path.normcase(scope)


def resolve_path(scope: str) -> Optional[str]:
    """
    Return the absolute, symlink-free form of a path scope, or None if it
    cannot be resolved safely. Callers that act on a path after a permission
    check should use this value, not the original string.
    """
    if "\x00" in scope:
        return None
    try:
        expanded = os.path.expanduser(scope)
        # "~nosuchuser/x" and "C:/x" on POSIX stay relative here. Resolving
        # them would silently attach the working folder, so refuse them.
        if not os.path.isabs(expanded):
            return None
        return os.path.normcase(os.path.realpath(expanded))
    except (ValueError, OSError, RuntimeError):
        return None


def _path_covers(granted: str, requested: str) -> bool:
    granted_real = resolve_path(granted)
    requested_real = resolve_path(requested)
    if granted_real is None or requested_real is None:
        return False
    if requested_real == granted_real:
        return True
    prefix = granted_real if granted_real.endswith(os.sep) else granted_real + os.sep
    return requested_real.startswith(prefix)


def _normalise_url_path(path: str) -> str:
    # Decode once so %2e%2e cannot hide a parent segment, then remove dot
    # segments. posixpath.normpath keeps a leading "//" so collapse it first.
    decoded = unquote(path)
    if not decoded.startswith("/"):
        decoded = "/" + decoded
    normalised = posixpath.normpath("/" + decoded.lstrip("/"))
    return normalised


_CONTROL_OR_SPACE = re.compile(r"[\x00-\x20\x7f-\x9f]")
_ENCODED_AMBIGUOUS = re.compile(r"%(2f|5c|25|00|c0|c1|u)", re.IGNORECASE)


def _url_is_unambiguous(url: str) -> bool:
    """Refuse URLs that different clients and servers read differently."""
    if _CONTROL_OR_SPACE.search(url) or "\\" in url:
        return False
    try:
        path = urlsplit(url).path
    except ValueError:
        return False
    if not path.isascii():
        return False  # write non-ASCII path characters percent-encoded
    return not (";" in path or "//" in path or _ENCODED_AMBIGUOUS.search(path))


def _url_parts(url: str):
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname
    if not scheme or not host:
        return None
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)
    return scheme, host.lower(), port, parts.username, parts.password, parts.path


def _url_covers(granted: str, requested: str) -> bool:
    if not (_url_is_unambiguous(granted) and _url_is_unambiguous(requested)):
        return False
    try:
        granted_parts = _url_parts(granted)
        requested_parts = _url_parts(requested)
    except ValueError:
        return False
    if granted_parts is None or requested_parts is None:
        return False

    g_scheme, g_host, g_port, g_user, g_pass, g_path = granted_parts
    r_scheme, r_host, r_port, r_user, r_pass, r_path = requested_parts

    if (g_scheme, g_host, g_port) != (r_scheme, r_host, r_port):
        return False
    if (r_user, r_pass) != (g_user, g_pass):
        return False

    granted_path = _normalise_url_path(g_path)
    requested_path = _normalise_url_path(r_path)
    if granted_path == "/":
        return True
    if requested_path == granted_path:
        return True
    return requested_path.startswith(granted_path.rstrip("/") + "/")


def scope_covers(granted: str, requested: str) -> bool:
    """True if a grant for `granted` allows an action on `requested`."""
    if not isinstance(granted, str) or not isinstance(requested, str):
        return False
    if granted == WILDCARD:
        return True
    if granted == requested:
        return True
    if requested == WILDCARD:
        return False

    if _is_url(granted):
        return _is_url(requested) and _url_covers(granted, requested)
    if _is_path(granted):
        if is_relative_path(granted):
            return False
        return _is_path(requested) and _path_covers(granted, requested)
    return False
