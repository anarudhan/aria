"""
ARIA Zero-Trust Permission Engine
===================================
The absolute guardian of user trust.

The single most important rule in ARIA: it never does anything a user
has not explicitly permitted. Not even things that seem harmless.
Not even things other agents in the coalition have permission to do.
Permissions are personal, scoped, time-bounded, and fully revokable.

Design philosophy:
    - Default is DENY. Everything starts refused.
    - Permissions are granted per capability, per user, per scope
    - Every permission has an expiry (never granted forever unless user insists)
    - Every permission check is logged in an append-only, hash-chained audit
      trail that verify_chain() checks (memory only and unanchored: see
      AuditEntry)
    - ARIA cannot grant itself permissions — only users can
    - Self-modification is an exception: ARIA can patch SECURITY FLAWS only,
      and even then it logs every change and users can review/revert

Permission Model (four dimensions):
    WHO    : user_id — whose permission is this
    WHAT   : capability — what action or resource
    SCOPE  : the specific resource (e.g. a single folder, not all files)
    WHEN   : expiry — how long it lasts

Capability categories:
    FILE_READ        Read files from disk
    FILE_WRITE       Write/create files on disk
    NETWORK          Make HTTP requests / API calls
    EMAIL_READ       Read emails
    EMAIL_SEND       Send emails
    TEAMS_READ       Read Teams messages
    TEAMS_POST       Post to Teams channels
    CODE_EXECUTE     Run code in a subprocess
    MEMORY_WRITE     Store data in ARIA's memory
    SELF_MODIFY      ARIA modifying its own code (security patches only)
    AGENT_SPAWN      Create a new sub-agent
    SYSTEM_INFO      Read system/environment information
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Optional

from .scope import (
    canonical_scope,
    grant_is_stable,
    scope_covers,
)

# ─── Capability Definitions ───────────────────────────────────────────────────

class Capability(Enum):
    FILE_READ       = "file_read"
    FILE_WRITE      = "file_write"
    NETWORK         = "network"
    EMAIL_READ      = "email_read"
    EMAIL_SEND      = "email_send"
    TEAMS_READ      = "teams_read"
    TEAMS_POST      = "teams_post"
    CODE_EXECUTE    = "code_execute"
    MEMORY_WRITE    = "memory_write"
    SELF_MODIFY     = "self_modify"         # Guarded: security patches only
    AGENT_SPAWN     = "agent_spawn"
    SYSTEM_INFO     = "system_info"
    CALENDAR_READ   = "calendar_read"
    CALENDAR_WRITE  = "calendar_write"
    DOCUMENT_CREATE = "document_create"
    WEB_SEARCH      = "web_search"
    WEB_BROWSE      = "web_browse"


# Human-readable descriptions shown during permission requests
CAPABILITY_DESCRIPTIONS = {
    Capability.FILE_READ:       "Read files from disk",
    Capability.FILE_WRITE:      "Create or modify files on disk",
    Capability.NETWORK:         "Make network requests to external services",
    Capability.EMAIL_READ:      "Read your emails from Outlook",
    Capability.EMAIL_SEND:      "Send emails on your behalf",
    Capability.TEAMS_READ:      "Read your Microsoft Teams messages",
    Capability.TEAMS_POST:      "Post messages to Microsoft Teams on your behalf",
    Capability.CODE_EXECUTE:    "Execute code in a sandboxed environment",
    Capability.MEMORY_WRITE:    "Store information in ARIA's memory",
    Capability.SELF_MODIFY:     "Patch ARIA's own code (security fixes only)",
    Capability.AGENT_SPAWN:     "Create specialized sub-agents for complex tasks",
    Capability.SYSTEM_INFO:     "Read system and environment information",
    Capability.CALENDAR_READ:   "Read your calendar events",
    Capability.CALENDAR_WRITE:  "Create or modify calendar events",
    Capability.DOCUMENT_CREATE: "Create documents (Word, PDF, etc.)",
    Capability.WEB_SEARCH:      "Search the web for information",
    Capability.WEB_BROWSE:      "Browse specific web pages",
}

DEFAULT_EXPIRY_HOURS = {
    Capability.FILE_READ:       8,
    Capability.FILE_WRITE:      4,
    Capability.NETWORK:         24,
    Capability.EMAIL_READ:      12,
    Capability.EMAIL_SEND:      1,     # Short — email sending is irreversible
    Capability.TEAMS_READ:      12,
    Capability.TEAMS_POST:      1,     # Short — posting is visible to others
    Capability.CODE_EXECUTE:    4,
    Capability.MEMORY_WRITE:    24,
    Capability.SELF_MODIFY:     0.5,   # 30 minutes — very short
    Capability.AGENT_SPAWN:     8,
    Capability.SYSTEM_INFO:     24,
    Capability.CALENDAR_READ:   12,
    Capability.CALENDAR_WRITE:  2,
    Capability.DOCUMENT_CREATE: 8,
    Capability.WEB_SEARCH:      24,
    Capability.WEB_BROWSE:      8,
}

RISK_LEVEL = {
    Capability.FILE_READ:       "low",
    Capability.FILE_WRITE:      "medium",
    Capability.NETWORK:         "low",
    Capability.EMAIL_READ:      "medium",
    Capability.EMAIL_SEND:      "high",
    Capability.TEAMS_READ:      "medium",
    Capability.TEAMS_POST:      "high",
    Capability.CODE_EXECUTE:    "high",
    Capability.MEMORY_WRITE:    "low",
    Capability.SELF_MODIFY:     "critical",
    Capability.AGENT_SPAWN:     "medium",
    Capability.SYSTEM_INFO:     "low",
    Capability.CALENDAR_READ:   "low",
    Capability.CALENDAR_WRITE:  "medium",
    Capability.DOCUMENT_CREATE: "low",
    Capability.WEB_SEARCH:      "low",
    Capability.WEB_BROWSE:      "low",
}


# ─── Permission Record ────────────────────────────────────────────────────────

@dataclass
class PermissionGrant:
    """A single permission grant from a user to ARIA."""
    grant_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id:        str = ""
    capability:     Capability = Capability.MEMORY_WRITE
    scope:          str = "*"          # Specific resource or "*" for all
    granted_at:     float = field(default_factory=time.time)
    expires_at:     float = 0.0        # 0 = no expiry (only if user chooses)
    granted_by:     str = "user"       # Always "user" — ARIA never self-grants
    task_context:   str = ""           # Why was this permission granted
    use_count:      int = 0
    max_uses:       Optional[int] = None   # None = unlimited

    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at

    @property
    def is_exhausted(self) -> bool:
        if self.max_uses is None:
            return False
        return self.use_count >= self.max_uses

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.is_exhausted

    def matches_scope(self, requested_scope: str) -> bool:
        """
        True if this grant covers the requested scope.

        Paths and URLs match on whole parts, never on a text prefix, so a
        grant for /home/u/project does not cover /home/u/project-secrets.
        Other scopes (ids, names) must be equal. See aria/permissions/scope.py.
        """
        if not grant_is_stable(self.scope):
            return False
        return scope_covers(self.scope, requested_scope)


# ─── Audit Entry ─────────────────────────────────────────────────────────────

# Value of prev_hash for the first entry of a chain, and of the engine's head
# hash while the chain is empty.
GENESIS_HASH = "genesis"

# Longest `detail` string record_audit() accepts, in UTF-8 bytes.
MAX_AUDIT_DETAIL_BYTES = 8 * 1024


@dataclass
class AuditEntry:
    """
    One record in the audit chain: a permission check, grant, denial or
    revoke, or an event added with PermissionEngine.record_audit().

    What the chain gives you. compute_hash() covers EVERY field except
    entry_hash itself, as the full SHA256 hex digest (64 characters) of a
    canonical JSON encoding. prev_hash holds the previous entry's entry_hash,
    so the entries form a chain, and PermissionEngine.verify_chain() walks it.
    An edit, reorder, insertion, duplication or removal (including from the
    end, which is checked against the engine's head hash) makes verify_chain()
    fail.

    What it does NOT give you:
      - The entries and the head hash live in process memory and are not
        anchored anywhere else. Whoever can rewrite process memory can edit
        entries, recompute every hash and the head value, and verify_chain()
        will then pass.
      - The chain is lost on restart. Saving it to disk is issue #68.
    See "The permission engine internals" in docs/architecture.md.

    `detail` is a canonical JSON string that the CALLER must already have
    redacted. The engine does not redact it.
    """
    entry_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp:  float = field(default_factory=time.time)
    user_id:    str = ""
    capability: str = ""
    scope:      str = ""
    action:     str = ""       # "granted", "denied", "expired", "revoked"
    reason:     str = ""
    agent_id:   str = ""
    task_id:    str = ""
    # entry_hash of the previous entry ("genesis" for the first one).
    prev_hash:  str = ""
    # Pipeline fields (issue #56, design 0042 section 2(f)).
    call_id:    str = ""
    stage:      str = ""
    detail:     str = ""       # canonical JSON, already redacted by the caller
    args_digest: str = ""
    # This entry's own hash, set when the entry is appended to the chain. It
    # is the only field that compute_hash() leaves out.
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """
        Full SHA256 hex digest (64 characters) of every field except
        entry_hash. The fields come from dataclasses.fields() of the CLASS
        (an instance cannot hide a field by shadowing __dataclass_fields__),
        so a field added later is covered automatically. The encoding is
        canonical JSON of a dict of all fields (sorted keys, no whitespace,
        non-ASCII written as is), turned into bytes with UTF-8 and
        "surrogatepass". It is unambiguous: one field's text cannot run into
        the next one's, and two different strings never give the same bytes.
        That includes a lone surrogate pair versus the one character it would
        spell if joined (U+D83D U+DE00 versus U+1F600). Escaping everything
        to ASCII would not have kept those apart.

        The hash of an entry that holds non-ASCII text is different from the
        one an earlier version computed. Nothing is saved to disk, so that
        only matters inside one process.

        Raises TypeError or ValueError if a field holds something JSON cannot
        encode (verify_chain() reports that as a bad entry).
        """
        data = {
            f.name: getattr(self, f.name)
            for f in fields(type(self))
            if f.name != "entry_hash"
        }
        text = json.dumps(
            data, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        )
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class ChainReport:
    """
    Result of PermissionEngine.verify_chain().

    ok:              True if the whole chain checks out.
    checked:         how many entries, counted from the start, passed before
                     the first problem (all of them when ok).
    first_bad_index: index in the audit list of the first entry that failed,
                     or None when ok. If an attacker rewrites an entry AND
                     recomputes its entry_hash, the break shows at the NEXT
                     entry (its prev_hash no longer matches). If the last entry
                     does not lead to the engine's head hash (entries removed
                     from the end, or the last one rewritten), this is the
                     index of the last entry present, or 0 if the list is empty.
    reason:          short human-readable explanation ("" when ok).
    """
    ok: bool
    checked: int
    first_bad_index: Optional[int]
    reason: str


def _check_text(name: str, value: object) -> str:
    """Return an exact `str` copy of value, or raise TypeError."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, not {type(value).__name__}")
    return str.__str__(value)  # exact str, even when value is a str subclass


_MAX_DETAIL_DEPTH = 32


def _check_json_value(value: object, path: str, depth: int = 0) -> None:
    """Allow only plain JSON data: str-keyed dicts, lists, str, int, finite float, bool, None."""
    if depth > _MAX_DETAIL_DEPTH:
        raise ValueError(f"detail is nested more than {_MAX_DETAIL_DEPTH} levels deep at {path}")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError(f"detail has a non-finite number at {path}")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"detail has a non-str key {k!r} ({type(k).__name__}) at {path}"
                )
            _check_json_value(v, f"{path}.{k}", depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_json_value(v, f"{path}[{i}]", depth + 1)
        return
    raise TypeError(f"detail is not JSON-serialisable: {type(value).__name__} at {path}")


def _canonical_detail(detail: object) -> str:
    """Turn record_audit's `detail` argument into the stored canonical string."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        text = str.__str__(detail)
    else:
        if not isinstance(detail, (dict, list, tuple)):
            raise TypeError(
                "detail must be None, a str, a dict or a list, "
                f"not {type(detail).__name__}"
            )
        _check_json_value(detail, "detail")
        try:
            text = json.dumps(
                detail, sort_keys=True, ensure_ascii=True,
                separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise TypeError(f"detail is not JSON-serialisable: {exc}") from exc
    size = len(text.encode("utf-8", "surrogatepass"))
    if size > MAX_AUDIT_DETAIL_BYTES:
        raise ValueError(
            f"detail is {size} bytes; the limit is {MAX_AUDIT_DETAIL_BYTES} bytes. "
            "It is never truncated silently: shorten it."
        )
    return text


# ─── Permission Engine ────────────────────────────────────────────────────────

class PermissionEngine:
    """
    ARIA's zero-trust permission guardian.

    Every capability check that is made flows through this class, and it
    denies by default: with no valid grant, check() returns False and logs
    the denial.

    What this class can and cannot do. It ANSWERS questions. It cannot stop
    code that never asks. A tool or agent body that does not call check() or
    require() is not blocked by it. Today these places ask: the orchestrator
    (required_capabilities and memory writes), the web agent, the improvement
    engine (AGENT_SPAWN, CODE_EXECUTE) and the patcher (SELF_MODIFY). Agent
    tool calls and most agent bodies do not. Enforcing the check at tool call
    time is tracked in issue #42. See "What is and is not enforced today" in
    docs/architecture.md.

    Audit trail. Grants, checks, denials and revokes, and events added with
    record_audit(), go into one hash chain. Each entry's SHA256 covers every
    field, and verify_chain() detects edits, reordering, insertion,
    duplication and removal (also from the end). The chain is in memory only
    and unanchored: someone who can rewrite process memory can recompute it
    all, and it is gone after a restart (saving it is issue #68).

    The intent is that ARIA does nothing with user data or resources without
    an explicit grant, apart from: reading its own source for self-analysis,
    using in-memory data from the current session, logging to the audit
    trail, and showing text responses to the user.
    """

    def __init__(self, persist_path: Optional[str] = None):
        # {user_id: {capability: [PermissionGrant]}}
        self._grants: dict[str, dict[str, list[PermissionGrant]]] = {}
        self._audit: list[AuditEntry] = []
        self._last_hash = GENESIS_HASH   # head of the chain
        # Guards the append plus the head update, so threads cannot fork the chain.
        self._audit_lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None

        # Self-modify has an extra confirmation layer
        self._self_modify_confirmations: dict[str, dict] = {}

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Permission Checking (callers must ask: see "What is and is not enforced") ─

    def check(
        self,
        user_id:    str,
        capability: Capability,
        scope:      str = "*",
        agent_id:   str = "aria",
        task_id:    str = "",
    ) -> bool:
        """
        Check if a user has granted ARIA permission for a capability.

        Code that touches user data or resources is meant to call this
        first. The engine cannot make it do so: callers that skip the check
        are not blocked (see the class docstring and docs/architecture.md).

        Returns True only if a valid, non-expired, in-scope grant exists.
        Everything else returns False and logs the denial.
        """
        grants = self._grants.get(user_id, {}).get(capability.value, [])

        for grant in grants:
            if grant.is_valid and grant.matches_scope(scope):
                grant.use_count += 1
                self._audit_log(
                    user_id=user_id, capability=capability.value,
                    scope=scope, action="allowed", agent_id=agent_id,
                    task_id=task_id,
                    reason=f"Grant {grant.grant_id} (uses={grant.use_count})"
                )
                self._save()
                return True

        # Denied — find out why
        reason = self._denial_reason(user_id, capability, scope)
        self._audit_log(
            user_id=user_id, capability=capability.value,
            scope=scope, action="denied", agent_id=agent_id,
            task_id=task_id, reason=reason
        )
        return False

    def require(
        self,
        user_id:    str,
        capability: Capability,
        scope:      str = "*",
        agent_id:   str = "aria",
        task_id:    str = "",
    ) -> None:
        """
        Like check() but raises PermissionDeniedError if not granted.
        Use this inside tool functions for clean error handling.
        """
        if not self.check(user_id, capability, scope, agent_id, task_id):
            raise PermissionDeniedError(
                user_id=user_id,
                capability=capability,
                scope=scope,
                description=CAPABILITY_DESCRIPTIONS[capability],
                risk=RISK_LEVEL[capability],
            )

    # ── Granting Permissions (user action only) ───────────────────────────────

    def grant(
        self,
        user_id:       str,
        capability:    Capability,
        scope:         str = "*",
        hours:         Optional[float] = None,
        max_uses:      Optional[int] = None,
        task_context:  str = "",
    ) -> PermissionGrant:
        """
        Grant a permission. This represents the user explicitly saying yes.

        In a real deployment, this is called ONLY from:
          - The UI when the user clicks "Allow"
          - The API when the user sends a permission grant request
          - Never from internal ARIA code

        The hours parameter defaults to the capability's safe default.
        Users can override it but ARIA will warn about long-lived grants.
        """
        # Stored in canonical form. Raises ValueError for a relative path.
        scope = canonical_scope(scope)

        effective_hours = hours if hours is not None else DEFAULT_EXPIRY_HOURS[capability]
        expires_at = time.time() + (effective_hours * 3600) if effective_hours > 0 else 0

        grant = PermissionGrant(
            user_id      = user_id,
            capability   = capability,
            scope        = scope,
            granted_at   = time.time(),
            expires_at   = expires_at,
            granted_by   = "user",
            task_context = task_context,
            max_uses     = max_uses,
        )

        if user_id not in self._grants:
            self._grants[user_id] = {}
        if capability.value not in self._grants[user_id]:
            self._grants[user_id][capability.value] = []

        self._grants[user_id][capability.value].append(grant)

        expiry_desc = f"{effective_hours}h" if expires_at else "no expiry"
        scope_desc  = f" for scope {scope!r}" if scope != "*" else ""
        print(
            f"[Permission] ✅ Granted '{capability.value}'{scope_desc} "
            f"to user '{user_id}' (expires: {expiry_desc})"
        )

        self._audit_log(
            user_id=user_id, capability=capability.value, scope=scope,
            action="granted", reason=f"User granted {expiry_desc}"
        )
        self._save()
        return grant

    # ── Revoking Permissions ──────────────────────────────────────────────────

    def revoke(self, user_id: str, capability: Capability, scope: str = "*") -> int:
        """
        Revoke all grants for a user/capability/scope combination.
        Returns number of grants revoked.
        """
        names = {scope}
        try:
            names.add(canonical_scope(scope))  # any spelling of the same path
        except ValueError:
            pass  # not canonical, so only an exact text match can revoke it
        grants = self._grants.get(user_id, {}).get(capability.value, [])
        before = len(grants)
        self._grants[user_id][capability.value] = [
            g for g in grants
            if not ((isinstance(g.scope, str) and g.scope in names) or scope == "*")
        ]
        revoked = before - len(self._grants[user_id][capability.value])
        if revoked:
            print(f"[Permission] 🔒 Revoked {revoked} grant(s) for '{capability.value}' from '{user_id}'")
            self._audit_log(
                user_id=user_id, capability=capability.value, scope=scope,
                action="revoked", reason="User revoked permission"
            )
            self._save()
        return revoked

    def revoke_all(self, user_id: str) -> int:
        """Revoke every permission for a user — full reset."""
        count = sum(len(v) for v in self._grants.get(user_id, {}).values())
        self._grants[user_id] = {}
        print(f"[Permission] 🔒 Revoked ALL {count} permissions for user '{user_id}'")
        self._audit_log(
            user_id=user_id, capability="ALL", scope="*",
            action="revoked_all", reason="User revoked all permissions"
        )
        self._save()
        return count

    def expire_stale(self):
        """Remove expired grants. Call periodically."""
        expired_count = 0
        for user_id, caps in self._grants.items():
            for cap_name, grants in caps.items():
                before = len(grants)
                caps[cap_name] = [g for g in grants if not g.is_expired]
                expired_count += before - len(caps[cap_name])
        if expired_count:
            print(f"[Permission] 🕐 Expired {expired_count} stale grants")

    # ── Self-Modify Special Flow ──────────────────────────────────────────────

    def request_self_modify_permission(
        self,
        user_id:     str,
        patch_id:    str,
        description: str,
        files:       list[str],
        severity:    str,
    ) -> str:
        """
        Request permission for a specific security patch.
        Returns a confirmation token the user must acknowledge.

        Self-modification has a separate, extra-explicit flow:
          1. ARIA detects a security flaw
          2. ARIA calls this method (does NOT modify code yet)
          3. A pending request is created
          4. User sees the request with full details of what will change
          5. User calls confirm_self_modify() to approve
          6. ARIA applies the patch and logs every byte changed
        """
        token = str(uuid.uuid4())[:16]
        self._self_modify_confirmations[token] = {
            "user_id":     user_id,
            "patch_id":    patch_id,
            "description": description,
            "files":       files,
            "severity":    severity,
            "requested_at": time.time(),
            "expires_at":  time.time() + 3600,   # 1 hour to confirm
        }
        print(
            f"\n[ARIA Self-Modify Request]\n"
            f"  Patch ID:    {patch_id}\n"
            f"  Severity:    {severity.upper()}\n"
            f"  Description: {description}\n"
            f"  Files:       {', '.join(files)}\n"
            f"  Token:       {token}\n"
            f"  Action:      Call permission_engine.confirm_self_modify('{token}') to approve\n"
            f"               Or call permission_engine.deny_self_modify('{token}') to reject\n"
        )
        self._audit_log(
            user_id=user_id, capability="self_modify", scope=str(files),
            action="requested", reason=f"Security patch {patch_id}: {description}"
        )
        return token

    def confirm_self_modify(self, token: str, user_id: str) -> PermissionGrant:
        """User explicitly confirms a self-modification patch."""
        req = self._self_modify_confirmations.get(token)
        if not req:
            raise ValueError(f"Unknown or expired confirmation token: {token}")
        if req["user_id"] != user_id:
            raise ValueError("Token does not belong to this user")
        if time.time() > req["expires_at"]:
            raise ValueError("Confirmation token has expired")

        del self._self_modify_confirmations[token]
        grant = self.grant(
            user_id      = user_id,
            capability   = Capability.SELF_MODIFY,
            scope        = req["patch_id"],
            hours        = 0.25,     # 15 minutes to apply the patch
            max_uses     = 1,        # One-time use
            task_context = req["description"],
        )
        print(f"[Permission] ✅ Self-modify confirmed for patch '{req['patch_id']}'")
        return grant

    def deny_self_modify(self, token: str, user_id: str):
        """User rejects a self-modification request."""
        req = self._self_modify_confirmations.pop(token, None)
        if req:
            print(f"[Permission] 🚫 Self-modify denied for patch '{req.get('patch_id')}'")
            self._audit_log(
                user_id=user_id, capability="self_modify", scope=str(req.get("files", [])),
                action="denied", reason="User rejected self-modification request"
            )

    # ── User-Facing Permission Summary ────────────────────────────────────────

    def list_grants(self, user_id: str) -> list[dict]:
        """Return all active grants for a user in a readable format."""
        results = []
        for cap_name, grants in self._grants.get(user_id, {}).items():
            for g in grants:
                if g.is_valid:
                    remaining = ""
                    if g.expires_at:
                        secs = g.expires_at - time.time()
                        remaining = f"{secs/3600:.1f}h remaining"
                    else:
                        remaining = "no expiry"
                    results.append({
                        "grant_id":   g.grant_id,
                        "capability": cap_name,
                        "scope":      g.scope,
                        "expires":    remaining,
                        "uses":       f"{g.use_count}/{g.max_uses or '∞'}",
                        "risk":       RISK_LEVEL.get(Capability(cap_name), "unknown"),
                    })
        return results

    def get_audit_trail(self, user_id: str, last_n: int = 50) -> list[dict]:
        """
        Return the last N audit entries for a user. Every value is a str or
        a float, so the result is always JSON-serialisable (json.dumps escapes
        control characters in the text fields). `detail` stays a JSON string.
        The keys time, capability, scope, action, reason and agent are the
        original ones; the rest were added with the hardened chain.
        """
        with self._audit_lock:
            user_entries = [e for e in self._audit if e.user_id == user_id]
        out = []
        for e in user_entries[-last_n:]:
            try:
                when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp))
            except (OverflowError, OSError, ValueError, TypeError):
                when = ""
            out.append({
                "time":       when,
                "capability": e.capability,
                "scope":      e.scope,
                "action":     e.action,
                "reason":     e.reason,
                "agent":      e.agent_id,
                "entry_id":   e.entry_id,
                "timestamp":  e.timestamp,
                "task_id":    e.task_id,
                "call_id":    e.call_id,
                "stage":      e.stage,
                "detail":     e.detail,
                "args_digest": e.args_digest,
                "prev_hash":  e.prev_hash,
                "entry_hash": e.entry_hash,
            })
        return out

    def record_audit(
        self,
        user_id:     str,
        capability:  str,
        scope:       str,
        action:      str,
        reason:      str = "",
        agent_id:    str = "",
        task_id:     str = "",
        call_id:     str = "",
        stage:       str = "",
        detail:      object = None,
        args_digest: str = "",
    ) -> AuditEntry:
        """
        Append an event to the SAME hash chain as grants and checks, and
        return the new entry.

        THIS DOES NOT REDACT. The caller must remove secrets (tokens,
        passwords, raw arguments, personal data) from every argument, and from
        `detail` in particular, BEFORE calling. What is passed in is stored and
        hashed as given.

        All arguments except `detail` must be str (TypeError otherwise).
        Control characters are allowed: they are stored and hashed as given.
        `detail` may be None, a str, or a dict or list of plain JSON data
        (str keys; str, int, finite float, bool, None values). A dict or list
        is stored as canonical JSON (sorted keys, ASCII, no spaces). Anything
        else raises TypeError. Over MAX_AUDIT_DETAIL_BYTES (8 KiB) raises
        ValueError; it is never cut short silently.
        """
        values = {
            "user_id": user_id, "capability": capability, "scope": scope,
            "action": action, "reason": reason, "agent_id": agent_id,
            "task_id": task_id, "call_id": call_id, "stage": stage,
            "args_digest": args_digest,
        }
        clean = {name: _check_text(name, v) for name, v in values.items()}
        text = _canonical_detail(detail)
        return self._audit_log(detail=text, **clean)

    def verify_chain(self) -> ChainReport:
        """
        Walk the audit chain from the start and report the first problem.

        Checks, for every entry in order: it is an AuditEntry; its prev_hash
        equals the previous entry's entry_hash ("genesis" for the first); its
        stored entry_hash equals a fresh compute_hash() over all its other
        fields. Finally the last entry's entry_hash must equal the engine's
        head hash, which is how entries removed from the END are caught (an
        empty list is only valid while the head is still "genesis").

        It does not trust the objects it checks. Every entry must be exactly
        an AuditEntry (a subclass could override compute_hash()), the hash is
        computed with AuditEntry.compute_hash() itself, and entry_hash,
        prev_hash and the head hash must be exactly str (a str subclass could
        claim to equal anything). It never raises for a bad entry (an
        attribute deleted, a field JSON cannot encode): it returns ok=False
        with a reason that names the problem but holds no field values.
        KeyboardInterrupt and SystemExit still propagate.

        Honest limit: the entries and the head hash are plain objects in this
        process. Anyone who can rewrite process memory can edit entries and
        recompute every hash and the head value, and this will then report ok.
        The chain is not saved or anchored anywhere (saving is issue #68).
        This detects accidents and edits that leave the head alone, not a
        determined attacker inside the process.
        """
        with self._audit_lock:
            entries = list(self._audit)
            head = self._last_hash

        # The objects are checked for what they EXACTLY are, not for what they
        # claim to be: a subclass can override compute_hash(), and a str
        # subclass can override __eq__. Both are rejected, and the hash is
        # computed with the class's own function. Nothing here may raise.
        if type(head) is not str:
            return ChainReport(
                False, len(entries), max(len(entries) - 1, 0),
                "the engine's head hash is not an exact str",
            )

        expected_prev = GENESIS_HASH
        for i, e in enumerate(entries):
            try:
                if type(e) is not AuditEntry:
                    return ChainReport(
                        False, i, i, f"entry {i} is not an exact AuditEntry"
                    )
                prev_hash = e.prev_hash
                stored_hash = e.entry_hash
                if type(prev_hash) is not str or type(stored_hash) is not str:
                    return ChainReport(
                        False, i, i,
                        f"entry {i}: prev_hash or entry_hash is not an exact str",
                    )
                if prev_hash != expected_prev:
                    return ChainReport(
                        False, i, i,
                        f"entry {i}: prev_hash does not match the previous entry "
                        "(entry reordered, inserted, duplicated or removed)",
                    )
                fresh = AuditEntry.compute_hash(e)
                if stored_hash != fresh:
                    return ChainReport(
                        False, i, i,
                        f"entry {i}: contents do not match its entry_hash (edited)",
                    )
                expected_prev = stored_hash
            except Exception as exc:  # noqa: BLE001 - a bad entry must be a report, not a crash
                # Only the exception type is reported, never its message, so
                # no field value can leak into the reason.
                return ChainReport(
                    False, i, i,
                    f"entry {i}: cannot be hashed or read ({type(exc).__name__})",
                )

        if head != expected_prev:
            last = max(len(entries) - 1, 0)
            return ChainReport(
                False, len(entries), last,
                "the last entry does not lead to the engine's head hash "
                "(entries removed from the end, or the last entry rewritten)",
            )
        return ChainReport(True, len(entries), None, "")

    def print_grant_summary(self, user_id: str):
        grants = self.list_grants(user_id)
        if not grants:
            print(f"[Permissions] No active grants for user '{user_id}'")
            return
        print(f"\n[Permissions] Active grants for '{user_id}':")
        for g in grants:
            print(
                f"  [{g['risk'].upper():<8}] {g['capability']:<20} "
                f"scope={g['scope']:<20} {g['expires']} | uses={g['uses']}"
            )

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _denial_reason(self, user_id: str, capability: Capability, scope: str) -> str:
        grants = self._grants.get(user_id, {}).get(capability.value, [])
        if not grants:
            return "No grant exists for this capability"
        expired  = [g for g in grants if g.is_expired]
        exhausted = [g for g in grants if g.is_exhausted]
        scope_mismatch = [g for g in grants if g.is_valid and not g.matches_scope(scope)]
        if expired:
            return f"Grant expired {(time.time() - expired[-1].expires_at)/60:.0f} minutes ago"
        if exhausted:
            return f"Grant exhausted (max_uses={exhausted[-1].max_uses} reached)"
        if scope_mismatch:
            return f"Scope mismatch: granted for '{scope_mismatch[-1].scope}', requested '{scope}'"
        return "Unknown denial reason"

    def _audit_log(
        self,
        user_id: str, capability: str, scope: str,
        action: str, reason: str = "",
        agent_id: str = "", task_id: str = "",
        call_id: str = "", stage: str = "",
        detail: str = "", args_digest: str = "",
    ) -> AuditEntry:
        """
        Append one entry to the chain. Internal: no validation (the engine's
        own callers pass strings). `detail` must already be a canonical string.
        Use record_audit() from outside.
        """
        # Internal callers may pass odd objects (a Path as scope); make them
        # text so a hash failure can never make check() raise.
        user_id, capability, scope, action, reason, agent_id, task_id = (
            v if isinstance(v, str) else str(v)
            for v in (user_id, capability, scope, action, reason, agent_id, task_id)
        )
        # The lock covers reading the head, hashing, appending and moving the
        # head, so two threads cannot both extend the same head.
        with self._audit_lock:
            entry = AuditEntry(
                user_id    = user_id,
                capability = capability,
                scope      = scope,
                action     = action,
                reason     = reason,
                agent_id   = agent_id,
                task_id    = task_id,
                prev_hash  = self._last_hash,
                call_id    = call_id,
                stage      = stage,
                detail     = detail,
                args_digest = args_digest,
            )
            entry.entry_hash = entry.compute_hash()
            self._audit.append(entry)
            self._last_hash = entry.entry_hash
            return entry

    # ── Persistence ───────────────────────────────────────────────────────────

    def _warn_about_dead_grants(self):
        """Say so when a saved grant can no longer match anything (honest reporting)."""
        for uid, caps in self._grants.items():
            for cap, grants in caps.items():
                for g in grants:
                    if g.is_valid and not grant_is_stable(g.scope):
                        print(
                            f"[Permission] ⚠️  Saved grant {g.grant_id} for '{cap}' "
                            f"(user '{uid}', scope {g.scope!r}) matches nothing: the path "
                            "or URL is not in canonical form, or changed since it was granted. "
                            "Grant it again if it is still wanted."
                        )

    def _save(self):
        if not self._persist_path:
            return
        data = {
            "grants": {
                uid: {
                    cap: [
                        {
                            "grant_id":    g.grant_id,
                            "user_id":     g.user_id,
                            "capability":  g.capability.value,
                            "scope":       g.scope,
                            "granted_at":  g.granted_at,
                            "expires_at":  g.expires_at,
                            "task_context": g.task_context,
                            "use_count":   g.use_count,
                            "max_uses":    g.max_uses,
                        }
                        for g in grants
                    ]
                    for cap, grants in caps.items()
                }
                for uid, caps in self._grants.items()
            }
        }
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self):
        if not self._persist_path.exists():
            return
        try:
            raw = self._persist_path.read_bytes()
            # Detect binary corruption — valid JSON always starts with { or [
            if raw and raw[0] not in (ord('{'), ord('['), ord(' '), ord('\n')):
                raise ValueError(f"File appears corrupted (starts with 0x{raw[0]:02x})")
            data = json.loads(raw.decode("utf-8"))
            for uid, caps in data.get("grants", {}).items():
                self._grants[uid] = {}
                for cap, grants in caps.items():
                    self._grants[uid][cap] = [
                        PermissionGrant(
                            grant_id    = g["grant_id"],
                            user_id     = g["user_id"],
                            capability  = Capability(g["capability"]),
                            scope       = g["scope"],
                            granted_at  = g["granted_at"],
                            expires_at  = g["expires_at"],
                            task_context= g["task_context"],
                            use_count   = g["use_count"],
                            max_uses    = g.get("max_uses"),
                        )
                        for g in grants
                    ]
            self._warn_about_dead_grants()
            print(f"[Permission] Loaded permissions from {self._persist_path}")
        except Exception as e:
            # Auto-recover — delete the corrupted file and start clean
            print(f"[Permission] Corrupted file detected ({e}) — resetting to clean state")
            try:
                self._persist_path.unlink()
            except Exception:
                pass


# ─── Permission Denied Error ──────────────────────────────────────────────────

class PermissionDeniedError(Exception):
    """Raised when ARIA tries to do something without user permission."""

    def __init__(
        self,
        user_id:     str,
        capability:  Capability,
        scope:       str = "*",
        description: str = "",
        risk:        str = "unknown",
    ):
        self.user_id     = user_id
        self.capability  = capability
        self.scope       = scope
        self.description = description or CAPABILITY_DESCRIPTIONS.get(capability, "")
        self.risk        = risk
        super().__init__(
            f"Permission denied: '{capability.value}' for user '{user_id}' (scope='{scope}'). "
            f"Description: {self.description}. "
            f"To grant: permission_engine.grant('{user_id}', Capability.{capability.name})"
        )

    def to_user_message(self) -> str:
        return (
            f"I need your permission to proceed. This action requires:\n\n"
            f"  Capability: {self.capability.value}\n"
            f"  What it does: {self.description}\n"
            f"  Risk level: {self.risk.upper()}\n"
            f"  Scope: {self.scope}\n\n"
            f"If you want to allow this, call:\n"
            f"  aria.permissions.grant('{self.user_id}', Capability.{self.capability.name})\n\n"
            f"ARIA will not proceed without your explicit approval."
        )
