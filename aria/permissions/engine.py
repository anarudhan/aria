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
    - Every permission check is logged in an append-only audit trail
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
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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
        if self.scope == "*":
            return True
        # Scope matching: granted scope must be an ancestor of requested
        return requested_scope == self.scope or requested_scope.startswith(self.scope)


# ─── Audit Entry ─────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Immutable record of every permission check, grant, or denial."""
    entry_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp:  float = field(default_factory=time.time)
    user_id:    str = ""
    capability: str = ""
    scope:      str = ""
    action:     str = ""       # "granted", "denied", "expired", "revoked"
    reason:     str = ""
    agent_id:   str = ""
    task_id:    str = ""
    # Hash of previous entry — makes audit trail tamper-evident
    prev_hash:  str = ""

    def compute_hash(self) -> str:
        data = f"{self.entry_id}{self.timestamp}{self.user_id}{self.capability}{self.action}{self.prev_hash}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ─── Permission Engine ────────────────────────────────────────────────────────

class PermissionEngine:
    """
    ARIA's zero-trust permission guardian.

    Every single capability check flows through this class.
    If check() returns False, the action does not happen. Period.
    No exceptions. No bypasses. No internal overrides.

    The only thing ARIA can do without user permission is:
      - Read its own source files for self-analysis (but not modify them)
      - Access in-memory data it created in the current session
      - Log to the audit trail
      - Display text responses to the user

    Everything else requires explicit user permission.
    """

    def __init__(self, persist_path: Optional[str] = None):
        # {user_id: {capability: [PermissionGrant]}}
        self._grants: dict[str, dict[str, list[PermissionGrant]]] = {}
        self._audit: list[AuditEntry] = []
        self._last_hash = "genesis"
        self._persist_path = Path(persist_path) if persist_path else None

        # Self-modify has an extra confirmation layer
        self._self_modify_confirmations: dict[str, dict] = {}

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Permission Checking (called before EVERY action) ──────────────────────

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

        This is called before EVERY tool execution, file access,
        network call, or any action that touches user data or resources.

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
        scope_desc  = f" for scope '{scope}'" if scope != "*" else ""
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
        grants = self._grants.get(user_id, {}).get(capability.value, [])
        before = len(grants)
        self._grants[user_id][capability.value] = [
            g for g in grants
            if not (g.scope == scope or scope == "*")
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
        """Return the last N audit entries for a user."""
        user_entries = [e for e in self._audit if e.user_id == user_id]
        return [
            {
                "time":       time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp)),
                "capability": e.capability,
                "scope":      e.scope,
                "action":     e.action,
                "reason":     e.reason,
                "agent":      e.agent_id,
            }
            for e in user_entries[-last_n:]
        ]

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
    ):
        entry = AuditEntry(
            user_id    = user_id,
            capability = capability,
            scope      = scope,
            action     = action,
            reason     = reason,
            agent_id   = agent_id,
            task_id    = task_id,
            prev_hash  = self._last_hash,
        )
        self._last_hash = entry.compute_hash()
        self._audit.append(entry)

    # ── Persistence ───────────────────────────────────────────────────────────

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
