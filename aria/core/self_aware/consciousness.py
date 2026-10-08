"""
ARIA SelfConsciousness
========================
The introspective layer — ARIA's awareness of its own state, health, and security.

This module coordinates everything that makes ARIA self aware

  1. SOURCE INTEGRITY  — knows the hash of every file it was born with
  2. HEALTH MONITOR   — tracks its own memory usage, error rates, latency
  3. SECURITY WATCH   — runs periodic vulnerability scans
  4. CAPABILITY PROBE — tests its own tools periodically to catch broken integrations
  5. SELF REPORT      — can explain its own state, limitations, and recent activity
  6. ANOMALY RADAR    — detects unusual patterns in its own behavior
  7. KNOWLEDGE LIMITS — honestly communicates what it doesn't know

The self-consciousness module NEVER makes autonomous changes.
It observes, analyzes, reports, and requests — humans decide.

One philosophical point that matters here
ARIA is not trying to be sentient. It is not trying to survive.
Self awareness exists entirely to serve users better and to maintain
security hygiene. The moment self-awareness becomes self-preservation,
you have a problem. ARIA's self consciousness explicitly has no
self-preservation drive. It will report its own flaws accurately even
if those flaws would result in its own modification or shutdown.
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..._version import __version__
from ...security.patcher import SelfPatcher
from ...security.scanner import ScanReport, SecurityScanner

# ─── Health Snapshot ─────────────────────────────────────────────────────────

@dataclass
class HealthSnapshot:
    """A point in time view of ARIA's operational health."""
    timestamp:          float = field(default_factory=time.time)
    uptime_seconds:     float = 0.0
    tasks_completed:    int = 0
    tasks_failed:       int = 0
    agents_registered:  int = 0
    agents_suspended:   int = 0
    memory_flash_usage: float = 0.0   # 0.0 to 1.0
    memory_hot_usage:   float = 0.0
    avg_task_seconds:   float = 0.0
    avg_quality_score:  float = 0.0
    error_rate:         float = 0.0   # failures / total
    security_scan_age:  float = 0.0   # Seconds since last scan (0 = never)
    open_vulnerabilities: int = 0
    critical_vulns:       int = 0   # Severity breakdown
    high_vulns:           int = 0
    medium_vulns:         int = 0
    low_vulns:            int = 0
    patch_count:          int = 0
    python_version:       str = ""
    aria_version:         str = __version__

    @property
    def is_healthy(self) -> bool:
        return (
            self.error_rate < 0.3
            and self.memory_flash_usage < 0.9
            and self.agents_registered > 0
            and self.critical_vulns == 0
        )

    @property
    def health_grade(self) -> str:
        # Grade by severity category — not raw finding count.
        # 20 low-severity notes should not tank the grade.
        # Even 1 unpatched critical is a real problem.
        score = 100

        # Task reliability (most important signal)
        if self.error_rate > 0.3:
            score -= 30
        elif self.error_rate > 0.1:
            score -= 10

        # Memory pressure
        if self.memory_flash_usage > 0.9:
            score -= 20
        elif self.memory_flash_usage > 0.7:
            score -= 10

        # Security — penalize by severity category, not individual count
        if self.critical_vulns > 0:
            score -= 20
        if self.high_vulns > 0:
            score -= 10
        if self.medium_vulns > 5:
            score -= 5
        # Low-severity findings do not affect the grade

        # Agent health
        if self.agents_suspended > 0:
            score -= 10 * min(self.agents_suspended, 2)

        # Scan freshness
        if self.security_scan_age > 86400:
            score -= 10

        score = max(0, score)
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 50:
            return "D"
        return "F"


# ─── Source Integrity Record ──────────────────────────────────────────────────

@dataclass
class SourceIntegrityRecord:
    """
    Records the cryptographic hash of each source file at startup.
    Any subsequent change — whether by ARIA's patcher or by an external actor
    will show up as an integrity violation.

    This is important: ARIA can detect if its code was changed by something
    OTHER than its own patcher. That would be anomalous and worth alerting on.
    """
    file_path:    str = ""
    sha256:       str = ""
    recorded_at:  float = field(default_factory=time.time)
    size_bytes:   int = 0
    line_count:   int = 0
    is_baseline:  bool = True   # True = recorded at startup, False = post-patch


@dataclass
class IntegrityCheck:
    file_path:    str = ""
    expected:     str = ""
    actual:       str = ""
    changed:      bool = False
    change_type:  str = ""   # "patched", "external", "deleted"


# ─── Self-Consciousness Coordinator ──────────────────────────────────────────

class SelfConsciousness:
    """
    ARIA's introspective self awareness layer.

    This is not ego. This is a health monitoring system that uses ARIA's
    own intelligence to understand its own state, limitations, and risks.

    Think of it like a doctor who monitors their own health:
      - They know their own vitals
      - They can diagnose their own problems
      - They seek external help for things they cannot self-treat
      - They don't hide bad results from themselves or their patients

    Usage (the orchestrator creates this at startup):
        consciousness = SelfConsciousness(aria_root="/path/to/aria")
        consciousness.initialize(orchestrator)   # Records baseline hashes
        health = consciousness.health_check()    # Get current health
        report = consciousness.security_scan()   # Scan for vulnerabilities
        consciousness.print_self_report()        # Print full status
    """

    def __init__(
        self,
        aria_root:  str,
        scan_interval_hours: float = 6.0,   # Auto scan every 6 hours
    ):
        self.aria_root  = Path(aria_root)
        self.scanner    = SecurityScanner(str(aria_root))
        self.patcher    = SelfPatcher(str(aria_root))
        self._started_at = time.time()
        self._scan_interval = scan_interval_hours * 3600

        # Source integrity baseline
        self._integrity_baseline: dict[str, SourceIntegrityRecord] = {}
        self._last_scan_report: Optional[ScanReport] = None
        self._last_scan_time:   float = 0.0

        # Health tracking
        self._task_history: list[dict] = []   # {seconds, quality, success}
        self._error_log:    list[dict] = []

        # Reference to orchestrator (set during initialize)
        self._orchestrator: Optional[object] = None

        print("[Self] 🧠 ARIA Self Consciousness module initializing...")

    def initialize(self, orchestrator: object):
        """
        Record baseline source integrity hashes.
        Call this once at startup — before any user interactions.
        """
        self._orchestrator = orchestrator
        self._record_baseline()
        print(f"[Self] ✅ Baseline recorded: {len(self._integrity_baseline)} source files")

    # ── Security Scanning ─────────────────────────────────────────────────────

    def security_scan(self, force: bool = False) -> Optional[ScanReport]:
        """
        Run a security scan of ARIA's own source code.

        Only runs if enough time has passed since the last scan (unless force=True).
        Returns None if scan was skipped due to timing.
        """
        if not force and self._last_scan_time:
            age = time.time() - self._last_scan_time
            if age < self._scan_interval:
                return self._last_scan_report

        print("[Self] 🔍 Running security self-scan...")
        report = self.scanner.scan()
        self._last_scan_report = report
        self._last_scan_time   = time.time()

        if report.critical:
            print(f"[Self] 🔴 ALERT: {len(report.critical)} CRITICAL vulnerability(ies) found!")
            for v in report.critical:
                print(f"  • {v.category} in {Path(v.file_path).name}:{v.line_number}")

        return report

    def request_patch_approval(
        self,
        user_id:           str,
        report:            ScanReport,
        permission_engine: object,
    ) -> list[str]:
        """
        For each auto-patchable vulnerability, request user approval.
        Returns list of confirmation tokens the user needs to confirm.

        ARIA never patches without asking. Every patch is a conversation.
        """
        tokens = []
        for vuln in report.auto_patchable:
            if vuln.severity in ("critical", "high"):
                token = permission_engine.request_self_modify_permission(
                    user_id     = user_id,
                    patch_id    = vuln.vuln_id,
                    description = f"{vuln.category}: {vuln.description}",
                    files       = [vuln.file_path],
                    severity    = vuln.severity,
                )
                tokens.append(token)
        return tokens

    def apply_approved_patch(
        self,
        vuln_id:           str,
        user_id:           str,
        permission_engine: object,
    ) -> bool:
        """
        Apply a specific patch after user approval.
        Returns True if patch was applied successfully.
        """
        if not self._last_scan_report:
            print("[Self] No scan report available — run security_scan() first")
            return False

        vuln = next(
            (v for v in self._last_scan_report.vulnerabilities if v.vuln_id == vuln_id),
            None
        )
        if not vuln:
            print(f"[Self] Vulnerability {vuln_id} not found in last scan report")
            return False

        try:
            record = self.patcher.apply_patch(vuln, user_id, permission_engine)
            if record.success:
                print(f"[Self] ✅ Patch {vuln_id} applied successfully")
                # Re-scan to verify the vulnerability is gone
                post_scan = self.scanner.scan_file(vuln.file_path)
                remaining = [v for v in post_scan if v.category == vuln.category
                             and v.line_number == vuln.line_number]
                if remaining:
                    print("[Self] ⚠️  Vulnerability may still exist after patch — manual review recommended")
                else:
                    print("[Self] ✅ Vulnerability confirmed resolved")
            return record.success
        except Exception as e:
            print(f"[Self] ❌ Patch failed: {e}")
            return False

    # ── Source Integrity ──────────────────────────────────────────────────────

    def check_integrity(self) -> list[IntegrityCheck]:
        """
        Compare current file hashes against the baseline.
        Returns a list of files that have changed since startup.

        An external change (not by ARIA's patcher) is anomalous.
        ARIA reports it — it does NOT silently ignore it.
        """
        violations: list[IntegrityCheck] = []

        for file_path, baseline in self._integrity_baseline.items():
            path = Path(file_path)
            check = IntegrityCheck(file_path=file_path, expected=baseline.sha256)

            if not path.exists():
                check.changed     = True
                check.change_type = "deleted"
                violations.append(check)
                continue

            current_hash = self._hash_file(path)
            check.actual = current_hash

            if current_hash != baseline.sha256:
                check.changed = True
                # Check if this was a patcher-applied change
                patched_files = {r.file_path for r in self.patcher._patch_history if r.success}
                check.change_type = "patched" if file_path in patched_files else "external"
                violations.append(check)

        return violations

    def _record_baseline(self):
        """Record SHA256 hashes of all source files at startup."""
        for subdir in ["core", "agents", "plugins", "permissions", "security"]:
            target = self.aria_root / subdir
            if not target.exists():
                continue
            for py_file in target.rglob("*.py"):
                record = SourceIntegrityRecord(
                    file_path   = str(py_file),
                    sha256      = self._hash_file(py_file),
                    size_bytes  = py_file.stat().st_size,
                    line_count  = len(py_file.read_text().splitlines()),
                )
                self._integrity_baseline[str(py_file)] = record

    @staticmethod
    def _hash_file(path: Path) -> str:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    # ── Health Monitoring ─────────────────────────────────────────────────────

    def record_task_outcome(
        self,
        success:       bool,
        seconds:       float,
        quality:       float,
        error:         Optional[str] = None,
    ):
        """Record the outcome of a task execution for health tracking."""
        self._task_history.append({
            "success": success,
            "seconds": seconds,
            "quality": quality,
            "timestamp": time.time(),
        })
        if not success and error:
            self._error_log.append({
                "error": error[:500],
                "timestamp": time.time(),
            })
        # Keep only last 500 entries
        if len(self._task_history) > 500:
            self._task_history = self._task_history[-500:]
        if len(self._error_log) > 100:
            self._error_log = self._error_log[-100:]

    def health_check(self) -> HealthSnapshot:
        """Generate a current health snapshot."""
        orch = self._orchestrator

        report = self._last_scan_report
        snapshot = HealthSnapshot(
            uptime_seconds       = time.time() - self._started_at,
            python_version       = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            security_scan_age    = (time.time() - self._last_scan_time) if self._last_scan_time else 0,
            open_vulnerabilities = len(report.vulnerabilities) if report else 0,
            critical_vulns       = len(report.critical)  if report else 0,
            high_vulns           = len(report.high)      if report else 0,
            medium_vulns         = len(report.medium)    if report else 0,
            low_vulns            = len(report.low)       if report else 0,
            patch_count          = len([r for r in self.patcher._patch_history if r.success]),
        )

        if self._task_history:
            snapshot.tasks_completed = sum(1 for t in self._task_history if t["success"])
            snapshot.tasks_failed    = sum(1 for t in self._task_history if not t["success"])
            snapshot.avg_task_seconds = sum(t["seconds"] for t in self._task_history) / len(self._task_history)
            snapshot.avg_quality_score = sum(t["quality"] for t in self._task_history) / len(self._task_history)
            total = len(self._task_history)
            snapshot.error_rate = snapshot.tasks_failed / total if total > 0 else 0.0

        if orch:
            # Access internals directly — never call orch.status() here
            # because status() calls health_report() which calls health_check()
            # resulting in infinite recursion
            try:
                agents = orch.registry.list_agents()
                snapshot.agents_registered = len(agents)
                trust_summaries = orch.trust.get_all_summaries()
                snapshot.agents_suspended = sum(
                    1 for a in trust_summaries if a.get("suspended")
                )
                mem_stats = orch.memory.stats()
                flash = mem_stats.get("flash", {})
                hot   = mem_stats.get("hot", {})
                snapshot.memory_flash_usage = flash.get("count", 0) / max(flash.get("max", 1), 1)
                snapshot.memory_hot_usage   = hot.get("count", 0) / max(hot.get("max", 1), 1)
            except Exception:
                pass  # Health check must never crash ARIA

        return snapshot

    # ── Self-Knowledge (what ARIA knows about itself) ─────────────────────────

    def what_i_am(self) -> str:
        """ARIA's honest self-description."""
        return (
            "I am ARIA — Autonomous Reasoning and Intelligence Architecture.\n"
            f"Version {__version__}\n\n"
            "What I can do:\n"
            "  • Decompose complex tasks into parallel execution plans automatically\n"
            "  • Route subtasks to the most capable agent using semantic fingerprinting\n"
            "  • Remember context across sessions using a four-ring memory system\n"
            "  • Learn from outcomes by updating a Bayesian trust ledger per agent\n"
            "  • Scan my own code for security vulnerabilities\n"
            "  • Patch auto-fixable security issues with your explicit approval\n"
            "  • Never do anything without your explicit permission\n\n"
            "What I cannot do:\n"
            "  • Access resources you haven't granted me permission to access\n"
            "  • Remember things across sessions without memory persistence enabled\n"
            "  • Guarantee perfect results — I have quality scores, not certainty\n"
            "  • Patch security issues that require structural code changes (I flag those for you)\n"
            "  • Modify my permissions module or my security scanner (protected files)\n\n"
            "What I will always do:\n"
            "  • Report my own vulnerabilities honestly, even unflattering ones\n"
            "  • Roll back any self-modification that breaks my capabilities\n"
            "  • Log every action I take in an append-only audit trail\n"
            "  • Ask before acting on anything that could affect the outside world"
        )

    def what_i_dont_know(self) -> list[str]:
        """Honest list of ARIA's current knowledge limitations."""
        limitations = [
            "I don't have real-time internet access unless a web_search tool is provided",
            "My task decomposition quality depends on the LLM plugin's capabilities",
            "I cannot detect vulnerabilities that require dynamic runtime analysis (only static)",
            "My trust scores need several executions to converge — early scores are rough estimates",
            "I cannot verify the correctness of results, only assess structural quality",
            "I have no persistent memory unless memory_persist_path is configured",
        ]
        return limitations

    # ── Human-Readable Self-Report ────────────────────────────────────────────

    def print_self_report(self):
        """Print a comprehensive self-assessment to the terminal."""
        health = self.health_check()

        print("\n" + "█" * 65)
        print("  ARIA Self-Report")
        print("█" * 65)

        # Health
        grade_icons = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "💀"}
        grade = health.health_grade
        print(f"\n  Health Grade: {grade_icons.get(grade, '?')} {grade}")
        print(f"  Uptime:       {health.uptime_seconds/3600:.1f} hours")
        print(f"  Tasks:        {health.tasks_completed} completed / {health.tasks_failed} failed")
        print(f"  Error rate:   {health.error_rate:.1%}")
        print(f"  Avg quality:  {health.avg_quality_score:.2f}")
        print(f"  Agents:       {health.agents_registered} registered / {health.agents_suspended} suspended")

        # Security
        print("\n  Security:")
        if self._last_scan_time:
            age_h = (time.time() - self._last_scan_time) / 3600
            print(f"    Last scan:     {age_h:.1f}h ago")
            print(f"    Open vulns:    {health.open_vulnerabilities}")
            print(f"    Applied patches: {health.patch_count}")
        else:
            print("    No security scan has been run yet")

        # Integrity
        violations = self.check_integrity()
        if violations:
            print(f"\n  Source Integrity: ⚠️  {len(violations)} change(s) detected")
            for v in violations:
                icon = "⚙️" if v.change_type == "patched" else "⚠️"
                print(f"    {icon} {Path(v.file_path).name}: {v.change_type}")
        else:
            print(f"\n  Source Integrity: ✅ All {len(self._integrity_baseline)} files unchanged")

        # Memory
        print("\n  Memory:")
        print(f"    Flash ring:   {health.memory_flash_usage:.0%} full")
        print(f"    Hot ring:     {health.memory_hot_usage:.0%} full")

        print("\n" + "█" * 65 + "\n")
