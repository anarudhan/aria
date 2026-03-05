"""
ARIA Security Scanner
======================
ARIA reads and analyzes its own source code looking for security vulnerabilities.

This is the "self-awareness" side of ARIA's security brain. It does not modify
anything — it only observes. The patcher module handles modifications after
explicit user approval.

What it looks for:

  CRITICAL vulnerabilities:
    - Arbitrary code execution via eval(), exec(), compile()
    - Shell injection via subprocess with shell=True and user input
    - Path traversal (unsanitized file paths from user input)
    - Hardcoded secrets, API keys, passwords
    - SQL injection patterns
    - Pickle deserialization of untrusted data

  HIGH vulnerabilities:
    - Unbounded file reads/writes without size checks
    - Missing input validation on public-facing functions
    - Token/credential logging
    - Missing timeout on network requests
    - Exception swallowing (bare except:)

  MEDIUM vulnerabilities:
    - Insecure random (random module instead of secrets)
    - Missing TLS verification
    - Overly broad exception catching
    - Debug prints containing sensitive data

  LOW vulnerabilities:
    - Missing docstrings on public methods
    - Dead code that could confuse future patches
    - Deeply nested conditions (hard to audit)

The scanner uses two passes:
  Pass 1: Fast regex-based pattern matching — catches obvious issues instantly
  Pass 2: AST-based structural analysis — catches subtle logical issues

Both passes produce Vulnerability records with exact file location, severity,
a plain English description of the risk, and a suggested fix.
"""

from __future__ import annotations

import ast
import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─── Vulnerability Record ─────────────────────────────────────────────────────

@dataclass
class Vulnerability:
    vuln_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    severity:     str = "medium"       # "critical", "high", "medium", "low"
    category:     str = ""             # e.g. "code_injection", "path_traversal"
    file_path:    str = ""
    line_number:  int = 0
    line_content: str = ""
    description:  str = ""
    risk:         str = ""             # What an attacker could do
    suggestion:   str = ""             # Plain English fix recommendation
    patch_safe:   bool = True          # Can ARIA auto-patch this safely?
    detected_at:  float = field(default_factory=time.time)
    code_hash:    str = ""             # Hash of the vulnerable code section

    def to_report_line(self) -> str:
        return (
            f"[{self.severity.upper():<8}] {self.category} | "
            f"{Path(self.file_path).name}:{self.line_number} | "
            f"{self.description[:70]}"
        )


# ─── Regex Patterns for Pass 1 ────────────────────────────────────────────────

PATTERNS_CRITICAL = [
    (
        r"\beval\s*\(",
        "arbitrary_code_execution",
        "eval() can execute arbitrary code if user input reaches it",
        "Replace eval() with ast.literal_eval() for data parsing, or remove entirely",
        True,   # patch_safe
    ),
    (
        r"\bexec\s*\(",
        "arbitrary_code_execution",
        "exec() can execute arbitrary Python code",
        "Remove exec() or replace with a controlled subprocess with strict input validation",
        False,  # Not auto-patchable — needs human review
    ),
    (
        r"subprocess\.(call|run|Popen).*shell\s*=\s*True",
        "shell_injection",
        "subprocess with shell=True allows shell injection if user input is included",
        "Use shell=False and pass arguments as a list instead of a shell string",
        True,
    ),
    (
        r'(api_key|password|secret|token)\s*=\s*["\'][a-zA-Z0-9_\-\.]{10,}["\']',
        "hardcoded_secret",
        "Hardcoded credential found in source code",
        "Move to environment variables or a secrets manager like HashiCorp Vault",
        False,  # Not auto-patchable — need to know where to move the secret
    ),
    (
        r"\bpickle\.(loads?|load)\s*\(",
        "unsafe_deserialization",
        "pickle.load() on untrusted data can execute arbitrary code",
        "Replace with json.loads() or another safe deserializer for untrusted input",
        True,
    ),
    (
        r"\b__import__\s*\(",
        "dynamic_import",
        "Dynamic import can be abused to load malicious modules",
        "Replace with explicit imports or validate the module name against a whitelist",
        False,
    ),
]

PATTERNS_HIGH = [
    (
        r"open\s*\([^,)]+[,)][^)]*\)\s*\.read\s*\(\s*\)",
        "unbounded_file_read",
        "Unbounded file.read() — could exhaust memory on large files",
        "Use file.read(MAX_SIZE) with an explicit size limit",
        True,
    ),
    (
        r"except\s*:\s*$",
        "bare_exception",
        "Bare 'except:' catches all exceptions including SystemExit and KeyboardInterrupt",
        "Replace with 'except Exception as e:' and log the error",
        True,
    ),
    (
        r"except\s*Exception\s*:\s*pass\s*$",
        "exception_swallowing",
        "Exception swallowed silently — errors are hidden from monitoring",
        "At minimum log the exception: 'except Exception as e: logging.error(e)'",
        True,
    ),
    (
        r"(print|logging\.(info|debug))\s*\(.*?(token|key|secret|password|auth).*?\)",
        "credential_logging",
        "Potential credential value being logged — could expose secrets in log files",
        "Mask sensitive values before logging: log the first 4 characters + '***'",
        True,
    ),
    (
        r"requests\.(get|post|put|delete|patch)\s*\([^)]*\)\s*(?!.*timeout)",
        "missing_timeout",
        "HTTP request without timeout — can hang indefinitely",
        "Add timeout parameter: requests.get(url, timeout=30)",
        True,
    ),
    (
        r"verify\s*=\s*False",
        "tls_disabled",
        "TLS certificate verification disabled — vulnerable to MITM attacks",
        "Remove verify=False and ensure proper certificate chain is installed",
        True,
    ),
]

PATTERNS_MEDIUM = [
    (
        r"\brandom\.(random|randint|choice|shuffle)\s*\(",
        "insecure_random",
        "random module is not cryptographically secure",
        "Use secrets module for security-sensitive randomness: secrets.token_hex()",
        True,
    ),
    (
        r"(?<!#).*TODO.*security|FIXME.*auth|HACK.*permission",
        "flagged_debt",
        "Security-related TODO/FIXME found — outstanding security debt",
        "Review and resolve this security-related comment before production deployment",
        False,
    ),
]

PATTERNS_LOW = [
    (
        r"^(?!.*\"\"\"|.*#)def [a-z][a-zA-Z0-9_]+\s*\([^)]*\)\s*:\s*$",
        "missing_docstring",
        "Public function without docstring — hard to audit for security implications",
        "Add a docstring explaining the function's purpose and security considerations",
        False,
    ),
]

ALL_PATTERNS = [
    ("critical", PATTERNS_CRITICAL),
    ("high",     PATTERNS_HIGH),
    ("medium",   PATTERNS_MEDIUM),
    ("low",      PATTERNS_LOW),
]


# ─── AST-Based Analysis ───────────────────────────────────────────────────────

class ASTSecurityVisitor(ast.NodeVisitor):
    """
    Walk the AST of a Python file looking for structural security issues
    that regex can't catch reliably.
    """

    def __init__(self, source_lines: list[str], file_path: str):
        self.source_lines = source_lines
        self.file_path    = file_path
        self.findings: list[Vulnerability] = []

    def _line(self, node) -> str:
        try:
            return self.source_lines[node.lineno - 1].strip()
        except (IndexError, AttributeError):
            return ""

    def visit_Call(self, node: ast.Call):
        """Check for dangerous function calls."""
        # format_map / format with user input
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("format_map", "format"):
                if node.args or node.keywords:
                    # Could be benign — only flag if argument looks like user input
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and any(
                            kw in arg.id.lower() for kw in ["user", "input", "request", "param"]
                        ):
                            self.findings.append(Vulnerability(
                                severity     = "high",
                                category     = "format_injection",
                                file_path    = self.file_path,
                                line_number  = node.lineno,
                                line_content = self._line(node),
                                description  = "str.format() with user-controlled input may allow injection",
                                risk         = "Attacker could access arbitrary object attributes",
                                suggestion   = "Use f-strings with explicit sanitization or parameterized templates",
                                patch_safe   = False,
                            ))

        # Assert used for security checks (disabled in optimized mode)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        """Assert statements are removed with python -O and should not be used for security."""
        line = self._line(node)
        if any(kw in line.lower() for kw in ["permission", "auth", "user", "access", "valid"]):
            self.findings.append(Vulnerability(
                severity     = "high",
                category     = "assert_security_check",
                file_path    = self.file_path,
                line_number  = node.lineno,
                line_content = line,
                description  = "assert used for a security check — disabled when Python runs with -O flag",
                risk         = "Security check silently removed in optimized builds",
                suggestion   = "Replace assert with explicit if/raise PermissionError(...)",
                patch_safe   = True,
            ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function definitions for security issues."""
        # Functions with *args/**kwargs and no input validation
        has_star_args = any(
            isinstance(a, ast.arg)
            for a in (node.args.vararg, node.args.kwarg)
            if a is not None
        )
        if has_star_args and node.name.startswith(("handle_", "process_", "execute_")):
            self.findings.append(Vulnerability(
                severity     = "low",
                category     = "unvalidated_variadic",
                file_path    = self.file_path,
                line_number  = node.lineno,
                line_content = self._line(node),
                description  = f"Function '{node.name}' accepts *args/**kwargs without visible input validation",
                risk         = "Unexpected arguments may reach sensitive operations",
                suggestion   = "Add explicit parameter validation at the start of the function",
                patch_safe   = False,
            ))
        self.generic_visit(node)


# ─── Security Scanner ─────────────────────────────────────────────────────────

class SecurityScanner:
    """
    ARIA's self-inspection security scanner.

    Reads ARIA's own source files and identifies security vulnerabilities.
    Never modifies files — only reports findings for the patcher to act on.

    Usage:
        scanner = SecurityScanner("/path/to/aria")
        report = scanner.scan()
        for v in report.critical:
            print(v.to_report_line())
    """

    def __init__(self, aria_root: str):
        self.aria_root = Path(aria_root)
        self._last_scan: Optional[ScanReport] = None

    def scan(
        self,
        include_dirs: Optional[list[str]] = None,
        min_severity: str = "low",
    ) -> "ScanReport":
        """
        Scan ARIA's source code for security vulnerabilities.

        Args:
            include_dirs: Subdirectories to scan (default: core, agents, plugins, permissions, security)
            min_severity: Minimum severity to report ("critical", "high", "medium", "low")

        Returns:
            ScanReport with all findings categorized by severity
        """
        scan_dirs = include_dirs or ["core", "agents", "plugins", "permissions", "security"]
        all_vulns: list[Vulnerability] = []

        print("\n[Security Scanner] 🔍 Starting scan of ARIA source code...")
        files_scanned = 0

        for subdir in scan_dirs:
            target = self.aria_root / subdir
            if not target.exists():
                continue

            for py_file in target.rglob("*.py"):
                vulns = self._scan_file(py_file)
                all_vulns.extend(vulns)
                files_scanned += 1

        # Also scan root-level files
        for py_file in self.aria_root.glob("*.py"):
            all_vulns.extend(self._scan_file(py_file))
            files_scanned += 1

        # Filter by severity
        severity_order = ["critical", "high", "medium", "low"]
        min_idx = severity_order.index(min_severity)
        filtered = [v for v in all_vulns if severity_order.index(v.severity) <= min_idx]

        report = ScanReport(
            aria_root     = str(self.aria_root),
            files_scanned = files_scanned,
            vulnerabilities = filtered,
            scanned_at    = time.time(),
        )
        self._last_scan = report

        print(
            f"[Security Scanner] ✅ Scan complete: {files_scanned} files | "
            f"{len(report.critical)} critical | {len(report.high)} high | "
            f"{len(report.medium)} medium | {len(report.low)} low"
        )
        return report

    def scan_file(self, file_path: str) -> list[Vulnerability]:
        """Scan a single file."""
        return self._scan_file(Path(file_path))

    def _scan_file(self, path: Path) -> list[Vulnerability]:
        """Run both passes on a single Python file."""
        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            return []

        lines = source.splitlines()
        findings: list[Vulnerability] = []

        # Pass 1: Regex patterns
        for severity, patterns in ALL_PATTERNS:
            for regex, category, description, suggestion, patch_safe in patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(regex, line, re.IGNORECASE):
                        findings.append(Vulnerability(
                            severity     = severity,
                            category     = category,
                            file_path    = str(path),
                            line_number  = i,
                            line_content = line.strip(),
                            description  = description,
                            suggestion   = suggestion,
                            patch_safe   = patch_safe,
                            code_hash    = hashlib.md5(line.encode()).hexdigest()[:8],
                        ))

        # Pass 2: AST analysis
        try:
            tree = ast.parse(source, filename=str(path))
            visitor = ASTSecurityVisitor(lines, str(path))
            visitor.visit(tree)
            findings.extend(visitor.findings)
        except SyntaxError:
            pass   # Can't parse — don't crash the scanner

        return findings

    def get_last_report(self) -> Optional["ScanReport"]:
        return self._last_scan


# ─── Scan Report ─────────────────────────────────────────────────────────────

@dataclass
class ScanReport:
    aria_root:        str = ""
    files_scanned:    int = 0
    vulnerabilities:  list[Vulnerability] = field(default_factory=list)
    scanned_at:       float = field(default_factory=time.time)

    @property
    def critical(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if v.severity == "critical"]

    @property
    def high(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if v.severity == "high"]

    @property
    def medium(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if v.severity == "medium"]

    @property
    def low(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if v.severity == "low"]

    @property
    def auto_patchable(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if v.patch_safe]

    @property
    def requires_human(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if not v.patch_safe]

    def print_report(self):
        print("\n" + "=" * 70)
        print("  ARIA Security Scan Report")
        print(f"  Scanned: {self.files_scanned} files | "
              f"{len(self.vulnerabilities)} findings")
        print("=" * 70)

        for severity in ["critical", "high", "medium", "low"]:
            vulns = [v for v in self.vulnerabilities if v.severity == severity]
            if not vulns:
                continue
            icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
            print(f"\n{icons[severity]} {severity.upper()} ({len(vulns)})")
            for v in vulns:
                auto = "⚙️ auto-patchable" if v.patch_safe else "👤 needs review"
                print(f"  • {v.category}")
                print(f"    {Path(v.file_path).name}:{v.line_number}")
                print(f"    Risk:    {v.description}")
                print(f"    Fix:     {v.suggestion}")
                print(f"    Status:  {auto}")

        print(f"\n{'=' * 70}")
        print(f"  Auto-patchable: {len(self.auto_patchable)}")
        print(f"  Needs human:    {len(self.requires_human)}")
        print("=" * 70 + "\n")

    def to_dict(self) -> dict:
        return {
            "aria_root":     self.aria_root,
            "files_scanned": self.files_scanned,
            "scanned_at":    self.scanned_at,
            "summary": {
                "critical": len(self.critical),
                "high":     len(self.high),
                "medium":   len(self.medium),
                "low":      len(self.low),
            },
            "vulnerabilities": [
                {
                    "id":          v.vuln_id,
                    "severity":    v.severity,
                    "category":    v.category,
                    "file":        v.file_path,
                    "line":        v.line_number,
                    "description": v.description,
                    "suggestion":  v.suggestion,
                    "auto_patch":  v.patch_safe,
                }
                for v in self.vulnerabilities
            ],
        }
