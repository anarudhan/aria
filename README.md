# ARIA — Autonomous Reasoning and Intelligence Architecture

> Version 1.1.0 | Zero Trust | Self Aware | Self Healing | Self Improving

ARIA is a production grade Python framework for building multi agent AI systems that are genuinely trustworthy. It is not a wrapper around another AI framework. It is built from scratch with four principles that most agent frameworks completely ignore.

**Zero trust by default.** ARIA cannot read your email, browse the web, write files, or do anything else until you explicitly grant it permission. Every permission has an expiry time and a use limit. There are no hidden capabilities.

**Self awareness.** ARIA knows what it is, what it can and cannot do, and reports its own health and vulnerabilities honestly  even unflattering ones. It hashes its own source files at startup and detects any unexpected changes.

**Self healing.** ARIA scans its own source code for security vulnerabilities, identifies which ones it can safely patch automatically, and applies those patches with your explicit approval. Every patch creates a backup and rolls back automatically if verification fails.

**Self improvement.** ARIA learns from every task it runs — what went wrong, which agents performed well, and what gaps exist in its capabilities. When it repeatedly fails at a class of tasks, it identifies the gap, proposes a solution, and with your permission synthesizes a brand new agent to fill that gap.

---

## What makes ARIA different

Most agent frameworks give you a way to chain LLM calls. ARIA gives you a system that improves itself, protects itself, and never surprises you with what it has access to.

| Feature | ARIA | Most frameworks |
|---|---|---|
| Zero trust permissions | Built in, default deny | Not present |
| Self security scanning | Yes, reads its own code | Not present |
| Self patching | Yes, with rollback | Not present |
| Learning from mistakes | Yes, 3 learning layers | Not present |
| Capability synthesis | Yes, generates new agents | Not present |
| Health monitoring | Yes, graded A-F | Not present |
| Source integrity checks | Yes, SHA256 per file | Not present |
| Human in the loop | Yes, for all side effects | Varies |
| LLM dependencies | Zero mandatory | Usually 1-3 required |

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                     │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│               ARIA ORCHESTRATOR                          │
│  Receives task → checks permissions → decomposes →       │
│  forms coalition → executes → learns → reports           │
└──┬──────────┬──────────┬──────────┬──────────┬───────────┘
   │          │          │          │          │
┌──▼──┐  ┌───▼───┐  ┌───▼───┐  ┌──▼──┐  ┌───▼────────────┐
│PERM │  │SELF   │  │SECU   │  │MEM  │  │SELF-IMPROVEMENT│
│ISSION  │AWARE  │  │RITY   │  │ORY  │  │LEARNING ENGINE │
│ENGINE  │NESS   │  │SCANNER│  │     │  │                │
└──┬──┘  └───────┘  └──┬────┘  └─────┘  └────────────────┘
   │                   │
   │  grants required  │  findings
   │                   │
┌──▼───────────────────▼──────────────────────────────────┐
│                   AGENT REGISTRY                        │
│  Registered agents with capability fingerprints         │
│  Trust ledger updated per outcome                       │
│  Empirical routing weights from learning engine         │
└──────────────────────────────────────────────────────────┘
```

---

## Quick start

### Install

```bash
# No dependencies required for the core framework
git clone https://github.com/anarudhan/aria.git
cd aria

# With your LLM provider (pick one)
pip install anthropic          # for Claude
pip install openai             # for GPT-4o
```

### Run the demo (no API key needed)

```bash
# The demo uses a mock LLM — shows all 9 features without any API key
PYTHONPATH=. python examples/demo_v11.py
```

### Minimal working example

```python
import asyncio
from aria import ARIAOrchestrator, ARIAConfig, ExecutionContract, aria_agent, Capability
from aria.plugins.llm_plugins import AnthropicPlugin

# 1. Define an agent
@aria_agent(
    capabilities=["research", "web search", "information retrieval"],
    domains=["research"],
)
async def researcher(node, context, memory):
    """Finds and summarizes information from available sources."""
    return f"Research complete for: {node.description}"

# 2. Build ARIA
aria = ARIAOrchestrator(
    config=ARIAConfig(),
    llm_plugin=AnthropicPlugin(api_key="your-key-here"),
)

# 3. Register agents
aria.register_agents([researcher])

# 4. Grant permissions (nothing works without this)
aria.permissions.grant("alice", Capability.MEMORY_WRITE, hours=8)
aria.permissions.grant("alice", Capability.WEB_SEARCH, hours=8)

# 5. Run a task
async def main():
    result = await aria.run(
        task="Research the latest developments in quantum computing",
        user_id="alice",
        contract=ExecutionContract(max_seconds=30),
    )
    print(result.final_answer)

asyncio.run(main())
```

---

## Core modules

### `aria/permissions/engine.py` — Zero Trust Permission Engine

Every capability ARIA has is gated by an explicit user grant. The engine uses a default-deny model — if a permission hasn't been granted, the action fails immediately before any LLM call is even made.

```python
# Grant a permission (with expiry and optional use limit)
aria.permissions.grant("user_id", Capability.EMAIL_SEND, hours=1, max_uses=5)

# Check a permission (returns bool)
allowed = aria.permissions.check("user_id", Capability.FILE_WRITE)

# Require a permission (raises PermissionDeniedError if denied)
aria.permissions.require("user_id", Capability.WEB_BROWSE, scope="https://example.com")

# Revoke all permissions for a user
aria.permissions.revoke_all("user_id")

# See the audit trail (append-only, hash-chained)
trail = aria.permissions.get_audit_trail("user_id")
```

Available capabilities:

```python
Capability.FILE_READ          # Read files from disk
Capability.FILE_WRITE         # Write files to disk
Capability.NETWORK            # Make network requests
Capability.EMAIL_READ         # Read email inbox
Capability.EMAIL_SEND         # Send emails (1-hour default expiry)
Capability.TEAMS_READ         # Read Teams channels and DMs
Capability.TEAMS_POST         # Post to Teams (1-hour default expiry)
Capability.CODE_EXECUTE       # Execute generated code
Capability.MEMORY_WRITE       # Store information in long-term memory
Capability.SELF_MODIFY        # Apply security patches (30-min expiry)
Capability.AGENT_SPAWN        # Register new agents
Capability.SYSTEM_INFO        # Read system information
Capability.CALENDAR_READ      # Read calendar
Capability.CALENDAR_WRITE     # Create calendar events
Capability.DOCUMENT_CREATE    # Create documents
Capability.WEB_SEARCH         # Search the web
Capability.WEB_BROWSE         # Browse specific URLs
```

---

### `aria/core/improvement.py` — Self Improvement Engine

Three learning layers that run after every task.

**Layer 1: Mistake learning**

After every task, ARIA analyses what happened — whether it succeeded, failed, produced poor quality results, or timed out. If the LLM is available, it does a root cause analysis. Otherwise it falls back to heuristic pattern matching. Either way, a lesson is recorded with the root cause and a concrete adjustment for next time. Before future similar tasks, relevant lessons are injected into the decomposition and planning stage.

```python
# Lessons are recalled automatically before each task.
# You can also query them directly:
lessons = aria.learning.recall_lessons("write a quarterly report", limit=5)
for lesson in lessons:
    print(f"{lesson.outcome}: {lesson.adjustment}")
```

**Layer 2: Capability gap detection**

When the same category of task fails 3 or more times with the root cause being a missing capability, ARIA identifies a gap and proposes a solution.

```python
# See open gaps
gaps = aria.learning.get_open_gaps()
for gap in gaps:
    print(f"{gap.task_type}: failed {gap.failure_count}x")
    print(f"Proposed: {gap.proposed_cap} — {gap.proposed_desc}")
```

**Layer 3: Capability synthesis**

With `AGENT_SPAWN` and `CODE_EXECUTE` permissions granted, ARIA can write a new Python agent using its LLM, show you the code for review, and load it into the running system after your explicit approval.

```python
# Synthesise a new agent to fill a gap
synth = await aria.learning.synthesize_capability(
    gap_id="abc123",
    user_id="alice",
    permission_engine=aria.permissions,
)
# Shows you the generated code. You review it.

# Approve it (your explicit action required)
aria.learning.approve_capability(synth.cap_id, "alice")

# Load it into the live system
await aria.learning.load_approved_capability(
    synth.cap_id, aria, aria.permissions, "alice"
)
# ARIA now has a new registered agent
```

**Routing refinement**

ARIA tracks real observed performance for every (agent, task_category) pair. After 5 executions, the empirical score blends with semantic fingerprint similarity for routing decisions. Agents that claim broad capabilities but consistently underperform get lower routing priority for those task types over time.

```python
# See the learning summary at any time
aria.learning.print_learning_report()
```

---

### `aria/security/scanner.py` — Security Self Scanner

ARIA reads its own source code and identifies vulnerabilities. Two passes: regex pattern matching for known dangerous patterns, and AST structural analysis for deeper issues.

```python
from aria.security.scanner import SecurityScanner

scanner = SecurityScanner("/path/to/aria")
report = scanner.scan()
report.print_report()

# Filter by severity
for vuln in report.critical:
    print(f"{vuln.file_path}:{vuln.line_number} — {vuln.description}")

# See what can be auto-patched vs what needs human review
print(f"Auto-patchable: {len(report.auto_patchable)}")
print(f"Needs human:    {len(report.requires_human)}")
```

Detected vulnerability categories: `arbitrary_code_execution`, `dynamic_import`, `hardcoded_secret`, `unsafe_deserialization`, `shell_injection`, `missing_timeout`, `tls_disabled`, `exception_swallowed`, `insecure_random`, `flagged_debt`, `missing_docstring`.

---

### `aria/security/patcher.py` — Self-Patcher

Applies security patches to ARIA's own code with a strict safety protocol: permission check, backup, apply, syntax verify, rollback on failure.

```python
# Run through the full scan-to-patch flow
report = scanner.scan()

# Request SELF_MODIFY permission (two-step approval)
token = aria.self_awareness.request_patch_approval("alice", report, aria.permissions)
# User sees full details of what will change and confirms

aria.self_awareness.apply_approved_patch(vuln_id, "alice", aria.permissions)
# Patch applied, backed up, verified. Rolls back automatically if broken.

# Roll back any patch by ID
patcher.rollback(patch_id)

# Emergency: roll back everything
patcher.rollback_all()
```

---

### `aria/core/self_aware/consciousness.py` — Self Consciousness

ARIA's introspective awareness of its own state, health, and integrity.

```python
# Full self-report (printed to stdout)
aria.self_report()

# What ARIA knows about itself (returned as string)
description = aria.what_i_am()

# Health snapshot
health = aria.health_report()
print(f"Grade: {health['grade']}")       # A, B, C, D, or F
print(f"Healthy: {health['is_healthy']}")
print(f"Error rate: {health['error_rate']:.1%}")

# Source integrity — detects unexpected file changes
violations = aria.integrity_check()
if violations:
    for v in violations:
        print(f"[{v['change_type']}] {v['file']}")
```

---

## Writing agents

```python
from aria.agents.base import aria_agent

@aria_agent(
    capabilities=["email management", "inbox reading", "reply drafting"],
    domains=["email", "communication"],
)
async def email_agent(node, context, memory):
    """
    Handles everything related to email.
    Reads inbox, searches threads, drafts replies.
    Never sends without human approval — drafts only.
    """
    task = node.description
    user_id = context.get("user_id")

    # Your implementation here
    # node.description  — the specific subtask for this agent
    # context           — dict with user_id, tenant_id, etc.
    # memory            — access to ARIA's memory system

    return "Agent result as a string"
```

Agents are automatically scored on quality after each execution. The trust ledger updates per agent per task type. Low performing agents get deprioritized for the task types they underperform on.

---

## LLM plugins

ARIA is LLM agnostic. Swap providers by changing the plugin.

```python
# Anthropic Claude (recommended)
from aria.plugins.llm_plugins import AnthropicPlugin
plugin = AnthropicPlugin(api_key="sk-ant-...")

# OpenAI GPT-4o
from aria.plugins.llm_plugins import OpenAIPlugin
plugin = OpenAIPlugin(api_key="sk-...")

# Mock (no API key needed — for testing and demos)
from aria.plugins.llm_plugins import MockPlugin
plugin = MockPlugin()
```

---

## Running the full demo

The demo shows all nine capabilities in sequence. No API key needed — it uses a mock LLM.

```bash
PYTHONPATH=. python examples/demo_v11.py
```

What each demo shows:

1. Zero trust permission model — granting capabilities, denying unchecked ones
2. Self awareness — health grade, uptime, agent inventory, memory rings
3. Security self scan — ARIA reading its own 21 files, finding 22 vulnerabilities in itself
4. Source integrity — SHA256 hash comparison of all source files
5. Task execution with learning — running a real task and recording a lesson from the outcome
6. Failure simulation and gap detection — 3 repeated failures trigger capability gap identification
7. Capability synthesis — ARIA writes a new agent, shows the code, you approve, it loads live
8. Empirical routing — blending semantic fingerprinting with real observed performance data
9. Full learning report — lessons, routing weights, gaps, synthesized capabilities

---

## Connecting your GitHub to Claude

Once you push this to GitHub, share the repo with Claude in your Claude Project and Claude will have direct access to the full source. To connect it:

1. Push to GitHub: `git push origin main`
2. In your Claude Project settings, add the GitHub repository as a connected source
3. Claude will index all files and can answer questions about specific functions, suggest improvements, write new agents, and debug issues with full context of the codebase

---

## Roadmap

These are clean extension points — each one is a new node in the agent graph, no rearchitecting needed.

- Microsoft Graph API integration (Outlook + Teams) — connects to the org AI assistant platform
- Tavily web search agent — live information retrieval
- Playwright web browsing agent — full page content extraction
- python-docx document agent — formatted Word document generation
- REST API server — FastAPI wrapper exposing ARIA over HTTP/WebSocket
- LangGraph bridge — run ARIA agents inside a LangGraph supervisor graph
- Async batch execution — run multiple independent tasks in parallel pipelines

---

## Project structure

```
aria/
├── aria/                          # Core framework package
│   ├── __init__.py                # Public API exports
│   ├── agents/
│   │   ├── __init__.py
│   │   └── base.py                # BaseAgent, aria_agent decorator, registry
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py              # ARIAConfig, TaskGenome, TaskResult, etc.
│   │   ├── orchestrator.py        # ARIAOrchestrator — the main entry point
│   │   ├── decomposer.py          # Task → DAG decomposition
│   │   ├── executor.py            # Parallel DAG execution engine
│   │   ├── memory.py              # Four-ring memory system
│   │   ├── registry.py            # Agent registry with semantic fingerprinting
│   │   ├── trust.py               # Bayesian trust ledger per agent
│   │   ├── improvement.py         # Self-improvement and learning engine
│   │   └── self_aware/
│   │       ├── __init__.py
│   │       └── consciousness.py   # Health monitoring, source integrity
│   ├── permissions/
│   │   ├── __init__.py
│   │   └── engine.py              # Zero-trust permission engine
│   ├── security/
│   │   ├── __init__.py
│   │   ├── scanner.py             # Source code vulnerability scanner
│   │   └── patcher.py             # Self-patcher with rollback
│   └── plugins/
│       ├── __init__.py
│       └── llm_plugins.py         # Anthropic, OpenAI, Mock plugins
├── examples/
│   ├── demo_v11.py                # Full v1.1.0 demo (all 9 features)
│   └── full_demo.py               # Original v1.0 demo
├── tests/
│   ├── test_permissions.py
│   ├── test_improvement.py
│   ├── test_scanner.py
│   └── test_orchestrator.py
├── docs/
│   └── architecture.md
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Version history

**v1.1.0** (March 2026)
- Zero trust permission engine with 17 capability types
- Self consciousness module with health grading and source integrity
- Security self scanner with two-pass analysis across 21 source files
- Self-patcher with automatic rollback on verification failure
- Self-improvement engine: mistake learning, capability gap detection, capability synthesis
- Empirical routing weights blending semantic fingerprinting with real performance data

**v1.0.0** (February 2026)
- Multi agent orchestrator with supervisor pattern
- TaskGenome DAG decomposition
- Four ring temporal memory system
- Bayesian trust ledger per agent
- LLM plugin system (Anthropic, OpenAI, Mock)

---

*Built with zero mandatory dependencies. Runs on Python 3.10+.*
