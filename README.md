# ARIA — Autonomous Reasoning and Intelligence Architecture

> Version 1.2.0 | Zero Trust | Self Aware | Self Healing | Self Improving | Web Native

ARIA is a production-grade Python framework for building multi-agent AI systems that are genuinely trustworthy. It is not a wrapper around another AI framework. It is built from scratch with four principles that most agent frameworks completely ignore.

**Zero trust by default.** ARIA cannot read your email, browse the web, write files, or do anything else until you explicitly grant it permission. Every permission has an expiry time and a use limit. There are no hidden capabilities.

**Self-awareness.** ARIA knows what it is, what it can and cannot do, and reports its own health and vulnerabilities honestly including the unflattering ones. It hashes its own source files at startup and detects any unexpected changes.

**Self-healing.** ARIA scans its own source code for security vulnerabilities, identifies which ones it can safely patch automatically, and applies those patches with your explicit approval. Every patch creates a backup and rolls back automatically if verification fails.

**Self-improvement.** ARIA learns from every task it runs including what went wrong, which agents performed well, and what gaps exist in its capabilities. When it repeatedly fails at a class of tasks, it identifies the gap, proposes a solution, and with your permission synthesizes a brand new agent to fill that gap.

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
| Health monitoring | Yes, graded A through F | Not present |
| Source integrity checks | Yes, SHA256 per file | Not present |
| Human in the loop | Yes, for all side effects | Varies |
| Web search | Claude-native, memory-first | Third-party API |
| Memory persistence | SQLite, four ring temporal | Flat store or none |
| LLM dependencies | Zero mandatory | Usually 1 to 3 required |

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
│PERM │  │SELF   │  │SECU   │  │MEM  │  │SELF IMPROVEMENT│
│ISSION  │AWARE  │  │RITY   │  │ORY  │  │LEARNING ENGINE │
│ENGINE  │NESS   │  │SCANNER│  │SQLite  │                │
└──┬──┘  └───────┘  └──┬────┘  └─────┘  └────────────────┘
   │                   │
┌──▼───────────────────▼──────────────────────────────────┐
│                   AGENT REGISTRY                        │
│  Registered agents with capability fingerprints         │
│  Trust ledger updated per outcome                       │
│  Empirical routing weights from learning engine         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  WEB AGENT (aria/agents/web_agent.py)           │   │
│  │  Memory-first → Claude search → DDG fallback    │   │
│  │  Domain trust scoring — persistent across runs  │   │
│  └─────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Quick start

### Install

```bash
# No dependencies required for the core framework
git clone https://github.com/anarudhan/aria.git
cd aria

# With Claude (recommended — enables Claude-native web search)
pip install anthropic httpx

# Optional — full JS page rendering for web browsing
pip install playwright && playwright install chromium
```

### Run the demo (no API key needed)

```bash
# The demo uses a mock LLM and shows all features without any API key
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

Every capability ARIA has is gated by an explicit user grant. The engine uses a default deny model. If a permission has not been granted, the action fails immediately before any LLM call is even made.

```python
# Grant a permission with expiry and optional use limit
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

### `aria/core/memory.py` — Four Ring Temporal Memory with SQLite

ARIA organises memory into four time-based rings modelled on CPU cache hierarchy. Memory moves between rings automatically. Frequently accessed entries are promoted to faster rings and old untouched entries are demoted.

```
FLASH  → Session only. RAM only. No persistence. Working variables.
HOT    → Last 2 hours.  SQLite. Fast recall.
WARM   → Last 24 hours. SQLite. Today's context.
COLD   → Up to 30 days. SQLite. Long-term knowledge.
```

The SQLite backend writes one row per `store()` call regardless of total entry count, consistently under 1ms per write at any scale. The previous JSON design rewrote the entire file on every write and degraded to roughly 100ms per write at 10,000 entries. FLASH stays RAM-only by design because session working memory should never outlive the conversation.

```python
from aria.core.memory import TemporalMemory
from aria.core.models import MemoryRing

memory = TemporalMemory(persist_path=".aria/memory.db")

# Store something that survives process restarts for HOT, WARM and COLD rings
entry_id = memory.store(
    "User prefers bullet point summaries",
    ring=MemoryRing.WARM,
    tags=["preference", "formatting"],
    user_id="alice",
)

# Recall by semantic similarity using TF-IDF with no embedding API needed
results = memory.recall("formatting preference", user_id="alice", top_k=5)

# Recall by exact tag
results = memory.recall_by_tag("preference", rings=[MemoryRing.HOT, MemoryRing.WARM])

# See ring stats
print(memory.stats())

# Run maintenance to demote stale entries and evict expired ones
memory.maintain()
```

---

### `aria/agents/web_agent.py` — Web Browsing and Search Agent

ARIA's native web agent. Not a tool wrapper but a thinking agent with memory, trust scoring, and permission gates built in from the ground up.

**Search priority**

ARIA checks her own temporal memory first and serves fresh cached results instantly with zero network cost. If nothing fresh is found she uses Claude-native search via Anthropic's `web_search_20260209` tool which is Brave-powered and returns a reasoned cited answer in one API call. If no API key is set she falls back to DuckDuckGo lite which requires zero dependencies.

**Browse strategy**

ARIA checks her own temporal memory first and serves fresh cached page content if the URL was browsed recently. If nothing is cached she tries Playwright for full JS rendering when it is installed, then falls back to requests and finally to urllib which is always available.

**What makes it different from other frameworks' web tools**

Every other framework gives you a function that fetches a URL and returns a string. ARIA's web agent checks its own memory before touching the internet, gates every action through the real permission engine, tracks domain reliability over time with Bayesian scoring, and stores what it learns in the right temporal ring for future use. The Claude search integration means ARIA gets a reasoned synthesised answer rather than raw results that need a second LLM pass to process.

```python
import os
from aria.agents.web_agent import create_web_agent
from aria.permissions.engine import Capability

# Grant permissions first — always required
aria.permissions.grant("alice", Capability.WEB_SEARCH, hours=4)
aria.permissions.grant("alice", Capability.WEB_BROWSE, hours=4)

# Create and register the web agent
web_agent = create_web_agent(
    permissions=aria.permissions,
    memory=aria.memory,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    freshness_hours=2,
    domain_trust_path=".aria/domain_trust.json",
    max_claude_searches=3,
)
aria.register_agent(web_agent)
```

Domain trust scoring lets ARIA learn which sources are reliable over time:

```python
print(web_agent.trust_report())
# [ARIA Web Agent — Domain Trust Report]
# ██████████  0.95  docs.anthropic.com
# █████████░  0.90  arxiv.org
# ███████░░░  0.72  medium.com
# ████░░░░░░  0.45  somespamsite.com
```

---

### `aria/core/improvement.py` — Self-Improvement Engine

Three learning layers that run after every task.

**Layer 1 — Mistake learning**

After every task ARIA analyses what happened. If the LLM is available it does a root cause analysis. Otherwise it falls back to heuristic pattern matching. A lesson is recorded with the root cause and a concrete adjustment for next time. Relevant lessons are injected into planning before future similar tasks.

```python
lessons = aria.learning.recall_lessons("write a quarterly report", limit=5)
for lesson in lessons:
    print(f"{lesson.outcome}: {lesson.adjustment}")
```

**Layer 2 — Capability gap detection**

When the same category of task fails 3 or more times with a missing capability as the root cause, ARIA identifies the gap and proposes a solution.

```python
gaps = aria.learning.get_open_gaps()
for gap in gaps:
    print(f"{gap.task_type}: failed {gap.failure_count}x")
    print(f"Proposed: {gap.proposed_cap} — {gap.proposed_desc}")
```

**Layer 3 — Capability synthesis**

With `AGENT_SPAWN` and `CODE_EXECUTE` permissions granted, ARIA writes a new Python agent, shows you the code for review, and loads it into the running system after your explicit approval.

```python
synth = await aria.learning.synthesize_capability(
    gap_id="abc123",
    user_id="alice",
    permission_engine=aria.permissions,
)
aria.learning.approve_capability(synth.cap_id, "alice")
await aria.learning.load_approved_capability(synth.cap_id, aria, aria.permissions, "alice")
```

---

### `aria/security/scanner.py` — Security Self Scanner

ARIA reads its own source code and identifies vulnerabilities using two passes — regex pattern matching and AST structural analysis.

```python
from aria.security.scanner import SecurityScanner

scanner = SecurityScanner("/path/to/aria")
report = scanner.scan()
report.print_report()

for vuln in report.critical:
    print(f"{vuln.file_path} line {vuln.line_number} — {vuln.description}")

print(f"Auto-patchable: {len(report.auto_patchable)}")
print(f"Needs human review: {len(report.requires_human)}")
```

---

### `aria/security/patcher.py` — Self Patcher

Applies security patches to ARIA's own code with a strict safety protocol. It performs a permission check, creates a backup, applies the transformation, verifies syntax, and rolls back automatically if verification fails.

```python
token = aria.self_awareness.request_patch_approval("alice", report, aria.permissions)
aria.self_awareness.apply_approved_patch(vuln_id, "alice", aria.permissions)

# Roll back a specific patch
patcher.rollback(patch_id)

# Emergency rollback of everything
patcher.rollback_all()
```

---

### `aria/core/self_aware/consciousness.py` — Self Consciousness

```python
aria.self_report()                    # Full self-report printed to stdout

health = aria.health_report()
print(f"Grade: {health['grade']}")   # A, B, C, D, or F

violations = aria.integrity_check()   # Detects unexpected file changes
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
    Reads inbox, searches threads, and drafts replies.
    Never sends without human approval.
    """
    task    = node.description
    user_id = context.get("user_id")
    return "Agent result as a string"
```

---

## LLM plugins

```python
# Anthropic Claude — recommended, also enables Claude-native web search
from aria.plugins.llm_plugins import AnthropicPlugin
plugin = AnthropicPlugin(api_key="sk-ant-...")

# OpenAI GPT-4o
from aria.plugins.llm_plugins import OpenAIPlugin
plugin = OpenAIPlugin(api_key="sk-...")

# Mock — no API key needed, great for testing and demos
from aria.plugins.llm_plugins import MockPlugin
plugin = MockPlugin()
```

---

## Running the full demo

```bash
PYTHONPATH=. python examples/demo_v11.py
```

---

## Project structure

```
aria/
├── aria/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseAgent, aria_agent decorator, registry
│   │   └── web_agent.py           # Web search and browsing agent (Claude-native)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── orchestrator.py
│   │   ├── decomposer.py
│   │   ├── executor.py
│   │   ├── memory.py              # Four-ring temporal memory with SQLite backend
│   │   ├── registry.py
│   │   ├── trust.py
│   │   ├── improvement.py
│   │   └── self_aware/
│   │       ├── __init__.py
│   │       └── consciousness.py
│   ├── permissions/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── security/
│   │   ├── __init__.py
│   │   ├── scanner.py
│   │   └── patcher.py
│   └── plugins/
│       ├── __init__.py
│       └── llm_plugins.py
├── examples/
│   ├── demo_v11.py
│   └── full_demo.py
├── tests/
│   ├── test_permissions.py
│   ├── test_improvement.py
│   ├── test_scanner.py
│   ├── test_orchestrator.py
│   └── test_web_agent.py
├── docs/
│   └── architecture.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Version history

**v1.2.0** (March 2026)

SQLite memory backend bringing 27x faster writes and proper persistence across restarts. FLASH ring stays RAM-only by design because session working memory should never touch disk. Web agent with Claude native search using tool version web_search_20260209 powered by Brave. Memory first retrieval checks temporal memory before any network call. Domain trust scoring uses Bayesian reliability tracking that persists across sessions. DuckDuckGo lite fallback works with zero dependencies when no API key is set. Playwright and requests and urllib form a graceful degradation chain for page browsing. Content fingerprinting detects page changes and avoids redundant re fetches. 70 tests passing across all modules.

**v1.1.0** (March 2026)

Zero-trust permission engine with 17 capability types. Self-consciousness module with health grading and source integrity checking. Security self-scanner with two-pass analysis across 21 source files. Self patcher with automatic rollback on verification failure. Self improvement engine covering mistake learning, gap detection, and capability synthesis. Empirical routing weights blending semantic fingerprinting with real observed performance data.

**v1.0.0** (February 2026)

Multi-agent orchestrator with supervisor pattern. TaskGenome DAG decomposition. Four-ring temporal memory system. Bayesian trust ledger per agent. LLM plugin system supporting Anthropic, OpenAI, and Mock.

---

*Built with zero mandatory dependencies. Runs on Python 3.10 and above.*

*Web search requires httpx and an Anthropic API key for Claude native search.*

*Page browsing with full JS rendering requires playwright.*