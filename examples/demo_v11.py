#!/usr/bin/env python3
"""
ARIA v1.1.0 — Full Demonstration
===================================
Run with:  python demo_v11.py

Shows everything:
  1. Zero-trust permission model (ARIA asks before doing anything)
  2. Self-awareness (health check, source integrity)
  3. Security scanning (ARIA reads its own code)
  4. Self-learning (lessons from mistakes, routing refinement)
  5. Capability gap detection (ARIA notices what it cannot do)
  6. Capability synthesis (ARIA proposes new agents to fill gaps)
  7. Multi-agent task decomposition (the original ARIA magic)

No API key needed — MockLLMPlugin handles all LLM calls.
"""

import sys
import asyncio
import time
from pathlib import Path

# Fix the module path so 'import aria' works from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Import ARIA ──────────────────────────────────────────────────────────────
from aria import (
    ARIAOrchestrator, ARIAConfig, ExecutionContract,
    TaskResult, TaskStatus, MemoryRing,
    aria_agent, Capability,
)
from aria.core.improvement import SelfImprovementEngine


# ─── Mock LLM Plugin (no API key needed) ─────────────────────────────────────

class MockLLMPlugin:
    """Simulates LLM responses for testing without an API key."""

    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        prompt_lower = prompt.lower()

        # Code generation for capability synthesis (MUST be first - most specific)
        if "you are generating a new python agent tool" in prompt_lower:
            return """from aria.agents.base import aria_agent

@aria_agent(
    capabilities=["data analysis", "statistics", "quantitative reasoning"],
    domains=["analysis"],
)
async def data_analyst_agent(node, context, memory):
    \"\"\"Specialized agent for data analysis and quantitative tasks.\"\"\"
    task = node.description
    return "Data analysis complete. Key statistical findings identified and summarized."
"""

        # Capability gap naming (second most specific)
        if "capability name" in prompt_lower or ("suggest" in prompt_lower and "capability" in prompt_lower):
            return "data_analyst_specialist | Agent specialized in data analysis, statistics, and quantitative reasoning"

        # Root cause analysis
        if "root_cause:" in prompt_lower or "analyze this ai task" in prompt_lower:
            return "ROOT_CAUSE: wrong_agent\nADJUSTMENT: Route this task type to a specialist agent with deeper domain knowledge"

        # Task decomposition
        if "decompose" in prompt_lower or "break" in prompt_lower:
            return """
{
  "nodes": [
    {"node_id": "n1", "description": "Research the topic", "required_capabilities": ["research"], "dependencies": []},
    {"node_id": "n2", "description": "Synthesize findings into a summary", "required_capabilities": ["writing"], "dependencies": ["n1"]},
    {"node_id": "n3", "description": "Format final output", "required_capabilities": ["writing"], "dependencies": ["n2"]}
  ]
}"""

        # General research
        if any(w in prompt_lower for w in ["research", "find", "look"]):
            return "Research complete. Found 12 relevant sources. Key insights: the topic shows strong growth trends."

        # General writing
        if any(w in prompt_lower for w in ["write", "draft", "summarize", "format"]):
            return "Document prepared with executive overview, detailed findings, and actionable recommendations."

        return f"Task processed successfully."

    async def chat(self, messages: list, max_tokens: int = 500) -> str:
        if messages:
            return await self.complete(messages[-1].get("content", ""), max_tokens)
        return "Ready."


# ─── Agent Definitions ────────────────────────────────────────────────────────

@aria_agent(
    capabilities=["research", "information retrieval", "web search", "fact finding"],
    domains=["research", "information"],

)
async def researcher(node, context, memory):
    """Research agent — finds information, searches sources, retrieves data."""
    task = node.description
    await asyncio.sleep(0.1)  # Simulate work
    return (
        f"Research complete for: '{task[:50]}'. "
        f"Found relevant information across 8 sources. "
        f"Key findings: primary trend identified with supporting evidence."
    )


@aria_agent(
    capabilities=["writing", "summarization", "formatting", "document creation", "drafting"],
    domains=["writing", "communication"],

)
async def writer(node, context, memory):
    """Writing agent — drafts documents, summaries, and structured content."""
    task = node.description
    await asyncio.sleep(0.08)
    return (
        f"Written content for: '{task[:50]}'. "
        f"Produced a 3-section document with executive summary, "
        f"detailed findings, and recommendations. Word count: ~450."
    )


@aria_agent(
    capabilities=["email management", "outlook", "inbox", "reply drafting", "email search"],
    domains=["email", "communication"],

)
async def email_agent(node, context, memory):
    """Email agent — reads Outlook inbox, drafts replies, searches emails."""
    task = node.description
    await asyncio.sleep(0.05)
    return f"Email task handled: '{task[:50]}'. Draft prepared and ready for approval."


@aria_agent(
    capabilities=["microsoft teams", "channel messages", "team chat", "post reply"],
    domains=["messaging", "collaboration"],

)
async def teams_agent(node, context, memory):
    """Teams agent — reads channels, drafts replies, summarizes discussions."""
    task = node.description
    await asyncio.sleep(0.05)
    return f"Teams task handled: '{task[:50]}'. Channel summary prepared."


# ─── Helper: Print Section Header ─────────────────────────────────────────────

def section(title: str):
    print(f"\n{'━' * 62}")
    print(f"  {title}")
    print("━" * 62)


# ─── Main Demo ────────────────────────────────────────────────────────────────

async def main():
    llm = MockLLMPlugin()

    # ── Build ARIA ────────────────────────────────────────────────────────────
    aria_root = str(Path(__file__).parent.parent)

    aria = ARIAOrchestrator(
        config    = ARIAConfig(enable_retrospective=True),
        llm_plugin = llm,
        aria_root  = aria_root,
    )

    # Attach the self-improvement engine
    aria.learning = SelfImprovementEngine(
        llm_plugin = llm,
        autonomous_synthesis = False,   # Propose only — user must approve
    )

    # Register agents
    aria.register_agents([researcher, writer, email_agent, teams_agent])

    print("\n✅ ARIA v1.1.0 initialized with 4 agents\n")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 1: Permission model
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 1: Zero-Trust Permission Model")

    print("ARIA starts with zero permissions. Everything must be granted explicitly.\n")

    # Grant permissions for this demo session
    aria.permissions.grant("sathis", Capability.MEMORY_WRITE, hours=8,
                           task_context="Development session")
    aria.permissions.grant("sathis", Capability.WEB_SEARCH, hours=8)
    aria.permissions.grant("sathis", Capability.DOCUMENT_CREATE, hours=8)
    aria.permissions.grant("sathis", Capability.AGENT_SPAWN, hours=4)
    aria.permissions.grant("sathis", Capability.CODE_EXECUTE, hours=4)

    # Show what's been granted
    print()
    aria.permissions.print_grant_summary("sathis")

    # Show what a denied permission looks like
    print("\nTrying to check EMAIL_SEND (not granted yet):")
    has_email = aria.permissions.check("sathis", Capability.EMAIL_SEND)
    print(f"  Email send permission: {'✅' if has_email else '❌ Denied (as expected)'}")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 2: Self-awareness and health
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 2: Self-Awareness and Health Check")

    aria.self_report()
    print("What ARIA knows about itself:")
    print(aria.what_i_am())


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 3: Security scan (ARIA reads its own code)
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 3: Security Self-Scan")

    print("ARIA scanning its own source code for vulnerabilities...\n")
    report = await aria.security_scan("sathis")
    print(f"\nScan summary:")
    print(f"  Critical: {report['summary']['critical']}")
    print(f"  High:     {report['summary']['high']}")
    print(f"  Medium:   {report['summary']['medium']}")
    print(f"  Low:      {report['summary']['low']}")
    print(f"  Files:    {report['files_scanned']}")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 4: Source integrity check
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 4: Source Integrity Verification")

    violations = aria.integrity_check()
    if violations:
        print(f"⚠️  {len(violations)} file change(s) detected:")
        for v in violations:
            print(f"  [{v['change_type']}] {v['file']}")
    else:
        print("✅ All source files match baseline hashes — no unexpected changes")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 5: Normal task execution with learning
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 5: Task Execution with Self-Learning")

    print("Running a research task — ARIA will learn from the outcome...\n")

    result = await aria.run(
        task    = "Research the latest AI trends in enterprise software and write a structured summary",
        user_id = "sathis",
        contract = ExecutionContract(max_seconds=30, max_cost_usd=1.0),
    )

    print(f"\nResult status:   {result.status.value}")
    print(f"Quality score:   {result.quality_score:.2f}")
    print(f"Answer preview:  {(result.final_answer or '')[:120]}...")

    # Manually trigger learning for this result (in production this is automatic)
    from aria.core.models import TaskGenome, TaskNode
    mock_genome = TaskGenome(original_task="Research AI trends and summarize")
    mock_genome.nodes["n1"] = TaskNode(node_id="n1", description="Research AI trends", status=TaskStatus.COMPLETED, result="Found trends", started_at=time.time()-1.2, completed_at=time.time())
    mock_genome.nodes["n2"] = TaskNode(node_id="n2", description="Write summary", status=TaskStatus.COMPLETED, result="Summary written", started_at=time.time()-0.8, completed_at=time.time())

    lessons = await aria.learning.learn_from_execution(
        task     = "Research the latest AI trends in enterprise software and write a structured summary",
        result   = result,
        coalition = {"n1": "researcher_agent", "n2": "writer_agent"},
        genome   = mock_genome,
        llm_plugin = llm,
    )
    print(f"\nLessons recorded: {len(lessons)}")
    for l in lessons:
        print(f"  • Outcome: {l.outcome} | Root: {l.root_cause}")
        print(f"    Adjustment: {l.adjustment}")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 6: Simulate repeated failures to trigger capability gap detection
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 6: Learning from Failures and Capability Gap Detection")

    print("Simulating repeated failures on 'data analysis' tasks...\n")
    print("(In production these would be real failed executions)\n")

    failed_result = TaskResult(
        original_task = "Analyze quarterly sales data and identify statistical anomalies",
        status        = TaskStatus.FAILED,
    )
    failed_result.error_messages.append("No agent with data analysis capabilities found")
    failed_result.quality_score = 0.1
    failed_result.total_seconds = 2.0

    # Simulate 3 failures to trigger gap detection
    for i in range(3):
        task_str = f"Perform statistical analysis on dataset {i+1} to find trends and outliers"
        print(f"  Failure {i+1}: '{task_str[:55]}'")
        await aria.learning.learn_from_execution(
            task      = task_str,
            result    = failed_result,
            coalition = {},
            genome    = mock_genome,
            llm_plugin = llm,
        )
        await asyncio.sleep(0.05)

    # Show the detected gap
    gaps = aria.learning.get_open_gaps()
    print(f"\n🔴 Capability gaps detected: {len(gaps)}")
    for g in gaps:
        print(f"  • Task type: {g.task_type}")
        print(f"    Failures:  {g.failure_count}x")
        print(f"    Proposed:  {g.proposed_cap}")
        print(f"    Gap ID:    {g.gap_id}")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 7: Capability synthesis (ARIA proposes a new agent)
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 7: Capability Synthesis — ARIA Grows a New Agent")

    if gaps:
        gap = gaps[0]
        print(f"ARIA detected a gap in '{gap.task_type}' tasks.")
        print("With AGENT_SPAWN permission, ARIA will synthesize a new specialized agent.\n")

        synth = await aria.learning.synthesize_capability(
            gap_id            = gap.gap_id,
            user_id           = "sathis",
            permission_engine = aria.permissions,
            llm_plugin        = llm,
        )

        if synth:
            print(f"\n  Synthesized capability ID: {synth.cap_id}")
            print(f"  Name: {synth.name}")
            print(f"  Status: {'Awaiting user approval'}")
            print(f"\n  Approve with: aria.learning.approve_capability('{synth.cap_id}', 'sathis')")
            print(f"  Load with:    await aria.learning.load_approved_capability('{synth.cap_id}', aria, aria.permissions, 'sathis')")

            # Auto-approve for demo purposes
            print("\n  [Demo] Auto-approving and loading the synthesized agent...")
            aria.learning.approve_capability(synth.cap_id, "sathis")
            loaded = await aria.learning.load_approved_capability(
                synth.cap_id, aria, aria.permissions, "sathis"
            )
            if loaded:
                print(f"\n  ✅ New agent registered! ARIA now has {len(aria.registry.list_agents())} agents")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 8: Routing weights (empirical learning in action)
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 8: Empirical Routing Intelligence")

    print("After enough executions, ARIA blends semantic fingerprinting")
    print("with real observed performance to pick the best agent.\n")

    # Show routing weights accumulated during the demo
    if aria.learning._routing_weights:
        print("Routing weights recorded this session:")
        for key, w in aria.learning._routing_weights.items():
            if w.executions > 0:
                reliable = "📊 reliable" if w.is_reliable else "🔄 accumulating"
                print(
                    f"  {w.agent_id:<25} on {w.task_category:<12} "
                    f"quality={w.avg_quality:.2f} success={w.success_rate:.0%} "
                    f"n={w.executions} {reliable}"
                )
    else:
        print("  (No weights yet — need more task executions to build empirical data)")

    # Show how blending works
    print("\nExample: blending semantic + empirical score for future routing:")
    semantic_score = 0.75  # From TF-IDF fingerprint matching
    empirical = aria.learning.get_empirical_score("researcher_agent", "research topic")
    blended   = aria.learning.blend_routing_score("researcher_agent", "research topic", semantic_score)
    print(f"  Semantic similarity: {semantic_score:.2f}")
    print(f"  Empirical score:     {empirical:.2f}")
    print(f"  Blended score:       {blended:.2f}")


    # ═══════════════════════════════════════════════════════════════════
    # DEMO 9: Full learning report
    # ═══════════════════════════════════════════════════════════════════
    section("DEMO 9: Full Self-Improvement Report")

    aria.learning.print_learning_report()


    # ═══════════════════════════════════════════════════════════════════
    # Final status
    # ═══════════════════════════════════════════════════════════════════
    section("Final System Status")

    aria.print_status()

    health = aria.health_report()
    print(f"\n  ARIA Health Grade: {health['grade']}")
    print(f"  Is healthy:        {'✅' if health['is_healthy'] else '⚠️ Issues detected'}")
    print(f"  Tasks completed:   {health['tasks_completed']}")
    print(f"  Error rate:        {health['error_rate']:.1%}")
    print(f"  Open vuln count:   {health['open_vulnerabilities']}")

    print("\n" + "═" * 62)
    print("  ARIA v1.1.0 Demo Complete")
    print("  Capabilities demonstrated:")
    print("    ✅ Zero-trust permission engine")
    print("    ✅ Self-awareness and health monitoring")
    print("    ✅ Security self-scanning")
    print("    ✅ Source integrity verification")
    print("    ✅ Learning from task outcomes")
    print("    ✅ Capability gap detection")
    print("    ✅ Capability synthesis (new agent from scratch)")
    print("    ✅ Empirical routing weight tracking")
    print("    ✅ Multi-agent task decomposition")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
