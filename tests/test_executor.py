"""
Unit tests for aria.core.executor.ExecutionEngine and print_execution_graph.

The engine is exercised with in-process fake agents, a real registry, a real
(in-memory) trust ledger and a real (RAM only) memory. No network or API keys.
"""
import asyncio

import pytest

from aria.core.executor import STATUS_ICONS, ExecutionEngine, print_execution_graph
from aria.core.memory import TemporalMemory
from aria.core.models import (
    AgentMessage,
    AgentStatus,
    CapabilityFingerprint,
    ExecutionContract,
    MemoryRing,
    MessageType,
    TaskGenome,
    TaskNode,
    TaskStatus,
)
from aria.core.registry import CapabilityRegistry
from aria.core.trust import TrustLedger

# ─── Test doubles and helpers ─────────────────────────────────────────────────


class FakeAgent:
    """Minimal agent: records every call and delegates to an optional behaviour."""

    def __init__(self, agent_id, behaviour=None):
        self.agent_id = agent_id
        self.behaviour = behaviour
        self.calls = []          # list of (node_id, context)
        self._outbox = asyncio.Queue()
        self._message_queue = asyncio.Queue()

    async def execute(self, node, context, memory):
        self.calls.append((node.node_id, context))
        if self.behaviour is not None:
            return await self.behaviour(node, context)
        return f"{self.agent_id} did {node.node_id}"


def add_agent(registry, agent_id, behaviour=None, max_concurrent=3) -> FakeAgent:
    agent = FakeAgent(agent_id, behaviour)
    registry.register(
        CapabilityFingerprint(
            agent_id=agent_id,
            capabilities=["research", "analysis"],
            domains=["research"],
            tools_available=[],
            input_types=["text"],
            output_types=["text"],
            max_concurrent_tasks=max_concurrent,
        ),
        agent,
    )
    return agent


def build_genome(layers, node_seconds=10.0, deps=None, task="research the market") -> TaskGenome:
    """Genome with one node per id in `layers` (a list of lists of node ids)."""
    genome = TaskGenome(original_task=task)
    deps = deps or {}
    for layer in layers:
        for node_id in layer:
            genome.add_node(TaskNode(
                node_id=node_id,
                description=f"research task {node_id}",
                dependencies=deps.get(node_id, []),
                contract=ExecutionContract(max_seconds=node_seconds),
            ))
    genome.execution_order = [list(layer) for layer in layers]
    return genome


def tally(ledger: TrustLedger, agent_id: str):
    """(successes, failures, timeouts) recorded for an agent across all categories."""
    records = list(ledger._records.get(agent_id, {}).values())
    return (
        sum(r.success_count for r in records),
        sum(r.failure_count for r in records),
        sum(r.timeout_count for r in records),
    )


async def boom(node, context):
    raise ValueError(f"boom in {node.node_id}")


async def sleeps_forever(node, context):
    await asyncio.sleep(30)


@pytest.fixture
def registry():
    return CapabilityRegistry()


@pytest.fixture
def ledger():
    return TrustLedger()


@pytest.fixture
def memory():
    return TemporalMemory()


@pytest.fixture
def engine(registry, ledger, memory):
    return ExecutionEngine(registry, ledger, memory, enable_graph_print=False)


def contract(**kw) -> ExecutionContract:
    kw.setdefault("max_seconds", 30)
    return ExecutionContract(**kw)


# ─── Happy path ───────────────────────────────────────────────────────────────

class TestSuccessfulRuns:

    async def test_single_node_completes(self, engine, registry, ledger):
        agent = add_agent(registry, "a1")
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "a1"}, contract())

        assert result.status == TaskStatus.COMPLETED
        assert result.genome_id == genome.genome_id
        assert result.original_task == "research the market"
        assert result.node_results == {"t1": "a1 did t1"}
        assert result.final_answer == "a1 did t1"
        assert result.agents_used == ["a1"]
        assert result.error_messages == []
        assert result.completed_at is not None
        assert result.total_seconds >= 0
        assert len(agent.calls) == 1

        node = genome.nodes["t1"]
        assert node.status == TaskStatus.COMPLETED
        assert node.result == "a1 did t1"
        assert node.attempt_count == 1
        assert node.duration_seconds is not None

    async def test_success_updates_trust_and_releases_agent(self, engine, registry, ledger):
        add_agent(registry, "a1")
        before = ledger.get_trust_score("a1", "research task")

        await engine.run(build_genome([["t1"]]), {"t1": "a1"}, contract())

        assert tally(ledger, "a1") == (1, 0, 0)
        assert ledger.get_trust_score("a1", "research task") > before
        reg = registry._agents["a1"]
        assert reg.active_task_count == 0
        assert reg.status == AgentStatus.IDLE

    async def test_layers_run_in_order_and_downstream_sees_upstream_results(
        self, engine, registry
    ):
        agent = add_agent(registry, "a1")
        genome = build_genome([["t1"], ["t2"]], deps={"t2": ["t1"]})

        result = await engine.run(genome, {"t1": "a1", "t2": "a1"}, contract())

        assert [c[0] for c in agent.calls] == ["t1", "t2"]
        first_ctx, second_ctx = agent.calls[0][1], agent.calls[1][1]
        assert "upstream_results" not in first_ctx
        assert "[Result from subtask t1]" in second_ctx["upstream_results"]
        assert "a1 did t1" in second_ctx["upstream_results"]
        # Final answer is the output of the last layer only.
        assert result.final_answer == "a1 did t2"

    async def test_upstream_results_are_truncated_to_2000_chars(self, engine, registry):
        async def long_output(node, context):
            return "x" * 5000 if node.node_id == "t1" else "short"

        agent = add_agent(registry, "a1", long_output)
        genome = build_genome([["t1"], ["t2"]])

        await engine.run(genome, {"t1": "a1", "t2": "a1"}, contract())

        upstream = agent.calls[1][1]["upstream_results"]
        assert upstream.count("x") == 2000

    async def test_nodes_in_a_layer_run_concurrently(self, engine, registry):
        started = {"t1": asyncio.Event(), "t2": asyncio.Event()}

        async def rendezvous(node, context):
            # Each node waits for the other one to start; this deadlocks (and
            # times out) if the layer were executed sequentially.
            other = "t2" if node.node_id == "t1" else "t1"
            started[node.node_id].set()
            await asyncio.wait_for(started[other].wait(), timeout=2)
            return node.node_id

        add_agent(registry, "a1", rendezvous)
        add_agent(registry, "a2", rendezvous)
        genome = build_genome([["t1", "t2"]])

        result = await engine.run(genome, {"t1": "a1", "t2": "a2"}, contract())

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results == {"t1": "t1", "t2": "t2"}

    async def test_final_answer_joins_all_last_layer_outputs(self, engine, registry):
        add_agent(registry, "a1")
        add_agent(registry, "a2")
        genome = build_genome([["t1", "t2"]])

        result = await engine.run(genome, {"t1": "a1", "t2": "a2"}, contract())

        assert sorted(result.final_answer.split("\n\n")) == ["a1 did t1", "a2 did t2"]
        assert sorted(result.agents_used) == ["a1", "a2"]

    async def test_max_parallel_agents_limits_concurrency(self, registry, ledger, memory):
        running = 0
        peak = 0

        async def tracked(node, context):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return "ok"

        add_agent(registry, "a1", tracked)
        limited = ExecutionEngine(registry, ledger, memory, enable_graph_print=False,
                                  max_parallel_agents=2)
        genome = build_genome([["t1", "t2", "t3", "t4"]])

        result = await limited.run(genome, {n: "a1" for n in genome.nodes}, contract())

        assert result.status == TaskStatus.COMPLETED
        assert len(result.node_results) == 4
        assert peak == 2
        assert registry._agents["a1"].active_task_count == 0

    async def test_initial_context_reaches_agents_but_not_upstream_summary(
        self, engine, registry
    ):
        sentinel = object()
        agent = add_agent(registry, "a1")
        engine._initial_context = {"llm_plugin": sentinel}
        genome = build_genome([["t1"], ["t2"]])

        await engine.run(genome, {"t1": "a1", "t2": "a1"}, contract())

        for _, ctx in agent.calls:
            assert ctx["llm_plugin"] is sentinel
        assert "llm_plugin" not in agent.calls[1][1]["upstream_results"]

    async def test_final_answer_is_stored_in_hot_memory(self, engine, registry, memory):
        add_agent(registry, "a1")
        genome = build_genome([["t1"]])

        await engine.run(genome, {"t1": "a1"}, contract())

        entries = memory.recall_by_tag(f"genome:{genome.genome_id}", rings=[MemoryRing.HOT])
        assert len(entries) == 1
        assert entries[0].agent_id == "orchestrator"
        assert "a1 did t1" in entries[0].content

    async def test_message_queue_registered_for_each_coalition_agent(self, engine, registry):
        add_agent(registry, "a1")
        add_agent(registry, "a2")
        await engine.run(build_genome([["t1", "t2"]]), {"t1": "a1", "t2": "a2"}, contract())
        assert set(engine._message_bus) == {"a1", "a2"}

    async def test_empty_genome_produces_no_results(self, engine):
        result = await engine.run(TaskGenome(original_task="nothing"), {}, contract())
        assert result.status == TaskStatus.COMPLETED
        assert result.final_answer == "No results produced"
        assert result.node_results == {}

    async def test_already_finished_nodes_are_skipped(self, engine, registry):
        agent = add_agent(registry, "a1")
        genome = build_genome([["t1"]])
        genome.nodes["t1"].status = TaskStatus.COMPLETED

        result = await engine.run(genome, {"t1": "a1"}, contract())

        assert agent.calls == []
        assert result.node_results == {}


# ─── Failure paths ────────────────────────────────────────────────────────────

class TestFailures:

    async def test_node_without_assigned_agent_fails(self, engine, registry):
        add_agent(registry, "a1")
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {}, contract())

        assert result.status == TaskStatus.FAILED
        assert genome.nodes["t1"].status == TaskStatus.FAILED
        assert "No agent assigned to node t1" in genome.nodes["t1"].error
        assert any("[t1]" in m and "No agent assigned" in m for m in result.error_messages)
        assert result.node_results == {}
        assert result.final_answer == "No results produced"

    async def test_agent_missing_from_registry_fails(self, engine):
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "ghost"}, contract(max_retries=0))

        assert result.status == TaskStatus.FAILED
        assert any("failed after 1 attempts" in m for m in result.error_messages)

    async def test_failing_agent_marks_node_failed_and_penalises_trust(
        self, engine, registry, ledger
    ):
        add_agent(registry, "a1", boom)
        genome = build_genome([["t1"]])
        before = ledger.get_trust_score("a1", "research task")

        result = await engine.run(genome, {"t1": "a1"}, contract(max_retries=0))

        assert result.status == TaskStatus.FAILED
        assert genome.nodes["t1"].status == TaskStatus.FAILED
        assert "failed after 1 attempts" in genome.nodes["t1"].error
        assert tally(ledger, "a1") == (0, 1, 0)
        assert ledger.get_trust_score("a1", "research task") < before
        assert registry._agents["a1"].active_task_count == 0
        assert registry._agents["a1"].status == AgentStatus.IDLE

    @pytest.mark.xfail(
        strict=True,
        reason="bug: executor.py run() overwrites node.error with the generic 'failed after N "
               "attempts' RuntimeError, so the agent's real exception message is only printed "
               "and never reaches node.error or result.error_messages",
    )
    async def test_root_cause_of_agent_failure_is_reported(self, engine, registry):
        add_agent(registry, "a1", boom)
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "a1"}, contract(max_retries=0))

        assert any("boom in t1" in m for m in result.error_messages)

    async def test_failure_retries_on_a_backup_agent(self, engine, registry, ledger):
        bad = add_agent(registry, "bad", boom)
        good = add_agent(registry, "good")
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "bad"}, contract(max_retries=2))

        assert result.status == TaskStatus.COMPLETED
        assert result.node_results == {"t1": "good did t1"}
        assert result.error_messages == []
        assert genome.nodes["t1"].attempt_count == 2
        assert len(bad.calls) == 1 and len(good.calls) == 1
        assert tally(ledger, "bad") == (0, 1, 0)
        assert tally(ledger, "good") == (1, 0, 0)

    async def test_zero_retries_means_no_failover(self, engine, registry):
        add_agent(registry, "bad", boom)
        good = add_agent(registry, "good")
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "bad"}, contract(max_retries=0))

        assert result.status == TaskStatus.FAILED
        assert good.calls == []
        assert genome.nodes["t1"].attempt_count == 1

    async def test_retries_stop_when_no_untried_agent_is_left(self, engine, registry):
        bad1 = add_agent(registry, "bad1", boom)
        bad2 = add_agent(registry, "bad2", boom)
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "bad1"}, contract(max_retries=5))

        assert result.status == TaskStatus.FAILED
        assert len(bad1.calls) == 1 and len(bad2.calls) == 1   # each agent tried once
        assert genome.nodes["t1"].attempt_count == 2
        assert any("failed after 2 attempts" in m for m in result.error_messages)

    async def test_timeout_fails_node_and_records_timeout(self, engine, registry, ledger):
        add_agent(registry, "slow", sleeps_forever)
        genome = build_genome([["t1"]], node_seconds=0.05)

        result = await asyncio.wait_for(
            engine.run(genome, {"t1": "slow"}, contract(max_retries=0)), timeout=5
        )

        assert result.status == TaskStatus.FAILED
        assert genome.nodes["t1"].status == TaskStatus.FAILED
        assert tally(ledger, "slow") == (0, 0, 1)
        assert registry._agents["slow"].active_task_count == 0

    @pytest.mark.xfail(
        strict=True,
        reason="bug: executor.py _run_node_with_retry gives every retry the same node "
               "contract, which is already expired after a timeout, so the failover to a "
               "backup agent after a timeout can never run",
    )
    async def test_timeout_fails_over_to_backup_agent(self, engine, registry):
        add_agent(registry, "slow", sleeps_forever)
        backup = add_agent(registry, "backup")
        genome = build_genome([["t1"]], node_seconds=0.2)

        result = await asyncio.wait_for(
            engine.run(genome, {"t1": "slow"}, contract(max_retries=2)), timeout=5
        )

        assert result.status == TaskStatus.COMPLETED
        assert len(backup.calls) == 1

    async def test_expired_node_contract_never_calls_the_agent(self, engine, registry):
        agent = add_agent(registry, "a1")
        genome = build_genome([["t1"]], node_seconds=0.0)

        result = await engine.run(genome, {"t1": "a1"}, contract())

        assert agent.calls == []
        assert result.status == TaskStatus.FAILED
        assert any("contract expired" in m for m in result.error_messages)

    async def test_expired_run_contract_runs_nothing(self, engine, registry):
        agent = add_agent(registry, "a1")
        genome = build_genome([["t1"], ["t2"]])

        result = await engine.run(genome, {"t1": "a1", "t2": "a1"}, contract(max_seconds=0))

        assert agent.calls == []
        assert all(n.status == TaskStatus.PENDING for n in genome.nodes.values())
        assert result.final_answer == "No results produced"
        assert result.node_results == {}

    @pytest.mark.xfail(
        strict=True,
        reason="bug: executor.py run() reports COMPLETED when the run contract has expired "
               "and no node was executed at all; it should surface a failure/timeout",
    )
    async def test_expired_run_contract_is_not_reported_as_completed(self, engine, registry):
        add_agent(registry, "a1")
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "a1"}, contract(max_seconds=0))

        assert result.status != TaskStatus.COMPLETED

    async def test_partial_failure_in_a_layer_still_completes(self, engine, registry):
        add_agent(registry, "good")
        add_agent(registry, "bad", boom)
        genome = build_genome([["t1", "t2"]])

        result = await engine.run(genome, {"t1": "good", "t2": "bad"}, contract(max_retries=0))

        assert result.status == TaskStatus.COMPLETED     # at least one node succeeded
        assert result.node_results == {"t1": "good did t1"}
        assert result.final_answer == "good did t1"
        assert len(result.error_messages) == 1 and "[t2]" in result.error_messages[0]
        assert genome.nodes["t2"].status == TaskStatus.FAILED

    async def test_failed_last_layer_falls_back_to_earlier_results(self, engine, registry):
        add_agent(registry, "good")
        add_agent(registry, "bad", boom)
        genome = build_genome([["t1"], ["t2"]])

        result = await engine.run(genome, {"t1": "good", "t2": "bad"}, contract(max_retries=0))

        assert result.final_answer == "good did t1"
        assert result.status == TaskStatus.COMPLETED

    async def test_all_nodes_failing_reports_failed(self, engine, registry):
        add_agent(registry, "bad", boom)
        genome = build_genome([["t1", "t2"]])

        result = await engine.run(genome, {"t1": "bad", "t2": "bad"}, contract(max_retries=0))

        assert result.status == TaskStatus.FAILED
        assert len(result.error_messages) == 2
        assert result.final_answer == "No results produced"

    async def test_error_messages_are_truncated(self, engine, registry):
        async def long_error(node, context):
            raise ValueError("e" * 1000)

        add_agent(registry, "bad", long_error)
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "bad"}, contract(max_retries=0))

        assert all(len(m) <= 200 + len("[t1] ") for m in result.error_messages)

    @pytest.mark.xfail(
        strict=True,
        reason="bug: executor.py run() checks isinstance(layer_result, Exception), but "
               "asyncio.CancelledError is a BaseException, so a node whose agent raises it "
               "is recorded as a successful result",
    )
    async def test_agent_raising_cancelled_error_is_not_a_success(self, engine, registry):
        async def cancelled(node, context):
            raise asyncio.CancelledError()

        add_agent(registry, "a1", cancelled)
        genome = build_genome([["t1"]])

        result = await engine.run(genome, {"t1": "a1"}, contract(max_retries=0))

        assert "t1" not in result.node_results

    async def test_memory_failure_is_reported_as_executor_error(self, registry, ledger):
        class BrokenMemory:
            def store_result(self, **kwargs):
                raise OSError("disk full")

        add_agent(registry, "a1")
        eng = ExecutionEngine(registry, ledger, BrokenMemory(), enable_graph_print=False)

        # store_result is called outside the try block, so the error propagates.
        with pytest.raises(OSError):
            await eng.run(build_genome([["t1"]]), {"t1": "a1"}, contract())


# ─── Execution graph printing ─────────────────────────────────────────────────

class TestExecutionGraph:

    def test_prints_nodes_agents_and_status_icons(self, capsys):
        genome = build_genome([["t1", "t2"], ["t3"]])
        genome.nodes["t1"].status = TaskStatus.COMPLETED
        genome.nodes["t2"].status = TaskStatus.FAILED

        print_execution_graph(genome, {"t1": "alpha", "t3": "gamma"})
        out = capsys.readouterr().out

        assert "Layer 1 (parallel)" in out
        assert "Layer 2 " in out and "Layer 2 (parallel)" not in out
        assert "[t1]" in out and "alpha" in out
        assert "unassigned" in out                       # t2 has no agent
        assert STATUS_ICONS[TaskStatus.COMPLETED] in out
        assert STATUS_ICONS[TaskStatus.FAILED] in out
        assert STATUS_ICONS[TaskStatus.PENDING] in out

    def test_every_task_status_has_an_icon(self):
        assert set(STATUS_ICONS) == set(TaskStatus)

    async def test_engine_prints_graph_when_enabled(self, registry, ledger, memory, capsys):
        add_agent(registry, "a1")
        eng = ExecutionEngine(registry, ledger, memory, enable_graph_print=True)

        await eng.run(build_genome([["t1"]]), {"t1": "a1"}, contract())

        assert "ARIA Execution" in capsys.readouterr().out

    async def test_engine_is_silent_about_graph_when_disabled(self, engine, registry, capsys):
        add_agent(registry, "a1")
        await engine.run(build_genome([["t1"]]), {"t1": "a1"}, contract())
        assert "ARIA Execution" not in capsys.readouterr().out


# ─── Coalition message bus ────────────────────────────────────────────────────

class TestMessageDelivery:

    async def _wait_for(self, queue: asyncio.Queue, timeout: float = 3.0):
        return await asyncio.wait_for(queue.get(), timeout=timeout)

    async def _run_bus(self, engine, genome_id, coalition):
        task = asyncio.ensure_future(engine._deliver_messages(genome_id, coalition))
        return task

    async def _stop(self, task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _msg(self, sender, recipient, genome_id="g1") -> AgentMessage:
        return AgentMessage(
            msg_type=MessageType.STATUS_UPDATE, sender_id=sender,
            recipient_id=recipient, genome_id=genome_id, payload={"hello": "world"},
        )

    async def test_direct_message_reaches_only_the_recipient(self, engine, registry):
        a1 = add_agent(registry, "a1")
        a2 = add_agent(registry, "a2")
        a3 = add_agent(registry, "a3")
        coalition = {"t1": "a1", "t2": "a2", "t3": "a3"}
        msg = self._msg("a1", "a2")
        await a1._outbox.put(msg)

        task = await self._run_bus(engine, "g1", coalition)
        try:
            delivered = await self._wait_for(a2._message_queue)
        finally:
            await self._stop(task)

        assert delivered is msg
        assert a1._outbox.empty()
        assert a3._message_queue.empty()
        assert a1._message_queue.empty()

    async def test_broadcast_reaches_everyone_except_the_sender(self, engine, registry):
        a1 = add_agent(registry, "a1")
        a2 = add_agent(registry, "a2")
        a3 = add_agent(registry, "a3")
        coalition = {"t1": "a1", "t2": "a2", "t3": "a3"}
        msg = self._msg("a1", None)
        await a1._outbox.put(msg)

        task = await self._run_bus(engine, "g1", coalition)
        try:
            got2 = await self._wait_for(a2._message_queue)
            got3 = await self._wait_for(a3._message_queue)
        finally:
            await self._stop(task)

        assert got2 is msg and got3 is msg
        assert a1._message_queue.empty()

    async def test_messages_from_other_genomes_are_dropped(self, engine, registry):
        a1 = add_agent(registry, "a1")
        a2 = add_agent(registry, "a2")
        await a1._outbox.put(self._msg("a1", "a2", genome_id="other-genome"))

        task = await self._run_bus(engine, "g1", {"t1": "a1", "t2": "a2"})
        try:
            await asyncio.sleep(0.4)
        finally:
            await self._stop(task)

        assert a1._outbox.empty()            # drained...
        assert a2._message_queue.empty()     # ...but never delivered

    async def test_message_to_unknown_recipient_is_discarded_without_crashing(
        self, engine, registry
    ):
        a1 = add_agent(registry, "a1")
        await a1._outbox.put(self._msg("a1", "nobody"))

        task = await self._run_bus(engine, "g1", {"t1": "a1"})
        try:
            await asyncio.sleep(0.4)
            assert not task.done()           # bus still alive
        finally:
            await self._stop(task)

        assert a1._outbox.empty()
