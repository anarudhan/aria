"""
Unit tests for aria.core.registry: TF-IDF helpers, CapabilityRegistry registration,
agent selection, coalition formation and status management.
"""
import math

import pytest

from aria.core.models import AgentStatus, CapabilityFingerprint, TaskNode
from aria.core.registry import (
    AgentRegistration,
    CapabilityRegistry,
    build_term_frequency,
    cosine_similarity,
    tokenize,
)
from aria.core.trust import TrustLedger

# ─── Helpers ──────────────────────────────────────────────────────────────────


def fp(agent_id, capabilities, domains=(), tools=(), inputs=("text",), outputs=("text",),
       max_concurrent=3) -> CapabilityFingerprint:
    return CapabilityFingerprint(
        agent_id=agent_id,
        capabilities=list(capabilities),
        domains=list(domains),
        tools_available=list(tools),
        input_types=list(inputs),
        output_types=list(outputs),
        max_concurrent_tasks=max_concurrent,
    )


class StubLedger:
    """Trust ledger stand-in with fixed per-agent scores."""

    def __init__(self, scores):
        self.scores = scores

    def get_trust_score(self, agent_id, task_description):
        return self.scores.get(agent_id, 0.6)


@pytest.fixture
def registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(
        fp("researcher", ["search the web for information", "research topics"],
           ["research"], tools=["tavily_search"]),
        object(),
    )
    reg.register(
        fp("coder", ["write python code", "debug programs and scripts"], ["coding"],
           tools=["python_repl"]),
        object(),
    )
    reg.register(
        fp("mailer", ["send email messages", "draft outlook replies"], ["email"],
           tools=["outlook"], inputs=("text", "json"), outputs=("text", "file")),
        object(),
    )
    return reg


# ─── TF-IDF utilities ─────────────────────────────────────────────────────────

class TestTokenize:

    def test_lowercases_and_splits_words(self):
        assert tokenize("Search The WEB") == ["search", "web"]

    def test_drops_stop_words_and_short_tokens(self):
        tokens = tokenize("it is an ox to do the thing")
        assert tokens == ["thing"]

    def test_keeps_underscored_and_hyphenated_identifiers(self):
        assert tokenize("use web_search and read-only mode") == [
            "use", "web_search", "read-only", "mode",
        ]

    def test_empty_and_punctuation_only_input(self):
        assert tokenize("") == []
        assert tokenize("!!! ... ???") == []

    def test_numbers_alone_are_not_tokens(self):
        assert tokenize("2024 12345") == []


class TestBuildTermFrequency:

    def test_frequencies_sum_to_one(self):
        tf = build_term_frequency(["alpha beta beta", "gamma"])
        assert sum(tf.values()) == pytest.approx(1.0)
        assert tf["beta"] == pytest.approx(0.5)
        assert tf["alpha"] == pytest.approx(0.25)

    def test_counts_across_multiple_texts(self):
        tf = build_term_frequency(["alpha", "alpha", "beta"])
        assert tf["alpha"] == pytest.approx(2 / 3)

    @pytest.mark.parametrize("texts", [[], [""], ["the and of"], ["a b"]])
    def test_no_usable_tokens_gives_empty_vector(self, texts):
        assert build_term_frequency(texts) == {}


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = {"a": 0.5, "b": 0.5}
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_scaling_does_not_change_similarity(self):
        assert cosine_similarity({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        score = cosine_similarity({"a": 1.0, "b": 1.0}, {"a": 1.0, "c": 1.0})
        assert score == pytest.approx(0.5)

    def test_is_symmetric(self):
        a, b = {"x": 0.2, "y": 0.8}, {"y": 0.5, "z": 0.5}
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_empty_vectors(self):
        assert cosine_similarity({}, {"a": 1.0}) == 0.0
        assert cosine_similarity({"a": 1.0}, {}) == 0.0
        assert cosine_similarity({}, {}) == 0.0

    def test_zero_magnitude_vector(self):
        assert cosine_similarity({"a": 0.0}, {"a": 1.0}) == 0.0


# ─── Registration ─────────────────────────────────────────────────────────────

class TestRegistration:

    def test_register_builds_term_frequencies_from_all_fields(self):
        reg = CapabilityRegistry()
        fingerprint = fp("a", ["crunch numbers"], ["finance"], tools=["spreadsheet"],
                         inputs=["csvfile"], outputs=["report"])
        reg.register(fingerprint, object())

        tf = fingerprint.term_frequencies
        assert set(tf) == {"crunch", "numbers", "finance", "spreadsheet", "csvfile", "report"}
        assert sum(tf.values()) == pytest.approx(1.0)

    def test_get_agent_returns_the_registered_instance(self):
        reg = CapabilityRegistry()
        ref = object()
        reg.register(fp("a", ["x capability"]), ref)
        assert reg.get_agent("a") is ref
        assert len(reg) == 1

    def test_get_unknown_agent_returns_none(self):
        assert CapabilityRegistry().get_agent("nope") is None

    def test_reregistering_an_id_replaces_the_agent_and_resets_load(self):
        reg = CapabilityRegistry()
        old, new = object(), object()
        reg.register(fp("a", ["first thing"]), old)
        reg.mark_busy("a")
        reg.register(fp("a", ["second thing"]), new)
        assert len(reg) == 1
        assert reg.get_agent("a") is new
        assert reg.list_agents()[0]["load"] == 0

    def test_unregister_removes_agent(self, registry):
        registry.unregister("coder")
        assert registry.get_agent("coder") is None
        assert len(registry) == 2
        assert registry.find_best_agent("write python code") != "coder"

    def test_unregister_unknown_agent_is_a_noop(self, registry):
        registry.unregister("ghost")
        assert len(registry) == 3

    def test_list_agents_reports_domains_status_and_load(self, registry):
        registry.mark_busy("researcher")
        listing = {a["agent_id"]: a for a in registry.list_agents()}
        assert set(listing) == {"researcher", "coder", "mailer"}
        assert listing["researcher"] == {
            "agent_id": "researcher", "domains": ["research"],
            "status": "idle", "load": 1, "max_load": 3,
        }
        assert listing["coder"]["load"] == 0

    def test_registration_is_available_only_when_idle_and_under_capacity(self):
        reg = AgentRegistration(fingerprint=fp("a", ["x"], max_concurrent=2), agent_ref=object())
        assert reg.is_available
        reg.active_task_count = 2
        assert not reg.is_available
        reg.active_task_count = 0
        reg.status = AgentStatus.SUSPENDED
        assert not reg.is_available


class TestIdf:

    def test_rare_terms_weigh_more_than_common_terms(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["shared playwright"]), object())
        reg.register(fp("b", ["shared writing"]), object())
        reg._rebuild_idf()

        assert reg._idf["playwright"] > reg._idf["shared"]
        assert reg._idf["shared"] == pytest.approx(1.0)
        assert reg._idf["playwright"] == pytest.approx(math.log(3 / 2) + 1.0)

    def test_idf_is_rebuilt_after_registry_changes(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["alpha"]), object())
        reg._rebuild_idf()
        assert reg._corpus_dirty is False

        reg.register(fp("b", ["beta"]), object())
        assert reg._corpus_dirty is True
        reg._rebuild_idf()
        assert "beta" in reg._idf

        reg.unregister("b")
        assert reg._corpus_dirty is True

    def test_empty_registry_has_empty_idf(self):
        reg = CapabilityRegistry()
        reg._rebuild_idf()
        assert reg._idf == {}


# ─── Agent selection ──────────────────────────────────────────────────────────

class TestFindBestAgent:

    @pytest.mark.parametrize("task,expected", [
        ("search the web for research information", "researcher"),
        ("write python code and debug scripts", "coder"),
        ("send an email reply in outlook", "mailer"),
    ])
    def test_routes_task_to_the_semantically_closest_agent(self, registry, task, expected):
        assert registry.find_best_agent(task) == expected

    def test_empty_registry_returns_none(self):
        assert CapabilityRegistry().find_best_agent("anything") is None

    def test_excluded_agents_are_skipped(self, registry):
        task = "write python code and debug scripts"
        assert registry.find_best_agent(task) == "coder"
        assert registry.find_best_agent(task, exclude_agents=["coder"]) != "coder"

    def test_all_agents_excluded_returns_none(self, registry):
        assert registry.find_best_agent("anything", exclude_agents=["researcher", "coder",
                                                                       "mailer"]) is None

    def test_required_tools_are_a_hard_filter(self, registry):
        # Text is about email, but the tool requirement forces the coder.
        task = "send an email reply in outlook"
        assert registry.find_best_agent(task, required_tools=["python_repl"]) == "coder"

    def test_agent_must_have_all_required_tools(self, registry):
        assert registry.find_best_agent("task", required_tools=["python_repl", "outlook"]) is None

    def test_unknown_required_tool_returns_none(self, registry):
        assert registry.find_best_agent("search", required_tools=["nonexistent"]) is None

    def test_required_input_and_output_types_filter(self, registry):
        assert registry.find_best_agent("task", required_input_type="json") == "mailer"
        assert registry.find_best_agent("task", required_output_type="file") == "mailer"
        assert registry.find_best_agent("task", required_input_type="video") is None
        assert registry.find_best_agent("task", required_output_type="audio") is None

    def test_busy_and_suspended_agents_are_not_selected(self, registry):
        registry.suspend_agent("coder")
        assert registry.find_best_agent("write python code and debug scripts") != "coder"

        for _ in range(3):
            registry.mark_busy("researcher")
        assert registry.find_best_agent("search the web for research") == "mailer"

    def test_no_available_agent_returns_none(self, registry):
        for agent_id in ("researcher", "coder", "mailer"):
            registry.suspend_agent(agent_id)
        assert registry.find_best_agent("anything") is None

    def test_trust_score_can_flip_the_choice_between_similar_agents(self):
        reg = CapabilityRegistry()
        reg.register(fp("alpha", ["analyse sales data"], ["analysis"]), object())
        reg.register(fp("beta", ["analyse sales data"], ["analysis"]), object())
        task = "analyse sales data"

        assert reg.find_best_agent(task, trust_ledger=StubLedger({"alpha": 0.9, "beta": 0.3})) \
            == "alpha"
        assert reg.find_best_agent(task, trust_ledger=StubLedger({"alpha": 0.3, "beta": 0.9})) \
            == "beta"

    def test_real_trust_ledger_demotes_agent_with_failures(self):
        reg = CapabilityRegistry()
        reg.register(fp("alpha", ["analyse sales data"], ["analysis"]), object())
        reg.register(fp("beta", ["analyse sales data"], ["analysis"]), object())
        ledger = TrustLedger()
        for _ in range(3):
            ledger.record_failure("alpha", "analyse sales data", 1.0)
            ledger.record_success("beta", "analyse sales data", 0.9, 1.0)

        assert reg.find_best_agent("analyse sales data", trust_ledger=ledger) == "beta"

    @pytest.mark.xfail(
        strict=True,
        reason="bug: registry.py find_best_agent starts best_score at -1.0, so an agent that "
               "the trust ledger has suspended (trust 0.0) is still returned when it is the "
               "only candidate, and the executor can fail over back to it",
    )
    def test_ledger_suspended_agent_is_not_returned(self):
        reg = CapabilityRegistry()
        reg.register(fp("flaky", ["analyse sales data"], ["analysis"]), object())
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(4):
            ledger.record_failure("flaky", "analyse sales data", 1.0)
        assert ledger.is_suspended("flaky")

        assert reg.find_best_agent("analyse sales data", trust_ledger=ledger) is None


# ─── Coalition formation ──────────────────────────────────────────────────────

def node(node_id, description, tools=(), deps=()) -> TaskNode:
    return TaskNode(node_id=node_id, description=description,
                    required_tools=list(tools), dependencies=list(deps))


class TestFindCoalition:

    def test_each_node_goes_to_the_best_matching_agent(self, registry):
        nodes = [
            node("t1", "search the web for research information"),
            node("t2", "write python code and debug scripts"),
            node("t3", "send an email reply in outlook"),
        ]
        assert registry.find_coalition(nodes) == {
            "t1": "researcher", "t2": "coder", "t3": "mailer",
        }

    def test_empty_node_list_gives_empty_assignment(self, registry):
        assert registry.find_coalition([]) == {}

    def test_nodes_nobody_can_serve_are_left_unassigned(self, registry):
        nodes = [node("t1", "write python code", tools=["python_repl"]),
                 node("t2", "do something", tools=["quantum_computer"])]
        assignment = registry.find_coalition(nodes)
        assert assignment == {"t1": "coder"}

    def test_empty_registry_assigns_nothing(self):
        assert CapabilityRegistry().find_coalition([node("t1", "anything")]) == {}

    def test_load_is_spread_across_equally_good_agents(self):
        reg = CapabilityRegistry()
        reg.register(fp("alpha", ["analyse sales data"]), object())
        reg.register(fp("beta", ["analyse sales data"]), object())
        nodes = [node(f"t{i}", "analyse sales data") for i in range(1, 5)]

        assignment = reg.find_coalition(nodes)

        assert set(assignment) == {"t1", "t2", "t3", "t4"}
        counts = {a: list(assignment.values()).count(a) for a in set(assignment.values())}
        assert counts == {"alpha": 2, "beta": 2}

    def test_nodes_with_fewer_dependencies_choose_first(self):
        reg = CapabilityRegistry()
        reg.register(fp("alpha", ["analyse sales data"]), object())
        reg.register(fp("beta", ["analyse sales data"]), object())
        nodes = [node("late", "analyse sales data", deps=["early"]),
                 node("early", "analyse sales data")]

        assignment = reg.find_coalition(nodes)

        assert assignment["early"] == "alpha"   # first pick, ties go to registration order
        assert assignment["late"] == "beta"

    def test_trust_ledger_is_respected(self):
        reg = CapabilityRegistry()
        reg.register(fp("alpha", ["analyse sales data"]), object())
        reg.register(fp("beta", ["analyse sales data"]), object())
        ledger = StubLedger({"alpha": 0.2, "beta": 0.95})
        assert reg.find_coalition([node("t1", "analyse sales data")], ledger) == {"t1": "beta"}

    def test_unavailable_agents_are_not_assigned(self, registry):
        registry.suspend_agent("coder")
        assignment = registry.find_coalition([node("t1", "write python code", tools=["python_repl"])])
        assert assignment == {}


# ─── Status management ────────────────────────────────────────────────────────

class TestStatusManagement:

    def test_agent_becomes_busy_at_max_concurrency_and_idle_when_drained(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["x thing"], max_concurrent=2), object())

        reg.mark_busy("a")
        assert reg.list_agents()[0]["status"] == "idle"
        reg.mark_busy("a")
        assert reg.list_agents()[0]["status"] == "busy"
        reg.mark_idle("a")
        reg.mark_idle("a")
        listing = reg.list_agents()[0]
        assert listing["status"] == "idle" and listing["load"] == 0

    def test_mark_idle_never_goes_negative(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["x thing"]), object())
        reg.mark_idle("a")
        reg.mark_idle("a")
        assert reg.list_agents()[0]["load"] == 0

    def test_status_calls_on_unknown_agent_are_noops(self, registry):
        registry.mark_busy("ghost")
        registry.mark_idle("ghost")
        registry.suspend_agent("ghost")
        assert len(registry) == 3

    def test_suspend_agent_sets_status(self, registry):
        registry.suspend_agent("coder")
        listing = {a["agent_id"]: a["status"] for a in registry.list_agents()}
        assert listing["coder"] == "suspended"
        assert listing["researcher"] == "idle"

    @pytest.mark.xfail(
        strict=True,
        reason="bug: registry.py mark_idle sets status to IDLE whenever the load reaches 0, "
               "so an agent suspended while running a task is silently reinstated when the "
               "task finishes (the executor calls mark_idle after every attempt)",
    )
    def test_mark_idle_does_not_lift_a_suspension(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["x thing"]), object())
        reg.mark_busy("a")
        reg.suspend_agent("a")
        reg.mark_idle("a")
        assert reg.list_agents()[0]["status"] == "suspended"

    @pytest.mark.xfail(
        strict=True,
        reason="bug: registry.py mark_idle only restores IDLE when the load hits 0, so an agent "
               "that reached max_concurrent_tasks stays BUSY (and unroutable) until it has "
               "finished every task even though it has spare capacity again",
    )
    def test_agent_with_spare_capacity_is_available_again(self):
        reg = CapabilityRegistry()
        reg.register(fp("a", ["x thing"], max_concurrent=2), object())
        reg.mark_busy("a")
        reg.mark_busy("a")
        reg.mark_idle("a")           # 1 of 2 slots in use
        assert reg.find_best_agent("x thing") == "a"
