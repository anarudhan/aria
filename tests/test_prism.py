"""
Tests for PRISM — Progressive Reasoning with Integrated Self-Monitoring

These tests verify that PRISM thinks honestly. Every test has a clear
intention — not just that the code runs, but that the cognitive logic
is correct. PRISM's value is its honesty, so tests that only check
that a number is returned without verifying its direction or meaning
would be worthless.

Test groups:
    PRISMComprehension  — Question 1: does ARIA understand the task?
    PRISMNovelty        — Question 2: how new is this?
    PRISMHistory        — Question 3: what does the track record say?
    PRISMStrategy       — Question 4: what approach to take?
    PRISMMonitoring     — Live drift detection during execution
    PRISMReflection     — Post-task calibration and self-knowledge update
    PRISMCalibration    — Long-term self-model accuracy reporting
    PRISMIntegration    — Full three-phase cycle end to end
    PRISMPersistence    — Save and reload calibration history
"""

import json
from unittest.mock import MagicMock

import pytest

from aria.core.prism import CalibrationRecord, DriftEvent, PRISMEngine, PRISMState

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def prism():
    """Fresh PRISM with no calibration history and no persistence."""
    return PRISMEngine(persist_path=None)


@pytest.fixture
def prism_with_history(tmp_path):
    """PRISM with a temporary persist path — tests real save/load."""
    persist_file = tmp_path / "aria_memory" / "memory.db"
    return PRISMEngine(persist_path=str(persist_file))


@pytest.fixture
def memory_entry():
    """A fake memory object that mimics what TemporalMemory returns."""
    m = MagicMock()
    m.content = "Researched AI trends and produced a 5-section report for the team"
    return m


@pytest.fixture
def lesson():
    """A fake lesson object that mimics what SelfImprovementEngine returns."""
    lsn = MagicMock()
    lsn.outcome = "success"
    lsn.matches = MagicMock(return_value=True)
    return lsn


# ── PRISMState Unit Tests ──────────────────────────────────────────────────────

class TestPRISMState:
    def test_overall_confidence_blends_three_scores(self):
        state = PRISMState()
        state.comprehension_confidence = 0.8
        state.historical_accuracy = 0.9
        state.novelty_score = 0.1   # Low novelty = familiar = boosts confidence
        conf = state.overall_confidence
        assert conf > 0.7, "High comp + high history + low novelty should yield high confidence"
        assert conf <= 0.95

    def test_overall_confidence_low_on_poor_inputs(self):
        state = PRISMState()
        state.comprehension_confidence = 0.25
        state.historical_accuracy = 0.2
        state.novelty_score = 0.9   # Very novel
        conf = state.overall_confidence
        assert conf < 0.5, "Low comp + low history + high novelty should yield low confidence"

    def test_overall_confidence_clamps_to_valid_range(self):
        state = PRISMState()
        state.comprehension_confidence = 1.0
        state.historical_accuracy = 1.0
        state.novelty_score = 0.0
        assert state.overall_confidence <= 0.95

        state.comprehension_confidence = 0.0
        state.historical_accuracy = 0.0
        state.novelty_score = 1.0
        assert state.overall_confidence >= 0.1

    def test_is_high_drift_triggers_on_one_high_event(self):
        state = PRISMState()
        state.drift_events.append(DriftEvent(severity="high"))
        assert state.is_high_drift is True

    def test_is_high_drift_triggers_on_three_medium_events(self):
        state = PRISMState()
        for _ in range(3):
            state.drift_events.append(DriftEvent(severity="medium"))
        assert state.is_high_drift is True

    def test_is_high_drift_false_on_low_events_only(self):
        state = PRISMState()
        for _ in range(5):
            state.drift_events.append(DriftEvent(severity="low"))
        assert state.is_high_drift is False

    def test_to_context_string_contains_strategy(self):
        state = PRISMState(task="research AI trends")
        state.comprehension_confidence = 0.7
        state.historical_accuracy = 0.6
        state.novelty_score = 0.4
        state.chosen_strategy = "exploratory"
        state.strategy_rationale = "Moderate footing but significant novelty"
        ctx = state.to_context_string()
        assert "exploratory" in ctx.lower()
        assert "PRISM" in ctx

    def test_to_dict_is_serialisable(self):
        state = PRISMState(task="send email to team", user_id="alice")
        state.chosen_strategy = "confident"
        state.actual_quality = 0.85
        d = state.to_dict()
        json.dumps(d)   # Must not raise
        assert "task_id" in d
        assert "chosen_strategy" in d
        assert d["user_id"] == "alice"


# ── Comprehension Tests ────────────────────────────────────────────────────────

class TestPRISMComprehension:
    def test_long_specific_task_scores_higher_than_vague_short_task(self, prism):
        conf_specific, _ = prism._assess_comprehension(
            "Write a formal email to the Q3 budget team requesting approval for the server upgrade"
        )
        conf_vague, _ = prism._assess_comprehension("do the thing")
        assert conf_specific > conf_vague

    def test_very_short_task_gets_penalised(self, prism):
        conf, notes = prism._assess_comprehension("ok")
        assert conf < 0.6
        assert any("short" in n.lower() for n in notes)

    def test_ambiguous_pronouns_reduce_confidence(self, prism):
        with_pronoun, notes_with = prism._assess_comprehension(
            "Reply to it and tell them we agree"
        )
        without_pronoun, _ = prism._assess_comprehension(
            "Reply to Sarah's email and tell the team we agree"
        )
        assert with_pronoun < without_pronoun
        assert any("pronoun" in n.lower() or "reference" in n.lower() for n in notes_with)

    def test_domain_vocabulary_raises_confidence(self, prism):
        technical, notes = prism._assess_comprehension(
            "Debug the API endpoint and fix the database connection error"
        )
        assert technical >= 0.6
        assert any("technical" in n.lower() for n in notes)

    def test_question_mark_gives_small_boost(self, prism):
        with_q, _ = prism._assess_comprehension("What is the current server status?")
        without_q, _ = prism._assess_comprehension("Tell me the current server status")
        assert with_q >= without_q

    def test_comprehension_score_stays_in_valid_range(self, prism):
        for task in [
            "",
            "x",
            "Write a complete technical specification document for the entire platform architecture including all integration points",
            "it",
            "maybe possibly somehow do something or other or something else",
        ]:
            conf, _ = prism._assess_comprehension(task)
            assert 0.25 <= conf <= 0.90, f"Score {conf} out of range for task: {task!r}"

    def test_specific_proper_nouns_boost_score(self, prism):
        with_nouns, _ = prism._assess_comprehension(
            "Schedule a meeting with Sarah Johnson and Michael Chen about the Orion Project"
        )
        without_nouns, _ = prism._assess_comprehension(
            "Schedule a meeting with someone about the project"
        )
        assert with_nouns >= without_nouns


# ── Novelty Tests ──────────────────────────────────────────────────────────────

class TestPRISMNovelty:
    def test_no_memories_means_high_novelty(self, prism):
        novelty, similar = prism._assess_novelty("write a research report", past_memories=[])
        assert novelty >= 0.80
        assert similar == []

    def test_highly_similar_memory_reduces_novelty(self, prism):
        m = MagicMock()
        m.content = "write a detailed research report on AI trends for the executive team"
        novelty, similar = prism._assess_novelty(
            "write a research report on current AI developments",
            past_memories=[m]
        )
        assert novelty < 0.80
        assert len(similar) >= 1

    def test_unrelated_memories_do_not_reduce_novelty(self, prism):
        m = MagicMock()
        m.content = "configure the kubernetes cluster networking settings"
        novelty, similar = prism._assess_novelty(
            "write an email to the board about Q3 financials",
            past_memories=[m]
        )
        assert novelty >= 0.70
        assert similar == []

    def test_more_similar_memories_means_lower_novelty(self, prism):
        def make_memory(content):
            m = MagicMock()
            m.content = content
            return m

        few_memories = [
            make_memory("research technology trends quarterly")
        ]
        many_memories = [
            make_memory("research technology trends in AI this quarter"),
            make_memory("research quarterly technology development overview"),
            make_memory("analyse technology research and trends report"),
            make_memory("technology trends research and summary document"),
            make_memory("prepare research on technology trends quarterly"),
        ]

        novelty_few, _ = prism._assess_novelty("research technology trends", few_memories)
        novelty_many, _ = prism._assess_novelty("research technology trends", many_memories)
        assert novelty_many < novelty_few

    def test_novelty_stays_in_valid_range(self, prism):
        for _ in range(10):
            m = MagicMock()
            m.content = "exactly the same task repeated again and again"
            novelty, _ = prism._assess_novelty("exactly the same task", [m] * 10)
            assert 0.05 <= novelty <= 0.95


# ── History Tests ──────────────────────────────────────────────────────────────

class TestPRISMHistory:
    def test_no_data_returns_bayesian_prior(self, prism):
        accuracy, sample = prism._assess_history("any task", None, None)
        assert 0.50 <= accuracy <= 0.70
        assert sample == 0

    def test_successful_lessons_raise_historical_accuracy(self, prism, lesson):
        lesson.outcome = "success"
        accuracy_with, sample = prism._assess_history("research report", [lesson] * 5, None)
        accuracy_without, _ = prism._assess_history("research report", None, None)
        assert accuracy_with > accuracy_without
        assert sample > 0

    def test_failed_lessons_lower_historical_accuracy(self, prism):
        failed_lesson = MagicMock()
        failed_lesson.outcome = "failure"
        failed_lesson.matches = MagicMock(return_value=True)

        accuracy_with_failures, _ = prism._assess_history(
            "research report", [failed_lesson] * 5, None
        )
        accuracy_prior, _ = prism._assess_history("research report", None, None)
        assert accuracy_with_failures < accuracy_prior

    def test_trust_summaries_influence_score(self, prism):
        high_trust = {"agent_1": {"overall_trust": 0.95}, "agent_2": {"overall_trust": 0.90}}
        low_trust  = {"agent_1": {"overall_trust": 0.20}, "agent_2": {"overall_trust": 0.15}}

        score_high, _ = prism._assess_history("any task", None, high_trust)
        score_low,  _ = prism._assess_history("any task", None, low_trust)
        assert score_high > score_low

    def test_prism_own_calibration_history_contributes(self, prism):
        # Seed calibration history with good results for research tasks
        for _ in range(10):
            prism._calibration_history.append(CalibrationRecord(
                task_category        = "research",
                chosen_strategy      = "confident",
                predicted_confidence = 0.8,
                actual_quality       = 0.85,
                calibration_error    = 0.05,
                verdict              = "well_calibrated",
            ))

        accuracy_with_history, _ = prism._assess_history(
            "research AI trends and summarise", None, None
        )
        # Reset and compare without history
        prism._calibration_history.clear()
        accuracy_without, _ = prism._assess_history(
            "research AI trends and summarise", None, None
        )
        assert accuracy_with_history > accuracy_without

    def test_historical_accuracy_stays_in_valid_range(self, prism):
        perfect_lessons = [MagicMock(outcome="success", matches=MagicMock(return_value=True))] * 20
        high_trust = {"a": {"overall_trust": 1.0}, "b": {"overall_trust": 1.0}}
        accuracy, _ = prism._assess_history("task", perfect_lessons, high_trust)
        assert 0.10 <= accuracy <= 0.95


# ── Strategy Tests ─────────────────────────────────────────────────────────────

class TestPRISMStrategy:
    def test_high_scores_yield_confident_strategy(self, prism):
        strategy, rationale = prism._select_strategy(
            comprehension=0.85, history=0.80, novelty=0.20
        )
        assert strategy == "confident"
        assert len(rationale) > 0

    def test_high_novelty_yields_exploratory_strategy(self, prism):
        strategy, _ = prism._select_strategy(
            comprehension=0.65, history=0.60, novelty=0.65
        )
        assert strategy == "exploratory"

    def test_mixed_low_scores_yield_cautious_strategy(self, prism):
        strategy, _ = prism._select_strategy(
            comprehension=0.45, history=0.45, novelty=0.80
        )
        assert strategy in ("cautious", "delegating")

    def test_very_low_scores_yield_delegating_strategy(self, prism):
        strategy, rationale = prism._select_strategy(
            comprehension=0.25, history=0.20, novelty=0.95
        )
        assert strategy == "delegating"
        assert "human" in rationale.lower() or "uncertain" in rationale.lower()

    def test_all_strategies_return_non_empty_rationale(self, prism):
        test_cases = [
            (0.85, 0.80, 0.20),   # confident
            (0.65, 0.60, 0.65),   # exploratory
            (0.45, 0.45, 0.80),   # cautious
            (0.25, 0.20, 0.95),   # delegating
        ]
        for comp, hist, nov in test_cases:
            strategy, rationale = prism._select_strategy(comp, hist, nov)
            assert isinstance(strategy, str)
            assert len(rationale) > 10


# ── Pre-Reason Integration Tests ───────────────────────────────────────────────

class TestPRISMPreReason:
    def test_pre_reason_returns_prism_state(self, prism):
        state = prism.pre_reason(
            task="Research the latest developments in transformer architecture",
            past_memories=[],
            user_id="alice",
        )
        assert isinstance(state, PRISMState)
        assert state.task == "Research the latest developments in transformer architecture"
        assert state.user_id == "alice"

    def test_pre_reason_populates_all_fields(self, prism, memory_entry):
        state = prism.pre_reason(
            task="Write an email to the board about Q3 results",
            past_memories=[memory_entry],
            user_id="bob",
        )
        assert state.comprehension_confidence > 0
        assert state.novelty_score > 0
        assert state.historical_accuracy > 0
        assert state.chosen_strategy in ("confident", "exploratory", "cautious", "delegating")
        assert len(state.strategy_rationale) > 0
        assert state.pre_reasoning_seconds >= 0

    def test_pre_reason_task_id_is_unique_per_call(self, prism):
        state1 = prism.pre_reason("task one", [], "alice")
        state2 = prism.pre_reason("task two", [], "alice")
        assert state1.task_id != state2.task_id

    def test_pre_reason_context_string_is_injected_cleanly(self, prism):
        state = prism.pre_reason("Analyse quarterly sales data", [], "dave")
        ctx = state.to_context_string()
        assert "[PRISM Pre-Reasoning]" in ctx
        assert "%" in ctx   # confidence percentages
        assert state.chosen_strategy in ctx


# ── Monitoring Tests ───────────────────────────────────────────────────────────

class TestPRISMMonitoring:
    def test_on_node_complete_no_drift_on_expected_quality(self, prism):
        state = PRISMState(chosen_strategy="confident")
        prism.on_node_complete(state, "node_1", "agent_a", quality=0.85, expected_quality=0.6)
        assert len(state.drift_events) == 0

    def test_on_node_complete_records_drift_on_low_quality(self, prism):
        state = PRISMState(chosen_strategy="confident")
        prism.on_node_complete(state, "node_1", "agent_a", quality=0.2, expected_quality=0.6)
        assert len(state.drift_events) == 1
        assert state.drift_events[0].event_type == "quality_shortfall"

    def test_on_node_complete_high_severity_on_very_low_quality(self, prism):
        state = PRISMState(chosen_strategy="confident")
        prism.on_node_complete(state, "node_1", "agent_a", quality=0.1, expected_quality=0.6)
        assert state.drift_events[0].severity == "high"

    def test_on_node_fail_records_drift_event(self, prism):
        state = PRISMState(chosen_strategy="cautious")
        prism.on_node_fail(state, "node_2", "agent_b", error="Connection timeout")
        assert len(state.drift_events) == 1
        assert state.drift_events[0].event_type == "node_failure"
        assert "agent_b" in state.drift_events[0].description

    def test_confident_strategy_failure_records_high_severity(self, prism):
        state = PRISMState(chosen_strategy="confident")
        prism.on_node_fail(state, "node_1", "agent_a")
        assert state.drift_events[0].severity == "high"

    def test_cautious_strategy_failure_records_lower_severity(self, prism):
        state = PRISMState(chosen_strategy="cautious")
        prism.on_node_fail(state, "node_1", "agent_a")
        # cautious strategy expects some failure — lower severity
        assert state.drift_events[0].severity in ("low", "medium")

    def test_high_drift_triggers_strategy_adjustment(self, prism):
        state = PRISMState(chosen_strategy="confident")
        # One high-severity event should trigger adjustment
        prism.on_node_fail(state, "node_1", "agent_a")
        assert state.strategy_adjusted is True
        assert state.adjusted_strategy == "cautious"
        assert len(state.monitoring_flags) > 0

    def test_strategy_adjustment_happens_only_once(self, prism):
        state = PRISMState(chosen_strategy="confident")
        prism.on_node_fail(state, "node_1", "agent_a")
        prism.on_node_fail(state, "node_2", "agent_b")
        prism.on_node_fail(state, "node_3", "agent_c")
        assert len(state.monitoring_flags) == 1  # Adjusted only once


# ── Reflection Tests ───────────────────────────────────────────────────────────

class TestPRISMReflection:
    def test_reflect_populates_all_fields(self, prism):
        state = PRISMState(task="Write a report")
        state.comprehension_confidence = 0.8
        state.historical_accuracy = 0.7
        state.novelty_score = 0.3
        state.chosen_strategy = "confident"

        result = prism.reflect(state, "success", 0.85)
        assert result.actual_outcome == "success"
        assert result.actual_quality == 0.85
        assert result.calibration_error >= 0
        assert result.calibration_verdict in ("well_calibrated", "overconfident", "underconfident")
        assert len(result.reflection_insight) > 0
        assert result.reflection_seconds >= 0

    def test_well_calibrated_when_prediction_matches_outcome(self, prism):
        state = PRISMState(task="send an email reply")
        state.comprehension_confidence = 0.75
        state.historical_accuracy = 0.70
        state.novelty_score = 0.30
        state.chosen_strategy = "confident"
        # overall_confidence ≈ 0.71 — quality 0.75 is close → well calibrated
        result = prism.reflect(state, "success", 0.75)
        assert result.calibration_error < 0.20

    def test_overconfident_verdict_when_predicted_high_but_quality_low(self, prism):
        state = PRISMState(task="write complex algorithm")
        state.comprehension_confidence = 0.90
        state.historical_accuracy = 0.90
        state.novelty_score = 0.05
        state.chosen_strategy = "confident"
        # overall_confidence will be very high, actual quality very low
        result = prism.reflect(state, "failure", 0.10)
        assert result.calibration_verdict == "overconfident"
        assert result.calibration_error > 0.3

    def test_underconfident_verdict_when_predicted_low_but_quality_high(self, prism):
        state = PRISMState(task="something vague and unclear")
        state.comprehension_confidence = 0.25
        state.historical_accuracy = 0.25
        state.novelty_score = 0.90
        state.chosen_strategy = "delegating"
        # overall_confidence will be low, actual quality surprisingly high
        result = prism.reflect(state, "success", 0.90)
        assert result.calibration_verdict == "underconfident"
        assert result.calibration_error > 0.2

    def test_reflect_adds_record_to_calibration_history(self, prism):
        initial_count = len(prism._calibration_history)
        state = PRISMState(task="test task", chosen_strategy="cautious")
        state.comprehension_confidence = 0.6
        state.historical_accuracy = 0.6
        state.novelty_score = 0.5
        prism.reflect(state, "success", 0.7)
        assert len(prism._calibration_history) == initial_count + 1

    def test_reflect_caps_history_at_500_records(self, prism):
        # Seed with 495 records
        for _ in range(495):
            prism._calibration_history.append(CalibrationRecord())

        # Add 10 more via reflect
        for _ in range(10):
            state = PRISMState(task="task", chosen_strategy="cautious")
            state.comprehension_confidence = 0.6
            state.historical_accuracy = 0.6
            state.novelty_score = 0.5
            prism.reflect(state, "success", 0.7)

        assert len(prism._calibration_history) <= 500

    def test_reflect_generates_meaningful_insight(self, prism):
        state = PRISMState(task="research AI trends", chosen_strategy="exploratory")
        state.comprehension_confidence = 0.65
        state.historical_accuracy = 0.60
        state.novelty_score = 0.60
        result = prism.reflect(state, "success", 0.30)  # Low quality despite moderate confidence
        assert len(result.reflection_insight) > 20
        assert result.reflection_insight  # Not empty


# ── Calibration Report Tests ───────────────────────────────────────────────────

class TestPRISMCalibrationReport:
    def test_empty_history_returns_zero_tasks(self, prism):
        report = prism.calibration_report()
        assert report["total_tasks"] == 0
        assert "message" in report

    def test_report_structure_with_data(self, prism):
        for _ in range(5):
            prism._calibration_history.append(CalibrationRecord(
                task_category        = "research",
                chosen_strategy      = "confident",
                predicted_confidence = 0.8,
                actual_quality       = 0.75,
                calibration_error    = 0.05,
                verdict              = "well_calibrated",
            ))

        report = prism.calibration_report()
        assert report["total_tasks"] == 5
        assert "avg_calibration_error" in report
        assert "calibration_grade" in report
        assert "by_category" in report
        assert "research" in report["by_category"]

    def test_calibration_grade_reflects_error_magnitude(self, prism):
        def make_records(error, count=10):
            prism._calibration_history.clear()
            for _ in range(count):
                prism._calibration_history.append(CalibrationRecord(
                    calibration_error=error,
                    verdict="well_calibrated" if error < 0.15 else "overconfident",
                ))

        make_records(0.05)
        report_good = prism.calibration_report()

        make_records(0.45)
        report_bad = prism.calibration_report()

        assert "A" in report_good["calibration_grade"]
        assert "F" in report_bad["calibration_grade"] or "D" in report_bad["calibration_grade"]

    def test_weakest_category_identified_correctly(self, prism):
        # Research tasks: high error
        for _ in range(5):
            prism._calibration_history.append(CalibrationRecord(
                task_category="research",
                calibration_error=0.45,
                verdict="overconfident",
            ))
        # Communication tasks: low error
        for _ in range(5):
            prism._calibration_history.append(CalibrationRecord(
                task_category="communication",
                calibration_error=0.05,
                verdict="well_calibrated",
            ))

        report = prism.calibration_report()
        assert report["weakest_category"] == "research"

    def test_overconfident_interpretation_when_mostly_overconfident(self, prism):
        for _ in range(8):
            prism._calibration_history.append(CalibrationRecord(
                calibration_error=0.40,
                verdict="overconfident",
            ))
        for _ in range(2):
            prism._calibration_history.append(CalibrationRecord(
                calibration_error=0.05,
                verdict="well_calibrated",
            ))

        report = prism.calibration_report()
        assert "overestimate" in report["interpretation"].lower()


# ── Persistence Tests ──────────────────────────────────────────────────────────

class TestPRISMPersistence:
    def test_save_and_reload_calibration(self, prism_with_history):
        prism_a = prism_with_history

        # Add records to prism_a
        for i in range(5):
            prism_a._calibration_history.append(CalibrationRecord(
                task_category        = "technical",
                chosen_strategy      = "confident",
                predicted_confidence = 0.8,
                actual_quality       = 0.75 + i * 0.01,
                calibration_error    = 0.05,
                verdict              = "well_calibrated",
            ))
        prism_a._save_calibration()

        # Create a second PRISM pointing to same path and verify it loaded
        prism_b = PRISMEngine(persist_path=prism_a._persist_path)
        assert len(prism_b._calibration_history) == 5
        assert all(r.task_category == "technical" for r in prism_b._calibration_history)

    def test_graceful_when_no_persist_path(self, prism):
        # No path — should not raise anything
        prism._save_calibration()
        prism._load_calibration()   # Neither should raise

    def test_calibration_file_is_valid_json(self, prism_with_history):
        prism = prism_with_history
        prism._calibration_history.append(CalibrationRecord(
            task_category="web",
            actual_quality=0.8,
        ))
        prism._save_calibration()

        # Find the file and parse it directly
        cal_path = prism._persist_path.parent / "prism_calibration.json"
        assert cal_path.exists()
        data = json.loads(cal_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert "task_category" in data[0]


# ── Full Integration Tests ─────────────────────────────────────────────────────

class TestPRISMIntegration:
    def test_full_cycle_success(self, prism):
        """Full pre-reason → monitor → reflect cycle for a successful task."""
        state = prism.pre_reason(
            task="Research current AI safety developments and write a summary",
            past_memories=[],
            user_id="alice",
        )

        assert state.chosen_strategy in ("confident", "exploratory", "cautious", "delegating")

        # Simulate execution — one node completes well, one fails
        prism.on_node_complete(state, "node_1", "researcher", quality=0.8)
        prism.on_node_complete(state, "node_2", "writer", quality=0.75)

        state = prism.reflect(state, "success", 0.78)

        assert state.actual_outcome == "success"
        assert state.actual_quality == 0.78
        assert state.calibration_verdict in ("well_calibrated", "overconfident", "underconfident")
        assert len(prism._calibration_history) == 1

    def test_full_cycle_failure_with_drift(self, prism):
        """Full cycle where execution fails and drift is detected."""
        state = prism.pre_reason(
            task="Deploy the production server upgrade",
            past_memories=[],
            user_id="bob",
        )

        # Multiple node failures — should trigger drift
        prism.on_node_fail(state, "node_1", "deploy_agent", error="Connection refused")
        prism.on_node_fail(state, "node_2", "monitor_agent", error="Health check failed")

        state = prism.reflect(state, "failure", 0.0)

        assert state.actual_outcome == "failure"
        assert len(state.drift_events) >= 1
        assert state.calibration_verdict in ("overconfident", "well_calibrated", "underconfident")
        assert len(prism._calibration_history) == 1

    def test_calibration_improves_over_many_tasks(self, prism):
        """
        After many tasks of the same type, PRISM's historical accuracy
        should start reflecting the actual success rate.
        """
        # Simulate 20 research tasks that mostly succeed
        for i in range(20):
            state = prism.pre_reason(
                task = "research and summarise AI developments",
                past_memories = [],
                user_id = "alice",
            )
            quality = 0.80 if i % 5 != 0 else 0.30
            outcome = "success" if quality > 0.5 else "failure"
            prism.reflect(state, outcome, quality)

        # Now a fresh pre-reason for a similar task should have non-trivial history
        final_state = prism.pre_reason(
            task = "research and summarise machine learning trends",
            past_memories = [],
            user_id = "alice",
        )
        assert final_state.historical_sample > 0
        assert 0.1 <= final_state.historical_accuracy <= 0.95

    def test_unique_task_id_per_pre_reason(self, prism):
        """Each call to pre_reason generates a fresh unique task_id."""
        states = [prism.pre_reason("same task text", [], "user") for _ in range(5)]
        ids = [s.task_id for s in states]
        assert len(set(ids)) == 5   # All unique

    def test_prism_state_dict_compatible_with_json(self, prism):
        """The dict form of a full PRISMState cycle must be JSON serialisable."""
        state = prism.pre_reason("write a technical document", [], "alice")
        prism.on_node_complete(state, "n1", "writer", quality=0.8)
        state = prism.reflect(state, "success", 0.8)
        serialised = json.dumps(state.to_dict())
        restored = json.loads(serialised)
        assert restored["actual_outcome"] == "success"


# ── Task Categorisation Tests ──────────────────────────────────────────────────

class TestPRISMCategorisation:
    @pytest.mark.parametrize("task,expected_category", [
        ("Debug the API endpoint and fix the connection error",       "technical"),
        ("Research the latest AI safety papers and summarise",        "research"),
        ("Write a reply to Sarah's email about the project deadline", "communication"),
        ("Create a formatted Word document with the meeting notes",   "document"),
        ("Remember that the standup is at 9am every Monday",         "memory"),
        ("Browse the web and find the latest news on OpenAI",        "web"),
        ("Hello",                                                      "general"),
    ])
    def test_categorisation_accuracy(self, prism, task, expected_category):
        category = prism._categorise_task(task)
        assert category == expected_category, (
            f"Task '{task[:40]}' categorised as '{category}', expected '{expected_category}'"
        )
