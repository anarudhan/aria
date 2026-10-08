"""
Unit tests for aria.core.trust: category detection, Bayesian scoring, suspension
and JSON persistence of the TrustLedger.
"""
import json

import pytest

from aria.core.trust import CATEGORY_KEYWORDS, TrustLedger, detect_category

RESEARCH = "research the latest AI news"
CODING = "debug the python script"


# ─── Category detection ───────────────────────────────────────────────────────

class TestDetectCategory:

    @pytest.mark.parametrize("task,expected", [
        ("investigate and research the market", "research"),
        ("write a draft report", "writing"),
        ("debug the python script and implement the function", "coding"),
        ("send an email reply to the inbox", "email"),
        ("schedule a meeting in the teams channel", "teams"),
        ("calculate statistics from the csv data", "data_analysis"),
        ("scrape the website and download the page", "web_browsing"),
        ("format the docx template", "document"),
        ("plan the project deadline", "planning"),
    ])
    def test_picks_category_with_most_keyword_hits(self, task, expected):
        assert detect_category(task) == expected

    @pytest.mark.parametrize("task", ["", "hello there", "the quick brown fox", "!!!"])
    def test_falls_back_to_general(self, task):
        assert detect_category(task) == "general"

    def test_is_case_insensitive(self):
        assert detect_category("DEBUG THE PYTHON SCRIPT") == "coding"

    def test_every_category_has_a_keyword_list(self):
        assert "general" in CATEGORY_KEYWORDS and CATEGORY_KEYWORDS["general"] == []


# ─── Scoring ──────────────────────────────────────────────────────────────────

class TestScoring:

    def test_unknown_agent_gets_the_prior(self):
        assert TrustLedger().get_trust_score("new_agent", RESEARCH) == pytest.approx(0.6)

    def test_successes_raise_trust_bayesian_style(self):
        ledger = TrustLedger()
        for _ in range(3):
            ledger.record_success("a", RESEARCH, quality=0.9, seconds=1.0)
        # alpha = 3 + 3, beta = 2 -> 6 / 8
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(0.75)

    def test_failures_lower_trust(self):
        ledger = TrustLedger()
        for _ in range(3):
            ledger.record_failure("a", RESEARCH, seconds=1.0)
        # alpha = 3, beta = 2 + 3 -> 3 / 8
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(0.375)

    def test_timeouts_count_against_trust_like_failures(self):
        ledger = TrustLedger()
        ledger.record_timeout("a", RESEARCH)
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(3 / 6)

    def test_one_success_does_not_make_an_agent_fully_trusted(self):
        ledger = TrustLedger()
        ledger.record_success("a", RESEARCH, 1.0, 1.0)
        assert 0.6 < ledger.get_trust_score("a", RESEARCH) < 0.8

    def test_scores_converge_towards_observed_success_rate(self):
        ledger = TrustLedger()
        for i in range(100):
            if i % 10 == 0:
                ledger.record_failure("a", RESEARCH, 1.0)
            else:
                ledger.record_success("a", RESEARCH, 0.9, 1.0)
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(0.9, abs=0.03)

    def test_trust_is_tracked_per_category(self):
        ledger = TrustLedger()
        for _ in range(4):
            ledger.record_success("a", RESEARCH, 0.9, 1.0)
            ledger.record_failure("a", CODING, 1.0)
        assert ledger.get_trust_score("a", RESEARCH) > 0.7
        assert ledger.get_trust_score("a", CODING) < 0.4

    def test_unseen_category_falls_back_to_average_of_known_categories(self):
        ledger = TrustLedger()
        for _ in range(4):
            ledger.record_success("a", RESEARCH, 0.9, 1.0)
            ledger.record_failure("a", CODING, 1.0)
        expected = (ledger.get_trust_score("a", RESEARCH) + ledger.get_trust_score("a", CODING)) / 2
        assert ledger.get_trust_score("a", "send an email") == pytest.approx(expected)

    def test_agents_do_not_affect_each_other(self):
        ledger = TrustLedger()
        for _ in range(5):
            ledger.record_failure("a", RESEARCH, 1.0)
        assert ledger.get_trust_score("b", RESEARCH) == pytest.approx(0.6)

    def test_rolling_average_of_quality_and_seconds(self):
        ledger = TrustLedger()
        ledger.record_success("a", RESEARCH, quality=0.8, seconds=10.0)
        ledger.record_success("a", RESEARCH, quality=1.0, seconds=20.0)
        record = ledger._records["a"]["research"]
        assert record.average_quality == pytest.approx(0.9)
        assert record.average_seconds == pytest.approx(15.0)
        assert record.success_count == 2 and record.total_executions == 2

    @pytest.mark.xfail(
        strict=True,
        reason="bug: trust.py record_success divides by total_executions (which includes "
               "failures and timeouts) but failures never feed the average, so average_quality "
               "depends on the order of outcomes: success-then-failure gives 1.0 while "
               "failure-then-success gives 0.5",
    )
    def test_average_quality_does_not_depend_on_outcome_order(self):
        success_first = TrustLedger()
        success_first.record_success("a", RESEARCH, quality=1.0, seconds=1.0)
        success_first.record_failure("a", RESEARCH, seconds=1.0)

        failure_first = TrustLedger()
        failure_first.record_failure("a", RESEARCH, seconds=1.0)
        failure_first.record_success("a", RESEARCH, quality=1.0, seconds=1.0)

        assert (success_first._records["a"]["research"].average_quality
                == pytest.approx(failure_first._records["a"]["research"].average_quality))

    def test_counters_per_outcome_type(self):
        ledger = TrustLedger()
        ledger.record_success("a", RESEARCH, 1.0, 1.0)
        ledger.record_failure("a", RESEARCH, 1.0)
        ledger.record_timeout("a", RESEARCH)
        record = ledger._records["a"]["research"]
        assert (record.success_count, record.failure_count, record.timeout_count) == (1, 1, 1)
        assert record.total_executions == 3


# ─── Suspension ───────────────────────────────────────────────────────────────

class TestSuspension:

    def test_agent_is_suspended_after_enough_failures(self):
        ledger = TrustLedger()          # threshold 0.2, min 5 executions
        for _ in range(10):
            ledger.record_failure("a", RESEARCH, 1.0)
        assert not ledger.is_suspended("a")        # 3 / 15 == 0.2, not below threshold
        ledger.record_failure("a", RESEARCH, 1.0)  # 3 / 16 < 0.2
        assert ledger.is_suspended("a")

    def test_no_suspension_before_minimum_executions(self):
        ledger = TrustLedger(suspension_threshold=0.9, min_executions_for_suspension=5)
        for _ in range(4):
            ledger.record_failure("a", RESEARCH, 1.0)
        assert not ledger.is_suspended("a")
        ledger.record_failure("a", RESEARCH, 1.0)
        assert ledger.is_suspended("a")

    def test_timeouts_can_trigger_suspension(self):
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(3):
            ledger.record_timeout("a", RESEARCH)
        assert ledger.is_suspended("a")

    def test_successes_never_suspend(self):
        ledger = TrustLedger(suspension_threshold=1.1, min_executions_for_suspension=1)
        for _ in range(5):
            ledger.record_success("a", RESEARCH, 1.0, 1.0)
        assert not ledger.is_suspended("a")

    def test_suspended_agent_scores_zero_in_every_category(self):
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(3):
            ledger.record_failure("a", RESEARCH, 1.0)
        assert ledger.get_trust_score("a", RESEARCH) == 0.0
        assert ledger.get_trust_score("a", CODING) == 0.0

    def test_suspension_is_per_agent(self):
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(3):
            ledger.record_failure("a", RESEARCH, 1.0)
        assert ledger.get_trust_score("b", RESEARCH) == pytest.approx(0.6)
        assert not ledger.is_suspended("b")

    def test_reinstate_lifts_suspension_and_restores_the_real_score(self):
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(3):
            ledger.record_failure("a", RESEARCH, 1.0)
        ledger.reinstate_agent("a")
        assert not ledger.is_suspended("a")
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(3 / 8)

    def test_reinstating_an_agent_that_is_not_suspended_is_harmless(self):
        ledger = TrustLedger()
        ledger.reinstate_agent("never_seen")
        assert not ledger.is_suspended("never_seen")

    def test_suspension_requires_the_category_to_reach_minimum_executions(self):
        # Failures spread over several categories do not add up for the check.
        ledger = TrustLedger(suspension_threshold=0.9, min_executions_for_suspension=3)
        for task in (RESEARCH, CODING, "send an email"):
            for _ in range(2):
                ledger.record_failure("a", task, 1.0)
        assert not ledger.is_suspended("a")


# ─── Summaries ────────────────────────────────────────────────────────────────

class TestSummaries:

    def test_summary_for_unknown_agent(self):
        assert TrustLedger().get_agent_summary("x") == {
            "agent_id": "x", "status": "no_history", "trust": 0.6,
        }

    def test_summary_lists_categories(self):
        ledger = TrustLedger()
        ledger.record_success("a", RESEARCH, 0.9, 12.34)
        ledger.record_failure("a", CODING, 1.0)

        summary = ledger.get_agent_summary("a")

        assert summary["agent_id"] == "a"
        assert summary["suspended"] is False
        assert set(summary["categories"]) == {"research", "coding"}
        research = summary["categories"]["research"]
        assert research["successes"] == 1 and research["failures"] == 0
        assert research["avg_quality"] == 0.9
        assert research["avg_seconds"] == 12.3
        assert summary["categories"]["coding"]["failures"] == 1

    def test_summary_flags_suspended_agents(self):
        ledger = TrustLedger(suspension_threshold=0.5, min_executions_for_suspension=3)
        for _ in range(3):
            ledger.record_failure("a", RESEARCH, 1.0)
        summary = ledger.get_agent_summary("a")
        assert summary["suspended"] is True
        assert summary["overall_trust"] == 0.0

    def test_get_all_summaries_covers_every_known_agent(self):
        ledger = TrustLedger()
        ledger.record_success("a", RESEARCH, 1.0, 1.0)
        ledger.record_failure("b", CODING, 1.0)
        assert {s["agent_id"] for s in ledger.get_all_summaries()} == {"a", "b"}

    def test_get_all_summaries_empty(self):
        assert TrustLedger().get_all_summaries() == []


# ─── Persistence ──────────────────────────────────────────────────────────────

class TestPersistence:

    def test_no_file_is_written_without_a_persist_path(self, tmp_path):
        TrustLedger().record_success("a", RESEARCH, 1.0, 1.0)
        assert list(tmp_path.iterdir()) == []

    def test_nothing_is_written_until_something_is_recorded(self, tmp_path):
        path = tmp_path / "trust.json"
        TrustLedger(persist_path=str(path))
        assert not path.exists()

    @pytest.mark.parametrize("record", ["success", "failure", "timeout"])
    def test_every_kind_of_outcome_is_persisted(self, tmp_path, record):
        path = tmp_path / "trust.json"
        ledger = TrustLedger(persist_path=str(path))
        if record == "success":
            ledger.record_success("a", RESEARCH, 0.5, 2.0)
        elif record == "failure":
            ledger.record_failure("a", RESEARCH, 2.0)
        else:
            ledger.record_timeout("a", RESEARCH)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["records"]["a"]["research"]["total_executions"] == 1

    def test_history_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "trust.json")
        first = TrustLedger(persist_path=path)
        first.record_success("a", RESEARCH, 0.8, 5.0)
        first.record_success("a", RESEARCH, 1.0, 7.0)
        first.record_failure("a", CODING, 1.0)
        first.record_timeout("a", CODING)

        second = TrustLedger(persist_path=path)

        assert second.get_trust_score("a", RESEARCH) == pytest.approx(
            first.get_trust_score("a", RESEARCH))
        assert second.get_trust_score("a", CODING) == pytest.approx(
            first.get_trust_score("a", CODING))
        record = second._records["a"]["research"]
        assert record.success_count == 2
        assert record.average_quality == pytest.approx(0.9)
        assert second._records["a"]["coding"].timeout_count == 1

    def test_suspension_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "trust.json")
        first = TrustLedger(persist_path=path, suspension_threshold=0.5,
                            min_executions_for_suspension=3)
        for _ in range(3):
            first.record_failure("a", RESEARCH, 1.0)
        assert first.is_suspended("a")

        second = TrustLedger(persist_path=path)
        assert second.is_suspended("a")
        assert second.get_trust_score("a", RESEARCH) == 0.0

    @pytest.mark.xfail(
        strict=True,
        reason="bug: trust.py reinstate_agent changes the in-memory suspended set but never "
               "calls _maybe_persist, so after a restart the agent is suspended again",
    )
    def test_reinstatement_survives_a_restart(self, tmp_path):
        path = str(tmp_path / "trust.json")
        first = TrustLedger(persist_path=path, suspension_threshold=0.5,
                            min_executions_for_suspension=3)
        for _ in range(3):
            first.record_failure("a", RESEARCH, 1.0)
        first.reinstate_agent("a")

        assert not TrustLedger(persist_path=path).is_suspended("a")

    def test_binary_corruption_resets_to_clean_state_and_removes_file(self, tmp_path):
        path = tmp_path / "trust.json"
        path.write_bytes(b"\x89PNG\r\n\x1a\n not json")

        ledger = TrustLedger(persist_path=str(path))

        assert not path.exists()
        assert ledger.get_all_summaries() == []
        assert ledger.get_trust_score("a", RESEARCH) == pytest.approx(0.6)

    @pytest.mark.parametrize("content", [
        "{ this is not valid json",
        '{"records": {"a": {"research": {"success_count": 1}}}}',   # missing fields
    ])
    def test_invalid_json_content_resets_to_clean_state(self, tmp_path, content):
        path = tmp_path / "trust.json"
        path.write_text(content, encoding="utf-8")

        ledger = TrustLedger(persist_path=str(path))

        assert not path.exists()
        assert ledger.get_all_summaries() == []

    def test_ledger_is_usable_and_persists_again_after_recovering_from_corruption(self, tmp_path):
        path = tmp_path / "trust.json"
        path.write_bytes(b"\x00\x01\x02garbage")
        ledger = TrustLedger(persist_path=str(path))

        ledger.record_success("a", RESEARCH, 1.0, 1.0)

        assert json.loads(path.read_text(encoding="utf-8"))["records"]["a"]
        assert TrustLedger(persist_path=str(path)).get_trust_score("a", RESEARCH) > 0.6

    def test_empty_file_loads_as_empty_ledger(self, tmp_path):
        path = tmp_path / "trust.json"
        path.write_text("", encoding="utf-8")
        ledger = TrustLedger(persist_path=str(path))
        assert ledger.get_all_summaries() == []

    def test_persisted_file_has_expected_shape(self, tmp_path):
        path = tmp_path / "trust.json"
        ledger = TrustLedger(persist_path=str(path))
        ledger.record_success("a", RESEARCH, 0.5, 2.0)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"records", "suspended"}
        assert set(data["records"]["a"]["research"]) == {
            "success_count", "failure_count", "timeout_count",
            "average_quality", "average_seconds", "total_executions",
        }
        assert data["suspended"] == []
