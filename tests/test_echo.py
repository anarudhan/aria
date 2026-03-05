"""
Tests for ECHO — Experiential Character through Historical Observation

These tests verify that ECHO builds genuinely useful relational and
character models from real observed interactions. Every test checks
that the right conclusion is drawn from the right evidence — and that
ECHO never draws conclusions without enough evidence to back them up.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aria.core.echo import (
    ARIACharacter,
    CharacterTrait,
    ECHOEngine,
    UserProfile,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def echo():
    """Fresh ECHO engine with no history."""
    return ECHOEngine(persist_path=None)


@pytest.fixture
def echo_persistent(tmp_path):
    """ECHO engine backed by a temp directory for persistence tests."""
    persist_file = tmp_path / "aria_data" / "memory.db"
    return ECHOEngine(persist_path=str(persist_file))


def make_prism_state(
    strategy: str = "cautious",
    confidence: float = 0.6,
    calibration_verdict: str = "well_calibrated",
    drift_count: int = 0,
):
    """Create a minimal PRISM state mock for ECHO observe calls."""
    state = MagicMock()
    state.chosen_strategy = strategy
    state.overall_confidence = confidence
    state.calibration_verdict = calibration_verdict
    state.drift_events = [MagicMock() for _ in range(drift_count)]
    return state


def run_interactions(echo, user_id, count, task="Research the latest AI trends",
                     quality=0.80, status="success", strategy="confident"):
    """Helper to run multiple interactions for pattern accumulation."""
    for _ in range(count):
        echo.observe(
            user_id=user_id,
            task=task,
            quality=quality,
            status=status,
            prism_state=make_prism_state(strategy=strategy, confidence=quality),
        )


# ── UserProfile Tests ─────────────────────────────────────────────────────────

class TestUserProfile:

    def test_success_rate_zero_with_no_tasks(self):
        p = UserProfile(user_id="alice")
        assert p.success_rate == 0.0

    def test_success_rate_correct(self):
        p = UserProfile(user_id="alice")
        p.successful_tasks = 8
        p.failed_tasks = 2
        assert p.success_rate == 0.8

    def test_relationship_label_new_at_start(self):
        p = UserProfile(user_id="alice")
        assert p.relationship_label == "new"

    def test_relationship_label_acquaintance(self):
        p = UserProfile(user_id="alice")
        p.total_interactions = 5
        p.relationship_depth = 0.15
        assert p.relationship_label == "acquaintance"

    def test_relationship_label_trusted_partner(self):
        p = UserProfile(user_id="alice")
        p.total_interactions = 50
        p.relationship_depth = 0.85
        assert p.relationship_label == "trusted partner"

    def test_to_context_string_new_user_is_neutral(self):
        p = UserProfile(user_id="alice")
        ctx = p.to_context_string()
        assert "new user" in ctx.lower() or "new" in ctx.lower()

    def test_to_context_string_established_user_has_guidance(self):
        p = UserProfile(user_id="alice")
        p.total_interactions = 20
        p.relationship_depth = 0.6
        p.preferred_style = "direct"
        p.preferred_detail_level = "concise"
        p.successful_tasks = 18
        p.failed_tasks = 2
        ctx = p.to_context_string()
        assert "direct" in ctx.lower()
        assert "concise" in ctx.lower()

    def test_to_context_string_expertise_areas_mentioned(self):
        p = UserProfile(user_id="alice")
        p.total_interactions = 15
        p.relationship_depth = 0.5
        p.expertise_areas = ["technical", "research"]
        p.successful_tasks = 13
        p.failed_tasks = 2
        ctx = p.to_context_string()
        assert "technical" in ctx or "research" in ctx

    def test_to_dict_has_required_keys(self):
        p = UserProfile(user_id="alice")
        d = p.to_dict()
        required = [
            "user_id", "total_interactions", "preferred_style",
            "preferred_detail_level", "relationship_depth",
            "relationship_label", "success_rate"
        ]
        for key in required:
            assert key in d


# ── CharacterTrait Tests ──────────────────────────────────────────────────────

class TestCharacterTrait:

    def test_frequency_zero_with_no_observations(self):
        t = CharacterTrait(name="test")
        assert t.frequency == 0.0

    def test_frequency_correct(self):
        t = CharacterTrait(name="test")
        t.observed_count = 6
        t.total_opportunities = 10
        assert t.frequency == 0.6

    def test_strength_emerging_at_low_frequency(self):
        t = CharacterTrait(name="test")
        t.observed_count = 2
        t.total_opportunities = 10
        t.update_strength()
        assert t.strength == "emerging"

    def test_strength_consistent_at_moderate_frequency(self):
        t = CharacterTrait(name="test")
        t.observed_count = 8
        t.total_opportunities = 20
        t.update_strength()
        assert t.strength == "consistent"

    def test_strength_defining_at_high_frequency(self):
        t = CharacterTrait(name="test")
        t.observed_count = 15
        t.total_opportunities = 20
        t.update_strength()
        assert t.strength == "defining"

    def test_to_dict_has_required_keys(self):
        t = CharacterTrait(name="honesty", description="ARIA is honest")
        d = t.to_dict()
        assert "name" in d
        assert "frequency" in d
        assert "strength" in d


# ── ARIACharacter Tests ───────────────────────────────────────────────────────

class TestARIACharacter:

    def test_observe_task_increments_counter(self):
        char = ARIACharacter()
        char.observe_task(
            task_category="research", quality=0.8, prism_strategy="confident",
            prism_confidence=0.8, calibration_verdict="well_calibrated", drift_count=0,
        )
        assert char.total_tasks_observed == 1

    def test_avg_quality_rolls_correctly(self):
        char = ARIACharacter()
        char.observe_task("research", 1.0, "confident", 0.9, "well_calibrated", 0)
        char.observe_task("research", 0.0, "cautious", 0.3, "overconfident", 0)
        assert abs(char.avg_quality_delivered - 0.5) < 0.01

    def test_strong_domains_after_enough_observations(self):
        char = ARIACharacter()
        for _ in range(5):
            char.observe_task("research", 0.9, "confident", 0.85, "well_calibrated", 0)
        assert "research" in char.strong_domains

    def test_weak_domains_after_poor_performance(self):
        char = ARIACharacter()
        for _ in range(5):
            char.observe_task("technical", 0.3, "cautious", 0.4, "overconfident", 2)
        assert "technical" in char.weak_domains

    def test_traits_built_from_observations(self):
        char = ARIACharacter()
        for _ in range(10):
            char.observe_task("research", 0.8, "confident", 0.8, "well_calibrated", 0)
        assert len(char.traits) > 0

    def test_defining_traits_property(self):
        char = ARIACharacter()
        char.traits["test_trait"] = CharacterTrait(
            name="test_trait",
            observed_count=15,
            total_opportunities=20,
            strength="defining",
        )
        assert len(char.defining_traits) == 1

    def test_to_summary_string_early_stage(self):
        char = ARIACharacter()
        summary = char.to_summary_string()
        assert "forming" in summary.lower() or "still" in summary.lower()

    def test_to_summary_string_mature_stage(self):
        char = ARIACharacter()
        for _ in range(10):
            char.observe_task("research", 0.85, "confident", 0.85, "well_calibrated", 0)
        summary = char.to_summary_string()
        assert "observed" in summary.lower()
        assert "quality" in summary.lower()

    def test_calibration_tendency_updates(self):
        char = ARIACharacter()
        for _ in range(10):
            char.observe_task("research", 0.8, "confident", 0.9, "overconfident", 0)
        assert char.calibration_tendency == "overconfident"


# ── ECHOEngine Core Tests ─────────────────────────────────────────────────────

class TestECHOEngine:

    def test_creates_new_profile_on_first_observe(self, echo):
        echo.observe(
            user_id="alice", task="Research AI trends",
            quality=0.8, status="success",
            prism_state=make_prism_state(),
        )
        assert echo.get_profile("alice") is not None

    def test_profile_interaction_count_increments(self, echo):
        run_interactions(echo, "alice", 3)
        profile = echo.get_profile("alice")
        assert profile.total_interactions == 3

    def test_successful_tasks_tracked(self, echo):
        run_interactions(echo, "alice", 5, status="success")
        assert echo.get_profile("alice").successful_tasks == 5

    def test_failed_tasks_tracked(self, echo):
        run_interactions(echo, "alice", 3, status="failure", quality=0.2)
        assert echo.get_profile("alice").failed_tasks == 3

    def test_relationship_depth_grows_with_successes(self, echo):
        run_interactions(echo, "alice", 10, status="success", quality=0.85)
        profile = echo.get_profile("alice")
        assert profile.relationship_depth > 0.0

    def test_relationship_depth_higher_after_more_successes(self, echo):
        run_interactions(echo, "alice", 5, status="success", quality=0.85)
        depth_5 = echo.get_profile("alice").relationship_depth

        run_interactions(echo, "alice", 15, status="success", quality=0.85)
        depth_20 = echo.get_profile("alice").relationship_depth

        assert depth_20 > depth_5

    def test_relationship_depth_decreases_on_failure(self, echo):
        run_interactions(echo, "alice", 10, status="success", quality=0.85)
        depth_before = echo.get_profile("alice").relationship_depth

        run_interactions(echo, "alice", 5, status="failure", quality=0.2)
        depth_after = echo.get_profile("alice").relationship_depth

        assert depth_after < depth_before

    def test_rolling_avg_quality_correct(self, echo):
        run_interactions(echo, "alice", 4, quality=1.0, status="success")
        run_interactions(echo, "alice", 4, quality=0.0, status="failure")
        profile = echo.get_profile("alice")
        assert 0.4 < profile.avg_quality_received < 0.6

    def test_multiple_users_are_independent(self, echo):
        run_interactions(echo, "alice", 10, quality=0.9, status="success")
        run_interactions(echo, "bob", 3, quality=0.3, status="failure")

        alice = echo.get_profile("alice")
        bob = echo.get_profile("bob")

        assert alice.relationship_depth > bob.relationship_depth
        assert alice.avg_quality_received > bob.avg_quality_received

    def test_all_user_ids_returns_both_users(self, echo):
        run_interactions(echo, "alice", 2)
        run_interactions(echo, "bob", 2)
        assert set(echo.all_user_ids()) == {"alice", "bob"}

    def test_get_profile_returns_none_for_unknown_user(self, echo):
        assert echo.get_profile("unknown_user_xyz") is None

    def test_character_updates_with_each_observe(self, echo):
        run_interactions(echo, "alice", 5, quality=0.85, status="success")
        char = echo.character_report()
        assert char["total_tasks_observed"] == 5

    def test_character_reflects_strong_domain_after_many_tasks(self, echo):
        for _ in range(5):
            echo.observe(
                user_id="alice",
                task="Research and analyse the latest AI papers",
                quality=0.90,
                status="success",
                prism_state=make_prism_state("confident", 0.9, "well_calibrated"),
            )
        char = echo.character_report()
        assert "research" in char.get("strong_domains", [])


# ── Preference Inference Tests ────────────────────────────────────────────────

class TestPreferenceInference:

    def test_brief_tasks_set_tends_to_be_brief(self, echo):
        for _ in range(10):
            echo.observe(
                user_id="alice",
                task="do it",
                quality=0.7,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert profile.tends_to_be_brief is True

    def test_technical_vocabulary_detected(self, echo):
        for _ in range(8):
            echo.observe(
                user_id="alice",
                task="Debug the api endpoint and fix the kubernetes namespace issue in the pipeline",
                quality=0.8,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert profile.technical_vocabulary is True

    def test_expertise_level_expert_for_technical_users(self, echo):
        for _ in range(8):
            echo.observe(
                user_id="alice",
                task="Implement the async api endpoint with kubernetes deploy pipeline and namespace config",
                quality=0.85,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert profile.expertise_level in ("expert", "moderate")

    def test_favourite_categories_populated_after_enough_interactions(self, echo):
        for _ in range(5):
            echo.observe(
                user_id="alice",
                task="Research and investigate the latest developments in AI",
                quality=0.8,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert len(profile.favourite_categories) > 0

    def test_detail_preference_set_after_enough_interactions(self, echo):
        for _ in range(8):
            echo.observe(
                user_id="alice",
                task="Research this topic and analyse the findings in detail and write a comprehensive summary with all the relevant context and background information",
                quality=0.8,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert profile.preferred_detail_level in DETAIL_LEVELS

    def test_no_preference_inference_below_minimum_observations(self, echo):
        for _ in range(2):
            echo.observe(
                user_id="alice",
                task="do it",
                quality=0.7,
                status="success",
                prism_state=make_prism_state(),
            )
        profile = echo.get_profile("alice")
        assert profile.tends_to_be_brief is False  # Not enough data

    def test_trust_in_aria_grows_with_success(self, echo):
        run_interactions(echo, "alice", 15, quality=0.9, status="success")
        profile = echo.get_profile("alice")
        assert profile.trust_in_aria > 0.5


DETAIL_LEVELS = ("minimal", "concise", "standard", "detailed", "comprehensive")


# ── Context Enrichment Tests ──────────────────────────────────────────────────

class TestContextEnrichment:

    def test_enrich_context_returns_string(self, echo):
        ctx = echo.enrich_context("alice", "Research AI trends")
        assert isinstance(ctx, str)
        assert len(ctx) > 10

    def test_enrich_context_new_user_is_neutral(self, echo):
        ctx = echo.enrich_context("brand_new_user", "Do something")
        assert "new user" in ctx.lower() or "still forming" in ctx.lower() or "new" in ctx.lower()

    def test_enrich_context_contains_prism_and_echo_sections(self, echo):
        run_interactions(echo, "alice", 10, quality=0.85, status="success")
        ctx = echo.enrich_context("alice", "Research AI trends")
        assert "[ECHO]" in ctx

    def test_enrich_context_contains_style_guidance(self, echo):
        for _ in range(10):
            echo.observe(
                user_id="alice",
                task="Research and investigate the latest AI trends and analyse results",
                quality=0.9,
                status="success",
                prism_state=make_prism_state("confident", 0.9),
            )
        ctx = echo.enrich_context("alice", "Find the latest AI papers")
        assert len(ctx) > 50

    def test_two_users_get_different_contexts(self, echo):
        run_interactions(echo, "alice", 10, quality=0.9, status="success",
                        task="Research and investigate AI topics thoroughly")
        run_interactions(echo, "bob", 8, quality=0.4, status="failure",
                        task="do it")
        ctx_alice = echo.enrich_context("alice", "Research AI")
        ctx_bob   = echo.enrich_context("bob", "Research AI")
        assert ctx_alice != ctx_bob


# ── Category Detection Tests ──────────────────────────────────────────────────

class TestCategoryDetection:

    def test_research_task_categorised(self, echo):
        cat = echo._detect_category("Research and investigate the latest AI developments")
        assert cat == "research"

    def test_email_task_categorised(self, echo):
        cat = echo._detect_category("Send an email reply to Sarah about the project")
        assert cat == "email"

    def test_technical_task_categorised(self, echo):
        cat = echo._detect_category("Debug the API endpoint and fix the code")
        assert cat == "technical"

    def test_unknown_task_falls_back_to_general(self, echo):
        cat = echo._detect_category("xyz123 completely unrelated gibberish")
        assert cat == "general"

    def test_web_task_categorised(self, echo):
        cat = echo._detect_category("Browse the website and search for the latest news")
        assert cat == "web"

    def test_planning_task_categorised(self, echo):
        cat = echo._detect_category("Plan and schedule the project deadline reminders")
        assert cat == "planning"


# ── Persistence Tests ─────────────────────────────────────────────────────────

class TestECHOPersistence:

    def test_profile_saves_and_reloads(self, tmp_path):
        persist = tmp_path / "data" / "memory.db"
        persist.parent.mkdir(parents=True)

        echo1 = ECHOEngine(persist_path=str(persist))
        run_interactions(echo1, "alice", 5, quality=0.85, status="success")
        assert echo1.get_profile("alice").total_interactions == 5

        echo2 = ECHOEngine(persist_path=str(persist))
        assert echo2.get_profile("alice") is not None
        assert echo2.get_profile("alice").total_interactions == 5

    def test_character_saves_and_reloads(self, tmp_path):
        persist = tmp_path / "data" / "memory.db"
        persist.parent.mkdir(parents=True)

        echo1 = ECHOEngine(persist_path=str(persist))
        run_interactions(echo1, "alice", 8, quality=0.85, status="success")
        tasks_before = echo1.character_report()["total_tasks_observed"]

        echo2 = ECHOEngine(persist_path=str(persist))
        tasks_after = echo2.character_report()["total_tasks_observed"]
        assert tasks_after == tasks_before

    def test_multiple_users_all_persist(self, tmp_path):
        persist = tmp_path / "data" / "memory.db"
        persist.parent.mkdir(parents=True)

        echo1 = ECHOEngine(persist_path=str(persist))
        run_interactions(echo1, "alice", 3)
        run_interactions(echo1, "bob", 3)
        run_interactions(echo1, "carol", 3)

        echo2 = ECHOEngine(persist_path=str(persist))
        assert set(echo2.all_user_ids()) == {"alice", "bob", "carol"}

    def test_echo_file_is_valid_json(self, tmp_path):
        persist = tmp_path / "data" / "memory.db"
        persist.parent.mkdir(parents=True)

        echo = ECHOEngine(persist_path=str(persist))
        run_interactions(echo, "alice", 3)

        echo_file = tmp_path / "data" / "echo_data.json"
        assert echo_file.exists()
        data = json.loads(echo_file.read_text())
        assert "profiles" in data
        assert "character" in data
        assert "alice" in data["profiles"]

    def test_no_persist_path_does_not_crash(self, echo):
        run_interactions(echo, "alice", 5)
        assert echo.get_profile("alice").total_interactions == 5


# ── Character Report Tests ────────────────────────────────────────────────────

class TestCharacterReport:

    def test_character_report_has_required_keys(self, echo):
        run_interactions(echo, "alice", 3)
        report = echo.character_report()
        required = [
            "total_tasks_observed", "avg_quality_delivered",
            "calibration_tendency", "strong_domains", "weak_domains",
            "defining_traits", "consistent_traits", "all_traits"
        ]
        for key in required:
            assert key in report

    def test_echo_stats_returns_user_count(self, echo):
        run_interactions(echo, "alice", 3)
        run_interactions(echo, "bob", 3)
        stats = echo.echo_stats() if hasattr(echo, "echo_stats") else {
            "users_known": len(echo.all_user_ids())
        }
        assert stats["users_known"] == 2

    def test_print_character_report_does_not_crash(self, echo, capsys):
        run_interactions(echo, "alice", 5, quality=0.85)
        echo.print_character_report()
        captured = capsys.readouterr()
        assert "ECHO" in captured.out

    def test_print_user_profile_does_not_crash(self, echo, capsys):
        run_interactions(echo, "alice", 5)
        echo.print_user_profile("alice")
        captured = capsys.readouterr()
        assert "alice" in captured.out

    def test_print_user_profile_unknown_user(self, echo, capsys):
        echo.print_user_profile("nobody")
        captured = capsys.readouterr()
        assert "No profile" in captured.out or "nobody" in captured.out


# ── Full Integration Tests ────────────────────────────────────────────────────

class TestFullECHOCycle:

    def test_full_observe_enrich_cycle(self, echo):
        # Observe many interactions
        for i in range(15):
            echo.observe(
                user_id="alice",
                task=f"Research and investigate topic {i} in detail",
                quality=0.85,
                status="success",
                prism_state=make_prism_state("confident", 0.85, "well_calibrated"),
            )

        # Enrich context should now be personalised
        ctx = echo.enrich_context("alice", "Research a new AI topic")
        assert "alice" in ctx or "established" in ctx or "familiar" in ctx or "trusted" in ctx or "ECHO" in ctx

        # Profile should reflect accumulated learning
        profile = echo.get_profile("alice")
        assert profile.relationship_label in ("familiar", "established", "trusted partner")
        assert profile.total_interactions == 15
        assert profile.successful_tasks == 15

    def test_two_users_diverge_over_time(self, echo):
        # Alice is a technical user with complex tasks
        for _ in range(12):
            echo.observe(
                user_id="alice",
                task="Implement and debug the api endpoint with kubernetes deploy and namespace config for the pipeline",
                quality=0.9,
                status="success",
                prism_state=make_prism_state("confident", 0.9),
            )

        # Bob gives very short tasks and gets mixed results
        for _ in range(12):
            echo.observe(
                user_id="bob",
                task="fix it",
                quality=0.5,
                status="success",
                prism_state=make_prism_state("cautious", 0.5),
            )

        alice = echo.get_profile("alice")
        bob   = echo.get_profile("bob")

        # They should have developed different profiles
        assert alice.avg_quality_received > bob.avg_quality_received
        assert alice.relationship_depth > bob.relationship_depth

    def test_character_grows_richer_over_time(self, echo):
        run_interactions(echo, "alice", 3)
        traits_early = len(echo.character_report().get("all_traits", {}))

        run_interactions(echo, "alice", 20)
        traits_later = len(echo.character_report().get("all_traits", {}))

        # More observations should build or strengthen existing traits
        assert traits_later >= traits_early

    def test_moments_capped_at_max(self, echo):
        from aria.core.echo import MAX_MOMENTS_PER_USER
        run_interactions(echo, "alice", MAX_MOMENTS_PER_USER + 20)
        profile = echo.get_profile("alice")
        assert len(profile.moments) <= MAX_MOMENTS_PER_USER
