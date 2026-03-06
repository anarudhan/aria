"""
PRISM — Progressive Reasoning with Integrated Self-Monitoring
=============================================================
Version 1.0 | Part of ARIA v1.2.0

PRISM is ARIA's cognitive engine. It runs alongside every single task
and governs how ARIA actually thinks rather than just what she does.

Most agent frameworks are executors. They receive a task, route it, run it,
return a result. PRISM makes ARIA something fundamentally different — a system
that is genuinely honest about what it knows, what it does not know, how
confident it really is, and what it is learning about itself over time.

Every task ARIA runs passes through three PRISM phases:

    PRE-REASONING
    Before any decomposition or execution begins, PRISM asks four honest
    questions about the task. Not rhetorically — with real analysis.
    How well do I understand what is being asked? How new is this to me?
    What does my actual history say about tasks like this? And given all of
    that, what approach should I take? The answers shape everything downstream.

    MONITORING
    While execution runs, PRISM watches for drift. When an agent fails
    unexpectedly, when results come back very different from what was predicted,
    when the execution path diverges from the strategy PRISM chose — these are
    logged as drift events. Enough drift and PRISM raises a flag.

    REFLECTION
    After every task, PRISM compares what it predicted to what actually
    happened. If it said confidence 0.8 and the task failed, it records a
    calibration error of 0.8. Over hundreds of tasks this builds a genuine
    self-knowledge — ARIA starts to know her own blind spots because the
    numbers show them.

The thing that makes PRISM special is not any single feature. It is that
over time ARIA develops a calibrated self-model. She knows which kinds of
tasks she tends to overestimate herself on. She knows which strategies
actually work for her versus which ones she only thinks work. She becomes
genuinely, measurably more self-aware with every task she runs.

Integration:
    PRISM integrates into the orchestrator at four points.
    Call prism.pre_reason() after memory recall, before decomposition.
    Pass the returned PRISMState context into the decomposer prompt.
    Call prism.on_node_complete() and prism.on_node_fail() during execution.
    Call prism.reflect() after quality assessment.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class DriftEvent:
    """
    Something happened during execution that PRISM did not predict.

    Drift events are the live signal that a task is going differently
    from the strategy PRISM chose. Low-severity drift is normal — tasks
    are messy. High-severity drift means the pre-reasoning was significantly
    wrong and PRISM should update its self-model accordingly.
    """
    event_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   float = field(default_factory=time.time)
    event_type:  str   = "unknown"
    description: str   = ""
    severity:    str   = "low"       # low, medium, high
    node_id:     Optional[str] = None
    agent_id:    Optional[str] = None


@dataclass
class PRISMState:
    """
    The complete cognitive state for one task cycle.

    This object is created at the start of pre-reasoning and updated
    throughout the execution lifecycle. It carries everything PRISM
    knows about this task — the predictions made before it ran, the
    events that happened during it, and the reflection after it finished.

    PRISMState objects are stored in ARIA's memory so future tasks on
    similar topics can learn from this one's calibration data.
    """

    task_id:   str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task:      str   = ""
    user_id:   str   = "default"
    timestamp: float = field(default_factory=time.time)

    # Pre-reasoning outputs — the four honest questions
    comprehension_confidence: float = 0.5
    comprehension_notes:      list  = field(default_factory=list)

    novelty_score:       float = 0.5
    similar_past_tasks:  list  = field(default_factory=list)

    historical_accuracy: float = 0.6   # Bayesian prior until we have real data
    historical_sample:   int   = 0     # How many past tasks inform this score

    chosen_strategy:    str = "cautious"
    strategy_rationale: str = ""

    pre_reasoning_seconds: float = 0.0

    # Monitoring outputs — updated live during execution
    drift_events:      list = field(default_factory=list)
    monitoring_flags:  list = field(default_factory=list)
    strategy_adjusted: bool = False
    adjusted_strategy: Optional[str] = None

    # Reflection outputs — filled in after execution completes
    actual_outcome:       str   = ""
    actual_quality:       float = 0.0
    calibration_error:    float = 0.0
    calibration_verdict:  str   = ""
    reflection_insight:   str   = ""
    reflection_seconds:   float = 0.0

    @property
    def is_high_drift(self) -> bool:
        """True if enough high-severity drift events occurred to warrant concern."""
        high = sum(1 for e in self.drift_events if e.severity == "high")
        medium = sum(1 for e in self.drift_events if e.severity == "medium")
        return high >= 1 or medium >= 3

    @property
    def overall_confidence(self) -> float:
        """
        A single honest confidence number for this task.

        This is not the comprehension confidence alone. It blends all three
        inputs — how well PRISM understood the task, how novel it was, and
        what the historical track record says. The result is ARIA's genuine
        probability estimate that this task will go well.
        """
        novelty_penalty = self.novelty_score * 0.3
        return max(0.1, min(0.95,
            (self.comprehension_confidence * 0.4) +
            (self.historical_accuracy * 0.4) +
            ((1.0 - novelty_penalty) * 0.2)
        ))

    def to_context_string(self) -> str:
        """
        Returns a plain-English summary of PRISM's pre-reasoning.

        This string is injected into the decomposer prompt so the LLM
        that plans the task knows exactly what PRISM thinks about it.
        Honest framing of uncertainty here leads to better planning.
        """
        lines = [
            "[PRISM Pre-Reasoning]",
            f"Overall confidence: {self.overall_confidence:.0%}",
            f"Strategy: {self.chosen_strategy}",
            f"Rationale: {self.strategy_rationale}",
        ]

        if self.comprehension_notes:
            lines.append(f"Comprehension notes: {', '.join(self.comprehension_notes[:3])}")

        if self.similar_past_tasks:
            lines.append(f"Similar past tasks found: {len(self.similar_past_tasks)}")

        if self.historical_sample > 0:
            lines.append(
                f"Historical accuracy on similar tasks: "
                f"{self.historical_accuracy:.0%} over {self.historical_sample} executions"
            )
        else:
            lines.append("No historical data for this task type — using prior estimate")

        strategy_guidance = {
            "confident":   "Proceed directly. Use established patterns. Be decisive.",
            "exploratory": "This is new territory. Document findings. Be curious and thorough.",
            "cautious":    "Go carefully. Hedge where uncertain. Flag ambiguities explicitly.",
            "delegating":  "High uncertainty detected. Provide best effort but recommend human review.",
        }
        lines.append(f"Execution guidance: {strategy_guidance.get(self.chosen_strategy, '')}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task_id":                 self.task_id,
            "task":                    self.task[:100],
            "user_id":                 self.user_id,
            "timestamp":               self.timestamp,
            "comprehension_confidence": self.comprehension_confidence,
            "comprehension_notes":     self.comprehension_notes,
            "novelty_score":           self.novelty_score,
            "similar_past_count":      len(self.similar_past_tasks),
            "historical_accuracy":     self.historical_accuracy,
            "historical_sample":       self.historical_sample,
            "chosen_strategy":         self.chosen_strategy,
            "strategy_rationale":      self.strategy_rationale,
            "overall_confidence":      self.overall_confidence,
            "drift_event_count":       len(self.drift_events),
            "is_high_drift":           self.is_high_drift,
            "actual_outcome":          self.actual_outcome,
            "actual_quality":          self.actual_quality,
            "calibration_error":       self.calibration_error,
            "calibration_verdict":     self.calibration_verdict,
            "reflection_insight":      self.reflection_insight,
        }


@dataclass
class CalibrationRecord:
    """
    One data point in PRISM's long-term calibration history.

    Over many tasks, these records tell ARIA exactly where she
    over-estimates and under-estimates herself. That self-knowledge
    feeds back into future pre-reasoning to make confidence scores
    progressively more accurate.
    """
    record_id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:          float = field(default_factory=time.time)
    task_category:      str   = "general"
    chosen_strategy:    str   = "cautious"
    predicted_confidence: float = 0.5
    actual_quality:     float  = 0.0
    calibration_error:  float  = 0.0
    verdict:            str    = ""


# ── PRISM Engine ───────────────────────────────────────────────────────────────

class PRISMEngine:
    """
    Progressive Reasoning with Integrated Self-Monitoring.

    PRISM is ARIA's honest inner voice. It runs before, during, and after
    every task, asking the questions that most agent frameworks never ask:
    Do I actually understand this? Have I done this before? Am I being
    honest about how confident I really am? Did this go the way I expected?

    Over time PRISM builds a calibration history that makes ARIA genuinely
    self-aware. Not in a philosophical sense but in a measurable one — she
    knows which task types she predicts well and which ones she misjudges,
    because the numbers show her.
    """

    # Keywords that signal specific domains — used in comprehension analysis
    _DOMAIN_VOCABULARY: dict[str, list[str]] = {
        "technical":    ["code", "api", "function", "algorithm", "database", "server",
                         "deploy", "debug", "architecture", "implementation", "script"],
        "research":     ["research", "find", "search", "analyse", "compare", "summarise",
                         "report", "study", "investigate", "evaluate", "review"],
        "communication": ["email", "message", "reply", "draft", "send", "teams",
                          "write", "respond", "compose", "notify", "inform"],
        "document":     ["document", "report", "word", "docx", "create", "format",
                         "write up", "prepare", "generate"],
        "memory":       ["remember", "recall", "store", "forget", "what did", "last time",
                         "previously", "history", "past"],
        "web":          ["browse", "website", "url", "search", "internet", "news",
                         "current", "latest", "find online"],
    }

    # Patterns that signal genuine ambiguity in a task description
    _AMBIGUITY_PATTERNS: list[tuple[str, str]] = [
        (r'\bit\b|\bthis\b|\bthat\b|\bthey\b', "unresolved pronoun reference"),
        (r'\bsomething\b|\bstuff\b|\bthing\b',  "vague noun"),
        (r'\bmaybe\b|\bpossibly\b|\bsomehow\b', "uncertain qualifier from user"),
        (r'\bor\b.*\bor\b',                      "multiple alternatives without clear preference"),
        (r'^\s*\w+\s*$',                         "very short task — may lack context"),
    ]

    # Strategy selection matrix based on the three score inputs
    _STRATEGY_RULES: list[tuple[float, float, float, str, str]] = [
        # (min_comprehension, min_history, max_novelty, strategy, rationale)
        (0.75, 0.70, 0.35, "confident",
         "High comprehension, strong track record, familiar territory"),
        (0.55, 0.50, 0.70, "exploratory",
         "Moderate footing but significant novelty — worth exploring carefully"),
        (0.40, 0.40, 1.00, "cautious",
         "Either the task is unclear or history suggests care is warranted"),
        (0.00, 0.00, 1.00, "delegating",
         "Too much uncertainty — best effort with strong human review recommendation"),
    ]

    def __init__(self, persist_path: Optional[str] = None):
        self._calibration_history: list[CalibrationRecord] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_calibration()

    # ── Phase 1: Pre-Reasoning ─────────────────────────────────────────────────

    def pre_reason(
        self,
        task:         str,
        past_memories: list,
        user_id:      str = "default",
        past_lessons:  Optional[list] = None,
        trust_summaries: Optional[dict] = None,
    ) -> PRISMState:
        """
        Ask the four honest questions before any execution begins.

        This is the heart of PRISM. Before ARIA plans or executes anything,
        this method gives her a genuine cognitive assessment of what she is
        about to attempt. The result shapes the decomposer prompt, the
        execution strategy, and ultimately how honestly ARIA communicates
        about her uncertainty.

        Args:
            task:            The natural language task description
            past_memories:   Memory recall results for this task
            user_id:         Who is running the task
            past_lessons:    Lessons from the improvement engine (optional)
            trust_summaries: Agent trust data from the trust ledger (optional)

        Returns:
            A PRISMState with pre-reasoning fully populated
        """
        start = time.monotonic()
        state = PRISMState(task=task, user_id=user_id)

        # Question 1 — Comprehension
        state.comprehension_confidence, state.comprehension_notes = (
            self._assess_comprehension(task)
        )

        # Question 2 — Novelty
        state.novelty_score, state.similar_past_tasks = (
            self._assess_novelty(task, past_memories)
        )

        # Question 3 — Historical accuracy
        state.historical_accuracy, state.historical_sample = (
            self._assess_history(task, past_lessons, trust_summaries)
        )

        # Question 4 — Strategy selection
        state.chosen_strategy, state.strategy_rationale = (
            self._select_strategy(
                state.comprehension_confidence,
                state.historical_accuracy,
                state.novelty_score,
            )
        )

        state.pre_reasoning_seconds = time.monotonic() - start

        self._print_pre_reasoning(state)
        return state

    def _assess_comprehension(self, task: str) -> tuple[float, list[str]]:
        """
        Question 1: How well do I actually understand what is being asked?

        This is purely linguistic analysis — no LLM, no API call.
        We look at clarity, specificity, ambiguity signals, and vocabulary
        overlap with known domains. The score is deliberately conservative.
        ARIA never claims perfect comprehension because perfect comprehension
        of natural language is a fiction.
        """
        notes = []
        score = 0.60  # Start at moderate confidence

        task_lower = task.lower()
        words = re.findall(r'\b\w+\b', task_lower)
        word_count = len(words)

        # Length signal — very short tasks are often underspecified
        if word_count < 5:
            score -= 0.15
            notes.append("very short task description")
        elif word_count > 15:
            score += 0.05  # More words usually means more context

        # Domain vocabulary overlap — do we recognise the territory?
        domain_hits = {}
        for domain, vocab in self._DOMAIN_VOCABULARY.items():
            hits = sum(1 for v in vocab if v in task_lower)
            if hits > 0:
                domain_hits[domain] = hits

        if domain_hits:
            top_domain = max(domain_hits, key=domain_hits.get)
            score += min(0.15, domain_hits[top_domain] * 0.05)
            notes.append(f"recognised as {top_domain} task")

        # Ambiguity patterns — signals that the task may have hidden assumptions
        for pattern, description in self._AMBIGUITY_PATTERNS:
            if re.search(pattern, task_lower):
                score -= 0.08
                notes.append(description)

        # Specificity bonus — concrete nouns and proper nouns suggest clear intent
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', task)
        if len(proper_nouns) >= 2:
            score += 0.05
            notes.append(f"specific entities mentioned ({', '.join(proper_nouns[:2])})")

        # Question marks signal the user is asking rather than instructing
        if '?' in task:
            score += 0.03

        # Clamp to honest range — we never claim perfect or zero comprehension
        score = max(0.25, min(0.90, score))

        if not notes:
            notes.append("clear and direct task description")

        return round(score, 3), notes

    def _assess_novelty(
        self,
        task: str,
        past_memories: list,
    ) -> tuple[float, list[str]]:
        """
        Question 2: How new is this to me?

        High novelty means ARIA should be more exploratory and more transparent
        about uncertainty. Low novelty means she can draw on established patterns
        and be more direct. This is not about difficulty — a hard task can be
        familiar. It is specifically about how much relevant past experience exists.
        """
        if not past_memories:
            return 0.85, []  # No memories at all — high novelty

        task_words = set(re.findall(r'\b\w{4,}\b', task.lower()))
        similar = []

        for memory in past_memories:
            content = getattr(memory, 'content', str(memory))
            memory_words = set(re.findall(r'\b\w{4,}\b', content.lower()))

            if not task_words or not memory_words:
                continue

            overlap = len(task_words & memory_words)
            union = len(task_words | memory_words)
            similarity = overlap / union if union > 0 else 0.0

            if similarity > 0.20:
                similar.append(content[:80])

        if not similar:
            return 0.80, []

        # More similar past tasks = lower novelty
        # The curve is logarithmic — going from 0 to 3 similar tasks matters a lot,
        # going from 10 to 13 matters much less
        novelty = max(0.05, 0.85 - (math.log1p(len(similar)) * 0.25))
        return round(novelty, 3), similar[:5]

    def _assess_history(
        self,
        task: str,
        past_lessons: Optional[list],
        trust_summaries: Optional[dict],
    ) -> tuple[float, int]:
        """
        Question 3: What does my actual track record say about tasks like this?

        This is the most important question for calibration. ARIA could feel
        confident about a task type she has historically struggled with. PRISM
        overrides that feeling with evidence. The score starts at the Bayesian
        prior (0.6) and updates based on real outcome data from the improvement
        engine and trust ledger.
        """
        prior = 0.60  # Bayesian prior — our starting estimate
        total_weight = 1.0  # The prior has a weight of 1

        score_sum = prior

        # Pull from improvement engine lessons if available
        if past_lessons:
            relevant = [
                lsn for lsn in past_lessons
                if hasattr(lsn, 'matches') and lsn.matches(task)
            ]

            if relevant:
                successes = sum(1 for lsn in relevant if getattr(lsn, 'outcome', '') == "success")
                sum(1 for lsn in relevant if getattr(lsn, 'outcome', '') == "failure")
                total     = len(relevant)

                if total > 0:
                    lesson_accuracy = successes / total
                    weight = min(3.0, math.log1p(total))
                    score_sum += lesson_accuracy * weight
                    total_weight += weight


        # Pull from trust summaries if available
        # trust_summaries is a list[dict] from get_all_summaries() — not a dict
        if trust_summaries:
            for summary in (trust_summaries if isinstance(trust_summaries, list) else trust_summaries.values()):
                if not isinstance(summary, dict):
                    continue
                agent_trust = summary.get('overall_trust', 0.6)
                score_sum += agent_trust * 0.5
                total_weight += 0.5

        # Pull from PRISM's own calibration history
        category = self._categorise_task(task)
        relevant_calibrations = [
            r for r in self._calibration_history
            if r.task_category == category
        ]

        if relevant_calibrations:
            recent = sorted(relevant_calibrations, key=lambda r: -r.timestamp)[:20]
            avg_quality = sum(r.actual_quality for r in recent) / len(recent)
            weight = min(2.0, math.log1p(len(recent)))
            score_sum += avg_quality * weight
            total_weight += weight

        accuracy = score_sum / total_weight
        sample = len(past_lessons or []) + len(relevant_calibrations)

        return round(max(0.1, min(0.95, accuracy)), 3), sample

    def _select_strategy(
        self,
        comprehension: float,
        history:       float,
        novelty:       float,
    ) -> tuple[str, str]:
        """
        Question 4: Given what I know, what approach should I take?

        This is not a heuristic that sounds reasonable. It is a decision
        table that ARIA applies consistently. The strategy shapes how the
        decomposer plans the task and how ARIA communicates uncertainty.

        The four strategies form a spectrum from most to least assured:
        confident, exploratory, cautious, delegating.
        """
        for min_comp, min_hist, max_nov, strategy, rationale in self._STRATEGY_RULES:
            if (comprehension >= min_comp and
                history >= min_hist and
                novelty <= max_nov):
                return strategy, rationale

        return "delegating", "Uncertainty across all three dimensions — strong human review recommended"

    # ── Phase 2: Monitoring ────────────────────────────────────────────────────

    def on_node_complete(
        self,
        state:    PRISMState,
        node_id:  str,
        agent_id: str,
        quality:  float,
        expected_quality: float = 0.6,
    ) -> None:
        """
        A node finished execution. Was the result what PRISM expected?

        If quality is significantly below what PRISM's strategy predicted,
        that is a drift event. Not an error — a signal that something in
        the pre-reasoning was off.
        """
        if quality < expected_quality - 0.25:
            event = DriftEvent(
                event_type  = "quality_shortfall",
                description = (
                    f"Node {node_id} completed but quality {quality:.2f} is below "
                    f"expected {expected_quality:.2f} for {state.chosen_strategy} strategy"
                ),
                severity    = "medium" if quality > 0.3 else "high",
                node_id     = node_id,
                agent_id    = agent_id,
            )
            state.drift_events.append(event)

            if state.is_high_drift and not state.strategy_adjusted:
                state.strategy_adjusted = True
                state.adjusted_strategy = "cautious"
                state.monitoring_flags.append(
                    f"Strategy adjusted to cautious after quality drift at node {node_id}"
                )

    def on_node_fail(
        self,
        state:    PRISMState,
        node_id:  str,
        agent_id: str,
        error:    str = "",
    ) -> None:
        """
        A node failed. PRISM logs this as a drift event with appropriate severity.

        A single failure is normal — tasks are messy. Multiple failures,
        especially on a task PRISM rated as confident, reveal a miscalibration
        that PRISM will record and learn from.
        """
        severity = "low"
        if state.chosen_strategy == "confident":
            severity = "high"   # Failure on a confident task is significant
        elif state.chosen_strategy == "exploratory":
            severity = "medium"

        event = DriftEvent(
            event_type  = "node_failure",
            description = f"Agent {agent_id} failed on node {node_id}. {error[:100]}",
            severity    = severity,
            node_id     = node_id,
            agent_id    = agent_id,
        )
        state.drift_events.append(event)

        if state.is_high_drift and not state.strategy_adjusted:
            state.strategy_adjusted = True
            state.adjusted_strategy = "cautious"
            state.monitoring_flags.append(
                "Multiple failures detected — execution strategy downgraded to cautious"
            )

    # ── Phase 3: Reflection ────────────────────────────────────────────────────

    def reflect(
        self,
        state:   PRISMState,
        outcome: str,
        quality: float,
    ) -> PRISMState:
        """
        After the task finishes, compare prediction to reality.

        This is where ARIA learns about herself. If PRISM predicted confidence
        0.8 and the task quality was 0.3, that is a calibration error of 0.5.
        Over time these errors reveal systematic biases — task types ARIA
        consistently over- or under-estimates herself on.

        The calibration record is stored and shapes future pre-reasoning.

        Args:
            state:   The PRISMState that was created during pre-reasoning
            outcome: "success", "failure", or "partial"
            quality: The actual quality score the orchestrator measured
        """
        start = time.monotonic()

        state.actual_outcome = outcome
        state.actual_quality = quality

        # Calibration error — how wrong was our confidence?
        predicted = state.overall_confidence
        actual    = quality

        state.calibration_error = round(abs(predicted - actual), 3)

        if state.calibration_error < 0.15:
            state.calibration_verdict = "well_calibrated"
        elif predicted > actual:
            state.calibration_verdict = "overconfident"
        else:
            state.calibration_verdict = "underconfident"

        # Reflection insight — one honest sentence about what PRISM learned
        state.reflection_insight = self._generate_insight(state)

        # Record this in calibration history for future pre-reasoning
        category = self._categorise_task(state.task)
        record = CalibrationRecord(
            task_category        = category,
            chosen_strategy      = state.chosen_strategy,
            predicted_confidence = state.overall_confidence,
            actual_quality       = quality,
            calibration_error    = state.calibration_error,
            verdict              = state.calibration_verdict,
        )
        self._calibration_history.append(record)

        # Keep history bounded — last 500 records
        if len(self._calibration_history) > 500:
            self._calibration_history = self._calibration_history[-500:]

        self._save_calibration()

        state.reflection_seconds = time.monotonic() - start

        self._print_reflection(state)
        return state

    # ── Calibration Reporting ──────────────────────────────────────────────────

    def calibration_report(self) -> dict:
        """
        PRISM's honest account of how well it predicts its own performance.

        This is the number that matters most for ARIA's self-awareness.
        A well-calibrated PRISM means ARIA's stated confidence actually
        corresponds to her real success rate. An uncalibrated PRISM means
        ARIA is either consistently overconfident or underconfident —
        which the report will show clearly.
        """
        if not self._calibration_history:
            return {
                "total_tasks":    0,
                "message":        "No calibration data yet. Run tasks to build PRISM's self-model.",
            }

        total = len(self._calibration_history)
        avg_error = sum(r.calibration_error for r in self._calibration_history) / total

        verdicts: dict[str, int] = {}
        for r in self._calibration_history:
            verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1

        by_category: dict[str, dict] = {}
        for r in self._calibration_history:
            cat = r.task_category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "total_error": 0.0, "total_quality": 0.0}
            by_category[cat]["count"] += 1
            by_category[cat]["total_error"] += r.calibration_error
            by_category[cat]["total_quality"] += r.actual_quality

        category_summary = {}
        for cat, data in by_category.items():
            n = data["count"]
            category_summary[cat] = {
                "tasks":          n,
                "avg_error":      round(data["total_error"] / n, 3),
                "avg_quality":    round(data["total_quality"] / n, 3),
                "calibration":    "good" if data["total_error"] / n < 0.2 else "needs improvement",
            }

        # Find worst calibrated category — where ARIA most misjudges herself
        worst = None
        if category_summary:
            worst = max(category_summary, key=lambda c: category_summary[c]["avg_error"])

        return {
            "total_tasks":         total,
            "avg_calibration_error": round(avg_error, 3),
            "calibration_grade":   self._calibration_grade(avg_error),
            "verdict_breakdown":   verdicts,
            "by_category":         category_summary,
            "weakest_category":    worst,
            "interpretation":      self._interpret_calibration(avg_error, verdicts),
        }

    def print_calibration_report(self) -> None:
        """Print the calibration report in human-readable format."""
        report = self.calibration_report()

        print("\n" + "=" * 62)
        print("  PRISM Calibration Report — How Well ARIA Knows Herself")
        print("=" * 62)

        if report["total_tasks"] == 0:
            print(f"\n  {report['message']}")
            print("=" * 62 + "\n")
            return

        print(f"\n  Tasks analysed:   {report['total_tasks']}")
        print(f"  Avg error:        {report['avg_calibration_error']:.3f}")
        print(f"  Calibration grade: {report['calibration_grade']}")
        print(f"\n  {report['interpretation']}")

        if report.get("verdict_breakdown"):
            print("\n  Verdict breakdown:")
            for verdict, count in sorted(report["verdict_breakdown"].items()):
                pct = count / report["total_tasks"] * 100
                print(f"    {verdict:<20} {count:>3} tasks ({pct:.0f}%)")

        if report.get("by_category"):
            print("\n  By task category:")
            for cat, data in sorted(report["by_category"].items(),
                                     key=lambda x: -x[1]["avg_error"]):
                flag = " ⚠️" if data["calibration"] == "needs improvement" else " ✅"
                print(f"    {cat:<18} avg error {data['avg_error']:.2f} "
                      f"avg quality {data['avg_quality']:.2f}{flag}")

        if report.get("weakest_category"):
            print(f"\n  Weakest category: {report['weakest_category']}")
            print("  This is where ARIA most often misjudges her own confidence.")

        print("=" * 62 + "\n")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _categorise_task(self, task: str) -> str:
        """Assign a category label to a task based on vocabulary overlap."""
        task_lower = task.lower()
        scores = {}
        for domain, vocab in self._DOMAIN_VOCABULARY.items():
            scores[domain] = sum(1 for v in vocab if v in task_lower)

        if max(scores.values(), default=0) == 0:
            return "general"

        return max(scores, key=scores.get)

    def _generate_insight(self, state: PRISMState) -> str:
        """Generate one honest sentence summarising what PRISM learned from this task."""
        verdict = state.calibration_verdict
        strategy = state.chosen_strategy
        error = state.calibration_error
        drift_count = len(state.drift_events)

        if verdict == "well_calibrated" and drift_count == 0:
            return (
                f"PRISM's {strategy} strategy was accurate — "
                f"predicted confidence matched outcome within {error:.0%}"
            )

        if verdict == "overconfident":
            return (
                f"PRISM overestimated confidence for this {self._categorise_task(state.task)} task "
                f"by {error:.0%} — future similar tasks should receive more conservative estimates"
            )

        if verdict == "underconfident":
            return (
                f"PRISM underestimated confidence for this {self._categorise_task(state.task)} task "
                f"by {error:.0%} — the {strategy} strategy was unnecessarily conservative"
            )

        if drift_count > 0:
            return (
                f"Execution drifted with {drift_count} unexpected events — "
                f"pre-reasoning did not fully anticipate task complexity"
            )

        return f"Task completed. Calibration error {error:.0%} recorded for future pre-reasoning."

    def _calibration_grade(self, avg_error: float) -> str:
        """Translate average calibration error to a simple letter grade."""
        if avg_error < 0.10:
            return "A — excellent self-knowledge"
        if avg_error < 0.20:
            return "B — good self-knowledge"
        if avg_error < 0.30:
            return "C — moderate self-knowledge"
        if avg_error < 0.40:
            return "D — limited self-knowledge"
        return "F — poor self-knowledge"

    def _interpret_calibration(self, avg_error: float, verdicts: dict) -> str:
        total = sum(verdicts.values())
        if total == 0:
            return "No data yet."

        over  = verdicts.get("overconfident",  0) / total
        under = verdicts.get("underconfident", 0) / total

        if avg_error < 0.15:
            return "ARIA has strong self-knowledge. Her confidence ratings are reliable."

        if over > 0.5:
            return (
                f"ARIA tends to overestimate herself ({over:.0%} of tasks). "
                "Her confidence ratings should be read with some discount."
            )

        if under > 0.5:
            return (
                f"ARIA tends to underestimate herself ({under:.0%} of tasks). "
                "She is often more capable than her confidence suggests."
            )

        return (
            f"ARIA's calibration is mixed. Average error {avg_error:.0%}. "
            "More tasks will clarify systematic patterns."
        )

    def _print_pre_reasoning(self, state: PRISMState) -> None:
        """Print PRISM's pre-reasoning summary to the terminal."""
        icons = {
            "confident":   "🟢",
            "exploratory": "🔵",
            "cautious":    "🟡",
            "delegating":  "🔴",
        }
        icon = icons.get(state.chosen_strategy, "⚪")
        print(
            f"[PRISM] {icon} Strategy: {state.chosen_strategy.upper()} | "
            f"Confidence: {state.overall_confidence:.0%} | "
            f"Novelty: {state.novelty_score:.0%} | "
            f"History: {state.historical_accuracy:.0%} "
            f"({state.historical_sample} samples)"
        )
        if state.comprehension_notes:
            print(f"[PRISM]    Notes: {', '.join(state.comprehension_notes[:2])}")

    def _print_reflection(self, state: PRISMState) -> None:
        """Print PRISM's reflection summary after execution."""
        verdict_icons = {
            "well_calibrated":  "✅",
            "overconfident":    "⚠️",
            "underconfident":   "💡",
        }
        icon = verdict_icons.get(state.calibration_verdict, "📊")
        print(
            f"[PRISM] {icon} Reflection: {state.calibration_verdict} | "
            f"Error: {state.calibration_error:.0%} | "
            f"Drift events: {len(state.drift_events)}"
        )
        print(f"[PRISM]    {state.reflection_insight}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_calibration(self) -> None:
        if self._persist_path is None:
            return
        try:
            path = self._persist_path.parent / "prism_calibration.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "record_id":           r.record_id,
                    "timestamp":           r.timestamp,
                    "task_category":       r.task_category,
                    "chosen_strategy":     r.chosen_strategy,
                    "predicted_confidence": r.predicted_confidence,
                    "actual_quality":      r.actual_quality,
                    "calibration_error":   r.calibration_error,
                    "verdict":             r.verdict,
                }
                for r in self._calibration_history[-500:]
            ]
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_calibration(self) -> None:
        if self._persist_path is None:
            return
        try:
            path = self._persist_path.parent / "prism_calibration.json"
            if not path.exists():
                return
            data = json.loads(path.read_text())
            for item in data:
                record = CalibrationRecord(
                    record_id            = item.get("record_id", str(uuid.uuid4())[:8]),
                    timestamp            = item.get("timestamp", time.time()),
                    task_category        = item.get("task_category", "general"),
                    chosen_strategy      = item.get("chosen_strategy", "cautious"),
                    predicted_confidence = item.get("predicted_confidence", 0.5),
                    actual_quality       = item.get("actual_quality", 0.0),
                    calibration_error    = item.get("calibration_error", 0.0),
                    verdict              = item.get("verdict", ""),
                )
                self._calibration_history.append(record)
        except Exception:
            pass
