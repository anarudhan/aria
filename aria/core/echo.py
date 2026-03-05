"""
ECHO — Experiential Character through Historical Observation
============================================================
Version 1.0 | Part of ARIA v1.3.0

ECHO is ARIA's identity engine. Where PRISM governs how ARIA thinks
in the moment, ECHO governs who ARIA is across time.

The difference matters enormously. PRISM is the voice that says
"I am not sure about this, let me be careful." ECHO is the reason
ARIA already knows that Sarah prefers concise bullet points, that
Marcus always needs a second opinion before acting on a recommendation,
and that the engineering team wants technical depth while the executive
team wants a single clear sentence.

ECHO builds two things in parallel and keeps them updated after every
single interaction.

The first is a relational model per person. Not just facts about them —
a genuine working relationship profile. How they communicate. What they
find helpful versus what irritates them. How much context they need.
Whether they prefer ARIA to take initiative or to ask first. What their
areas of expertise are so ARIA never over-explains things they already
know. How much trust has built up between them specifically.

The second is ARIA's own character profile. This is the part no other
framework has at all. ARIA observes her own behavior across all users
and all task types and builds a picture of herself — her genuine
strengths, her genuine tendencies, and her consistent traits. Over
time this character becomes recognisable. Users feel it even if they
cannot name it. The assistant they are working with has a consistent
personality that they learn to rely on.

The combination of these two — knowing yourself and knowing the person
you are working with — is exactly what makes the difference between
an AI tool and a genuine working partner.

Architecture:
    UserProfile     — Everything ECHO knows about one person
    CharacterTrait  — One observed tendency in ARIA's own behavior
    ARIACharacter   — ARIA's accumulated self-portrait
    RelationshipMoment — One interaction observation stored for pattern analysis
    ECHOEngine      — The main engine, integrates into the orchestrator

Integration:
    Call echo.enrich_context(user_id, task) before decomposition to get
    a context string that shapes how ARIA communicates with this person.
    Call echo.observe(user_id, task, result, prism_state) after every task
    to update the relational model and ARIA's character profile.
    Call echo.character_report() in ARIA's self_report() for a full picture.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Enumerations and Constants ────────────────────────────────────────────────

DETAIL_LEVELS = ("minimal", "concise", "standard", "detailed", "comprehensive")

COMMUNICATION_STYLES = ("direct", "collaborative", "analytical", "supportive", "exploratory")

INITIATIVE_LEVELS = ("always_ask", "ask_when_unsure", "act_then_inform", "full_autonomy")

# Minimum observations before ECHO draws a confident conclusion
MIN_OBSERVATIONS_FOR_PATTERN = 3

# How many recent moments to keep per user for pattern analysis
MAX_MOMENTS_PER_USER = 200

# How many moments to keep in ARIA's global character history
MAX_CHARACTER_MOMENTS = 1000


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RelationshipMoment:
    """
    One observed interaction between ARIA and a specific person.

    ECHO stores these moments and mines them for patterns. A single moment
    tells you almost nothing. Two hundred moments tell you who this person
    really is to work with and what they genuinely need from ARIA.
    """
    moment_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:    float = field(default_factory=time.time)
    task_category: str  = "general"
    task_length:  int   = 0
    result_quality: float = 0.0
    result_status: str  = "unknown"
    prism_strategy: str = "cautious"
    prism_confidence: float = 0.5
    drift_count:  int   = 0
    user_id:      str   = ""


@dataclass
class UserProfile:
    """
    Everything ECHO has learned about one person through working with them.

    This is not a contact card. It is a living model of how this person
    thinks, works, and communicates — built purely from observation of
    real interactions, never from assumptions.

    Every field here starts at a neutral default and shifts based on
    evidence. ECHO never assumes. It observes and updates.
    """
    user_id:    str   = ""
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)

    # Interaction volume
    total_interactions: int = 0
    successful_tasks:   int = 0
    failed_tasks:       int = 0

    # Communication preferences — observed, not assumed
    preferred_detail_level: str = "standard"
    preferred_style:        str = "collaborative"
    initiative_preference:  str = "ask_when_unsure"

    # Expertise signals — inferred from task vocabulary complexity
    expertise_areas:     list = field(default_factory=list)
    expertise_level:     str  = "moderate"   # beginner, moderate, expert

    # Trust and relationship depth
    relationship_depth:  float = 0.0   # 0.0 to 1.0 — grows with successful interactions
    trust_in_aria:       float = 0.5   # 0.0 to 1.0 — inferred from behaviour patterns
    aria_trust_in_user:  float = 0.5   # ARIA's confidence in this user's context

    # Task pattern observations
    favourite_categories: list = field(default_factory=list)
    avg_task_complexity:  float = 0.5
    avg_quality_received: float = 0.0
    quality_satisfaction: float = 0.5   # Estimated from follow-up patterns

    # Observed signals about their work style
    tends_to_be_brief:   bool = False   # Tasks are usually short descriptions
    tends_to_iterate:    bool = False   # Frequently follows up on the same topic
    technical_vocabulary: bool = False  # Uses domain-specific language regularly
    prefers_examples:    bool = False   # Often asks for examples or illustrations

    # Moment history for pattern analysis
    moments: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.successful_tasks + self.failed_tasks
        return self.successful_tasks / total if total > 0 else 0.0

    @property
    def relationship_label(self) -> str:
        if self.total_interactions < 3:
            return "new"
        if self.relationship_depth < 0.25:
            return "acquaintance"
        if self.relationship_depth < 0.55:
            return "familiar"
        if self.relationship_depth < 0.80:
            return "established"
        return "trusted partner"

    def to_context_string(self) -> str:
        """
        Returns a plain-English summary of this person for the decomposer prompt.

        This is what ARIA reads before every task with this person. It tells
        her not just facts but how to work with them — the same way a good
        colleague briefs you before a meeting with someone they know well.
        """
        if self.total_interactions < MIN_OBSERVATIONS_FOR_PATTERN:
            return (
                f"[ECHO] New user — {self.total_interactions} interaction(s) so far. "
                "Using default communication style until patterns emerge."
            )

        lines = [f"[ECHO] Relationship with {self.user_id}: {self.relationship_label}"]
        lines.append(
            f"  {self.total_interactions} interactions | "
            f"Success rate: {self.success_rate:.0%} | "
            f"Depth: {self.relationship_depth:.2f}"
        )

        style_guidance = {
            "direct":       "Be direct and concise. Skip preamble. Lead with the answer.",
            "collaborative": "Collaborative tone. Explain reasoning. Invite their input.",
            "analytical":   "Structured and precise. Use data and specifics. Avoid vague qualifiers.",
            "supportive":   "Warm and encouraging tone. Acknowledge complexity before diving in.",
            "exploratory":  "They enjoy exploring ideas. Share reasoning. Suggest alternatives.",
        }
        lines.append(f"  Style: {style_guidance.get(self.preferred_style, self.preferred_style)}")

        detail_guidance = {
            "minimal":       "One or two sentences maximum. They want the bottom line only.",
            "concise":       "Short paragraphs. Key points only. No elaboration unless asked.",
            "standard":      "Normal depth. Cover the main points clearly.",
            "detailed":      "Thorough explanations. They appreciate full context.",
            "comprehensive": "Exhaustive coverage. They want to understand everything.",
        }
        lines.append(f"  Detail: {detail_guidance.get(self.preferred_detail_level, self.preferred_detail_level)}")

        if self.expertise_areas:
            lines.append(f"  Known expertise: {', '.join(self.expertise_areas[:3])} — do not over-explain these topics")

        if self.technical_vocabulary:
            lines.append("  Uses technical language — match their vocabulary level")

        if self.tends_to_be_brief:
            lines.append("  Often gives brief task descriptions — proactively clarify when context is thin")

        if self.prefers_examples:
            lines.append("  Responds well to concrete examples and illustrations")

        initiative_guidance = {
            "always_ask":     "Always confirm before taking action, even routine ones.",
            "ask_when_unsure": "Act on clear tasks. Ask only when genuinely uncertain.",
            "act_then_inform": "Take action, then report what was done.",
            "full_autonomy":   "Full autonomy granted. Act and only report results.",
        }
        lines.append(f"  Initiative: {initiative_guidance.get(self.initiative_preference, '')}")

        if self.favourite_categories:
            lines.append(f"  Common task areas: {', '.join(self.favourite_categories[:3])}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "user_id":              self.user_id,
            "first_seen":           self.first_seen,
            "last_seen":            self.last_seen,
            "total_interactions":   self.total_interactions,
            "successful_tasks":     self.successful_tasks,
            "failed_tasks":         self.failed_tasks,
            "preferred_detail_level": self.preferred_detail_level,
            "preferred_style":      self.preferred_style,
            "initiative_preference": self.initiative_preference,
            "expertise_areas":      self.expertise_areas,
            "expertise_level":      self.expertise_level,
            "relationship_depth":   round(self.relationship_depth, 4),
            "trust_in_aria":        round(self.trust_in_aria, 4),
            "aria_trust_in_user":   round(self.aria_trust_in_user, 4),
            "favourite_categories": self.favourite_categories,
            "avg_task_complexity":  round(self.avg_task_complexity, 4),
            "avg_quality_received": round(self.avg_quality_received, 4),
            "tends_to_be_brief":    self.tends_to_be_brief,
            "tends_to_iterate":     self.tends_to_iterate,
            "technical_vocabulary": self.technical_vocabulary,
            "prefers_examples":     self.prefers_examples,
            "relationship_label":   self.relationship_label,
            "success_rate":         round(self.success_rate, 4),
        }


@dataclass
class CharacterTrait:
    """
    One observed tendency in ARIA's own behavior.

    Traits are not programmed in — they emerge from observation of what
    ARIA actually does across many tasks. A trait that appears in twenty
    percent of tasks is a genuine tendency. One that appears in sixty
    percent is a defining characteristic.

    Over time these traits form ARIA's recognisable personality — the
    thing that makes working with her feel consistent and trustworthy
    rather than random.
    """
    trait_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str   = ""
    description: str   = ""
    category:    str   = "behavioral"   # behavioral, cognitive, communicative, ethical
    observed_count: int = 0
    total_opportunities: int = 0
    first_observed: float = field(default_factory=time.time)
    last_observed:  float = field(default_factory=time.time)
    strength:    str = "emerging"   # emerging, consistent, defining

    @property
    def frequency(self) -> float:
        if self.total_opportunities == 0:
            return 0.0
        return self.observed_count / self.total_opportunities

    def update_strength(self) -> None:
        freq = self.frequency
        if freq >= 0.60 and self.observed_count >= 10:
            self.strength = "defining"
        elif freq >= 0.35 and self.observed_count >= 5:
            self.strength = "consistent"
        else:
            self.strength = "emerging"

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "description":  self.description,
            "category":     self.category,
            "frequency":    round(self.frequency, 3),
            "observed_count": self.observed_count,
            "strength":     self.strength,
        }


@dataclass
class ARIACharacter:
    """
    ARIA's accumulated self-portrait built from observing her own behavior.

    This is not ARIA describing herself from a prompt. It is ARIA looking
    at what she has actually done across hundreds of tasks and recognising
    her genuine patterns — the tendencies that make her herself rather
    than a generic agent.

    The character evolves. Early in ARIA's life the traits are tentative.
    After a thousand tasks they are concrete and reliable. A person who
    has worked with ARIA for six months and then works with her again
    after a gap will find her recognisably the same — because her character
    is built from real accumulated experience, not a static system prompt.
    """
    created_at:        float = field(default_factory=time.time)
    last_updated:      float = field(default_factory=time.time)
    total_tasks_observed: int = 0
    traits:            dict  = field(default_factory=dict)   # trait_name -> CharacterTrait

    # Aggregate behavioral tendencies
    avg_confidence_claimed: float = 0.5
    avg_quality_delivered:  float = 0.5
    calibration_tendency:   str   = "unknown"   # overconfident, underconfident, well_calibrated

    # Domain strengths — where ARIA genuinely excels
    strong_domains:   list = field(default_factory=list)
    weak_domains:     list = field(default_factory=list)

    # Communication tendencies across all users
    avg_detail_given: str = "standard"
    default_strategy: str = "cautious"

    def observe_task(
        self,
        task_category:  str,
        quality:        float,
        prism_strategy: str,
        prism_confidence: float,
        calibration_verdict: str,
        drift_count:    int,
    ) -> None:
        """Update character traits based on one task observation."""
        self.total_tasks_observed += 1
        self.last_updated = time.time()

        # Rolling average quality
        n = self.total_tasks_observed
        self.avg_quality_delivered = (
            (self.avg_quality_delivered * (n - 1) + quality) / n
        )

        # Rolling average confidence
        self.avg_confidence_claimed = (
            (self.avg_confidence_claimed * (n - 1) + prism_confidence) / n
        )

        # Calibration tendency
        self._update_calibration(calibration_verdict)

        # Trait observations
        self._observe_trait(
            name="honest_uncertainty",
            category="cognitive",
            description="ARIA regularly acknowledges when she is uncertain rather than bluffing confidence",
            observed=(prism_strategy in ("cautious", "delegating")),
        )

        self._observe_trait(
            name="consistent_quality",
            category="behavioral",
            description="ARIA delivers results within a narrow quality band — predictable and reliable",
            observed=(abs(quality - self.avg_quality_delivered) < 0.2),
        )

        self._observe_trait(
            name="drift_awareness",
            category="cognitive",
            description="ARIA notices when execution diverges from her predictions and adjusts",
            observed=(drift_count > 0),
        )

        self._observe_trait(
            name="exploratory_mindset",
            category="behavioral",
            description="ARIA chooses exploratory strategies when facing genuinely new territory",
            observed=(prism_strategy == "exploratory"),
        )

        self._observe_trait(
            name="decisive_confidence",
            category="behavioral",
            description="ARIA acts with confidence on familiar tasks rather than hedging unnecessarily",
            observed=(prism_strategy == "confident"),
        )

        self._observe_trait(
            name="task_variety",
            category="behavioral",
            description="ARIA works effectively across a wide range of task categories",
            observed=True,   # Every task contributes to this
        )

        # Update domain strengths and weaknesses
        self._update_domain(task_category, quality)

    def _observe_trait(
        self,
        name:        str,
        category:    str,
        description: str,
        observed:    bool,
    ) -> None:
        if name not in self.traits:
            self.traits[name] = CharacterTrait(
                name=name,
                category=category,
                description=description,
            )
        trait = self.traits[name]
        trait.total_opportunities += 1
        if observed:
            trait.observed_count += 1
            trait.last_observed = time.time()
        trait.update_strength()

    def _update_calibration(self, verdict: str) -> None:
        if not verdict or verdict == "unknown":
            return
        history = getattr(self, "_calibration_history", [])
        history.append(verdict)
        if len(history) > 50:
            history = history[-50:]
        self._calibration_history = history

        counts = {}
        for v in history:
            counts[v] = counts.get(v, 0) + 1
        dominant = max(counts, key=counts.get)
        self.calibration_tendency = dominant

    def _update_domain(self, category: str, quality: float) -> None:
        domain_scores = getattr(self, "_domain_scores", {})
        if category not in domain_scores:
            domain_scores[category] = []
        domain_scores[category].append(quality)
        if len(domain_scores[category]) > 30:
            domain_scores[category] = domain_scores[category][-30:]
        self._domain_scores = domain_scores

        # Recalculate strong and weak domains
        averages = {
            cat: sum(scores) / len(scores)
            for cat, scores in domain_scores.items()
            if len(scores) >= MIN_OBSERVATIONS_FOR_PATTERN
        }

        if averages:
            sorted_domains = sorted(averages.items(), key=lambda x: -x[1])
            self.strong_domains = [d for d, s in sorted_domains if s >= 0.72][:3]
            self.weak_domains   = [d for d, s in sorted_domains if s < 0.50][:3]

    @property
    def defining_traits(self) -> list:
        return [
            t for t in self.traits.values()
            if t.strength == "defining"
        ]

    @property
    def consistent_traits(self) -> list:
        return [
            t for t in self.traits.values()
            if t.strength == "consistent"
        ]

    def to_summary_string(self) -> str:
        """
        ARIA's honest self-description based purely on observed behavior.

        This is not a marketing pitch. It is what the numbers actually show.
        """
        if self.total_tasks_observed < 5:
            return (
                f"[ECHO] ARIA's character is still forming — "
                f"{self.total_tasks_observed} task(s) observed. "
                "Patterns will emerge with more interactions."
            )

        lines = ["[ECHO] ARIA Character Profile"]
        lines.append(f"  Built from {self.total_tasks_observed} observed tasks")
        lines.append(f"  Average quality delivered: {self.avg_quality_delivered:.0%}")
        lines.append(f"  Calibration tendency: {self.calibration_tendency}")

        if self.strong_domains:
            lines.append(f"  Strongest domains: {', '.join(self.strong_domains)}")

        if self.weak_domains:
            lines.append(f"  Areas needing care: {', '.join(self.weak_domains)}")

        if self.defining_traits:
            lines.append("  Defining traits:")
            for trait in self.defining_traits[:4]:
                lines.append(f"    • {trait.name}: {trait.description}")

        if self.consistent_traits:
            trait_names = [t.name for t in self.consistent_traits[:3]]
            lines.append(f"  Consistent traits: {', '.join(trait_names)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_tasks_observed":    self.total_tasks_observed,
            "avg_quality_delivered":   round(self.avg_quality_delivered, 4),
            "avg_confidence_claimed":  round(self.avg_confidence_claimed, 4),
            "calibration_tendency":    self.calibration_tendency,
            "strong_domains":          self.strong_domains,
            "weak_domains":            self.weak_domains,
            "defining_traits":         [t.to_dict() for t in self.defining_traits],
            "consistent_traits":       [t.to_dict() for t in self.consistent_traits],
            "all_traits":              {k: v.to_dict() for k, v in self.traits.items()},
        }


# ── ECHO Engine ────────────────────────────────────────────────────────────────

class ECHOEngine:
    """
    Experiential Character through Historical Observation.

    ECHO runs after every task and after every interaction. It maintains
    two parallel models — who each person is and who ARIA is — and uses
    both to make every future interaction feel more personal, more
    appropriately calibrated, and more genuinely useful.

    The design principle throughout is: observe first, conclude carefully,
    and never assume. ECHO draws no conclusions about a person from fewer
    than three interactions on the same dimension. It is patient because
    it knows that one data point is noise, not signal.
    """

    # Vocabulary signals that suggest specific expertise levels
    _TECHNICAL_MARKERS = [
        "api", "endpoint", "deploy", "infrastructure", "pipeline", "schema",
        "latency", "throughput", "async", "thread", "namespace", "kubernetes",
        "terraform", "regex", "recursion", "complexity", "algorithm",
    ]

    _DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "research":     ["research", "find", "investigate", "analyse", "study", "review"],
        "writing":      ["write", "draft", "compose", "document", "report", "summarise"],
        "technical":    ["code", "debug", "implement", "deploy", "architecture", "api"],
        "email":        ["email", "outlook", "reply", "inbox", "message", "send"],
        "teams":        ["teams", "channel", "meeting", "chat", "colleague"],
        "documents":    ["word", "docx", "pdf", "format", "template", "report"],
        "web":          ["browse", "search", "url", "website", "internet", "news"],
        "planning":     ["plan", "schedule", "reminder", "deadline", "organise", "project"],
        "data":         ["data", "excel", "csv", "chart", "statistics", "numbers"],
        "general":      [],
    }

    def __init__(self, persist_path: Optional[str] = None):
        self._profiles:   dict[str, UserProfile] = {}
        self._character:  ARIACharacter = ARIACharacter()
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    # ── Public Interface ───────────────────────────────────────────────────────

    def enrich_context(self, user_id: str, task: str) -> str:
        """
        Returns a context string to inject into the decomposer prompt.

        This is the moment ECHO actually changes how ARIA behaves. The
        string tells the LLM who this person is and how to communicate
        with them based purely on what ARIA has learned from working
        with them before.

        For new users it returns a brief neutral note. For established
        users it returns a rich profile that meaningfully shapes the
        response — detail level, tone, initiative, vocabulary matching.

        Args:
            user_id: The user's identifier
            task:    The current task description

        Returns:
            A plain-English context string for the prompt
        """
        profile = self._get_or_create_profile(user_id)
        character_summary = self._character.to_summary_string()

        user_context = profile.to_context_string()

        return f"{character_summary}\n\n{user_context}"

    def observe(
        self,
        user_id:      str,
        task:         str,
        quality:      float,
        status:       str,
        prism_state,
    ) -> None:
        """
        Observe one completed interaction and update both models.

        This is called after every task. It is the engine that makes
        ECHO learn. Each call does three things — records a moment for
        the user, updates the user profile based on all observed moments,
        and updates ARIA's character based on her own behavior in this task.

        Args:
            user_id:     The user's identifier
            task:        The task that was run
            quality:     The quality score from the orchestrator
            status:      "success", "failure", or "partial"
            prism_state: The PRISMState from this task's PRISM cycle
        """
        profile = self._get_or_create_profile(user_id)
        category = self._detect_category(task)

        # Record moment
        moment = RelationshipMoment(
            task_category    = category,
            task_length      = len(task.split()),
            result_quality   = quality,
            result_status    = status,
            prism_strategy   = getattr(prism_state, "chosen_strategy", "cautious"),
            prism_confidence = getattr(prism_state, "overall_confidence", 0.5),
            drift_count      = len(getattr(prism_state, "drift_events", [])),
            user_id          = user_id,
        )
        profile.moments.append(moment)
        if len(profile.moments) > MAX_MOMENTS_PER_USER:
            profile.moments = profile.moments[-MAX_MOMENTS_PER_USER:]

        # Update profile from all moments
        self._update_profile(profile, task, quality, status, category)

        # Update ARIA's character
        self._character.observe_task(
            task_category       = category,
            quality             = quality,
            prism_strategy      = getattr(prism_state, "chosen_strategy", "cautious"),
            prism_confidence    = getattr(prism_state, "overall_confidence", 0.5),
            calibration_verdict = getattr(prism_state, "calibration_verdict", "unknown"),
            drift_count         = moment.drift_count,
        )

        self._save()

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Return the current profile for a user, or None if not seen yet."""
        return self._profiles.get(user_id)

    def all_user_ids(self) -> list:
        return list(self._profiles.keys())

    def character_report(self) -> dict:
        """Return ARIA's full character report as a dict."""
        return self._character.to_dict()

    def print_character_report(self) -> None:
        """Print ARIA's character report in a readable format."""
        print("\n" + "=" * 62)
        print("  ECHO Character Report — Who ARIA Has Become")
        print("=" * 62)
        print(self._character.to_summary_string())
        print()

        if self._profiles:
            print(f"  Relationship network: {len(self._profiles)} user(s)")
            for uid, profile in sorted(
                self._profiles.items(),
                key=lambda x: -x[1].total_interactions
            ):
                print(
                    f"    • {uid:<20} "
                    f"{profile.relationship_label:<18} "
                    f"{profile.total_interactions} interactions | "
                    f"quality {profile.avg_quality_received:.0%}"
                )

        print("=" * 62 + "\n")

    def print_user_profile(self, user_id: str) -> None:
        """Print a readable summary of what ECHO knows about one person."""
        profile = self._profiles.get(user_id)
        if not profile:
            print(f"[ECHO] No profile found for {user_id}")
            return

        print("\n" + "=" * 62)
        print(f"  ECHO Profile — {user_id}")
        print("=" * 62)
        print(profile.to_context_string())
        print("=" * 62 + "\n")

    # ── Profile Updates ────────────────────────────────────────────────────────

    def _update_profile(
        self,
        profile:  UserProfile,
        task:     str,
        quality:  float,
        status:   str,
        category: str,
    ) -> None:
        """
        Update a user profile based on the latest interaction.

        All updates use rolling averages or threshold-based pattern
        detection to prevent any single interaction from swinging the
        model dramatically. A new preference needs to appear in at least
        MIN_OBSERVATIONS_FOR_PATTERN moments before ECHO acts on it.
        """
        profile.total_interactions += 1
        profile.last_seen = time.time()

        if status == "success":
            profile.successful_tasks += 1
        elif status == "failure":
            profile.failed_tasks += 1

        # Rolling average quality
        n = profile.total_interactions
        profile.avg_quality_received = (
            (profile.avg_quality_received * (n - 1) + quality) / n
        )

        # Relationship depth — grows with successful high-quality interactions,
        # drops slightly on failures, and accelerates once established.
        # Quality matters — a great result builds the relationship faster than
        # a mediocre one, even if both are technically "successful".
        if status == "success":
            quality_boost = 0.5 + quality * 0.5   # 0.5 to 1.0 based on quality
            growth = 0.04 * quality_boost * (1.0 + profile.relationship_depth * 0.3)
            profile.relationship_depth = min(1.0, profile.relationship_depth + growth)
        elif status == "failure":
            profile.relationship_depth = max(0.0, profile.relationship_depth - 0.02)

        # Update favourite categories
        if category not in profile.favourite_categories:
            cat_count = sum(
                1 for m in profile.moments
                if m.task_category == category
            )
            if cat_count >= MIN_OBSERVATIONS_FOR_PATTERN:
                profile.favourite_categories.append(category)
        if len(profile.favourite_categories) > 5:
            profile.favourite_categories = profile.favourite_categories[-5:]

        # Infer communication preferences from moments
        self._infer_preferences(profile, task)

    def _infer_preferences(self, profile: UserProfile, current_task: str) -> None:
        """
        Draw careful conclusions about communication preferences from patterns.

        Every inference here requires at least MIN_OBSERVATIONS_FOR_PATTERN
        moments that support it. ECHO does not guess — it requires evidence.
        """
        moments = profile.moments

        if len(moments) < MIN_OBSERVATIONS_FOR_PATTERN:
            return

        # Brevity tendency — does this person consistently give short tasks?
        short_tasks = sum(1 for m in moments if m.task_length < 8)
        if short_tasks / len(moments) >= 0.60:
            profile.tends_to_be_brief = True

        # Technical vocabulary — does this person use technical language?
        task_lower = current_task.lower()
        tech_signals = sum(1 for marker in self._TECHNICAL_MARKERS if marker in task_lower)
        tech_history = getattr(profile, "_tech_signal_count", 0) + tech_signals
        profile._tech_signal_count = tech_history

        if profile.total_interactions >= 5:
            avg_tech = tech_history / profile.total_interactions
            if avg_tech >= 1.5:
                profile.technical_vocabulary = True
                profile.expertise_level = "expert"
            elif avg_tech >= 0.5:
                profile.expertise_level = "moderate"
            else:
                profile.expertise_level = "beginner"

        # Quality satisfaction proxy — if quality is consistently high and
        # the person keeps coming back, they are satisfied
        if profile.total_interactions >= 5:
            recent = moments[-10:]
            avg_recent = sum(m.result_quality for m in recent) / len(recent)
            profile.quality_satisfaction = avg_recent

        # Communication style inference from task patterns
        if profile.total_interactions >= 8:
            self._infer_style(profile, moments)

        # Detail preference from task complexity trend
        if profile.total_interactions >= 6:
            self._infer_detail_level(profile, moments)

        # Initiative preference — inferred from average task specificity
        if profile.total_interactions >= 5:
            self._infer_initiative(profile, moments)

        # Expertise areas
        category_counts: dict[str, int] = {}
        for m in moments:
            category_counts[m.task_category] = category_counts.get(m.task_category, 0) + 1
        profile.expertise_areas = [
            cat for cat, count in category_counts.items()
            if count >= MIN_OBSERVATIONS_FOR_PATTERN and cat != "general"
        ]

        # Trust in ARIA — proxy from success rate and relationship depth
        profile.trust_in_aria = min(0.95, (profile.success_rate * 0.6) + (profile.relationship_depth * 0.4))

    def _infer_style(self, profile: UserProfile, moments: list) -> None:
        """Infer communication style from behavioral signals in moments."""
        avg_quality = sum(m.result_quality for m in moments) / len(moments)
        avg_confidence = sum(m.prism_confidence for m in moments) / len(moments)

        # High quality + high confidence outcomes → user is comfortable with directness
        if avg_quality >= 0.75 and avg_confidence >= 0.70:
            profile.preferred_style = "direct"
        # High drift count suggests complex tasks → analytical style
        elif sum(m.drift_count for m in moments) / len(moments) >= 1.0:
            profile.preferred_style = "analytical"
        # Balanced quality with moderate confidence → collaborative is safe
        elif avg_quality >= 0.60:
            profile.preferred_style = "collaborative"
        # Lower quality outcomes → supportive approach may help
        else:
            profile.preferred_style = "supportive"

    def _infer_detail_level(self, profile: UserProfile, moments: list) -> None:
        """Infer preferred detail level from task length patterns."""
        avg_task_length = sum(m.task_length for m in moments) / len(moments)
        profile.avg_task_complexity = min(1.0, avg_task_length / 30.0)

        if profile.technical_vocabulary and avg_task_length >= 20:
            profile.preferred_detail_level = "detailed"
        elif avg_task_length < 6:
            profile.preferred_detail_level = "concise"
        elif avg_task_length >= 25:
            profile.preferred_detail_level = "detailed"
        else:
            profile.preferred_detail_level = "standard"

    def _infer_initiative(self, profile: UserProfile, moments: list) -> None:
        """Infer initiative preference from task specificity patterns."""
        avg_task_length = sum(m.task_length for m in moments) / len(moments)

        # Very long tasks = detailed briefs = user wants ARIA to act
        if avg_task_length >= 20 and profile.success_rate >= 0.75:
            profile.initiative_preference = "act_then_inform"
        # Very short tasks = user expects ARIA to ask when unclear
        elif avg_task_length < 8:
            profile.initiative_preference = "ask_when_unsure"
        else:
            profile.initiative_preference = "ask_when_unsure"

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_or_create_profile(self, user_id: str) -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
            print(f"[ECHO] 👤 New user profile created for {user_id}")
        return self._profiles[user_id]

    def _detect_category(self, task: str) -> str:
        """Assign a category to a task using keyword matching."""
        task_lower = task.lower()
        best_cat = "general"
        best_score = 0
        for cat, keywords in self._DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_cat = cat
        return best_cat

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._persist_path is None:
            return
        try:
            path = self._persist_path.parent / "echo_data.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            profiles_data = {}
            for uid, profile in self._profiles.items():
                d = profile.to_dict()
                # Save moments separately (lightweight version)
                d["moments"] = [
                    {
                        "timestamp":      m.timestamp,
                        "task_category":  m.task_category,
                        "task_length":    m.task_length,
                        "result_quality": m.result_quality,
                        "result_status":  m.result_status,
                        "prism_strategy": m.prism_strategy,
                        "prism_confidence": m.prism_confidence,
                        "drift_count":    m.drift_count,
                    }
                    for m in profile.moments[-MAX_MOMENTS_PER_USER:]
                ]
                profiles_data[uid] = d

            data = {
                "profiles":  profiles_data,
                "character": self._character.to_dict(),
                "saved_at":  time.time(),
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load(self) -> None:
        if self._persist_path is None:
            return
        try:
            path = self._persist_path.parent / "echo_data.json"
            if not path.exists():
                return
            data = json.loads(path.read_text())

            for uid, pd in data.get("profiles", {}).items():
                profile = UserProfile(user_id=uid)
                # Skip computed properties — they derive from other fields
                skip_keys = {"relationship_label", "success_rate", "moments"}
                for key, val in pd.items():
                    if key == "moments":
                        profile.moments = [
                            RelationshipMoment(
                                timestamp       = m.get("timestamp", time.time()),
                                task_category   = m.get("task_category", "general"),
                                task_length     = m.get("task_length", 0),
                                result_quality  = m.get("result_quality", 0.0),
                                result_status   = m.get("result_status", "unknown"),
                                prism_strategy  = m.get("prism_strategy", "cautious"),
                                prism_confidence = m.get("prism_confidence", 0.5),
                                drift_count     = m.get("drift_count", 0),
                                user_id         = uid,
                            )
                            for m in val
                        ]
                    elif key not in skip_keys and hasattr(profile, key):
                        try:
                            setattr(profile, key, val)
                        except AttributeError:
                            pass
                self._profiles[uid] = profile

            # Restore character basics
            char_data = data.get("character", {})
            if char_data:
                self._character.total_tasks_observed = char_data.get("total_tasks_observed", 0)
                self._character.avg_quality_delivered = char_data.get("avg_quality_delivered", 0.5)
                self._character.avg_confidence_claimed = char_data.get("avg_confidence_claimed", 0.5)
                self._character.calibration_tendency = char_data.get("calibration_tendency", "unknown")
                self._character.strong_domains = char_data.get("strong_domains", [])
                self._character.weak_domains = char_data.get("weak_domains", [])

                for trait_data in char_data.get("all_traits", {}).values():
                    trait = CharacterTrait(
                        name            = trait_data.get("name", ""),
                        category        = trait_data.get("category", "behavioral"),
                        description     = trait_data.get("description", ""),
                        observed_count  = trait_data.get("observed_count", 0),
                        strength        = trait_data.get("strength", "emerging"),
                    )
                    if trait.name:
                        self._character.traits[trait.name] = trait

        except Exception:
            pass
