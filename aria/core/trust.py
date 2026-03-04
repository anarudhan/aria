"""
ARIA Trust Ledger
-----------------
Tracks agent performance history using Bayesian trust scoring.

The core insight here: you should never fully trust or fully distrust an agent
based on one or two outcomes. Bayesian updating means a new agent starts with a
moderate prior, and each outcome moves the score incrementally toward reality.

An agent with 50 successes and 5 failures is trustworthy.
An agent with 1 success and 0 failures is unknown — not trustworthy yet.
An agent with 2 successes and 8 failures should be suspended.

The ledger is also category-aware. An agent might be excellent at "research"
tasks but poor at "code generation". The trust system tracks this separately
so an agent isn't penalized for tasks outside its sweet spot.

Task categories are auto-detected by tokenizing the task description and
matching it to predefined category keywords. No LLM call needed.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from .models import TrustRecord
from .registry import tokenize


# ─── Category Detection ───────────────────────────────────────────────────────

# Maps keywords to high-level task categories.
# This lightweight approach avoids an LLM call just to categorize a task.
CATEGORY_KEYWORDS = {
    "research":       ["research", "find", "search", "look", "investigate", "analyze", "study", "explore"],
    "writing":        ["write", "draft", "compose", "create", "document", "report", "summarize", "explain"],
    "coding":         ["code", "program", "script", "implement", "debug", "function", "class", "api", "test"],
    "email":          ["email", "outlook", "mail", "send", "reply", "inbox", "message", "compose"],
    "teams":          ["teams", "channel", "meeting", "chat", "microsoft", "colleague", "slack"],
    "data_analysis":  ["data", "csv", "excel", "chart", "graph", "statistics", "numbers", "calculate"],
    "web_browsing":   ["browse", "url", "website", "scrape", "fetch", "download", "web", "page"],
    "document":       ["word", "docx", "pdf", "document", "file", "format", "template", "presentation"],
    "planning":       ["plan", "schedule", "calendar", "reminder", "deadline", "project", "task", "organise"],
    "general":        [],   # fallback
}


def detect_category(task_description: str) -> str:
    """
    Detect the task category from its description using keyword matching.
    Returns the best-matching category name, or 'general' if no match.
    """
    tokens = set(tokenize(task_description))
    best_cat = "general"
    best_score = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in tokens)
        if score > best_score:
            best_score = score
            best_cat = category

    return best_cat


# ─── Trust Ledger ─────────────────────────────────────────────────────────────

class TrustLedger:
    """
    Maintains the trust history for all agents across all task categories.
    
    The ledger is the long-term memory of ARIA's routing intelligence.
    Every completed task leaves a record here, and future routing decisions
    use these records to weight agent selection.
    
    Persistence: optionally writes to a JSON file on disk so trust history
    survives across ARIA restarts. Without persistence, every restart resets
    trust to the prior (0.6 for all agents).
    
    Usage:
        ledger = TrustLedger(persist_path="./aria_trust.json")
        ledger.record_success("agent_web", "research the latest AI news", quality=0.9, seconds=12.0)
        score = ledger.get_trust_score("agent_web", "find information about X")
    """

    def __init__(
        self,
        persist_path: Optional[str] = None,
        suspension_threshold: float = 0.2,
        min_executions_for_suspension: int = 5,
    ):
        self._records: dict[str, dict[str, TrustRecord]] = defaultdict(dict)
        # Structure: {agent_id: {category: TrustRecord}}
        self._persist_path = Path(persist_path) if persist_path else None
        self._suspension_threshold = suspension_threshold
        self._min_executions = min_executions_for_suspension
        self._suspended_agents: set[str] = set()

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Recording Outcomes ─────────────────────────────────────────────────────

    def record_success(
        self,
        agent_id: str,
        task_description: str,
        quality: float,       # 0.0 to 1.0
        seconds: float,
        tokens: int = 0,
    ):
        record = self._get_or_create(agent_id, task_description)
        record.success_count += 1
        record.total_executions += 1
        # Rolling average for quality and time
        record.average_quality = self._rolling_avg(record.average_quality, quality, record.total_executions)
        record.average_seconds = self._rolling_avg(record.average_seconds, seconds, record.total_executions)
        record.last_updated = time.time()
        self._maybe_persist()

    def record_failure(self, agent_id: str, task_description: str, seconds: float):
        record = self._get_or_create(agent_id, task_description)
        record.failure_count += 1
        record.total_executions += 1
        record.last_updated = time.time()
        self._check_suspension(agent_id, record)
        self._maybe_persist()

    def record_timeout(self, agent_id: str, task_description: str):
        record = self._get_or_create(agent_id, task_description)
        record.timeout_count += 1
        record.total_executions += 1
        record.last_updated = time.time()
        self._check_suspension(agent_id, record)
        self._maybe_persist()

    # ── Trust Queries ──────────────────────────────────────────────────────────

    def get_trust_score(self, agent_id: str, task_description: str) -> float:
        """
        Get the trust score for an agent on a given task type.
        
        If the agent has no history for this specific category, falls back
        to its overall score, then to the prior (0.6).
        """
        if agent_id in self._suspended_agents:
            return 0.0

        category = detect_category(task_description)
        agent_records = self._records.get(agent_id, {})

        # Category-specific score
        if category in agent_records:
            return agent_records[category].trust_score

        # Overall score if no category-specific history
        if agent_records:
            scores = [r.trust_score for r in agent_records.values()]
            return sum(scores) / len(scores)

        # Prior for unknown agents
        return 0.6

    def is_suspended(self, agent_id: str) -> bool:
        return agent_id in self._suspended_agents

    def get_agent_summary(self, agent_id: str) -> dict:
        """Return a human-readable summary of an agent's trust history."""
        agent_records = self._records.get(agent_id, {})
        if not agent_records:
            return {"agent_id": agent_id, "status": "no_history", "trust": 0.6}

        overall_trust = self.get_trust_score(agent_id, "general")
        categories = {
            cat: {
                "trust": record.trust_score,
                "successes": record.success_count,
                "failures": record.failure_count,
                "avg_quality": round(record.average_quality, 2),
                "avg_seconds": round(record.average_seconds, 1),
            }
            for cat, record in agent_records.items()
        }

        return {
            "agent_id":     agent_id,
            "overall_trust": round(overall_trust, 3),
            "suspended":    agent_id in self._suspended_agents,
            "categories":   categories,
        }

    def get_all_summaries(self) -> list[dict]:
        all_ids = set(self._records.keys()) | self._suspended_agents
        return [self.get_agent_summary(aid) for aid in all_ids]

    def reinstate_agent(self, agent_id: str):
        """Manually lift suspension — e.g. after the agent is upgraded."""
        self._suspended_agents.discard(agent_id)
        print(f"[Trust Ledger] Agent '{agent_id}' reinstated")

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _get_or_create(self, agent_id: str, task_description: str) -> TrustRecord:
        category = detect_category(task_description)
        if category not in self._records[agent_id]:
            self._records[agent_id][category] = TrustRecord(
                agent_id=agent_id,
                task_category=category
            )
        return self._records[agent_id][category]

    @staticmethod
    def _rolling_avg(current: float, new_value: float, n: int) -> float:
        """Incremental rolling average — avoids storing all historical values."""
        return current + (new_value - current) / n

    def _check_suspension(self, agent_id: str, record: TrustRecord):
        if (record.total_executions >= self._min_executions
                and record.trust_score < self._suspension_threshold):
            self._suspended_agents.add(agent_id)
            print(f"[Trust Ledger] ⚠️  Agent '{agent_id}' auto-suspended "
                  f"(trust={record.trust_score:.2f} on '{record.task_category}')")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _maybe_persist(self):
        if self._persist_path:
            self._save()

    def _save(self):
        data = {
            "records": {
                agent_id: {
                    cat: {
                        "success_count":    rec.success_count,
                        "failure_count":    rec.failure_count,
                        "timeout_count":    rec.timeout_count,
                        "average_quality":  rec.average_quality,
                        "average_seconds":  rec.average_seconds,
                        "total_executions": rec.total_executions,
                    }
                    for cat, rec in cats.items()
                }
                for agent_id, cats in self._records.items()
            },
            "suspended": list(self._suspended_agents),
        }
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        try:
            data = json.loads(self._persist_path.read_text())
            for agent_id, cats in data.get("records", {}).items():
                for cat, rec_data in cats.items():
                    rec = TrustRecord(agent_id=agent_id, task_category=cat)
                    rec.success_count    = rec_data["success_count"]
                    rec.failure_count    = rec_data["failure_count"]
                    rec.timeout_count    = rec_data["timeout_count"]
                    rec.average_quality  = rec_data["average_quality"]
                    rec.average_seconds  = rec_data["average_seconds"]
                    rec.total_executions = rec_data["total_executions"]
                    self._records[agent_id][cat] = rec
            self._suspended_agents = set(data.get("suspended", []))
            print(f"[Trust Ledger] Loaded trust history from {self._persist_path}")
        except Exception as e:
            print(f"[Trust Ledger] Could not load trust history: {e}")
