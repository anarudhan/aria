"""
ARIA Temporal Memory Rings
--------------------------
Four-ring memory architecture inspired by CPU cache hierarchy.

Most agent frameworks give you one flat memory store — either everything
is relevant or nothing is. ARIA organizes memory into four time-based rings
that work like CPU L1/L2/L3/L4 cache:

    FLASH  → Current session only. Microsecond access. No persistence.
              Holds: current task context, working variables, intermediate results.

    HOT    → Last hour of activity. Very fast access.
              Holds: recent conversation, last few task results, current user prefs.

    WARM   → Last 24 hours. Normal access speed.
              Holds: today's emails, recent decisions, ongoing project context.

    COLD   → Long-term archive. Slower access, searched with lower priority.
              Holds: historical decisions, past documents, learned user patterns.

Memory moves between rings automatically:
    - Frequently accessed memories get promoted (COLD → WARM → HOT)
    - Old, untouched memories get demoted (HOT → WARM → COLD → evicted)
    - Every search checks FLASH and HOT first, then WARM, then COLD (if needed)

Search uses TF-IDF cosine similarity — the same engine as the registry.
This means no embedding API calls for memory retrieval, and sub-millisecond
search for typical memory sizes.
"""

from __future__ import annotations

import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any

from .models import MemoryEntry, MemoryRing
from .registry import tokenize, build_term_frequency, cosine_similarity


# ─── Ring Configuration ────────────────────────────────────────────────────────

RING_MAX_AGE_HOURS = {
    MemoryRing.FLASH: 0.5,     # 30 minutes — cleared aggressively
    MemoryRing.HOT:   2.0,     # 2 hours
    MemoryRing.WARM:  24.0,    # 1 day
    MemoryRing.COLD:  720.0,   # 30 days
}

RING_DEFAULT_MAX_ENTRIES = {
    MemoryRing.FLASH: 200,
    MemoryRing.HOT:   1000,
    MemoryRing.WARM:  5000,
    MemoryRing.COLD:  20000,
}

PROMOTION_THRESHOLD_ACCESS = 3   # Access count that triggers promotion
DEMOTION_IDLE_HOURS = {
    MemoryRing.FLASH: 0.5,
    MemoryRing.HOT:   2.0,
    MemoryRing.WARM:  24.0,
}


# ─── Temporal Memory ──────────────────────────────────────────────────────────

class TemporalMemory:
    """
    ARIA's four-ring temporal memory system.
    
    Example usage:
        memory = TemporalMemory()
        
        # Store something in FLASH (current session)
        memory.store("User asked about Q3 budget", tags=["budget", "Q3"])
        
        # Recall relevant memories for a query
        results = memory.recall("budget discussion", rings=[FLASH, HOT], top_k=5)
        
        # Explicitly promote something to HOT for the next hour
        memory.promote(entry_id, MemoryRing.HOT)
    """

    def __init__(
        self,
        max_entries: Optional[dict[MemoryRing, int]] = None,
        persist_path: Optional[str] = None,
    ):
        self._rings: dict[MemoryRing, dict[str, MemoryEntry]] = {
            ring: {} for ring in MemoryRing
        }
        self._max_entries = max_entries or dict(RING_DEFAULT_MAX_ENTRIES)
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Storage ────────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        ring: MemoryRing = MemoryRing.FLASH,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        genome_id: Optional[str] = None,
    ) -> str:
        """Store content in the specified ring. Returns the entry_id."""
        entry = MemoryEntry(
            content    = content,
            ring       = ring,
            tags       = tags or [],
            metadata   = metadata or {},
            user_id    = user_id,
            agent_id   = agent_id,
            genome_id  = genome_id,
            term_vector = build_term_frequency([content] + (tags or [])),
        )

        ring_store = self._rings[ring]

        # Evict oldest entry if at capacity
        if len(ring_store) >= self._max_entries[ring]:
            self._evict_oldest(ring)

        ring_store[entry.entry_id] = entry
        return entry.entry_id

    def store_result(
        self,
        genome_id: str,
        node_id: str,
        result: Any,
        agent_id: str,
        ring: MemoryRing = MemoryRing.HOT,
    ) -> str:
        """Convenience method for storing task node results."""
        content = f"Task result for node {node_id}: {str(result)[:500]}"
        return self.store(
            content   = content,
            ring      = ring,
            tags      = [f"genome:{genome_id}", f"node:{node_id}", f"agent:{agent_id}"],
            metadata  = {"genome_id": genome_id, "node_id": node_id, "result": str(result)[:2000]},
            agent_id  = agent_id,
            genome_id = genome_id,
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        rings: Optional[list[MemoryRing]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        genome_id: Optional[str] = None,
        min_similarity: float = 0.05,
    ) -> list[MemoryEntry]:
        """
        Retrieve the most semantically relevant memories for a query.
        
        Search order: FLASH → HOT → WARM → COLD (stops when enough results found).
        This mimics cache hit behaviour — if the answer is in FLASH, we never
        even touch WARM or COLD, saving significant compute.
        
        Each retrieved entry is automatically touched (access count incremented).
        Frequently touched entries get promoted to faster rings.
        """
        search_rings = rings or [MemoryRing.FLASH, MemoryRing.HOT, MemoryRing.WARM, MemoryRing.COLD]
        query_vec = build_term_frequency([query])

        scored: list[tuple[float, MemoryEntry]] = []

        for ring in search_rings:
            for entry in self._rings[ring].values():
                # Filters
                if user_id  and entry.user_id  and entry.user_id  != user_id:  continue
                if agent_id and entry.agent_id and entry.agent_id != agent_id: continue
                if genome_id and entry.genome_id and entry.genome_id != genome_id: continue

                sim = cosine_similarity(query_vec, entry.term_vector)
                if sim >= min_similarity:
                    scored.append((sim, entry))

            # Early exit if we have enough results from fast rings
            if len(scored) >= top_k * 2:
                break

        # Sort by similarity, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]

        # Touch retrieved entries — may trigger promotion
        for entry in results:
            entry.touch()
            self._maybe_promote(entry)

        return results

    def recall_by_tag(
        self,
        tag: str,
        rings: Optional[list[MemoryRing]] = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve entries by exact tag match."""
        search_rings = rings or list(MemoryRing)
        results = []
        for ring in search_rings:
            for entry in self._rings[ring].values():
                if tag in entry.tags:
                    results.append(entry)
        return results[:top_k]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific entry by ID across all rings."""
        for ring_store in self._rings.values():
            if entry_id in ring_store:
                entry = ring_store[entry_id]
                entry.touch()
                return entry
        return None

    # ── Promotion and Demotion ─────────────────────────────────────────────────

    def promote(self, entry_id: str, target_ring: MemoryRing) -> bool:
        """Manually promote an entry to a faster ring."""
        for ring, ring_store in self._rings.items():
            if entry_id in ring_store:
                entry = ring_store.pop(entry_id)
                entry.ring = target_ring
                self._rings[target_ring][entry_id] = entry
                return True
        return False

    def _maybe_promote(self, entry: MemoryEntry):
        """Auto-promote frequently accessed entries."""
        if entry.access_count < PROMOTION_THRESHOLD_ACCESS:
            return

        ring_order = [MemoryRing.COLD, MemoryRing.WARM, MemoryRing.HOT, MemoryRing.FLASH]
        current_idx = ring_order.index(entry.ring) if entry.ring in ring_order else 0

        if current_idx < len(ring_order) - 1:
            target = ring_order[current_idx + 1]
            self.promote(entry.entry_id, target)

    def maintain(self):
        """
        Run ring maintenance — demote old entries and evict expired ones.
        Call this periodically (every few minutes in production).
        """
        now = time.time()
        demoted = 0
        evicted = 0

        # Check entries in HOT and WARM for demotion
        for ring, idle_hours in DEMOTION_IDLE_HOURS.items():
            if ring not in self._rings:
                continue

            to_demote = []
            for entry_id, entry in self._rings[ring].items():
                idle_time = (now - entry.last_accessed) / 3600
                if idle_time > idle_hours:
                    to_demote.append((entry_id, entry))

            for entry_id, entry in to_demote:
                del self._rings[ring][entry_id]
                ring_order = [MemoryRing.FLASH, MemoryRing.HOT, MemoryRing.WARM, MemoryRing.COLD]
                current_idx = ring_order.index(ring)
                if current_idx < len(ring_order) - 1:
                    target = ring_order[current_idx + 1]
                    entry.ring = target
                    self._rings[target][entry_id] = entry
                    demoted += 1

        # Evict entries older than their ring's max age
        for ring, max_age_hours in RING_MAX_AGE_HOURS.items():
            to_evict = [
                eid for eid, entry in self._rings[ring].items()
                if entry.age_hours > max_age_hours
            ]
            for eid in to_evict:
                del self._rings[ring][eid]
                evicted += 1

        if demoted or evicted:
            print(f"[Memory] Maintenance: {demoted} entries demoted, {evicted} evicted")

    def _evict_oldest(self, ring: MemoryRing):
        """Remove the oldest accessed entry in a ring to make space."""
        ring_store = self._rings[ring]
        if not ring_store:
            return
        oldest_id = min(ring_store, key=lambda eid: ring_store[eid].last_accessed)
        del ring_store[oldest_id]

    # ── Inspection ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            ring.value: {
                "count": len(self._rings[ring]),
                "max":   self._max_entries[ring],
            }
            for ring in MemoryRing
        }

    def clear_ring(self, ring: MemoryRing):
        self._rings[ring].clear()

    def clear_all(self):
        for ring in MemoryRing:
            self._rings[ring].clear()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self):
        if not self._persist_path:
            return
        data = {}
        for ring in [MemoryRing.HOT, MemoryRing.WARM, MemoryRing.COLD]:
            data[ring.value] = [
                {
                    "entry_id":    e.entry_id,
                    "content":     e.content,
                    "tags":        e.tags,
                    "metadata":    e.metadata,
                    "ring":        e.ring.value,
                    "created_at":  e.created_at,
                    "last_accessed": e.last_accessed,
                    "access_count": e.access_count,
                    "user_id":     e.user_id,
                    "agent_id":    e.agent_id,
                    "genome_id":   e.genome_id,
                    "term_vector": e.term_vector,
                }
                for e in self._rings[ring].values()
            ]
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        try:
            data = json.loads(self._persist_path.read_text())
            for ring_name, entries in data.items():
                ring = MemoryRing(ring_name)
                for e_data in entries:
                    entry = MemoryEntry(
                        entry_id     = e_data["entry_id"],
                        content      = e_data["content"],
                        tags         = e_data.get("tags", []),
                        metadata     = e_data.get("metadata", {}),
                        ring         = ring,
                        created_at   = e_data["created_at"],
                        last_accessed = e_data["last_accessed"],
                        access_count = e_data["access_count"],
                        user_id      = e_data.get("user_id"),
                        agent_id     = e_data.get("agent_id"),
                        genome_id    = e_data.get("genome_id"),
                        term_vector  = e_data.get("term_vector", {}),
                    )
                    self._rings[ring][entry.entry_id] = entry
            print(f"[Memory] Loaded memory from {self._persist_path}")
        except Exception as e:
            print(f"[Memory] Could not load memory: {e}")
