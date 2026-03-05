"""
ARIA Temporal Memory Rings
--------------------------
Four-ring memory architecture inspired by CPU cache hierarchy.

    FLASH  → Session only. No persistence. Working variables.
    HOT    → Last 2 hours. Fast access.
    WARM   → Last 24 hours. Normal access.
    COLD   → Up to 30 days. Long-term archive.

Memory moves between rings automatically via promotion and demotion.
Search uses TF-IDF cosine similarity — no embedding API, sub-millisecond.

Storage Backend — SQLite (stdlib, zero new dependencies):
    Previous design rewrote the entire JSON file on every store() call.
    At 1000 entries that cost ~10ms per write. SQLite writes one row per
    store() call — consistently under 1ms at any scale. RAM dicts stay
    as the primary TF-IDF search index. SQLite is the durable backing
    store that survives process restarts.

    FLASH stays RAM-only intentionally — session-only working memory.
    Public API is 100% identical to previous JSON version.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from .models import MemoryEntry, MemoryRing
from .registry import build_term_frequency, cosine_similarity

# ─── Ring Configuration ───────────────────────────────────────────────────────

RING_MAX_AGE_HOURS = {
    MemoryRing.FLASH: 0.5,
    MemoryRing.HOT:   2.0,
    MemoryRing.WARM:  24.0,
    MemoryRing.COLD:  720.0,
}

RING_DEFAULT_MAX_ENTRIES = {
    MemoryRing.FLASH: 200,
    MemoryRing.HOT:   1000,
    MemoryRing.WARM:  5000,
    MemoryRing.COLD:  20000,
}

PROMOTION_THRESHOLD_ACCESS = 3
DEMOTION_IDLE_HOURS = {
    MemoryRing.FLASH: 0.5,
    MemoryRing.HOT:   2.0,
    MemoryRing.WARM:  24.0,
}

# ─── SQLite Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id        TEXT    PRIMARY KEY,
    content         TEXT    NOT NULL,
    ring            TEXT    NOT NULL,
    tags            TEXT    NOT NULL DEFAULT '[]',
    metadata        TEXT    NOT NULL DEFAULT '{}',
    term_vector     TEXT    NOT NULL DEFAULT '{}',
    user_id         TEXT,
    agent_id        TEXT,
    genome_id       TEXT,
    created_at      REAL    NOT NULL,
    last_accessed   REAL    NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ring        ON memory_entries(ring);
CREATE INDEX IF NOT EXISTS idx_user_id     ON memory_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_created_at  ON memory_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_last_access ON memory_entries(last_accessed);
"""

_INSERT = """
INSERT OR REPLACE INTO memory_entries
    (entry_id, content, ring, tags, metadata, term_vector,
     user_id, agent_id, genome_id, created_at, last_accessed, access_count)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_ACCESS = """
UPDATE memory_entries SET last_accessed = ?, access_count = access_count + 1
WHERE entry_id = ?
"""

_SELECT_ALL_PERSISTENT = """
SELECT entry_id, content, ring, tags, metadata, term_vector,
       user_id, agent_id, genome_id, created_at, last_accessed, access_count
FROM memory_entries WHERE ring IN ('hot', 'warm', 'cold')
"""


# ─── Temporal Memory ──────────────────────────────────────────────────────────

class TemporalMemory:
    """
    ARIA four-ring temporal memory backed by SQLite.

    Example:
        memory = TemporalMemory(persist_path=".aria/memory.db")

        entry_id = memory.store(
            "User prefers bullet point summaries",
            ring=MemoryRing.WARM,
            tags=["preference", "formatting"],
            user_id="alice",
        )

        results = memory.recall("formatting preference", user_id="alice")
    """

    def __init__(
        self,
        max_entries:  Optional[dict[MemoryRing, int]] = None,
        persist_path: Optional[str] = None,
    ):
        self._rings: dict[MemoryRing, dict[str, MemoryEntry]] = {
            ring: {} for ring in MemoryRing
        }
        self._max_entries = max_entries or dict(RING_DEFAULT_MAX_ENTRIES)
        self._db_path: Optional[Path] = Path(persist_path) if persist_path else None
        self._conn: Optional[sqlite3.Connection] = None

        if self._db_path:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._load_from_db()

    @contextmanager
    def _db(self):
        if self._conn is None:
            yield None
            return
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    @staticmethod
    def _row_to_entry(row) -> MemoryEntry:
        (entry_id, content, ring, tags, metadata, term_vector,
         user_id, agent_id, genome_id, created_at, last_accessed, access_count) = row
        return MemoryEntry(
            entry_id=entry_id, content=content, ring=MemoryRing(ring),
            tags=json.loads(tags), metadata=json.loads(metadata),
            term_vector=json.loads(term_vector),
            user_id=user_id, agent_id=agent_id, genome_id=genome_id,
            created_at=created_at, last_accessed=last_accessed,
            access_count=access_count,
        )

    @staticmethod
    def _entry_to_row(e: MemoryEntry) -> tuple:
        return (
            e.entry_id, e.content, e.ring.value,
            json.dumps(e.tags), json.dumps(e.metadata), json.dumps(e.term_vector),
            e.user_id, e.agent_id, e.genome_id,
            e.created_at, e.last_accessed, e.access_count,
        )

    def _load_from_db(self):
        if self._conn is None:
            return
        cur = self._conn.execute(_SELECT_ALL_PERSISTENT)
        loaded = 0
        for row in cur.fetchall():
            entry = self._row_to_entry(row)
            self._rings[entry.ring][entry.entry_id] = entry
            loaded += 1
        if loaded:
            print(f"[Memory] Loaded {loaded} entries from {self._db_path.name}")

    # ── Public API ────────────────────────────────────────────────────────────

    def store(
        self,
        content:   str,
        ring:      MemoryRing = MemoryRing.FLASH,
        tags:      Optional[list[str]] = None,
        metadata:  Optional[dict] = None,
        user_id:   Optional[str] = None,
        agent_id:  Optional[str] = None,
        genome_id: Optional[str] = None,
    ) -> str:
        """Store content in the specified ring. Returns entry_id."""
        entry = MemoryEntry(
            content=content, ring=ring, tags=tags or [],
            metadata=metadata or {}, user_id=user_id,
            agent_id=agent_id, genome_id=genome_id,
            term_vector=build_term_frequency([content] + (tags or [])),
        )
        ring_store = self._rings[ring]
        if len(ring_store) >= self._max_entries[ring]:
            self._evict_oldest(ring)
        ring_store[entry.entry_id] = entry

        # SQLite write — one row, under 1ms, regardless of total entry count
        if ring != MemoryRing.FLASH:
            with self._db() as cur:
                if cur is not None:
                    cur.execute(_INSERT, self._entry_to_row(entry))
        return entry.entry_id

    def store_result(
        self,
        genome_id: str,
        node_id:   str,
        result:    Any,
        agent_id:  str,
        ring:      MemoryRing = MemoryRing.HOT,
    ) -> str:
        content = f"Task result for node {node_id}: {str(result)[:500]}"
        return self.store(
            content=content, ring=ring,
            tags=[f"genome:{genome_id}", f"node:{node_id}", f"agent:{agent_id}"],
            metadata={"genome_id": genome_id, "node_id": node_id, "result": str(result)[:2000]},
            agent_id=agent_id, genome_id=genome_id,
        )

    def recall(
        self,
        query:          str,
        rings:          Optional[list[MemoryRing]] = None,
        top_k:          int = 5,
        user_id:        Optional[str] = None,
        agent_id:       Optional[str] = None,
        genome_id:      Optional[str] = None,
        min_similarity: float = 0.05,
    ) -> list[MemoryEntry]:
        """
        Recall most relevant memories using TF-IDF cosine similarity.
        Searches FLASH → HOT → WARM → COLD, stopping early when enough results found.
        """
        search_rings = rings or [MemoryRing.FLASH, MemoryRing.HOT, MemoryRing.WARM, MemoryRing.COLD]
        query_vec = build_term_frequency([query])
        scored: list[tuple[float, MemoryEntry]] = []

        for ring in search_rings:
            for entry in self._rings[ring].values():
                if user_id and entry.user_id and entry.user_id != user_id:
                    continue
                if agent_id and entry.agent_id and entry.agent_id != agent_id:
                    continue
                if genome_id and entry.genome_id and entry.genome_id != genome_id:
                    continue
                sim = cosine_similarity(query_vec, entry.term_vector)
                if sim >= min_similarity:
                    scored.append((sim, entry))
            if len(scored) >= top_k * 2:
                break

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]

        for entry in results:
            entry.touch()
            if entry.ring != MemoryRing.FLASH and self._conn:
                with self._db() as cur:
                    if cur is not None:
                        cur.execute(_UPDATE_ACCESS, (entry.last_accessed, entry.entry_id))
            self._maybe_promote(entry)

        return results

    def recall_by_tag(
        self,
        tag:   str,
        rings: Optional[list[MemoryRing]] = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        search_rings = rings or list(MemoryRing)
        results = []
        for ring in search_rings:
            for entry in self._rings[ring].values():
                if tag in entry.tags:
                    results.append(entry)
        return results[:top_k]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        for ring_store in self._rings.values():
            if entry_id in ring_store:
                entry = ring_store[entry_id]
                entry.touch()
                return entry
        return None

    def promote(self, entry_id: str, target_ring: MemoryRing) -> bool:
        for ring, ring_store in self._rings.items():
            if entry_id in ring_store:
                entry = ring_store.pop(entry_id)
                entry.ring = target_ring
                self._rings[target_ring][entry_id] = entry
                if target_ring != MemoryRing.FLASH and self._conn:
                    with self._db() as cur:
                        if cur is not None:
                            cur.execute(
                                "UPDATE memory_entries SET ring=? WHERE entry_id=?",
                                (target_ring.value, entry_id)
                            )
                return True
        return False

    def _maybe_promote(self, entry: MemoryEntry):
        if entry.access_count < PROMOTION_THRESHOLD_ACCESS:
            return
        ring_order = [MemoryRing.COLD, MemoryRing.WARM, MemoryRing.HOT, MemoryRing.FLASH]
        if entry.ring not in ring_order:
            return
        idx = ring_order.index(entry.ring)
        if idx < len(ring_order) - 1:
            self.promote(entry.entry_id, ring_order[idx + 1])

    def maintain(self):
        """Run ring maintenance — demote stale entries, evict expired ones."""
        now = time.time()
        demoted = evicted = 0

        for ring, idle_hours in DEMOTION_IDLE_HOURS.items():
            to_demote = [
                (eid, e) for eid, e in self._rings[ring].items()
                if (now - e.last_accessed) / 3600 > idle_hours
            ]
            ring_order = [MemoryRing.FLASH, MemoryRing.HOT, MemoryRing.WARM, MemoryRing.COLD]
            for eid, entry in to_demote:
                del self._rings[ring][eid]
                idx = ring_order.index(ring)
                if idx < len(ring_order) - 1:
                    target = ring_order[idx + 1]
                    entry.ring = target
                    self._rings[target][eid] = entry
                    if self._conn:
                        with self._db() as cur:
                            if cur is not None:
                                cur.execute("UPDATE memory_entries SET ring=? WHERE entry_id=?",
                                            (target.value, eid))
                    demoted += 1

        for ring, max_age_hours in RING_MAX_AGE_HOURS.items():
            to_evict = [eid for eid, e in self._rings[ring].items()
                        if e.age_hours > max_age_hours]
            for eid in to_evict:
                del self._rings[ring][eid]
                if ring != MemoryRing.FLASH and self._conn:
                    with self._db() as cur:
                        if cur is not None:
                            cur.execute("DELETE FROM memory_entries WHERE entry_id=?", (eid,))
                evicted += 1

        if demoted or evicted:
            print(f"[Memory] Maintenance: {demoted} demoted, {evicted} evicted")

    def _evict_oldest(self, ring: MemoryRing):
        ring_store = self._rings[ring]
        if not ring_store:
            return
        oldest_id = min(ring_store, key=lambda eid: ring_store[eid].last_accessed)
        del ring_store[oldest_id]
        if ring != MemoryRing.FLASH and self._conn:
            with self._db() as cur:
                if cur is not None:
                    cur.execute("DELETE FROM memory_entries WHERE entry_id=?", (oldest_id,))

    def stats(self) -> dict:
        s = {
            ring.value: {"count": len(self._rings[ring]), "max": self._max_entries[ring]}
            for ring in MemoryRing
        }
        s["backend"] = f"SQLite — {self._db_path.name}" if self._db_path else "RAM only"
        return s

    def clear_ring(self, ring: MemoryRing):
        self._rings[ring].clear()
        if ring != MemoryRing.FLASH and self._conn:
            with self._db() as cur:
                if cur is not None:
                    cur.execute("DELETE FROM memory_entries WHERE ring=?", (ring.value,))

    def clear_all(self):
        for ring in MemoryRing:
            self._rings[ring].clear()
        if self._conn:
            with self._db() as cur:
                if cur is not None:
                    cur.execute("DELETE FROM memory_entries")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
