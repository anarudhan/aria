"""
Unit tests for aria.core.memory.TemporalMemory and inject_genesis_memory.

Persistence basics (rings saved to SQLite, FLASH never saved, restart survival) are
already covered in tests/test_web_agent.py and are not repeated here. Everything uses
RAM-only memory or databases under tmp_path.
"""
import sqlite3
import time

import pytest

from aria.core.memory import (
    PROMOTION_THRESHOLD_ACCESS,
    RING_DEFAULT_MAX_ENTRIES,
    TemporalMemory,
    inject_genesis_memory,
)
from aria.core.models import ARIA_ORIGIN, MemoryRing

HOUR = 3600.0


def db_rows(path, sql="SELECT entry_id, ring FROM memory_entries"):
    con = sqlite3.connect(str(path))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def age(memory: TemporalMemory, entry_id: str, created_hours=0.0, idle_hours=0.0):
    """Backdate an entry so maintenance sees it as old / idle (no sleeping needed)."""
    entry = memory._rings[memory.get(entry_id).ring][entry_id]
    now = time.time()
    entry.created_at = now - created_hours * HOUR
    entry.last_accessed = now - idle_hours * HOUR
    return entry


@pytest.fixture
def mem():
    return TemporalMemory()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory.db"


# ─── store / get ──────────────────────────────────────────────────────────────

class TestStoreAndGet:

    def test_store_returns_id_and_get_returns_entry(self, mem):
        entry_id = mem.store("remember the milk", ring=MemoryRing.HOT, tags=["shopping"],
                             metadata={"k": 1}, user_id="u", agent_id="a", genome_id="g")
        entry = mem.get(entry_id)
        assert entry.entry_id == entry_id
        assert entry.content == "remember the milk"
        assert entry.ring == MemoryRing.HOT
        assert entry.tags == ["shopping"]
        assert entry.metadata == {"k": 1}
        assert (entry.user_id, entry.agent_id, entry.genome_id) == ("u", "a", "g")

    def test_default_ring_is_flash(self, mem):
        assert mem.get(mem.store("scratch")).ring == MemoryRing.FLASH

    def test_term_vector_includes_content_and_tags(self, mem):
        entry = mem.get(mem.store("quarterly revenue", tags=["finance"]))
        assert {"quarterly", "revenue", "finance"} <= set(entry.term_vector)

    def test_ids_are_unique(self, mem):
        ids = {mem.store("same content") for _ in range(20)}
        assert len(ids) == 20

    def test_get_unknown_id_returns_none(self, mem):
        assert mem.get("does-not-exist") is None

    def test_get_updates_access_tracking(self, mem):
        entry_id = mem.store("x thing")
        mem.get(entry_id)
        entry = mem.get(entry_id)
        assert entry.access_count == 2

    def test_none_tags_and_metadata_become_empty_containers(self, mem):
        entry = mem.get(mem.store("plain"))
        assert entry.tags == [] and entry.metadata == {}


# ─── recall ───────────────────────────────────────────────────────────────────

class TestRecall:

    def test_most_relevant_entry_comes_first(self, mem):
        mem.store("python debugging tips for async code", ring=MemoryRing.HOT)
        mem.store("python", ring=MemoryRing.HOT)
        mem.store("recipe for chocolate cake", ring=MemoryRing.HOT)

        results = mem.recall("python debugging async")

        assert results[0].content == "python debugging tips for async code"
        assert all("chocolate" not in r.content for r in results)

    def test_no_match_returns_empty_list(self, mem):
        mem.store("completely unrelated text", ring=MemoryRing.HOT)
        assert mem.recall("quantum chromodynamics") == []

    def test_empty_memory_and_empty_query(self, mem):
        assert mem.recall("anything") == []
        mem.store("something stored", ring=MemoryRing.HOT)
        assert mem.recall("") == []

    def test_top_k_limits_results(self, mem):
        for i in range(10):
            mem.store(f"report number {i} about sales", ring=MemoryRing.HOT)
        assert len(mem.recall("sales report", top_k=3)) == 3

    def test_min_similarity_filters_weak_matches(self, mem):
        mem.store("alpha beta gamma delta epsilon zeta eta theta", ring=MemoryRing.HOT)
        assert mem.recall("alpha", min_similarity=0.05) != []
        assert mem.recall("alpha", min_similarity=0.9) == []

    def test_rings_argument_restricts_search(self, mem):
        mem.store("budget forecast", ring=MemoryRing.HOT)
        mem.store("budget archive", ring=MemoryRing.COLD)
        results = mem.recall("budget", rings=[MemoryRing.COLD])
        assert [r.content for r in results] == ["budget archive"]

    def test_user_isolation(self, mem):
        mem.store("alice private note about taxes", ring=MemoryRing.HOT, user_id="alice")
        mem.store("bob private note about taxes", ring=MemoryRing.HOT, user_id="bob")
        mem.store("shared note about taxes", ring=MemoryRing.HOT)

        contents = {r.content for r in mem.recall("taxes note", user_id="alice")}

        assert "alice private note about taxes" in contents
        assert "shared note about taxes" in contents
        assert "bob private note about taxes" not in contents

    def test_agent_and_genome_filters(self, mem):
        mem.store("finding from researcher", ring=MemoryRing.HOT, agent_id="researcher",
                  genome_id="g1")
        mem.store("finding from writer", ring=MemoryRing.HOT, agent_id="writer",
                  genome_id="g2")

        by_agent = mem.recall("finding", agent_id="writer")
        by_genome = mem.recall("finding", genome_id="g1")

        assert [r.content for r in by_agent] == ["finding from writer"]
        assert [r.content for r in by_genome] == ["finding from researcher"]

    def test_recall_touches_returned_entries(self, mem):
        entry_id = mem.store("touch me please", ring=MemoryRing.WARM)
        mem.recall("touch me")
        assert mem.get(entry_id).access_count == 2     # one from recall, one from get

    def test_recall_updates_access_count_in_sqlite(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        persisted.store("persistent insight", ring=MemoryRing.WARM)
        persisted.recall("persistent insight")
        persisted.close()

        rows = db_rows(db_path, "SELECT access_count FROM memory_entries")
        assert rows == [(1,)]

    def test_entries_are_promoted_after_repeated_access(self, mem):
        entry_id = mem.store("popular fact about oceans", ring=MemoryRing.COLD)
        for _ in range(PROMOTION_THRESHOLD_ACCESS):
            mem.recall("oceans fact")
        assert mem.get(entry_id).ring == MemoryRing.WARM

    def test_promotion_is_written_to_sqlite(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        entry_id = persisted.store("popular fact about oceans", ring=MemoryRing.COLD)
        for _ in range(PROMOTION_THRESHOLD_ACCESS):
            persisted.recall("oceans fact")
        persisted.close()

        assert db_rows(db_path) == [(entry_id, "warm")]

    def test_flash_entries_are_recalled_before_colder_rings_fill_results(self, mem):
        for i in range(2):
            mem.store(f"deploy checklist {i}", ring=MemoryRing.FLASH)
        mem.store("deploy checklist archived", ring=MemoryRing.COLD)

        results = mem.recall("deploy checklist", top_k=1)

        assert results[0].ring == MemoryRing.FLASH


class TestRecallByTag:

    def test_returns_entries_with_the_tag(self, mem):
        mem.store("one", tags=["alpha"], ring=MemoryRing.HOT)
        mem.store("two", tags=["alpha", "beta"], ring=MemoryRing.WARM)
        mem.store("three", tags=["beta"], ring=MemoryRing.COLD)

        assert {e.content for e in mem.recall_by_tag("alpha")} == {"one", "two"}
        assert mem.recall_by_tag("missing") == []

    def test_respects_rings_and_top_k(self, mem):
        for i in range(5):
            mem.store(f"hot {i}", tags=["t"], ring=MemoryRing.HOT)
        mem.store("cold", tags=["t"], ring=MemoryRing.COLD)

        assert len(mem.recall_by_tag("t", top_k=2)) == 2
        assert [e.content for e in mem.recall_by_tag("t", rings=[MemoryRing.COLD])] == ["cold"]


# ─── promote ──────────────────────────────────────────────────────────────────

class TestPromote:

    def test_moves_entry_between_rings(self, mem):
        entry_id = mem.store("move me", ring=MemoryRing.COLD)
        assert mem.promote(entry_id, MemoryRing.HOT) is True
        assert mem.get(entry_id).ring == MemoryRing.HOT
        assert entry_id in mem._rings[MemoryRing.HOT]
        assert entry_id not in mem._rings[MemoryRing.COLD]

    def test_unknown_entry_returns_false(self, mem):
        assert mem.promote("ghost", MemoryRing.HOT) is False

    def test_persisted_ring_is_updated(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        entry_id = persisted.store("move me", ring=MemoryRing.COLD)
        persisted.promote(entry_id, MemoryRing.HOT)
        persisted.close()
        assert db_rows(db_path) == [(entry_id, "hot")]

    @pytest.mark.xfail(
        strict=True,
        reason="bug: memory.py promote() only runs UPDATE on the SQLite row, but FLASH "
               "entries have no row, so an entry promoted from FLASH to a persistent ring "
               "is lost on restart",
    )
    def test_flash_entry_promoted_to_persistent_ring_survives_restart(self, db_path):
        first = TemporalMemory(persist_path=str(db_path))
        entry_id = first.store("important scratch finding", ring=MemoryRing.FLASH)
        first.promote(entry_id, MemoryRing.HOT)
        first.close()

        second = TemporalMemory(persist_path=str(db_path))
        assert second.get(entry_id) is not None


# ─── capacity ─────────────────────────────────────────────────────────────────

class TestCapacity:

    @staticmethod
    def sizes(overrides):
        merged = dict(RING_DEFAULT_MAX_ENTRIES)
        merged.update(overrides)
        return merged

    def test_full_ring_evicts_the_least_recently_accessed_entry(self):
        mem = TemporalMemory(max_entries=self.sizes({MemoryRing.HOT: 2}))
        first = mem.store("first", ring=MemoryRing.HOT)
        second = mem.store("second", ring=MemoryRing.HOT)
        mem._rings[MemoryRing.HOT][first].last_accessed -= 100   # first is the stalest

        third = mem.store("third", ring=MemoryRing.HOT)

        assert set(mem._rings[MemoryRing.HOT]) == {second, third}
        assert mem.get(first) is None

    def test_eviction_keeps_ring_at_capacity(self):
        mem = TemporalMemory(max_entries=self.sizes({MemoryRing.WARM: 3}))
        for i in range(10):
            mem.store(f"entry {i}", ring=MemoryRing.WARM)
        assert len(mem._rings[MemoryRing.WARM]) == 3

    def test_eviction_deletes_the_sqlite_row(self, db_path):
        mem = TemporalMemory(max_entries=self.sizes({MemoryRing.HOT: 1}),
                             persist_path=str(db_path))
        first = mem.store("first", ring=MemoryRing.HOT)
        second = mem.store("second", ring=MemoryRing.HOT)
        mem.close()

        assert db_rows(db_path) == [(second, "hot")]
        assert first != second

    def test_other_rings_are_unaffected_by_a_full_ring(self):
        mem = TemporalMemory(max_entries=self.sizes({MemoryRing.HOT: 1}))
        cold = mem.store("cold one", ring=MemoryRing.COLD)
        mem.store("hot one", ring=MemoryRing.HOT)
        mem.store("hot two", ring=MemoryRing.HOT)
        assert mem.get(cold) is not None


# ─── maintain ─────────────────────────────────────────────────────────────────

class TestMaintain:

    def test_maintain_on_empty_memory_is_a_noop(self, mem):
        mem.maintain()
        assert all(v["count"] == 0 for k, v in mem.stats().items() if k != "backend")

    def test_fresh_entries_are_untouched(self, mem):
        ids = [mem.store(f"fresh {r.value}", ring=r) for r in MemoryRing]
        mem.maintain()
        assert [mem.get(i).ring for i in ids] == list(MemoryRing)

    def test_idle_flash_entry_is_demoted_to_hot(self, mem):
        entry_id = mem.store("idle scratch", ring=MemoryRing.FLASH)
        age(mem, entry_id, created_hours=0.1, idle_hours=1.0)
        mem.maintain()
        assert mem.get(entry_id).ring == MemoryRing.HOT

    def test_idle_hot_entry_is_demoted_to_warm(self, mem):
        entry_id = mem.store("idle hot", ring=MemoryRing.HOT)
        age(mem, entry_id, created_hours=3.0, idle_hours=3.0)
        mem.maintain()
        assert mem.get(entry_id).ring == MemoryRing.WARM

    def test_demotion_is_written_to_sqlite(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        entry_id = persisted.store("idle hot", ring=MemoryRing.HOT)
        age(persisted, entry_id, created_hours=3.0, idle_hours=3.0)
        persisted.maintain()
        persisted.close()
        assert db_rows(db_path) == [(entry_id, "warm")]

    def test_recently_used_entry_is_not_demoted(self, mem):
        entry_id = mem.store("busy hot", ring=MemoryRing.HOT)
        age(mem, entry_id, created_hours=1.0, idle_hours=0.1)
        mem.maintain()
        assert mem.get(entry_id).ring == MemoryRing.HOT

    @pytest.mark.parametrize("ring,max_age", [
        (MemoryRing.FLASH, 0.5), (MemoryRing.HOT, 2.0),
        (MemoryRing.WARM, 24.0), (MemoryRing.COLD, 720.0),
    ])
    def test_entries_older_than_ring_limit_are_evicted(self, mem, ring, max_age):
        old = mem.store("too old", ring=ring)
        young = mem.store("young enough", ring=ring)
        age(mem, old, created_hours=max_age + 1, idle_hours=0.0)
        age(mem, young, created_hours=max_age / 2, idle_hours=0.0)

        mem.maintain()

        assert mem.get(old) is None
        assert mem.get(young) is not None

    def test_eviction_removes_sqlite_rows(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        old = persisted.store("expired", ring=MemoryRing.COLD)
        keep = persisted.store("kept", ring=MemoryRing.COLD)
        age(persisted, old, created_hours=800, idle_hours=0.0)

        persisted.maintain()
        persisted.close()

        assert db_rows(db_path) == [(keep, "cold")]

    def test_maintain_reports_activity(self, mem, capsys):
        entry_id = mem.store("expired", ring=MemoryRing.COLD)
        age(mem, entry_id, created_hours=800)
        mem.maintain()
        assert "1 evicted" in capsys.readouterr().out

    @pytest.mark.xfail(
        strict=True,
        reason="bug: memory.py maintain() moves an idle FLASH entry to HOT in RAM and runs "
               "UPDATE on a SQLite row that does not exist, so the demoted entry is never "
               "persisted and disappears on restart",
    )
    def test_entry_demoted_from_flash_survives_restart(self, db_path):
        first = TemporalMemory(persist_path=str(db_path))
        entry_id = first.store("idle scratch", ring=MemoryRing.FLASH)
        age(first, entry_id, created_hours=0.1, idle_hours=1.0)
        first.maintain()
        assert first.get(entry_id).ring == MemoryRing.HOT
        first.close()

        assert TemporalMemory(persist_path=str(db_path)).get(entry_id) is not None


# ─── store_result ─────────────────────────────────────────────────────────────

class TestStoreResult:

    def test_formats_content_tags_and_metadata(self, mem):
        entry_id = mem.store_result("g1", "t2", "the answer is 42", "researcher")
        entry = mem.get(entry_id)
        assert entry.content == "Task result for node t2: the answer is 42"
        assert entry.ring == MemoryRing.HOT
        assert entry.tags == ["genome:g1", "node:t2", "agent:researcher"]
        assert entry.metadata == {"genome_id": "g1", "node_id": "t2",
                                  "result": "the answer is 42"}
        assert entry.agent_id == "researcher" and entry.genome_id == "g1"

    def test_long_results_are_truncated(self, mem):
        entry = mem.get(mem.store_result("g", "n", "z" * 5000, "a"))
        assert len(entry.content) == len("Task result for node n: ") + 500
        assert len(entry.metadata["result"]) == 2000

    def test_non_string_result_is_stringified(self, mem):
        entry = mem.get(mem.store_result("g", "n", {"a": [1, 2]}, "a"))
        assert "{'a': [1, 2]}" in entry.content

    def test_ring_can_be_chosen_and_results_are_findable_by_genome_tag(self, mem):
        mem.store_result("g9", "n1", "warm result", "a", ring=MemoryRing.WARM)
        found = mem.recall_by_tag("genome:g9")
        assert len(found) == 1 and found[0].ring == MemoryRing.WARM


# ─── stats / clear / close ────────────────────────────────────────────────────

class TestStatsAndClearing:

    def test_stats_for_ram_only_memory(self, mem):
        mem.store("a", ring=MemoryRing.HOT)
        mem.store("b", ring=MemoryRing.HOT)
        stats = mem.stats()
        assert stats["hot"] == {"count": 2, "max": RING_DEFAULT_MAX_ENTRIES[MemoryRing.HOT]}
        assert stats["flash"]["count"] == 0
        assert stats["backend"] == "RAM only"

    def test_stats_name_the_sqlite_file(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        assert persisted.stats()["backend"].endswith("memory.db")
        persisted.close()

    def test_clear_ring_only_clears_that_ring(self, mem):
        mem.store("hot", ring=MemoryRing.HOT)
        keep = mem.store("warm", ring=MemoryRing.WARM)
        mem.clear_ring(MemoryRing.HOT)
        assert mem.stats()["hot"]["count"] == 0
        assert mem.get(keep) is not None

    def test_clear_ring_deletes_sqlite_rows_of_that_ring_only(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        persisted.store("hot", ring=MemoryRing.HOT)
        keep = persisted.store("warm", ring=MemoryRing.WARM)
        persisted.clear_ring(MemoryRing.HOT)
        persisted.close()
        assert db_rows(db_path) == [(keep, "warm")]

    def test_clear_all_empties_ram_and_sqlite(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        for ring in MemoryRing:
            persisted.store(f"entry in {ring.value}", ring=ring)
        persisted.clear_all()
        persisted.close()

        assert db_rows(db_path) == []
        assert all(persisted._rings[r] == {} for r in MemoryRing)

    def test_close_is_idempotent_and_memory_stays_usable_in_ram(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        persisted.close()
        persisted.close()
        entry_id = persisted.store("after close", ring=MemoryRing.HOT)
        assert persisted.get(entry_id) is not None


# ─── persistence details and failure paths ────────────────────────────────────

class TestPersistenceDetails:

    def test_creates_missing_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "memory.db"
        mem = TemporalMemory(persist_path=str(nested))
        mem.close()
        assert nested.exists()

    def test_all_fields_round_trip_through_sqlite(self, db_path):
        first = TemporalMemory(persist_path=str(db_path))
        entry_id = first.store(
            "full fidelity entry", ring=MemoryRing.COLD, tags=["x", "y"],
            metadata={"nested": {"k": [1, 2, 3]}}, user_id="u1", agent_id="a1", genome_id="g1",
        )
        first.close()

        entry = TemporalMemory(persist_path=str(db_path)).get(entry_id)

        assert entry.content == "full fidelity entry"
        assert entry.ring == MemoryRing.COLD
        assert entry.tags == ["x", "y"]
        assert entry.metadata == {"nested": {"k": [1, 2, 3]}}
        assert (entry.user_id, entry.agent_id, entry.genome_id) == ("u1", "a1", "g1")
        assert "full" in entry.term_vector

    def test_restored_entries_are_searchable(self, db_path):
        first = TemporalMemory(persist_path=str(db_path))
        first.store("kubernetes deployment notes", ring=MemoryRing.WARM, user_id="u1")
        first.close()

        second = TemporalMemory(persist_path=str(db_path))
        assert len(second.recall("kubernetes deployment", user_id="u1")) == 1
        assert second.recall("kubernetes deployment", user_id="someone-else") == []

    def test_database_that_is_not_sqlite_raises_a_clear_error(self, tmp_path):
        bad = tmp_path / "memory.db"
        bad.write_text("this is definitely not a sqlite database " * 20, encoding="utf-8")
        with pytest.raises(sqlite3.DatabaseError):
            TemporalMemory(persist_path=str(bad))

    def test_unserialisable_metadata_fails_loudly_and_leaves_no_row(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        with pytest.raises(TypeError):
            persisted.store("bad metadata", ring=MemoryRing.HOT, metadata={"x": object()})

        # The connection must still be usable after the rolled back write.
        good = persisted.store("good one", ring=MemoryRing.HOT)
        persisted.close()
        assert db_rows(db_path) == [(good, "hot")]


# ─── Genesis memory ───────────────────────────────────────────────────────────

class TestGenesisMemory:

    def test_origin_is_injected_into_flash_with_genesis_tags(self, mem):
        inject_genesis_memory(mem)

        entries = mem.recall_by_tag("GENESIS")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.ring == MemoryRing.FLASH
        assert {"origin", "creators", "identity", "immutable"} <= set(entry.tags)
        assert ARIA_ORIGIN["origin_statement"] in entry.content
        assert ARIA_ORIGIN["version"] in entry.content
        assert entry.metadata["genesis"] is True and entry.metadata["immutable"] is True
        assert entry.metadata["born"] == ARIA_ORIGIN["born"]

    def test_origin_can_be_recalled_by_question(self, mem):
        inject_genesis_memory(mem)
        results = mem.recall("who created ARIA and who made her")
        assert results and "GENESIS" in results[0].tags

    def test_genesis_is_never_written_to_sqlite(self, db_path):
        persisted = TemporalMemory(persist_path=str(db_path))
        inject_genesis_memory(persisted)
        persisted.close()
        assert db_rows(db_path) == []
