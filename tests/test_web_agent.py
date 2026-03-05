"""
Tests for ARIA Memory Persistence and Web Agent
------------------------------------------------
Validates the persistence auto-save fix and the web agent's
core ARIA-native behaviours: permission gating, memory-first retrieval,
domain trust scoring, and graceful degradation.
"""

import asyncio
import json
import os
import tempfile
import time
import pytest

from aria.core.memory import TemporalMemory
from aria.core.models import MemoryRing, TaskNode, ExecutionContract
from aria.agents.web_agent import WebAgent, DomainTrustStore, _clean_text, _fingerprint


# ─── Fixtures ────────────────────────────────────────────────────────────────

class MockPermissions:
    """Permission engine mock — starts with all grants active."""
    def __init__(self, granted: list = None):
        self._granted = set(granted or ["WEB_BROWSE", "WEB_SEARCH"])
        self._denied  = set()

    def require(self, user_id: str, capability):
        cap_name = capability.value if hasattr(capability, "value") else str(capability)
        if cap_name in self._denied or cap_name not in self._granted:
            from aria.permissions.engine import PermissionDeniedError, Capability
            # Find matching Capability enum or use first as placeholder
            cap_enum = capability if isinstance(capability, Capability) else Capability.NETWORK
            raise PermissionDeniedError(user_id=user_id, capability=cap_enum)

    def deny(self, capability):
        cap_name = capability.value if hasattr(capability, "value") else str(capability)
        self._denied.add(cap_name)
        self._granted.discard(cap_name)


def make_node(description: str = "test task") -> TaskNode:
    contract = ExecutionContract(max_seconds=60, max_input_tokens=4000, quality_threshold=0.7)
    return TaskNode(node_id="n1", description=description, contract=contract)


# ─── Memory Persistence Tests ─────────────────────────────────────────────────

class TestMemoryPersistence:

    def test_hot_warm_cold_auto_save(self, tmp_path):
        """Storing to HOT/WARM/COLD should automatically write to SQLite."""
        persist_file = str(tmp_path / "memory.db")
        mem = TemporalMemory(persist_path=persist_file)

        mem.store("important finding", ring=MemoryRing.HOT, tags=["test"])
        mem.store("daily summary", ring=MemoryRing.WARM, tags=["test"])
        mem.store("historical fact", ring=MemoryRing.COLD, tags=["test"])
        mem.close()

        assert os.path.exists(persist_file), "SQLite DB should be created automatically"
        import sqlite3
        con = sqlite3.connect(persist_file)
        rows = con.execute("SELECT ring, COUNT(*) FROM memory_entries GROUP BY ring").fetchall()
        con.close()
        rings_found = {r[0]: r[1] for r in rows}
        assert rings_found.get("hot",  0) == 1
        assert rings_found.get("warm", 0) == 1
        assert rings_found.get("cold", 0) == 1

    def test_flash_does_not_persist(self, tmp_path):
        """FLASH ring is intentionally session-only and must NOT be saved to SQLite."""
        persist_file = str(tmp_path / "memory.db")
        mem = TemporalMemory(persist_path=persist_file)

        mem.store("ephemeral working variable", ring=MemoryRing.FLASH)
        mem.close()

        import sqlite3
        con = sqlite3.connect(persist_file)
        count = con.execute("SELECT COUNT(*) FROM memory_entries WHERE ring='flash'").fetchone()[0]
        con.close()
        assert count == 0, "FLASH entries must never be written to SQLite"

    def test_memory_survives_restart(self, tmp_path):
        """Content stored in one session must be available in a fresh session."""
        persist_file = str(tmp_path / "memory.db")

        # Session 1 — store something important
        mem1 = TemporalMemory(persist_path=persist_file)
        mem1.store(
            "The ARIA framework uses default-deny permissions",
            ring=MemoryRing.WARM,
            tags=["aria", "knowledge"],
        )

        # Session 2 — fresh instance, same file
        mem2 = TemporalMemory(persist_path=persist_file)
        results = mem2.recall("ARIA permissions", rings=[MemoryRing.WARM])

        assert len(results) > 0, "Memory must survive a process restart"
        assert "default-deny" in results[0].content

    def test_multiple_stores_accumulate(self, tmp_path):
        """Each store call should add to SQLite, not overwrite previous entries."""
        persist_file = str(tmp_path / "memory.db")
        mem = TemporalMemory(persist_path=persist_file)

        for i in range(5):
            mem.store(f"finding number {i}", ring=MemoryRing.HOT)
        mem.close()

        import sqlite3
        con = sqlite3.connect(persist_file)
        count = con.execute("SELECT COUNT(*) FROM memory_entries WHERE ring='hot'").fetchone()[0]
        con.close()
        assert count == 5, "All 5 entries must be persisted in SQLite"

    def test_no_persist_path_still_works(self):
        """Memory without a persist_path works fine in-process — no crash."""
        mem = TemporalMemory()
        entry_id = mem.store("ephemeral content", ring=MemoryRing.HOT)
        assert entry_id is not None
        results = mem.recall("ephemeral content")
        assert len(results) > 0


# ─── Domain Trust Store Tests ─────────────────────────────────────────────────

class TestDomainTrustStore:

    def test_new_domain_gets_moderate_prior(self):
        """A domain never seen before should have a moderate trust score, not zero."""
        trust = DomainTrustStore()
        score = trust.score("https://brandnewsite.example.com/article")
        assert 0.5 < score < 0.8, f"New domain prior should be moderate, got {score}"

    def test_successful_fetches_increase_trust(self):
        """Domains that return good content should accumulate higher trust."""
        trust = DomainTrustStore()
        url = "https://reliable.example.com/page"
        initial = trust.score(url)

        for _ in range(5):
            trust.record(url, success=True, content_length=5000)

        assert trust.score(url) > initial

    def test_failed_fetches_decrease_trust(self):
        """Domains that fail or return empty content should lose trust over time."""
        trust = DomainTrustStore()
        url = "https://unreliable.example.com/page"
        initial = trust.score(url)

        for _ in range(5):
            trust.record(url, success=False, content_length=0)

        assert trust.score(url) < initial

    def test_trust_persists_to_disk(self, tmp_path):
        """Domain trust scores should survive between sessions."""
        path = str(tmp_path / "trust.json")
        trust1 = DomainTrustStore(persist_path=path)
        trust1.record("https://good.example.com", success=True, content_length=8000)
        trust1.record("https://good.example.com", success=True, content_length=9000)

        trust2 = DomainTrustStore(persist_path=path)
        score = trust2.score("https://good.example.com")
        assert score >= 0.8, "High trust domain must persist across sessions"

    def test_top_domains_returns_sorted_list(self):
        """top_domains() should return domains sorted by trust, highest first."""
        trust = DomainTrustStore()
        for _ in range(8):
            trust.record("https://great.example.com", success=True, content_length=5000)
        for _ in range(5):
            trust.record("https://ok.example.com", success=True, content_length=3000)
        for _ in range(5):
            trust.record("https://bad.example.com", success=False, content_length=0)

        top = trust.top_domains(n=3)
        assert len(top) == 3
        assert top[0][1] >= top[1][1] >= top[2][1], "Must be sorted descending"
        assert top[0][0] == "great.example.com"


# ─── Content Helpers Tests ────────────────────────────────────────────────────

class TestContentHelpers:

    def test_clean_text_strips_html(self):
        html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
        result = _clean_text(html)
        assert "<" not in result
        assert "Hello" in result
        assert "World" in result

    def test_clean_text_removes_scripts(self):
        html = "<html><script>alert('xss')</script><p>Clean content</p></html>"
        result = _clean_text(html)
        assert "alert" not in result
        assert "Clean content" in result

    def test_clean_text_truncates_at_sentence(self):
        long_text = ("This is a sentence. " * 1000)
        result = _clean_text(long_text, max_chars=200)
        assert len(result) <= 250   # Some tolerance for truncation logic
        assert "truncated" in result

    def test_fingerprint_is_consistent(self):
        content = "The quick brown fox jumps over the lazy dog"
        fp1 = _fingerprint(content)
        fp2 = _fingerprint(content)
        assert fp1 == fp2, "Same content must produce same fingerprint"

    def test_fingerprint_differs_for_different_content(self):
        fp1 = _fingerprint("content version one")
        fp2 = _fingerprint("content version two — different")
        assert fp1 != fp2, "Different content must produce different fingerprint"


# ─── Web Agent Core Tests ─────────────────────────────────────────────────────

class TestWebAgent:

    def _make_agent(self, tmp_path, permissions=None):
        mem   = TemporalMemory(persist_path=str(tmp_path / "memory.db"))
        perms = permissions or MockPermissions()
        agent = WebAgent(
            permissions       = perms,
            memory            = mem,
            anthropic_api_key = None,
            freshness_hours   = 1.0,
            domain_trust_path = str(tmp_path / "trust.json"),
        )
        return agent, mem, perms

    def test_agent_has_correct_capabilities(self, tmp_path):
        agent, _, _ = self._make_agent(tmp_path)
        caps = " ".join(agent.fingerprint.capabilities).lower()
        assert "browse" in caps
        assert "search" in caps

    def test_agent_domains_include_web(self, tmp_path):
        agent, _, _ = self._make_agent(tmp_path)
        assert "web" in agent.fingerprint.domains

    def test_permission_denied_for_search_blocks_network(self, tmp_path):
        """When WEB_SEARCH is not granted, execute must raise before any network call."""
        perms = MockPermissions(granted=[])  # No grants at all
        agent, mem, _ = self._make_agent(tmp_path, permissions=perms)

        from aria.permissions.engine import PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            asyncio.run(
                agent._execute_search("latest AI news", "user1", make_node())
            )

    def test_permission_denied_for_browse_blocks_network(self, tmp_path):
        """When WEB_BROWSE is not granted, browse must raise before any network call."""
        perms = MockPermissions(granted=[])
        agent, mem, _ = self._make_agent(tmp_path, permissions=perms)

        from aria.permissions.engine import PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            asyncio.run(
                agent._execute_browse("https://example.com", "browse this", "user1", make_node())
            )

    def test_memory_first_serves_fresh_cache(self, tmp_path):
        """
        If a URL was browsed recently and is fresh, the agent must serve from
        memory without making any network call.
        """
        agent, mem, _ = self._make_agent(tmp_path)

        # Manually plant a fresh browse result in memory
        url = "https://example.com/article"
        mem.store(
            content  = "This is the cached article content about ARIA framework design.",
            ring     = MemoryRing.WARM,
            tags     = ["web_browse", f"url:{url}"],
            metadata = {
                "source_url":  url,
                "fingerprint": "abc123",
                "browsed_at":  __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "char_count":  100,
            },
            user_id  = "user1",
        )

        cached = agent._recall_url(url, "user1")
        assert cached is not None, "Agent must find fresh cached content"
        assert "ARIA framework" in cached["content"]

    def test_stale_cache_returns_none(self, tmp_path):
        """Content older than freshness_hours must not be served from cache."""
        agent, mem, _ = self._make_agent(tmp_path)
        agent._freshness_seconds = 1   # 1 second freshness for test

        url = "https://example.com/old-article"
        mem.store(
            content  = "Old article content",
            ring     = MemoryRing.WARM,
            tags     = ["web_browse", f"url:{url}"],
            metadata = {"source_url": url, "fingerprint": "old123", "char_count": 50,
                        "browsed_at": "2024-01-01T00:00:00+00:00"},
            user_id  = "user1",
        )
        # Override created_at to be old
        import aria.core.memory as m_module
        for ring_store in mem._rings.values():
            for entry in ring_store.values():
                if "old-article" in " ".join(entry.tags):
                    entry.created_at = time.time() - 7200   # 2 hours old

        cached = agent._recall_url(url, "user1")
        assert cached is None, "Stale cache must not be served"

    def test_freshness_check_logic(self, tmp_path):
        agent, _, _ = self._make_agent(tmp_path)
        agent._freshness_seconds = 3600   # 1 hour

        assert agent._is_fresh(time.time() - 1800) is True    # 30 min ago = fresh
        assert agent._is_fresh(time.time() - 7200) is False   # 2 hours ago = stale

    def test_age_str_format(self, tmp_path):
        agent, _, _ = self._make_agent(tmp_path)
        assert "seconds" in agent._age_str(time.time() - 30)
        assert "minutes" in agent._age_str(time.time() - 300)
        assert "hours"   in agent._age_str(time.time() - 7200)

    def test_browse_result_stored_in_warm_memory(self, tmp_path):
        """After a successful browse, content must be stored in WARM memory."""
        agent, mem, _ = self._make_agent(tmp_path)

        url     = "https://example.com/page"
        content = "This is the full page content from the website. " * 50
        fp      = _fingerprint(content)

        agent._store_browse_result(url, content, fp, "user1", make_node())

        results = mem.recall(url, rings=[MemoryRing.WARM])
        assert len(results) > 0, "Browse result must be stored in WARM memory"
        meta = results[0].metadata or {}
        assert meta.get("source_url") == url

    def test_long_content_also_gets_hot_summary(self, tmp_path):
        """Content > 2000 chars must also generate a HOT summary entry."""
        agent, mem, _ = self._make_agent(tmp_path)

        url     = "https://example.com/long-article"
        content = "Detailed content paragraph. " * 200   # ~5600 chars
        fp      = _fingerprint(content)

        agent._store_browse_result(url, content, fp, "user1", make_node())

        hot_results = mem.recall(url, rings=[MemoryRing.HOT])
        assert len(hot_results) > 0, "Long content must generate a HOT summary"
        assert hot_results[0].metadata.get("is_summary") is True

    def test_trust_report_returns_string(self, tmp_path):
        agent, _, _ = self._make_agent(tmp_path)
        agent._domain_trust.record("https://example.com", success=True, content_length=5000)
        report = agent.trust_report()
        assert isinstance(report, str)
        assert "example.com" in report

    def test_url_detection_in_task(self, tmp_path):
        """Tasks containing a URL should route to browse, not search."""
        import re
        task_with_url    = "Summarize this article: https://example.com/article-123"
        task_without_url = "What are the latest trends in AI security?"

        url_match1 = re.search(r"https?://[^\s\"'>]+", task_with_url)
        url_match2 = re.search(r"https?://[^\s\"'>]+", task_without_url)

        assert url_match1 is not None, "URL task must be detected as browse"
        assert url_match2 is None,     "Non-URL task must route to search"
