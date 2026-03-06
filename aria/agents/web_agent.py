"""
ARIA Web Agent
--------------
Web browsing and search — built the ARIA way.

What makes this different from every other framework's web tool:

  1. PERMISSION GATE FIRST
     WEB_BROWSE and WEB_SEARCH are real ARIA capabilities. No grant means
     zero network activity. The permission check is the first line of
     execution — not a guardrail added later.

  2. MEMORY-FIRST RETRIEVAL
     Before touching the internet, ARIA searches her own temporal memory.
     Fresh cached content is served instantly. No redundant network calls.
     Freshness window is configurable per deployment.

  3. CLAUDE-NATIVE SEARCH (primary engine)
     ARIA uses Claude's own web search tool via the Anthropic API. This is
     not a third-party search API bolted on — it is the same intelligence
     that powers Claude reasoning, now extended to the live web. Claude
     decides when a search is needed, formulates the query, retrieves
     results, reasons over them, and returns a cited synthesised answer —
     all in one API call. No separate LLM pass to process raw results.

     Tool version: web_search_20250305 (current live version)
     Backend: Brave Search (Anthropic-operated infrastructure)
     Cost: $10 per 1,000 searches + standard token costs

  4. DOMAIN TRUST SCORING
     Every domain that returns content gets scored over time using the same
     Bayesian approach as ARIA's agent trust ledger. Trust is persistent
     across sessions and feeds into how results are presented.

  5. AUTOMATIC MEMORY STORAGE
     Search results and browsed content flow into the right temporal ring:
       - Search summaries  → HOT  (fast recent recall)
       - Full page content → WARM (available for the day)
     On the next similar query, ARIA serves from memory first.

  6. GRACEFUL DEGRADATION
     Claude API key set?   → Claude native search (primary, best quality)
     No Claude key?        → DuckDuckGo lite HTML scraping (zero-dependency)

     Playwright installed? → Full JS-rendered page content
     No Playwright?        → requests + BeautifulSoup fallback
     No requests?          → urllib from stdlib (always available)

     ARIA always gives you something useful. Never fails silently.

Usage:
    from aria.agents.web_agent import create_web_agent
    from aria.permissions.engine import Capability

    aria.permissions.grant(user_id, Capability.WEB_SEARCH, hours=4)
    aria.permissions.grant(user_id, Capability.WEB_BROWSE, hours=4)

    agent = create_web_agent(
        permissions=aria.permissions,
        memory=aria.memory,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        freshness_hours=2,
        domain_trust_path=".aria/domain_trust.json",
    )
    aria.register_agent(agent)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

from ..core.models import MemoryRing, TaskNode
from .base import BaseAgent

# ─── Content Helpers ──────────────────────────────────────────────────────────

def _fingerprint(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


def _clean_text(raw: str, max_chars: int = 12000) -> str:
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style[^>]*>.*?</style>",  " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    raw = raw.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) <= max_chars:
        return raw
    truncated = raw[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1] + " [content truncated]"
    return truncated + "... [content truncated]"


# ─── Page Fetch Strategies ────────────────────────────────────────────────────

async def _fetch_with_playwright(url: str, timeout_ms: int = 15000) -> str:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError("Playwright not installed. Run: pip install playwright && playwright install chromium")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (compatible; ARIA-WebAgent/1.1; research-bot)"
            })
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            await asyncio.sleep(1.5)
            content = await page.content()
            return _clean_text(content)
        finally:
            await browser.close()


async def _fetch_with_requests(url: str, timeout: int = 12) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ARIA-WebAgent/1.1; research-bot)"}
    try:
        import requests
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.get(url, headers=headers, timeout=timeout)
        )
        response.raise_for_status()
        return _clean_text(response.text)
    except ImportError:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _clean_text(resp.read().decode("utf-8", errors="replace"))


async def _fetch_page(url: str) -> tuple[str, str]:
    """Try Playwright first, fall back to requests/urllib. Returns (content, method)."""
    try:
        return await _fetch_with_playwright(url), "playwright"
    except ImportError:
        pass
    except Exception:
        pass
    content = await _fetch_with_requests(url)
    return content, "requests"


# ─── Search Engines ───────────────────────────────────────────────────────────

async def _search_claude(
    query: str,
    api_key: str,
    max_searches: int = 3,
) -> tuple[str, list[str]]:
    """
    Search using Claude's native web search tool via the Anthropic API.

    Claude reasons over the query, decides what to search, executes it
    via Anthropic's Brave-powered infrastructure, synthesises the results,
    and returns a cited answer — all in one API call.

    Returns (synthesised_answer, list_of_cited_urls).

    This is fundamentally different from Tavily or DuckDuckGo — those return
    raw results that still need to be reasoned over. Claude search returns
    a reasoned answer with citations attached.

    Tool: web_search_20250305 (current live version)
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
        "tools": [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max_searches,
        }],
        "messages": [{
            "role": "user",
            "content": (
                f"Search the web and provide a comprehensive, well-cited answer to this query:\n\n"
                f"{query}\n\n"
                "Include specific facts, figures, and source URLs in your answer."
            ),
        }],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":        api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta":    "web-search-2025-03-05",
                "content-type":      "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract the text answer from the response content blocks
    answer_parts = []
    cited_urls = []

    for block in data.get("content", []):
        if block.get("type") == "text":
            answer_parts.append(block.get("text", ""))
        elif block.get("type") == "web_search_tool_result":
            for result in block.get("content", []):
                url = result.get("url", "")
                if url and url not in cited_urls:
                    cited_urls.append(url)

    answer = "\n".join(answer_parts).strip()
    return answer or "Claude search returned no text answer.", cited_urls


async def _search_tavily(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """
    Tavily Search API fallback — reliable, structured, real-time results.
    Returns a list of dicts with title, url, content fields.
    Used when Claude native search fails and before DuckDuckGo.
    Get a free API key at https://tavily.com (1000 searches/month free).
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx required for Tavily — pip install httpx")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        api_key,
                "query":          query,
                "max_results":    max_results,
                "search_depth":   "basic",
                "include_answer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    # Include Tavily's synthesised answer if present
    if data.get("answer"):
        results.append({
            "title":   "Tavily Summary",
            "url":     "",
            "content": data["answer"],
        })
    for r in data.get("results", []):
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "content": r.get("content", "")[:500],
        })
    return results


async def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """
    Zero-dependency DuckDuckGo lite fallback. No API key needed.
    Used only when no Anthropic API key is configured.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ARIA-WebAgent/1.1)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return [{"url": "", "title": "Search failed", "content": "Search request failed", "score": 0.0}]

    results = []
    links   = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', html)
    snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', html, re.DOTALL)

    for i, (link_url, title) in enumerate(links[:max_results]):
        snippet = _clean_text(snippets[i]) if i < len(snippets) else ""
        results.append({
            "url": link_url, "title": title.strip(),
            "content": snippet, "score": 1.0 - (i * 0.1),
        })
    return results or [{"url": "", "title": "No results", "content": "Search returned no results", "score": 0.0}]


# ─── Domain Trust Store ───────────────────────────────────────────────────────

class DomainTrustStore:
    """
    Tracks reliability of web domains over time using Bayesian scoring.
    Same prior as ARIA's agent trust ledger: alpha=2, beta=1.
    Persists across sessions.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._scores: dict[str, dict] = {}
        self._persist_path = persist_path
        if persist_path:
            self._load()

    def _domain(self, url: str) -> str:
        try:
            return urllib.parse.urlparse(url).netloc.lower()
        except Exception:
            return "unknown"

    def record(self, url: str, success: bool, content_length: int = 0):
        domain = self._domain(url)
        if domain not in self._scores:
            self._scores[domain] = {"alpha": 2, "beta": 1, "total": 0}
        entry = self._scores[domain]
        entry["total"] += 1
        if success and content_length > 100:
            entry["alpha"] += 1
        else:
            entry["beta"] += 1
        if self._persist_path:
            self._save()

    def score(self, url: str) -> float:
        domain = self._domain(url)
        if domain not in self._scores:
            return 0.67
        e = self._scores[domain]
        return e["alpha"] / (e["alpha"] + e["beta"])

    def top_domains(self, n: int = 10) -> list[tuple[str, float]]:
        scored = [(d, self.score(f"https://{d}")) for d in self._scores]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:n]

    def _save(self):
        try:
            with open(self._persist_path, "w") as f:
                json.dump(self._scores, f, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            with open(self._persist_path) as f:
                self._scores = json.load(f)
        except Exception:
            self._scores = {}


# ─── ARIA Web Agent ───────────────────────────────────────────────────────────

class WebAgent(BaseAgent):
    """
    ARIA's native web browsing and search agent.

    Search engine priority:
        1. ARIA temporal memory  (zero network cost, instant)
        2. Claude native search  (when anthropic_api_key is set — best quality)
        3. DuckDuckGo lite       (zero-dependency fallback)

    Browse strategy:
        1. ARIA temporal memory  (if URL was browsed recently and is fresh)
        2. Playwright            (if installed — full JS rendering)
        3. requests / urllib     (always available)
    """

    def __init__(
        self,
        permissions,
        memory,
        anthropic_api_key:  Optional[str] = None,
        freshness_hours:    float = 2.0,
        max_content_chars:  int = 12000,
        domain_trust_path:  Optional[str] = None,
        max_claude_searches: int = 3,
    ):
        super().__init__(
            agent_id="aria_web_agent",
            capabilities=[
                "browse the web",
                "search the internet for live current information",
                "fetch content from a specific URL",
                "find today's news and latest articles",
                "web search for real-time data",
                "look up current prices or recent events online",
                "summarize a web page from a URL",
                "search for news today this week this year",
            ],
            domains=["web", "search", "browsing", "live-data", "current-events"],
            tools_available=["web_search", "browse_url", "cached_recall"],
            input_types=["text", "url"],
            output_types=["text", "structured"],
            max_concurrent_tasks=2,
            preferred_task_size="medium",
        )
        self._permissions        = permissions
        self._memory             = memory
        self._anthropic_key      = anthropic_api_key
        self._tavily_key         = None   # Set via set_tavily_key() or ARIA_TAVILY_KEY env var
        self._freshness_seconds  = freshness_hours * 3600
        self._max_content_chars  = max_content_chars
        self._domain_trust       = DomainTrustStore(domain_trust_path)
        self._max_claude_searches = max_claude_searches

    async def execute(self, node: TaskNode, context: dict, memory: object) -> Any:
        task    = node.description.strip()
        user_id = context.get("user_id", "default")

        # Detect formatting/writing tasks that should NOT trigger a web search.
        # When the decomposer sends a "format the results" instruction to the web
        # agent, we return the upstream search results directly so the system
        # can use them — rather than searching the web for the formatting words.
        format_signals = [
            "format", "summarise", "summarize", "present", "write up",
            "structure the", "compile", "organise", "organize",
            "retrieved news", "retrieved results", "retrieved stories",
            "create a briefing", "write a briefing", "make it readable",
            "friendly response", "warm tone", "collaborative tone",
        ]
        task_lower = task.lower()
        is_formatting = any(sig in task_lower for sig in format_signals)

        if is_formatting:
            upstream = context.get("upstream_results", "")
            if upstream:
                return (
                    "[Web Agent — formatting task, returning upstream results]\n\n"
                    + upstream
                )

        url_match = re.search(r"https?://[^\s\"'>]+", task)
        if url_match:
            return await self._execute_browse(url_match.group(0), task, user_id, node)
        return await self._execute_search(task, user_id, node)

    # ── Browse ────────────────────────────────────────────────────────────────

    async def _execute_browse(
        self, url: str, task: str, user_id: str, node: TaskNode
    ) -> str:
        from ..permissions.engine import Capability
        self._permissions.require(user_id, Capability.WEB_BROWSE)

        cached = self._recall_url(url, user_id)
        if cached:
            return self._format_browse_result(url, cached["content"], cached["fingerprint"], from_cache=True)

        start = time.monotonic()
        try:
            content, method = await _fetch_page(url)
            elapsed = time.monotonic() - start
            fp = _fingerprint(content)
            self._domain_trust.record(url, success=True, content_length=len(content))
            self._store_browse_result(url, content, fp, user_id, node)
            return self._format_browse_result(url, content, fp, from_cache=False, elapsed=elapsed, method=method)
        except Exception as ex:
            self._domain_trust.record(url, success=False)
            return (
                f"[ARIA Web Agent] Could not browse {url}\n"
                f"Error: {ex}\n"
                f"Domain trust: {self._domain_trust.score(url):.2f}"
            )

    # ── Search ────────────────────────────────────────────────────────────────

    async def _execute_search(self, query: str, user_id: str, node: TaskNode) -> str:
        from ..permissions.engine import Capability
        self._permissions.require(user_id, Capability.WEB_SEARCH)

        # For live/news queries — always go to network, never serve stale cache
        _live_keywords = [
            "news", "latest", "today", "current", "right now", "breaking",
            "this week", "this month", "2026", "2025", "recent", "update",
        ]
        _query_lower = query.lower()
        _is_live_query = any(kw in _query_lower for kw in _live_keywords)

        # Memory first — serve from cache only for non-live queries
        if not _is_live_query:
            recent = self._memory.recall(
                query=query,
                rings=[MemoryRing.HOT, MemoryRing.WARM],
                top_k=3,
                user_id=user_id,
            )
            if recent and recent[0].access_count > 0:
                top = recent[0]
                if "web_search" in top.tags and self._is_fresh(top.created_at):
                    return (
                        f"[ARIA Web Agent — from memory, no network call]\n\n"
                        f"{top.content}\n\n"
                        f"(Recalled from {top.ring.value} memory, "
                        f"searched {self._age_str(top.created_at)} ago)"
                    )

        start = time.monotonic()

        # Claude native search — primary engine
        # Read key fresh from environment at search time in case it was
        # set after the process started or not passed at init
        import os as _os
        active_key = self._anthropic_key or _os.environ.get("ANTHROPIC_API_KEY")
        if active_key:
            try:
                answer, cited_urls = await _search_claude(
                    query, active_key, self._max_claude_searches
                )
                elapsed = time.monotonic() - start

                # Score domains from Claude's cited sources
                for url in cited_urls:
                    self._domain_trust.record(url, success=True, content_length=500)

                # Format output with trust scores on citations
                url_lines = []
                for url in cited_urls:
                    trust = self._domain_trust.score(url)
                    url_lines.append(f"  {url}  [trust: {trust:.2f}]")

                formatted = (
                    f"[ARIA Web Agent — Claude search, {elapsed:.1f}s]\n"
                    f"Query: {query}\n"
                    f"Searches used: up to {self._max_claude_searches}\n\n"
                    f"{answer}"
                )
                if url_lines:
                    formatted += "\n\nSources cited:\n" + "\n".join(url_lines)

                # Store in HOT memory for fast recall
                self._memory.store(
                    content=formatted,
                    ring=MemoryRing.HOT,
                    tags=["web_search", f"query:{query[:80]}", "engine:claude"],
                    metadata={
                        "query": query,
                        "engine": "claude_native",
                        "cited_urls": cited_urls,
                        "searched_at": datetime.now(timezone.utc).isoformat(),
                    },
                    user_id=user_id,
                )
                return formatted

            except Exception as _claude_err:
                print(f"  [Web] Claude search failed ({type(_claude_err).__name__}: {_claude_err}) — trying Tavily")

        # Tavily fallback — reliable structured results, real-time
        import os as _os2
        tavily_key = self._tavily_key or _os2.environ.get("ARIA_TAVILY_KEY", "")
        if tavily_key:
            try:
                results = await _search_tavily(query, tavily_key)
                elapsed = time.monotonic() - start
                engine  = "Tavily"
                print(f"  [Web] Tavily search succeeded in {elapsed:.1f}s")
            except Exception as _tav_err:
                print(f"  [Web] Tavily failed ({type(_tav_err).__name__}: {_tav_err}) — falling back to DuckDuckGo")
                try:
                    results = await _search_duckduckgo(query)
                    elapsed = time.monotonic() - start
                    engine  = "DuckDuckGo"
                except Exception as ex:
                    return f"[ARIA Web Agent] All search engines failed: {ex}"
        else:
            # DuckDuckGo fallback — zero dependencies, no key needed
            try:
                results = await _search_duckduckgo(query)
                elapsed = time.monotonic() - start
                engine  = "DuckDuckGo"
            except Exception as ex:
                return f"[ARIA Web Agent] All search engines failed: {ex}"

        import os as _os
        lines = [
            f"[ARIA Web Agent — {engine} fallback, {elapsed:.1f}s]",
        ]
        if not (self._anthropic_key or _os.environ.get("ANTHROPIC_API_KEY")):
            lines.append("Note: Set ANTHROPIC_API_KEY for Claude-native search (better quality)")
        lines.append(f"Query: {query}\n")
        for i, r in enumerate(results, 1):
            url     = r.get("url", "")
            trust   = self._domain_trust.score(url) if url else 0.5
            snippet = r.get("content", "")[:500]
            lines.append(f"{i}. {r.get('title', 'No title')}")
            if url:
                lines.append(f"   {url}  [trust: {trust:.2f}]")
                self._domain_trust.record(url, success=bool(snippet), content_length=len(snippet))
            lines.append(f"   {snippet}\n")

        formatted = "\n".join(lines)
        self._memory.store(
            content=formatted,
            ring=MemoryRing.HOT,
            tags=["web_search", f"query:{query[:80]}", f"engine:{engine}"],
            metadata={"query": query, "engine": engine,
                      "searched_at": datetime.now(timezone.utc).isoformat()},
            user_id=user_id,
        )
        return formatted

    # ── Memory Helpers ────────────────────────────────────────────────────────

    def _recall_url(self, url: str, user_id: str) -> Optional[dict]:
        results = self._memory.recall(
            query=url,
            rings=[MemoryRing.WARM, MemoryRing.COLD],
            top_k=5,
            user_id=user_id,
        )
        for entry in results:
            meta = entry.metadata or {}
            if meta.get("source_url") == url and self._is_fresh(entry.created_at):
                return {"content": entry.content, "fingerprint": meta.get("fingerprint", "")}
        return None

    def _store_browse_result(
        self, url: str, content: str, fingerprint: str, user_id: str, node: TaskNode
    ):
        truncated = content[:self._max_content_chars]
        metadata  = {
            "source_url": url, "fingerprint": fingerprint,
            "browsed_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(content),
        }
        self._memory.store(
            content=truncated, ring=MemoryRing.WARM,
            tags=["web_browse", f"url:{url[:120]}"],
            metadata=metadata, user_id=user_id,
        )
        if len(content) > 2000:
            summary = content[:500] + "... [see WARM memory for full content]"
            self._memory.store(
                content=summary, ring=MemoryRing.HOT,
                tags=["web_browse", "summary", f"url:{url[:120]}"],
                metadata={**metadata, "is_summary": True},
                user_id=user_id,
            )

    def _is_fresh(self, created_at: float) -> bool:
        return (time.time() - created_at) < self._freshness_seconds

    def _age_str(self, created_at: float) -> str:
        age = time.time() - created_at
        if age < 60:
            return f"{age:.0f} seconds"
        if age < 3600:
            return f"{age/60:.0f} minutes"
        return f"{age/3600:.1f} hours"

    def _format_browse_result(
        self, url: str, content: str, fingerprint: str,
        from_cache: bool = False, elapsed: float = 0.0, method: str = "",
    ) -> str:
        trust  = self._domain_trust.score(url)
        source = "from memory cache" if from_cache else f"fetched via {method} in {elapsed:.1f}s"
        header = [
            f"[ARIA Web Agent — {source}]",
            f"URL: {url}",
            f"Domain trust: {trust:.2f}  |  Fingerprint: {fingerprint[:12]}...",
            f"Content: {len(content):,} chars\n",
        ]
        return "\n".join(header) + content

    def trust_report(self) -> str:
        top = self._domain_trust.top_domains(n=15)
        if not top:
            return "[ARIA Web Agent] No domain trust data yet."
        lines = ["[ARIA Web Agent — Domain Trust Report]", ""]
        for domain, score in top:
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {bar}  {score:.2f}  {domain}")
        return "\n".join(lines)


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_web_agent(
    permissions,
    memory,
    anthropic_api_key:   Optional[str] = None,
    freshness_hours:     float = 2.0,
    domain_trust_path:   Optional[str] = None,
    max_claude_searches: int = 3,
) -> WebAgent:
    """
    Create a configured ARIA WebAgent.

    Args:
        permissions:          ARIA PermissionEngine instance
        memory:               ARIA TemporalMemory instance
        anthropic_api_key:    Anthropic API key for Claude-native search.
                              If None, falls back to DuckDuckGo lite.
        freshness_hours:      Serve from memory cache if browsed within this window.
        domain_trust_path:    File path for persistent domain trust scores.
        max_claude_searches:  Max web searches Claude can run per query ($10/1000).

    Example:
        agent = create_web_agent(
            permissions=aria.permissions,
            memory=aria.memory,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            freshness_hours=1.0,
            domain_trust_path=".aria/domain_trust.json",
        )
        aria.register_agent(agent)
    """
    return WebAgent(
        permissions=permissions,
        memory=memory,
        anthropic_api_key=anthropic_api_key,
        freshness_hours=freshness_hours,
        domain_trust_path=domain_trust_path,
        max_claude_searches=max_claude_searches,
    )
