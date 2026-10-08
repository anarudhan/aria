"""
Tests for the spill store and its wiring into LLMAgent (issue #40).

Every test works inside pytest's tmp_path; nothing touches the repo tree or the
home folder. Offsets and lengths are CHARACTERS throughout.
"""

import errno
import os
import re
import time

import pytest

from aria.agents.base import LLMAgent
from aria.core import spill as spill_module
from aria.core.models import TaskNode
from aria.core.spill import (
    READ_TOOL_NAME,
    SpillError,
    SpillNotFound,
    SpillStore,
    get_default_store,
    reset_default_store,
)

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")


@pytest.fixture
def store(tmp_path):
    return SpillStore(tmp_path / "spill")


@pytest.fixture(autouse=True)
def _isolate_default_store(tmp_path, monkeypatch):
    """No test may create the default store in the real temp area or home."""
    monkeypatch.setenv("ARIA_SPILL_DIR", str(tmp_path / "default-spill"))
    reset_default_store()
    yield
    reset_default_store()


def spill_files(store):
    return sorted(p for p in os.listdir(store.base_dir) if p.endswith(".spill"))


# ─── Small text, large text ──────────────────────────────────────────────────

class TestPreview:
    def test_small_text_unchanged(self, store):
        r = store.preview_or_spill("hello", 100)
        assert r.text == "hello" and str(r) == "hello"
        assert r.omitted_chars == 0 and not r.spilled and not r.truncated
        assert r.locator is None and r.warning is None
        assert spill_files(store) == []

    def test_exact_fit_unchanged_and_one_over_spills(self, store):
        assert store.preview_or_spill("x" * 50, 50).spilled is False
        r = store.preview_or_spill("x" * 51, 50)
        assert r.spilled and r.omitted_chars == 1

    def test_empty_text(self, store):
        assert store.preview_or_spill("", 0).text == ""

    def test_negative_max_rejected(self, store):
        with pytest.raises(ValueError):
            store.preview_or_spill("abc", -1)

    def test_large_text_counts_head_tail_marker(self, store):
        text = "".join(chr(ord("a") + i % 26) for i in range(5000))
        r = store.preview_or_spill(text, 300)
        assert r.spilled and r.truncated
        assert r.total_chars == 5000
        assert r.omitted_chars == 5000 - 300
        head_n, tail_n = 200, 100
        assert r.text.startswith(text[:head_n])
        assert r.text.endswith(text[-tail_n:])
        # The marker states the exact number omitted, the locator and how to read more.
        assert f"{r.omitted_chars} characters omitted" in r.text
        assert "SHORTENED" in r.text
        assert r.locator is not None and r.locator in r.text
        assert f'{READ_TOOL_NAME}(locator="{r.locator}", offset={head_n}' in r.text
        assert "count characters" in r.text
        # What is shown plus what is omitted is the whole text.
        assert head_n + tail_n + r.omitted_chars == 5000

    def test_head_and_tail_join_with_omitted_middle_to_original(self, store):
        text = "".join(str(i % 10) for i in range(1000))
        r = store.preview_or_spill(text, 90)
        middle = store.read(r.locator, offset=60, length=r.omitted_chars)
        assert text[:60] + middle + text[-30:] == text

    def test_zero_max_chars_omits_everything_but_saves_it(self, store):
        r = store.preview_or_spill("secret text", 0)
        assert r.omitted_chars == 11
        assert store.read(r.locator, 0, 100) == "secret text"

    def test_each_spill_gets_its_own_locator_and_file(self, store):
        a = store.preview_or_spill("a" * 100, 10)
        b = store.preview_or_spill("a" * 100, 10)
        assert a.locator != b.locator
        assert len(spill_files(store)) == 2

    def test_locator_is_opaque_hex_not_a_path(self, store):
        r = store.preview_or_spill("z" * 100, 10)
        assert re.fullmatch(r"[0-9a-f]{32}", r.locator)
        assert str(store.base_dir) not in r.text


# ─── Reading back ────────────────────────────────────────────────────────────

class TestRead:
    def test_round_trip_with_offset_and_length(self, store):
        text = "0123456789" * 100
        loc = store.preview_or_spill(text, 20).locator
        assert store.read(loc, 0, 1000) == text
        assert store.read(loc, 15, 10) == text[15:25]
        assert store.read(loc, 995, 100) == text[995:]
        assert store.read(loc, 5000, 10) == ""

    def test_pages_cover_everything(self, store):
        text = "".join(chr(65 + i % 26) for i in range(2345))
        loc = store.preview_or_spill(text, 50).locator
        out, off = [], 0
        while True:
            page = store.read_page(loc, off, 400)
            out.append(page.text)
            assert page.total_chars == 2345
            if not page.has_more:
                break
            off = page.end
        assert "".join(out) == text

    def test_length_is_clamped_and_reported(self, tmp_path):
        s = SpillStore(tmp_path / "s", max_read_chars=100)
        loc = s.preview_or_spill("q" * 1000, 10).locator
        page = s.read_page(loc, 0, 10_000)
        assert len(page.text) == 100 and page.end == 100 and page.has_more

    @pytest.mark.parametrize("offset,length", [(-1, 10), (0, 0), (0, -5), ("1", 5), (0, 2.5), (True, 5)])
    def test_bad_numbers_rejected(self, store, offset, length):
        loc = store.preview_or_spill("x" * 100, 10).locator
        with pytest.raises(SpillError):
            store.read(loc, offset, length)

    def test_unicode_multibyte_offsets_are_characters(self, store):
        text = ("héllo wörld 日本語テキスト 😀🎉 ") * 40
        assert len(text.encode("utf-8")) > len(text)
        r = store.preview_or_spill(text, 99)
        head_n = 66
        assert r.text.startswith(text[:head_n])
        assert r.text.endswith(text[-33:])
        assert r.omitted_chars == len(text) - 99
        assert store.read(r.locator, head_n, 25) == text[head_n:head_n + 25]
        assert store.read(r.locator, 0, 10_000) == text

    def test_lone_surrogate_and_crlf_survive(self, store):
        text = "a\r\nb\ud800c\r\n" * 20
        loc = store.preview_or_spill(text, 10).locator
        assert store.read(loc, 0, 10_000) == text

    def test_describe_bytes_are_utf8_bytes(self, store):
        text = "日" * 100
        store.preview_or_spill(text, 10)
        d = store.describe()
        assert d["files"] == 1 and d["bytes"] == 300


# ─── Locator safety ──────────────────────────────────────────────────────────

class TestLocatorSafety:
    @pytest.mark.parametrize("bad", [
        "../x",
        "../../etc/passwd",
        "..",
        "/etc/passwd",
        "C:\\Windows\\win.ini",
        "",
        "a" * 100_000,
        "ab\x00cd",
        "0" * 32 + "\x00",
        "0" * 32 + "\n",
        "A" * 32,                  # uppercase hex is not what we issue
        "g" * 32,
        "0" * 31,
        "0" * 33,
        "0" * 32 + ".spill",
        None,
        123,
        b"0" * 32,
    ])
    def test_invalid_ids_rejected(self, store, bad):
        store.preview_or_spill("x" * 100, 10)
        with pytest.raises(SpillError):
            store.read(bad)

    def test_well_formed_but_unknown_id_rejected(self, store):
        with pytest.raises(SpillNotFound):
            store.read("0" * 32)

    def test_foreign_locator_rejected(self, tmp_path):
        a = SpillStore(tmp_path / "a")
        b = SpillStore(tmp_path / "b")
        loc = a.preview_or_spill("x" * 100, 10).locator
        with pytest.raises(SpillNotFound):
            b.read(loc)

    def test_foreign_file_in_same_folder_not_readable(self, store):
        """A file someone else dropped in the folder is not an id we issued."""
        planted = "1" * 32
        (store.base_dir / (planted + ".spill")).write_text("planted")
        with pytest.raises(SpillNotFound):
            store.read(planted)

    def test_second_store_on_same_folder_cannot_read_first_stores_ids(self, tmp_path):
        a = SpillStore(tmp_path / "s")
        b = SpillStore(tmp_path / "s")
        loc = a.preview_or_spill("x" * 100, 10).locator
        with pytest.raises(SpillNotFound):
            b.read(loc)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_symlink_swapped_in_is_refused(self, store, tmp_path):
        loc = store.preview_or_spill("x" * 100, 10).locator
        outside = tmp_path / "outside.txt"
        outside.write_text("do not read me")
        path = store.base_dir / (loc + ".spill")
        path.unlink()
        path.symlink_to(outside)
        with pytest.raises(SpillError):
            store.read(loc)

    @pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
    def test_symlinked_base_dir_is_not_used(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        s = SpillStore(link)
        assert not s.usable
        assert os.listdir(real) == []


# ─── Permissions and exclusive creation ──────────────────────────────────────

class TestFilesystem:
    @posix_only
    def test_folder_is_0700_and_files_0600(self, tmp_path):
        old = os.umask(0)           # a permissive umask must not weaken the modes
        try:
            s = SpillStore(tmp_path / "private")
            loc = s.preview_or_spill("x" * 100, 10).locator
        finally:
            os.umask(old)
        assert (os.stat(s.base_dir).st_mode & 0o777) == 0o700
        assert (os.stat(s.base_dir / (loc + ".spill")).st_mode & 0o777) == 0o600

    @posix_only
    def test_existing_loose_folder_is_tightened(self, tmp_path):
        d = tmp_path / "loose"
        d.mkdir(mode=0o755)
        os.chmod(d, 0o755)
        SpillStore(d)
        assert (os.stat(d).st_mode & 0o777) == 0o700

    def test_exclusive_creation_never_overwrites(self, store, monkeypatch):
        victim = "7" * 32
        path = store.base_dir / (victim + ".spill")
        path.write_text("precious")
        ids = iter([victim, victim, "8" * 32])
        monkeypatch.setattr(spill_module.secrets, "token_hex", lambda n: next(ids))
        r = store.preview_or_spill("n" * 100, 10)
        assert path.read_text() == "precious"       # untouched
        assert r.locator == "8" * 32                # retried with a fresh id
        assert store.read(r.locator, 0, 200) == "n" * 100

    def test_all_ids_colliding_keeps_text_inline(self, store, monkeypatch):
        victim = "7" * 32
        path = store.base_dir / (victim + ".spill")
        path.write_text("precious")
        monkeypatch.setattr(spill_module.secrets, "token_hex", lambda n: victim)
        text = "n" * 100
        r = store.preview_or_spill(text, 10)
        assert path.read_text() == "precious"
        assert r.text == text and not r.spilled and r.omitted_chars == 0
        assert store.warnings


# ─── Failure keeps text inline ───────────────────────────────────────────────

class TestSaveFailure:
    def test_disk_full_keeps_full_text_and_warns(self, store, monkeypatch):
        def boom(*a, **k):
            raise OSError(errno.ENOSPC, "No space left on device")

        text = "important " * 200
        monkeypatch.setattr(spill_module.os, "open", boom)
        r = store.preview_or_spill(text, 50)
        monkeypatch.undo()
        assert r.text == text                        # nothing dropped
        assert r.omitted_chars == 0 and not r.spilled and r.locator is None
        assert r.warning and "inline" in r.warning
        assert len(store.warnings) == 1 and "No space left" in store.warnings[0]
        assert store.describe()["warnings"] == 1
        assert spill_files(store) == []

    def test_failure_after_file_created_removes_partial_file(self, store, monkeypatch):
        if not hasattr(os, "fchmod"):
            pytest.skip("needs fchmod")

        def boom(fd, mode):
            raise PermissionError("nope")

        monkeypatch.setattr(spill_module.os, "fchmod", boom)
        text = "y" * 500
        r = store.preview_or_spill(text, 10)
        monkeypatch.undo()
        assert r.text == text and store.warnings
        assert spill_files(store) == []

    def test_unusable_folder_keeps_text_inline(self, tmp_path):
        blocker = tmp_path / "iam_a_file"
        blocker.write_text("x")
        s = SpillStore(blocker)
        assert not s.usable and s.warnings
        text = "w" * 300
        r = s.preview_or_spill(text, 10)
        assert r.text == text and r.omitted_chars == 0

    def test_failure_does_not_count_as_spill(self, store, monkeypatch):
        monkeypatch.setattr(spill_module.os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
        store.preview_or_spill("a" * 100, 10)
        monkeypatch.undo()
        assert store.describe()["spills_this_session"] == 0

    def test_warnings_list_is_bounded(self, store):
        for i in range(250):
            store._warn(f"w{i}")
        assert len(store.warnings) == 100
        assert store.warnings[-1] == "w249"
        assert store.warnings_total == 250


# ─── Retention ───────────────────────────────────────────────────────────────

class TestRetention:
    def _age(self, store, locator, seconds):
        p = store.base_dir / (locator + ".spill")
        t = time.time() - seconds
        os.utime(p, (t, t))

    def test_cleanup_removes_only_old_files(self, tmp_path):
        s = SpillStore(tmp_path / "s", retention_seconds=3600)
        old = s.preview_or_spill("o" * 100, 10).locator
        new = s.preview_or_spill("n" * 100, 10).locator
        self._age(s, old, 7200)
        assert s.cleanup() == 1
        assert spill_files(s) == [new + ".spill"]
        with pytest.raises(SpillNotFound):
            s.read(old)
        assert s.read(new, 0, 5) == "nnnnn"

    def test_cleanup_with_explicit_age(self, store):
        a = store.preview_or_spill("a" * 100, 10).locator
        self._age(store, a, 120)
        assert store.cleanup(max_age_seconds=3600) == 0
        assert store.cleanup(max_age_seconds=60) == 1

    def test_cleanup_never_touches_files_it_did_not_create(self, store):
        other = store.base_dir / "notes.txt"
        other.write_text("keep")
        odd = store.base_dir / ("Z" * 32 + ".spill")
        odd.write_text("keep")
        old = time.time() - 10**6
        os.utime(other, (old, old))
        os.utime(odd, (old, old))
        assert store.cleanup(0) == 0
        assert other.exists() and odd.exists()

    def test_clear_all(self, store):
        for _ in range(3):
            store.preview_or_spill("x" * 100, 10)
        (store.base_dir / "keep.txt").write_text("k")
        assert store.clear_all() == 3
        assert spill_files(store) == []
        assert (store.base_dir / "keep.txt").exists()
        assert store.describe()["files"] == 0

    def test_negative_arguments_rejected(self, store, tmp_path):
        with pytest.raises(ValueError):
            store.cleanup(-1)
        with pytest.raises(ValueError):
            SpillStore(tmp_path / "x", retention_seconds=-1)

    def test_describe(self, store):
        store.preview_or_spill("a" * 100, 10)
        store.preview_or_spill("b" * 200, 10)
        d = store.describe()
        assert d["files"] == 2 and d["bytes"] == 300
        assert d["spills_this_session"] == 2
        assert d["characters_omitted_from_previews"] == 90 + 190
        assert d["usable"] is True and d["warnings"] == 0


# ─── Default store ───────────────────────────────────────────────────────────

class TestDefaultStore:
    def test_default_store_uses_env_folder_and_is_shared(self, tmp_path):
        s = get_default_store()
        assert s.base_dir == tmp_path / "default-spill"
        assert get_default_store() is s

    def test_default_folder_is_outside_the_repo(self, monkeypatch):
        monkeypatch.delenv("ARIA_SPILL_DIR")
        d = spill_module._default_dir()
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert not str(d).startswith(repo + os.sep)


# ─── Wiring into LLMAgent ────────────────────────────────────────────────────

class ScriptedLLM:
    """Plays back canned responses and records the messages it was shown."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None, max_tokens=0):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)


def make_agent(store, llm, tools=None):
    return LLMAgent(
        agent_id="spiller", capabilities=["x"], domains=["d"],
        tools=tools, llm=llm, spill_store=store,
    )


class TestAgentWiring:
    async def test_small_dependency_output_is_unchanged(self, store):
        llm = ScriptedLLM([{"content": "FINAL_ANSWER: ok"}])
        agent = make_agent(store, llm)
        await agent.execute(TaskNode(description="t"), {"n1": "short result"}, None)
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "[n1]: short result" in prompt
        assert "omitted" not in prompt
        assert spill_files(store) == []
        assert READ_TOOL_NAME not in agent._tools

    async def test_large_dependency_output_is_previewed_honestly(self, store):
        big = "S" + "m" * 5000 + "E"
        llm = ScriptedLLM([{"content": "FINAL_ANSWER: ok"}])
        agent = make_agent(store, llm)
        await agent.execute(TaskNode(description="t"), {"n1": big}, None)
        prompt = llm.calls[0]["messages"][0]["content"]
        assert "[n1]: S" in prompt
        assert f"{len(big) - 1000} characters omitted" in prompt
        assert "SHORTENED" in prompt
        assert len(spill_files(store)) == 1
        # The model is offered the read tool for this very call.
        assert READ_TOOL_NAME in prompt
        assert READ_TOOL_NAME in [t["name"] for t in llm.calls[0]["tools"]]
        # The advertised fingerprint is not changed by the helper tool.
        assert READ_TOOL_NAME not in agent.fingerprint.tools_available

    async def test_tool_result_spill_and_model_reads_the_rest(self, store):
        payload = "".join(f"line {i:05d}\n" for i in range(600))   # 6600 chars

        def big_tool():
            """Return a lot of text."""
            return payload

        class ReadingLLM(ScriptedLLM):
            """Turn 1 calls the tool; turn 2 reads the spill id from what it was shown."""

            async def chat(self, messages, tools=None, max_tokens=0):
                n = len(self.calls)
                self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
                if n == 0:
                    return {"content": "", "tool_calls": [{"name": "big_tool", "arguments": {}}]}
                if n == 1:
                    loc = re.search(r"spill id ([0-9a-f]{32})", messages[-1]["content"]).group(1)
                    args = {"locator": loc, "offset": 667, "length": 50}
                    return {"content": "", "tool_calls": [{"name": READ_TOOL_NAME, "arguments": args}]}
                return {"content": "FINAL_ANSWER: done"}

        llm = ReadingLLM([])
        agent = make_agent(store, llm, tools=[big_tool])
        assert await agent.execute(TaskNode(description="t"), {}, None) == "done"

        # The tool message is str(list of dicts), so newlines show escaped; that is
        # existing behaviour. Undo it here to compare the text itself.
        shown = llm.calls[1]["messages"][-1]["content"].replace("\\n", "\n")
        assert "5600 characters omitted" in shown
        assert payload[:667] in shown and payload[-333:] in shown
        assert READ_TOOL_NAME in [t["name"] for t in llm.calls[1]["tools"]]
        read_back = llm.calls[2]["messages"][-1]["content"]      # tool message, turn 3
        assert payload[667:717] in read_back.replace("\\n", "\n")
        assert "characters 667 to 717 of 6600" in read_back

    async def test_read_spilled_pages_on(self, store):
        agent = make_agent(store, ScriptedLLM([]))
        loc = store.preview_or_spill("0123456789" * 10, 10).locator
        res = agent.read_spilled(loc, 10, 20)
        assert res.startswith("0123456789" * 2)
        assert "characters 10 to 30 of 100" in res
        assert f'{READ_TOOL_NAME}(locator="{loc}", offset=30)' in res

    async def test_read_spilled_reaches_end_and_reports_it(self, store):
        agent = make_agent(store, ScriptedLLM([]))
        loc = store.preview_or_spill("abcdefghij", 3).locator
        res = agent.read_spilled(loc, 5, 100)
        assert res.startswith("fghij") and "end of text" in res

    async def test_read_spilled_returns_error_text_for_bad_locator(self, store):
        agent = make_agent(store, ScriptedLLM([]))
        assert agent.read_spilled("../etc/passwd").startswith("Error:")
        assert agent.read_spilled("0" * 32).startswith("Error:")

    async def test_small_tool_result_and_errors_unchanged(self, store):
        def small():
            return "tiny"

        def broken():
            raise RuntimeError("kaput")

        async def asmall():
            return "async tiny"

        agent = make_agent(store, ScriptedLLM([]), tools=[small, broken, asmall])
        out = await agent._execute_tools([
            {"name": "small", "arguments": {}},
            {"name": "broken", "arguments": {}},
            {"name": "asmall", "arguments": {}},
            {"name": "ghost", "arguments": {}},
        ])
        by = {r["tool"]: r for r in out}
        assert by["small"] == {"tool": "small", "result": "tiny"}
        assert by["asmall"] == {"tool": "asmall", "result": "async tiny"}
        assert by["broken"] == {"tool": "broken", "error": "kaput"}
        assert "not available" in by["ghost"]["error"]
        assert spill_files(store) == []

    async def test_large_async_tool_result_spills(self, store):
        async def abig():
            return "A" * 4000

        agent = make_agent(store, ScriptedLLM([]), tools=[abig])
        out = await agent._execute_tools([{"name": "abig", "arguments": {}}])
        assert "3000 characters omitted" in out[0]["result"]
        assert len(spill_files(store)) == 1

    async def test_failing_save_keeps_whole_tool_result(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        broken_store = SpillStore(blocker)

        def big():
            return "B" * 4000

        agent = make_agent(broken_store, ScriptedLLM([]), tools=[big])
        out = await agent._execute_tools([{"name": "big", "arguments": {}}])
        assert out[0]["result"] == "B" * 4000
        assert broken_store.warnings
        assert READ_TOOL_NAME not in agent._tools

    async def test_agent_without_store_uses_default_in_configured_folder(self, tmp_path):
        agent = LLMAgent(agent_id="a", capabilities=["x"], domains=["d"], llm=ScriptedLLM([]))
        assert agent.spill_store is get_default_store()
        assert agent.spill_store.base_dir == tmp_path / "default-spill"


def test_cut_marks_truncation_with_count():
    from aria.agents.base import _cut

    assert _cut("abc", 5) == "abc"
    out = _cut("x" * 120, 100)
    assert out.startswith("x" * 100) and "20 more characters not shown" in out
