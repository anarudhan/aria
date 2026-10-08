"""
Tests for the hardened audit chain (issue #56, part of #42).

Covers: a full SHA256 over every field, verify_chain() against every kind of
tampering, the public record_audit() API, thread safety, and backward
compatibility of _audit_log() and get_audit_trail().

The honest limit is tested too: someone who rewrites the entries AND the
engine's head hash in memory can make verify_chain() pass.
"""
import copy
import dataclasses
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from aria.permissions.engine import (
    MAX_AUDIT_DETAIL_BYTES,
    AuditEntry,
    Capability,
    ChainReport,
    PermissionDeniedError,
    PermissionEngine,
)

ALL_FIELDS = [f.name for f in dataclasses.fields(AuditEntry)]
HASHED_FIELDS = [n for n in ALL_FIELDS if n != "entry_hash"]


def _full_entry() -> AuditEntry:
    """An entry with every field set to a distinct, non-default value."""
    return AuditEntry(
        entry_id="id-1", timestamp=1700000000.5, user_id="alice",
        capability="file_read", scope="/home/alice", action="allowed",
        reason="because", agent_id="agent-7", task_id="task-9",
        prev_hash="genesis", call_id="call-3", stage="tool.policy",
        detail='{"k":"v"}', args_digest="abc123",
    )


def _mixed_engine() -> PermissionEngine:
    """An engine whose chain holds every kind of entry."""
    e = PermissionEngine()
    e.grant("alice", Capability.FILE_READ, scope="/tmp", hours=1)            # 0 granted
    e.check("alice", Capability.FILE_READ, "/tmp/x", "agent-1", "task-1")    # 1 allowed
    e.check("alice", Capability.EMAIL_SEND, "*", "agent-2", "task-2")        # 2 denied
    e.record_audit(                                                          # 3
        "alice", "tool", "/tmp/x", "tool.policy", reason="ok", agent_id="agent-1",
        task_id="task-1", call_id="call-1", stage="policy",
        detail={"b": 2, "a": [1, 2]}, args_digest="deadbeef",
    )
    e.revoke("alice", Capability.FILE_READ, "/tmp")                          # 4 revoked
    e.record_audit("bob", "tool", "*", "tool.end", call_id="call-2")         # 5
    return e


def _mutate(entry: AuditEntry, name: str) -> None:
    value = getattr(entry, name)
    if isinstance(value, float):
        setattr(entry, name, value + 1.0)
    else:
        setattr(entry, name, value + "x")


# ─── Hash properties ──────────────────────────────────────────────────────────

def test_hash_is_full_sha256_hex():
    h = _full_entry().compute_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_every_stored_hash_is_64_chars():
    e = _mixed_engine()
    assert len(e._audit) == 6
    for entry in e._audit:
        assert len(entry.entry_hash) == 64
        assert len(entry.prev_hash) in (7, 64)  # "genesis" or a full hash
    assert len(e._last_hash) == 64


def test_hash_is_deterministic():
    a, b = _full_entry(), _full_entry()
    assert a.compute_hash() == b.compute_hash() == a.compute_hash()


def test_hash_matches_documented_encoding():
    """Independent re-computation: canonical JSON of all fields but entry_hash."""
    entry = _full_entry()
    data = {n: getattr(entry, n) for n in HASHED_FIELDS}
    text = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    assert entry.compute_hash() == hashlib.sha256(text.encode("ascii")).hexdigest()


@pytest.mark.parametrize("name", HASHED_FIELDS)
def test_every_field_changes_the_hash(name):
    """
    Iterates dataclasses.fields(AuditEntry): a field added later and left out
    of compute_hash() makes this test fail.
    """
    entry = _full_entry()
    before = entry.compute_hash()
    _mutate(entry, name)
    assert entry.compute_hash() != before, f"field {name!r} is not covered by compute_hash()"


def test_stored_hash_is_the_only_field_not_hashed():
    entry = _full_entry()
    before = entry.compute_hash()
    entry.entry_hash = "anything at all"
    assert entry.compute_hash() == before
    # And the parametrised test above really saw every other field.
    assert set(HASHED_FIELDS) | {"entry_hash"} == {f.name for f in dataclasses.fields(AuditEntry)}
    assert "scope" in HASHED_FIELDS and "detail" in HASHED_FIELDS and "call_id" in HASHED_FIELDS


def test_fields_cannot_bleed_into_each_other():
    """Concatenation would hash these the same; canonical JSON must not."""
    a = _full_entry()
    b = _full_entry()
    a.user_id, a.capability = "a", "bc"
    b.user_id, b.capability = "ab", "c"
    assert a.compute_hash() != b.compute_hash()
    c, d = _full_entry(), _full_entry()
    c.scope, c.reason = "x", ""
    d.scope, d.reason = "", "x"
    assert c.compute_hash() != d.compute_hash()
    # Quote and brace characters cannot forge a boundary either.
    e1, e2 = _full_entry(), _full_entry()
    e1.reason, e1.agent_id = 'a","agent_id":"b', ""
    e2.reason, e2.agent_id = "a", "b"
    assert e1.compute_hash() != e2.compute_hash()


def test_type_confusion_changes_the_hash():
    a, b = _full_entry(), _full_entry()
    a.timestamp = 5.0
    b.timestamp = "5.0"  # type: ignore[assignment]
    assert a.compute_hash() != b.compute_hash()


# ─── verify_chain basics ─────────────────────────────────────────────────────

def test_empty_chain_is_ok():
    report = PermissionEngine().verify_chain()
    assert report == ChainReport(ok=True, checked=0, first_bad_index=None, reason="")


def test_intact_chain_is_ok():
    e = _mixed_engine()
    report = e.verify_chain()
    assert report.ok is True
    assert report.checked == 6
    assert report.first_bad_index is None
    assert report.reason == ""


def test_report_is_frozen():
    report = PermissionEngine().verify_chain()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.ok = False  # type: ignore[misc]


def test_chain_links_and_head():
    e = _mixed_engine()
    assert e._audit[0].prev_hash == "genesis"
    for prev, cur in zip(e._audit, e._audit[1:]):
        assert cur.prev_hash == prev.entry_hash
    assert e._last_hash == e._audit[-1].entry_hash


def test_verify_does_not_change_anything():
    e = _mixed_engine()
    snapshot = copy.deepcopy(e._audit)
    head = e._last_hash
    e.verify_chain()
    assert e._audit == snapshot
    assert e._last_hash == head


# ─── Tampering: edit any field of any entry ──────────────────────────────────

@pytest.mark.parametrize("name", ALL_FIELDS)
def test_editing_any_field_of_any_entry_is_detected(name):
    e = _mixed_engine()
    for i in range(len(e._audit)):
        entry = e._audit[i]
        original = getattr(entry, name)
        _mutate(entry, name)
        report = e.verify_chain()
        assert report.ok is False, f"edit of {name} on entry {i} not detected"
        assert report.first_bad_index == i, (name, i, report)
        assert report.checked == i
        assert str(i) in report.reason
        setattr(entry, name, original)
        assert e.verify_chain().ok is True


def test_edit_with_recomputed_entry_hash_is_caught_at_next_entry():
    e = _mixed_engine()
    target = e._audit[2]
    target.scope = "/etc/shadow"
    target.entry_hash = target.compute_hash()
    report = e.verify_chain()
    assert not report.ok
    assert report.first_bad_index == 3


def test_edit_with_recomputed_hash_of_last_entry_is_caught_by_head():
    e = _mixed_engine()
    last = e._audit[-1]
    last.reason = "rewritten"
    last.entry_hash = last.compute_hash()
    report = e.verify_chain()
    assert not report.ok
    assert report.first_bad_index == len(e._audit) - 1
    assert "head" in report.reason


def test_edit_that_changes_type_is_detected():
    e = _mixed_engine()
    e._audit[1].timestamp = str(e._audit[1].timestamp)  # type: ignore[assignment]
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 1


def test_unhashable_field_is_reported_not_raised():
    e = _mixed_engine()
    e._audit[2].scope = {1, 2, 3}  # type: ignore[assignment]  # a set is not JSON
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 2
    assert "cannot be hashed" in report.reason


def test_nan_timestamp_is_reported_not_raised():
    e = _mixed_engine()
    e._audit[2].timestamp = float("nan")
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 2


def test_non_entry_object_in_list_is_detected():
    e = _mixed_engine()
    e._audit[3] = {"fake": "entry"}  # type: ignore[call-overload]
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 3


# ─── Tampering: structure ────────────────────────────────────────────────────

def test_swapping_two_entries_is_detected():
    e = _mixed_engine()
    e._audit[1], e._audit[2] = e._audit[2], e._audit[1]
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 1


def test_swapping_first_and_last_is_detected():
    e = _mixed_engine()
    e._audit[0], e._audit[-1] = e._audit[-1], e._audit[0]
    assert not e.verify_chain().ok


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_removing_a_non_last_entry_is_detected(index):
    e = _mixed_engine()
    del e._audit[index]
    report = e.verify_chain()
    assert not report.ok
    assert report.first_bad_index == index


def test_removing_the_last_entry_is_detected():
    e = _mixed_engine()
    del e._audit[-1]
    report = e.verify_chain()
    assert not report.ok
    assert report.first_bad_index == len(e._audit) - 1
    assert "head" in report.reason


def test_removing_several_entries_from_the_end_is_detected():
    e = _mixed_engine()
    del e._audit[3:]
    assert not e.verify_chain().ok


def test_removing_the_only_entry_is_detected():
    e = PermissionEngine()
    e.record_audit("u", "c", "s", "a")
    assert e.verify_chain().ok
    e._audit.clear()
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 0


def test_removing_everything_is_detected():
    e = _mixed_engine()
    e._audit.clear()
    assert not e.verify_chain().ok


def test_duplicating_an_entry_object_is_detected():
    e = _mixed_engine()
    e._audit.insert(3, e._audit[2])
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 3


def test_duplicating_an_entry_copy_is_detected():
    e = _mixed_engine()
    e._audit.insert(3, copy.copy(e._audit[2]))
    assert not e.verify_chain().ok


def test_duplicating_the_last_entry_is_detected():
    e = _mixed_engine()
    e._audit.append(copy.copy(e._audit[-1]))
    assert not e.verify_chain().ok


def test_inserting_a_forged_entry_in_the_middle_is_detected():
    e = _mixed_engine()
    forged = AuditEntry(
        user_id="alice", capability="file_read", scope="*", action="allowed",
        prev_hash=e._audit[1].entry_hash,       # correct link back
    )
    forged.entry_hash = forged.compute_hash()   # self-consistent
    e._audit.insert(2, forged)
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 3  # old entry 2 no longer links


def test_appending_a_forged_entry_without_moving_the_head_is_detected():
    e = _mixed_engine()
    forged = AuditEntry(user_id="alice", action="allowed", prev_hash=e._last_hash)
    forged.entry_hash = forged.compute_hash()
    e._audit.append(forged)
    report = e.verify_chain()
    assert not report.ok
    assert "head" in report.reason


def test_replacing_the_head_alone_is_detected():
    e = _mixed_engine()
    e._last_hash = "0" * 64
    assert not e.verify_chain().ok


def test_wrong_genesis_is_detected():
    e = _mixed_engine()
    e._audit[0].prev_hash = "not-genesis"
    e._audit[0].entry_hash = e._audit[0].compute_hash()
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 0


def test_whole_chain_and_head_rewrite_passes_documented_limit():
    """
    The honest limit: the chain is in memory and unanchored. Someone who can
    rewrite the entries AND the head can make verify_chain() pass. This test
    pins that behaviour so nobody documents more than is true.
    """
    e = _mixed_engine()
    prev = "genesis"
    for entry in e._audit:
        entry.scope = "/rewritten"
        entry.prev_hash = prev
        entry.entry_hash = entry.compute_hash()
        prev = entry.entry_hash
    e._last_hash = prev
    assert e.verify_chain().ok is True


# ─── Continuity across kinds of entries ──────────────────────────────────────

def test_chain_continues_across_grants_checks_denials_revokes_and_record_audit():
    e = PermissionEngine()
    e.grant("alice", Capability.WEB_SEARCH, hours=1)
    assert e.check("alice", Capability.WEB_SEARCH) is True
    assert e.check("alice", Capability.EMAIL_SEND) is False
    with pytest.raises(PermissionDeniedError):
        e.require("alice", Capability.FILE_WRITE, "/tmp")
    e.record_audit("alice", "web_search", "*", "tool.executed", call_id="c1")
    e.revoke("alice", Capability.WEB_SEARCH)
    e.revoke_all("alice")
    e.record_audit("alice", "web_search", "*", "tool.end", call_id="c1")

    actions = [x.action for x in e._audit]
    assert "granted" in actions and "allowed" in actions and "denied" in actions
    assert "revoked" in actions and "tool.executed" in actions and "tool.end" in actions
    assert len(actions) >= 7
    assert e.verify_chain().ok
    assert e.verify_chain().checked == len(e._audit)
    for prev, cur in zip(e._audit, e._audit[1:]):
        assert cur.prev_hash == prev.entry_hash


def test_self_modify_flow_stays_on_the_chain():
    e = PermissionEngine()
    token = e.request_self_modify_permission("alice", "patch-1", "fix", ["a.py"], "high")
    e.confirm_self_modify(token, "alice")
    token2 = e.request_self_modify_permission("alice", "patch-2", "fix", ["b.py"], "high")
    e.deny_self_modify(token2, "alice")
    assert len(e._audit) >= 2
    assert e.verify_chain().ok


def test_record_audit_and_grants_share_one_chain():
    e = PermissionEngine()
    first = e.record_audit("u", "c", "s", "a1")
    e.grant("u", Capability.WEB_SEARCH, hours=1)
    last = e.record_audit("u", "c", "s", "a2")
    assert first.prev_hash == "genesis"
    assert e._audit[1].prev_hash == first.entry_hash
    assert last.prev_hash == e._audit[1].entry_hash
    assert e._last_hash == last.entry_hash


def test_record_audit_returns_the_appended_entry():
    e = PermissionEngine()
    entry = e.record_audit(
        "alice", "tool", "/x", "tool.received", reason="r", agent_id="ag",
        task_id="t", call_id="c", stage="s", detail={"z": 1, "a": 2},
        args_digest="dg",
    )
    assert e._audit[-1] is entry
    assert isinstance(entry, AuditEntry)
    assert (entry.user_id, entry.capability, entry.scope, entry.action) == (
        "alice", "tool", "/x", "tool.received")
    assert (entry.reason, entry.agent_id, entry.task_id) == ("r", "ag", "t")
    assert (entry.call_id, entry.stage, entry.args_digest) == ("c", "s", "dg")
    assert entry.detail == '{"a":2,"z":1}'
    assert entry.entry_hash == entry.compute_hash()


# ─── record_audit validation ─────────────────────────────────────────────────

STR_ARGS = ["user_id", "capability", "scope", "action", "reason", "agent_id",
            "task_id", "call_id", "stage", "args_digest"]


@pytest.mark.parametrize("name", STR_ARGS)
@pytest.mark.parametrize("bad", [None, 5, 1.5, b"bytes", ["x"], {"a": 1}, object()])
def test_record_audit_rejects_non_string_arguments(name, bad):
    e = PermissionEngine()
    kwargs = {n: "v" for n in STR_ARGS}
    kwargs[name] = bad
    with pytest.raises(TypeError, match=name):
        e.record_audit(**kwargs)
    assert e._audit == []
    assert e._last_hash == "genesis"


def test_record_audit_stores_plain_str_for_str_subclass():
    class Sneaky(str):
        def __str__(self):
            return "changed"

    e = PermissionEngine()
    entry = e.record_audit(Sneaky("u"), "c", "s", "a")
    assert type(entry.user_id) is str and entry.user_id == "u"
    assert e.verify_chain().ok


def test_detail_none_and_empty():
    e = PermissionEngine()
    assert e.record_audit("u", "c", "s", "a").detail == ""
    assert e.record_audit("u", "c", "s", "a", detail="").detail == ""


def test_detail_string_is_stored_verbatim():
    e = PermissionEngine()
    assert e.record_audit("u", "c", "s", "a", detail="not json {").detail == "not json {"


def test_detail_dict_and_list_become_canonical_json():
    e = PermissionEngine()
    d1 = e.record_audit("u", "c", "s", "a", detail={"b": [1, 2.5, None, True], "a": "é"}).detail
    d2 = e.record_audit("u", "c", "s", "a", detail={"a": "é", "b": [1, 2.5, None, True]}).detail
    assert d1 == d2 == '{"a":"\\u00e9","b":[1,2.5,null,true]}'
    assert e.record_audit("u", "c", "s", "a", detail=[3, 1]).detail == "[3,1]"
    assert e.record_audit("u", "c", "s", "a", detail=("x",)).detail == '["x"]'


@pytest.mark.parametrize("bad", [
    {1, 2},
    object(),
    b"bytes",
    42,
    1.5,
    {"k": {1, 2}},
    {"k": object()},
    {1: "int key"},
    {"k": float("nan")},
    {"k": float("inf")},
    [float("-inf")],
    {"k": b"x"},
    {"k": lambda: 0},
])
def test_detail_rejects_non_json_values(bad):
    e = PermissionEngine()
    with pytest.raises(TypeError, match="detail"):
        e.record_audit("u", "c", "s", "a", detail=bad)
    assert e._audit == []


def test_detail_rejects_circular_structure():
    loop: list = []
    loop.append(loop)
    with pytest.raises((TypeError, ValueError), match="detail"):
        PermissionEngine().record_audit("u", "c", "s", "a", detail=loop)


def test_detail_rejects_excessive_nesting():
    deep: list = []
    cur = deep
    for _ in range(100):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    with pytest.raises(ValueError, match="nested"):
        PermissionEngine().record_audit("u", "c", "s", "a", detail=deep)


def test_detail_cap_is_8_kib_and_not_silent():
    assert MAX_AUDIT_DETAIL_BYTES == 8192
    e = PermissionEngine()
    ok = e.record_audit("u", "c", "s", "a", detail="x" * 8192)
    assert len(ok.detail) == 8192                       # exactly at the cap: kept whole
    with pytest.raises(ValueError, match="8192"):
        e.record_audit("u", "c", "s", "a", detail="x" * 8193)
    with pytest.raises(ValueError, match="detail"):
        e.record_audit("u", "c", "s", "a", detail={"k": "y" * 9000})
    assert len(e._audit) == 1
    assert e.verify_chain().ok


def test_detail_cap_counts_utf8_bytes():
    e = PermissionEngine()
    e.record_audit("u", "c", "s", "a", detail="é" * 4096)       # 8192 bytes
    with pytest.raises(ValueError):
        e.record_audit("u", "c", "s", "a", detail="é" * 4097)   # 8194 bytes
    with pytest.raises(ValueError):
        e.record_audit("u", "c", "s", "a", detail="\ud800" * 3000)  # 3 bytes each


def test_record_audit_does_not_redact():
    """Documented: redaction is the caller's job."""
    e = PermissionEngine()
    entry = e.record_audit("u", "c", "s", "a", reason="token=sk-secret", detail={"k": "sk-secret"})
    assert "sk-secret" in entry.reason
    assert "sk-secret" in entry.detail


# ─── Control characters and JSON-safe output ─────────────────────────────────

NASTY = "a\x00b\x01\x1b[31m\r\n\t  \x7f\ud800\U0001f600\"\\'"


def test_control_characters_are_hashed_and_chain_verifies():
    e = PermissionEngine()
    e.record_audit(NASTY, NASTY, NASTY, NASTY, reason=NASTY, agent_id=NASTY,
                   task_id=NASTY, call_id=NASTY, stage=NASTY, detail=NASTY,
                   args_digest=NASTY)
    entry = e._audit[0]
    assert entry.scope == NASTY
    assert e.verify_chain().ok
    entry.reason = NASTY.replace("\x00", "\x01")
    assert not e.verify_chain().ok


def test_trail_is_json_safe_with_control_characters():
    e = PermissionEngine()
    e.record_audit(NASTY, NASTY, NASTY, NASTY, reason=NASTY, agent_id=NASTY,
                   task_id=NASTY, call_id=NASTY, stage=NASTY, detail={"k": NASTY},
                   args_digest=NASTY)
    trail = e.get_audit_trail(NASTY)
    assert len(trail) == 1
    text = json.dumps(trail)
    assert text.isascii()
    assert json.loads(text)[0]["scope"] == NASTY
    assert json.loads(json.loads(text)[0]["detail"])["k"] == NASTY


def test_trail_survives_a_bad_timestamp():
    e = PermissionEngine()
    e.record_audit("u", "c", "s", "a")
    e._audit[0].timestamp = float("nan")
    trail = e.get_audit_trail("u")
    assert trail[0]["time"] == ""


# ─── Backward compatibility ──────────────────────────────────────────────────

def test_get_audit_trail_keeps_original_keys():
    e = PermissionEngine()
    e.grant("alice", Capability.FILE_READ, scope="/tmp", hours=1)
    e.check("alice", Capability.FILE_READ, "/tmp/f", agent_id="ag-1")
    e.check("alice", Capability.EMAIL_SEND, agent_id="ag-2")
    trail = e.get_audit_trail("alice")
    assert [t["action"] for t in trail] == ["granted", "allowed", "denied"]
    for item in trail:
        for key in ("time", "capability", "scope", "action", "reason", "agent"):
            assert isinstance(item[key], str)
    assert trail[1]["agent"] == "ag-1"
    assert trail[1]["scope"] == "/tmp/f"
    assert trail[1]["capability"] == "file_read"
    assert "Grant" in trail[1]["reason"]
    assert time.strptime(trail[0]["time"], "%Y-%m-%d %H:%M:%S")
    # New keys are additive.
    assert trail[1]["entry_hash"] == e._audit[1].entry_hash
    assert trail[1]["call_id"] == ""


def test_get_audit_trail_last_n_and_user_filter():
    e = PermissionEngine()
    for i in range(5):
        e.record_audit("alice", "c", "s", f"a{i}")
        e.record_audit("bob", "c", "s", f"b{i}")
    assert [t["action"] for t in e.get_audit_trail("alice", last_n=2)] == ["a3", "a4"]
    assert len(e.get_audit_trail("nobody")) == 0


def test_audit_log_old_signature_still_works():
    e = PermissionEngine()
    e._audit_log("u", "c", "s", "act")
    e._audit_log("u", "c", "s", "act", "why", "agent", "task")
    e._audit_log(user_id="u", capability="c", scope="s", action="act",
                 reason="r", agent_id="a", task_id="t")
    e._audit_log("u", "c", "s", "act", call_id="x", stage="y",
                 detail='{"a":1}', args_digest="z")
    assert len(e._audit) == 4
    assert e._audit[3].call_id == "x" and e._audit[3].detail == '{"a":1}'
    assert e.verify_chain().ok


def test_check_with_a_non_string_scope_does_not_raise_from_the_audit():
    e = PermissionEngine()
    e.grant("u", Capability.FILE_READ, scope="/tmp", hours=1)
    e._audit_log("u", "c", Path("/tmp/x"), "act")  # type: ignore[arg-type]
    assert e._audit[-1].scope == str(Path("/tmp/x"))
    assert e.verify_chain().ok


# ─── Concurrency ─────────────────────────────────────────────────────────────

def test_threads_cannot_fork_the_chain():
    e = PermissionEngine()
    threads_n, per_thread = 4, 500
    barrier = threading.Barrier(threads_n)
    errors: list = []

    def worker(n: int) -> None:
        try:
            barrier.wait()
            for i in range(per_thread):
                if i % 2:
                    e.record_audit(f"u{n}", "c", "s", "tool.x", call_id=f"{n}-{i}")
                else:
                    e._audit_log(f"u{n}", "c", "s", "act", call_id=f"{n}-{i}")
        except BaseException as exc:  # noqa: BLE001 - surface any failure
            errors.append(exc)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)

    assert errors == []
    assert len(e._audit) == threads_n * per_thread
    report = e.verify_chain()
    assert report.ok, report
    assert report.checked == threads_n * per_thread
    assert len({x.entry_hash for x in e._audit}) == threads_n * per_thread


def test_verify_and_trail_while_writing():
    e = PermissionEngine()
    stop = threading.Event()
    results: list = []

    def writer() -> None:
        for i in range(2000):
            e.record_audit("u", "c", "s", "a", call_id=str(i))
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            results.append(e.verify_chain().ok)
            e.get_audit_trail("u")

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        w, r = threading.Thread(target=writer), threading.Thread(target=reader)
        r.start()
        w.start()
        w.join()
        r.join()
    finally:
        sys.setswitchinterval(old)
    assert all(results)
    assert e.verify_chain().checked == 2000


# ─── Performance ─────────────────────────────────────────────────────────────

def test_long_chain_verifies_quickly():
    e = PermissionEngine()
    for i in range(10_000):
        e.record_audit("u", "c", "/s", "a", reason=f"r{i}", call_id=str(i), detail={"i": i})
    assert len(e._audit) == 10_000
    start = time.perf_counter()
    report = e.verify_chain()
    elapsed = time.perf_counter() - start
    assert report.ok and report.checked == 10_000
    assert elapsed < 2.0, f"verify_chain took {elapsed:.2f}s"


def test_long_chain_tamper_in_the_middle_is_found():
    e = PermissionEngine()
    for i in range(10_000):
        e.record_audit("u", "c", "/s", "a", call_id=str(i))
    e._audit[7_321].scope = "/tampered"
    report = e.verify_chain()
    assert not report.ok and report.first_bad_index == 7_321 and report.checked == 7_321
