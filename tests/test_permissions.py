"""
Tests for ARIA Zero-Trust Permission Engine
"""
import time

import pytest

from aria.permissions.engine import Capability, PermissionDeniedError, PermissionEngine


@pytest.fixture
def engine():
    return PermissionEngine()


def test_default_deny(engine):
    assert engine.check("alice", Capability.EMAIL_SEND) is False
    assert engine.check("alice", Capability.FILE_READ) is False
    assert engine.check("bob", Capability.WEB_SEARCH) is False


def test_grant_and_check(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    assert engine.check("alice", Capability.WEB_SEARCH) is True


def test_user_isolation(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    assert engine.check("alice", Capability.WEB_SEARCH) is True
    assert engine.check("bob", Capability.WEB_SEARCH) is False


def test_revoke_all(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    engine.grant("alice", Capability.EMAIL_READ, hours=1)
    engine.revoke_all("alice")
    assert engine.check("alice", Capability.WEB_SEARCH) is False
    assert engine.check("alice", Capability.EMAIL_READ) is False


def test_revoke_by_capability(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    assert engine.check("alice", Capability.WEB_SEARCH) is True
    engine.revoke("alice", Capability.WEB_SEARCH)
    assert engine.check("alice", Capability.WEB_SEARCH) is False


def test_require_raises_on_denied(engine):
    with pytest.raises(PermissionDeniedError):
        engine.require("alice", Capability.EMAIL_SEND)


def test_require_passes_when_granted(engine):
    engine.grant("alice", Capability.EMAIL_SEND, hours=1)
    engine.require("alice", Capability.EMAIL_SEND)


def test_max_uses_limit(engine):
    engine.grant("alice", Capability.EMAIL_SEND, hours=1, max_uses=2)
    assert engine.check("alice", Capability.EMAIL_SEND) is True
    assert engine.check("alice", Capability.EMAIL_SEND) is True
    assert engine.check("alice", Capability.EMAIL_SEND) is False


def test_expiry(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=0.000001)
    time.sleep(0.01)
    assert engine.check("alice", Capability.WEB_SEARCH) is False


def test_list_grants_returns_list(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    engine.grant("alice", Capability.EMAIL_READ, hours=1)
    grants = engine.list_grants("alice")
    assert isinstance(grants, list)
    assert len(grants) >= 2


def test_audit_trail_recorded(engine):
    engine.grant("alice", Capability.WEB_SEARCH, hours=1)
    engine.check("alice", Capability.WEB_SEARCH)
    engine.check("alice", Capability.EMAIL_SEND)
    trail = engine.get_audit_trail("alice")
    assert len(trail) >= 2


def test_wildcard_scope(engine):
    engine.grant("alice", Capability.WEB_SEARCH, scope="*", hours=1)
    assert engine.check("alice", Capability.WEB_SEARCH, scope="https://example.com") is True


def test_grant_summary_output(engine, capsys):
    engine.grant("alice", Capability.WEB_SEARCH, hours=8)
    engine.print_grant_summary("alice")
    captured = capsys.readouterr()
    assert "alice" in captured.out
    assert "web_search" in captured.out
