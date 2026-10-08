"""
Permission scope matching (issue #36).

A grant for one resource must never cover a different resource that only
shares a text prefix with it.
"""
import os

import pytest

from aria.permissions.engine import Capability, PermissionEngine
from aria.permissions.scope import resolve_path, scope_covers

# ─── Wildcard and exact ids ───────────────────────────────────────────────────


def test_wildcard_grant_covers_everything():
    assert scope_covers("*", "/any/path")
    assert scope_covers("*", "https://example.com")
    assert scope_covers("*", "vuln-123")
    assert scope_covers("*", "*")


def test_specific_grant_never_covers_a_request_for_everything():
    assert not scope_covers("/home/u/project", "*")
    assert not scope_covers("https://example.com", "*")
    assert not scope_covers("vuln-1", "*")


def test_ids_match_only_when_equal():
    assert scope_covers("vuln-1", "vuln-1")
    assert not scope_covers("vuln-1", "vuln-12")
    assert not scope_covers("vuln-12", "vuln-1")
    assert not scope_covers("cap", "capability")


def test_id_grant_does_not_cover_a_path_or_url():
    assert not scope_covers("project", "/project")
    assert not scope_covers("example.com", "https://example.com")


def test_non_string_inputs_are_denied():
    assert not scope_covers(None, "/a")  # type: ignore[arg-type]
    assert not scope_covers("/a", None)  # type: ignore[arg-type]
    assert not scope_covers(5, 5)  # type: ignore[arg-type]


# ─── Paths ────────────────────────────────────────────────────────────────────


def test_path_grant_covers_itself_and_children():
    assert scope_covers("/home/u/project", "/home/u/project")
    assert scope_covers("/home/u/project", "/home/u/project/a.txt")
    assert scope_covers("/home/u/project", "/home/u/project/sub/deep/b.txt")


def test_path_grant_does_not_cover_a_sibling_that_shares_a_prefix():
    assert not scope_covers("/home/u/project", "/home/u/project-secrets/key.pem")
    assert not scope_covers("/home/u/project", "/home/u/projectX")
    assert not scope_covers("/home/u/project", "/home/u/project2/a.txt")


def test_path_grant_does_not_cover_parent_or_unrelated_paths():
    assert not scope_covers("/home/u/project", "/home/u")
    assert not scope_covers("/home/u/project", "/home")
    assert not scope_covers("/home/u/project", "/etc/passwd")


def test_trailing_slash_on_the_grant_is_equivalent():
    assert scope_covers("/home/u/project/", "/home/u/project/a.txt")
    assert scope_covers("/home/u/project/", "/home/u/project")
    assert not scope_covers("/home/u/project/", "/home/u/project-secrets")


def test_duplicate_separators_do_not_matter():
    assert scope_covers("/home/u/project", "/home//u///project/a.txt")


def test_parent_traversal_cannot_leave_the_granted_folder():
    assert not scope_covers("/home/u/project", "/home/u/project/../secret.txt")
    assert not scope_covers("/home/u/project", "/home/u/project/sub/../../other")
    assert scope_covers("/home/u/project", "/home/u/project/sub/../a.txt")


def test_dot_segments_in_the_grant_are_resolved_too():
    assert scope_covers("/home/u/project/../project", "/home/u/project/a.txt")
    assert not scope_covers("/home/u/project/sub/..", "/home/u/other")


def test_root_grant_covers_every_absolute_path():
    assert scope_covers("/", "/etc/passwd")
    assert scope_covers("/", "/")


def test_path_grant_does_not_cover_a_url_with_the_same_text():
    assert not scope_covers("/home/u", "https://home/u")


def test_null_byte_is_denied():
    assert not scope_covers("/home/u/project", "/home/u/project/a\x00b")
    assert not scope_covers("/home/u/project\x00", "/home/u/project")
    assert resolve_path("/a\x00b") is None


def test_tilde_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert scope_covers("~/notes", str(tmp_path / "notes" / "a.txt"))
    assert not scope_covers("~/notes", str(tmp_path / "notes-private" / "a.txt"))


def test_relative_path_grants_never_cover_anything_but_their_own_text(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert not scope_covers("./work", str(tmp_path / "work" / "a.txt"))
    assert not scope_covers(".", str(tmp_path / "a.txt"))
    assert not scope_covers("..", str(tmp_path.parent / "a.txt"))
    assert not scope_covers("../sibling", str(tmp_path.parent / "sibling" / "a.txt"))


def test_relative_grant_with_cwd_at_root_cannot_cover_the_system(monkeypatch):
    monkeypatch.chdir("/")
    assert not scope_covers(".", "/etc/passwd")
    assert not scope_covers("./etc", "/etc/passwd")


needs_symlinks = pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="symlinks not available"
)


@needs_symlinks
def test_symlink_inside_the_grant_that_points_outside_is_denied(tmp_path):
    granted = tmp_path / "granted"
    outside = tmp_path / "outside"
    granted.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    try:
        os.symlink(outside, granted / "link")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")

    assert scope_covers(str(granted), str(granted / "real.txt"))
    assert not scope_covers(str(granted), str(granted / "link" / "secret.txt"))
    assert not scope_covers(str(granted), str(granted / "link"))


@needs_symlinks
def test_symlink_to_a_protected_file_is_judged_by_its_real_location(tmp_path):
    protected = tmp_path / "protected"
    open_dir = tmp_path / "open"
    protected.mkdir()
    open_dir.mkdir()
    (protected / "engine.py").write_text("x")
    try:
        os.symlink(protected / "engine.py", open_dir / "innocent.py")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")

    assert not scope_covers(str(open_dir), str(open_dir / "innocent.py"))
    assert scope_covers(str(protected), str(open_dir / "innocent.py"))


@needs_symlinks
def test_a_granted_symlink_means_its_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")

    assert scope_covers(str(link), str(target / "a.txt"))


@needs_symlinks
def test_path_that_does_not_exist_yet_is_still_judged_correctly(tmp_path):
    granted = tmp_path / "granted"
    granted.mkdir()
    assert scope_covers(str(granted), str(granted / "new" / "file.txt"))
    assert not scope_covers(str(granted), str(tmp_path / "granted-new" / "file.txt"))


# ─── URLs ─────────────────────────────────────────────────────────────────────


def test_url_grant_covers_its_own_paths():
    assert scope_covers("https://example.com", "https://example.com")
    assert scope_covers("https://example.com", "https://example.com/")
    assert scope_covers("https://example.com", "https://example.com/a/b?q=1#frag")


def test_url_grant_does_not_cover_a_lookalike_host():
    assert not scope_covers("https://example.com", "https://example.com.evil.com/")
    assert not scope_covers("https://example.com", "https://example.community")
    assert not scope_covers("https://example.com", "https://evil.com/example.com")


def test_url_grant_does_not_cover_a_subdomain_unless_granted():
    assert not scope_covers("https://example.com", "https://api.example.com/")
    assert scope_covers("https://api.example.com", "https://api.example.com/v1")


def test_url_embedded_credentials_cannot_redirect_the_host():
    assert not scope_covers("https://example.com", "https://example.com@evil.com/")
    assert not scope_covers("https://example.com", "https://example.com:pw@evil.com/")
    assert not scope_covers("https://example.com", "https://user:pw@example.com/")


def test_url_scheme_must_match():
    assert not scope_covers("https://example.com", "http://example.com")
    assert not scope_covers("https://example.com", "ftp://example.com")


def test_url_port_must_match_with_default_ports_normalised():
    assert scope_covers("https://example.com", "https://example.com:443/a")
    assert scope_covers("https://example.com:443", "https://example.com/a")
    assert not scope_covers("https://example.com", "https://example.com:8443/a")
    assert not scope_covers("https://example.com:8443", "https://example.com/a")


def test_url_host_comparison_ignores_case():
    assert scope_covers("https://Example.COM", "https://example.com/a")


def test_url_path_grant_matches_whole_segments_only():
    assert scope_covers("https://example.com/api", "https://example.com/api")
    assert scope_covers("https://example.com/api", "https://example.com/api/v1/users")
    assert scope_covers("https://example.com/api/", "https://example.com/api/v1")
    assert not scope_covers("https://example.com/api", "https://example.com/apiary")
    assert not scope_covers("https://example.com/api", "https://example.com/other")


def test_url_dot_segments_cannot_leave_the_granted_path():
    assert not scope_covers("https://example.com/api", "https://example.com/api/../admin")
    assert not scope_covers("https://example.com/api", "https://example.com/api/%2e%2e/admin")
    assert not scope_covers("https://example.com/api", "https://example.com/api/%2E%2E/admin")
    assert scope_covers("https://example.com/api", "https://example.com/api/v1/../v2")


def test_url_garbage_is_denied():
    assert not scope_covers("https://example.com", "https://")
    assert not scope_covers("https://example.com", "https://example.com:notaport/")
    assert not scope_covers("https://example.com", "://example.com")
    assert not scope_covers("https://", "https://example.com")


def test_url_and_path_do_not_cross():
    assert not scope_covers("https://example.com", "/example.com")
    assert not scope_covers("/home/u", "https://example.com/home/u")


# ─── Through the permission engine ───────────────────────────────────────────


@pytest.fixture
def engine():
    return PermissionEngine()


def test_engine_reproduction_from_issue_36(engine):
    engine.grant("u", Capability.FILE_WRITE, scope="/home/u/project", hours=1)

    assert engine.check("u", Capability.FILE_WRITE, scope="/home/u/project/a.txt")
    assert not engine.check("u", Capability.FILE_WRITE, scope="/home/u/project-secrets/key.pem")
    assert not engine.check("u", Capability.FILE_WRITE, scope="/home/u/projectX")
    assert not engine.check("u", Capability.FILE_WRITE, scope="/home/u/other")


def test_engine_denial_is_recorded_and_explained(engine):
    engine.grant("u", Capability.FILE_WRITE, scope="/home/u/project", hours=1)

    assert not engine.check("u", Capability.FILE_WRITE, scope="/home/u/project-secrets")

    last = engine.get_audit_trail("u")[-1]
    assert last["action"] == "denied"
    assert last["scope"] == "/home/u/project-secrets"
    assert "scope mismatch" in last["reason"].lower()


def test_engine_url_grant_from_the_readme_still_works(engine):
    engine.grant("u", Capability.WEB_BROWSE, scope="https://example.com", hours=1)

    assert engine.check("u", Capability.WEB_BROWSE, scope="https://example.com/page")
    assert not engine.check("u", Capability.WEB_BROWSE, scope="https://example.com.evil.com")


def test_engine_id_scopes_are_exact(engine):
    engine.grant("u", Capability.CODE_EXECUTE, scope="cap-1", hours=1)

    assert engine.check("u", Capability.CODE_EXECUTE, scope="cap-1")
    assert not engine.check("u", Capability.CODE_EXECUTE, scope="cap-10")


def test_engine_wildcard_grant_still_covers_everything(engine):
    engine.grant("u", Capability.WEB_SEARCH, scope="*", hours=1)

    assert engine.check("u", Capability.WEB_SEARCH, scope="https://example.com")
    assert engine.check("u", Capability.WEB_SEARCH, scope="anything")


# ─── URL ambiguity found in review ───────────────────────────────────────────

AMBIGUOUS_REQUESTS = [
    "https://example.com/api/..\\admin",
    "https://example.com/api/%5c..%5cadmin",
    "https://example.com/api/%5C..%5Cadmin",
    "https://example.com/api/%2e%2e%5cadmin",
    "https://example.com/api/..;/admin",
    "https://example.com/api/%252e%252e/admin",
    "https://example.com/api%2fadmin",
    "https://example.com/api/%2Fadmin",
    "https://example.com/api/a%00b",
    "https://example.com//api/x",
    "https://example.com/api//x",
    "https://exa\tmple.com/api/x",
    "ht\ttps://example.com/api/x",
    " https://example.com/api/x",
    "https://example.com/api/x\n",
    "https://example.com/api/x y",
]


@pytest.mark.parametrize("request_url", AMBIGUOUS_REQUESTS)
def test_ambiguous_urls_are_not_covered(request_url):
    assert not scope_covers("https://example.com/api", request_url)


def test_ambiguous_grant_urls_cover_nothing_but_themselves():
    grant = "https://example.com/api;v=1"
    assert not scope_covers(grant, "https://example.com/api;v=1/child")
    assert scope_covers(grant, grant)


def test_plain_urls_still_work_after_the_ambiguity_rules():
    assert scope_covers("https://example.com/api", "https://example.com/api/v1/users?id=7#top")
    assert scope_covers("https://example.com", "https://example.com/a-b_c.d/e")


# ─── Grants keep one meaning (found in review) ───────────────────────────────


def test_engine_refuses_relative_path_grants(engine):
    for scope in [".", "..", "./src", "../other"]:
        with pytest.raises(ValueError):
            engine.grant("u", Capability.FILE_WRITE, scope=scope, hours=1)


def test_engine_stores_path_grants_as_absolute_real_paths(engine, tmp_path):
    grant = engine.grant("u", Capability.FILE_WRITE, scope=str(tmp_path / "a" / ".." / "b"), hours=1)
    assert grant.scope == resolve_path(str(tmp_path / "b"))


def test_home_grant_is_fixed_when_created_not_when_checked(engine, monkeypatch, tmp_path):
    home_a = tmp_path / "home_a"
    home_b = tmp_path / "home_b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setenv("HOME", str(home_a))
    engine.grant("u", Capability.FILE_WRITE, scope="~", hours=1)

    monkeypatch.setenv("HOME", str(home_b))
    assert engine.check("u", Capability.FILE_WRITE, scope=str(home_a / "x.txt"))
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(home_b / "x.txt"))


@needs_symlinks
def test_grant_on_a_symlink_is_stored_as_its_target(engine, tmp_path):
    target = tmp_path / "target"
    other = tmp_path / "other"
    target.mkdir()
    other.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")

    engine.grant("u", Capability.FILE_WRITE, scope=str(link), hours=1)
    link.unlink()
    os.symlink(other, link)  # retarget after the grant

    assert engine.check("u", Capability.FILE_WRITE, scope=str(target / "a.txt"))
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(other / "a.txt"))
    # The link now points at "other", which was never granted.
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(link / "a.txt"))


@needs_symlinks
def test_replacing_a_granted_folder_with_a_symlink_stops_the_grant(engine, tmp_path):
    granted = tmp_path / "granted"
    secret = tmp_path / "secret"
    granted.mkdir()
    secret.mkdir()
    (secret / "key.pem").write_text("x")
    engine.grant("u", Capability.FILE_WRITE, scope=str(granted), hours=1)
    assert engine.check("u", Capability.FILE_WRITE, scope=str(granted / "a.txt"))

    granted.rmdir()
    try:
        os.symlink(secret, granted)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")

    assert not engine.check("u", Capability.FILE_WRITE, scope=str(granted / "key.pem"))
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(secret / "key.pem"))


def test_revoke_matches_a_path_given_in_a_different_spelling(engine, tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    engine.grant("u", Capability.FILE_WRITE, scope=str(folder), hours=1)

    revoked = engine.revoke("u", Capability.FILE_WRITE, scope=str(folder) + "/")
    assert revoked == 1
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(folder / "a.txt"))


def test_revoke_with_a_relative_path_removes_nothing_and_does_not_crash(engine, tmp_path):
    engine.grant("u", Capability.FILE_WRITE, scope=str(tmp_path), hours=1)
    assert engine.revoke("u", Capability.FILE_WRITE, scope=".") == 0


def test_saved_path_grants_survive_a_restart(tmp_path):
    store = tmp_path / "perms.json"
    folder = tmp_path / "work"
    folder.mkdir()
    first = PermissionEngine(persist_path=str(store))
    first.grant("u", Capability.FILE_WRITE, scope=str(folder), hours=1)

    second = PermissionEngine(persist_path=str(store))
    assert second.check("u", Capability.FILE_WRITE, scope=str(folder / "a.txt"))
    assert not second.check("u", Capability.FILE_WRITE, scope=str(tmp_path / "work-old" / "a.txt"))


def test_old_saved_relative_grant_is_kept_but_covers_nothing(tmp_path, monkeypatch):
    import json

    store = tmp_path / "perms.json"
    store.write_text(json.dumps({"grants": {"u": {"file_write": [{
        "grant_id": "g1", "user_id": "u", "capability": "file_write",
        "scope": "./src", "granted_at": 0, "expires_at": 0,
        "task_context": "", "use_count": 0, "max_uses": None,
    }]}}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    engine = PermissionEngine(persist_path=str(store))
    assert store.exists()  # not discarded as corrupt
    assert not engine.check("u", Capability.FILE_WRITE, scope=str(tmp_path / "src" / "a.py"))
