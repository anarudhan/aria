"""
Unit tests for aria.security.patcher: line-level patch strategies and the SelfPatcher
safety protocol (permission, protected files, backup, syntax check, rollback, audit log).

Every file the patcher touches lives under pytest's tmp_path.
"""
import json
import os
import pathlib
import time

import pytest

from aria.permissions.engine import Capability, PermissionDeniedError, PermissionEngine
from aria.security import patcher as patcher_module
from aria.security.patcher import (
    PATCH_STRATEGIES,
    PROTECTED_FILES,
    PatchRecord,
    SelfPatcher,
    _patch_bare_except,
    _patch_eval,
    _patch_insecure_random,
    _patch_missing_timeout,
    _patch_shell_injection,
    _patch_swallowed_exception,
    _patch_tls_disabled,
)
from aria.security.scanner import ScanReport, SecurityScanner, Vulnerability

SOURCE = (
    "import os\n"                       # line 1
    "\n"                                # line 2
    "def parse(s):\n"                   # line 3
    "    return eval(s)  # caf\u00e9\n"  # line 4
    "\n"                                # line 5
    "def other(s):\n"                   # line 6
    "    return eval(s)\n"              # line 7
)

USER = "alice"


# ─── Fixtures and helpers ─────────────────────────────────────────────────────

@pytest.fixture
def root(tmp_path):
    path = tmp_path / "aria_root"
    (path / "core").mkdir(parents=True)
    return path


@pytest.fixture
def target(root):
    path = root / "core" / "mod.py"
    path.write_text(SOURCE, encoding="utf-8")
    return path


@pytest.fixture
def engine():
    return PermissionEngine()


@pytest.fixture
def patcher(root):
    return SelfPatcher(str(root))


def vuln(path, line, category="arbitrary_code_execution", **kw) -> Vulnerability:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    content = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    return Vulnerability(
        severity=kw.pop("severity", "critical"),
        category=category,
        file_path=str(path),
        line_number=line,
        line_content=content,
        description=f"{category} test finding",
        suggestion="fix it",
        **kw,
    )


def allow(engine, v: Vulnerability):
    """Grant one-time SELF_MODIFY for exactly this vulnerability."""
    engine.grant(USER, Capability.SELF_MODIFY, scope=v.vuln_id, hours=1, max_uses=1)


def apply(patcher, engine, v):
    allow(engine, v)
    return patcher.apply_patch(v, USER, engine)


@pytest.fixture
def unique_backup_names(monkeypatch):
    """Make every timestamp differ so backups of the same file never collide."""
    real = time.strftime
    counter = iter(range(1, 10_000))

    def fake(fmt, *args):
        if fmt == "%Y%m%d_%H%M%S":
            return f"20250101_{next(counter):06d}"
        return real(fmt, *args)

    monkeypatch.setattr(patcher_module.time, "strftime", fake)


@pytest.fixture
def frozen_backup_clock(monkeypatch):
    """Every backup in the test gets the same second-resolution timestamp."""
    real = time.strftime

    def fake(fmt, *args):
        return "20250101_000000" if fmt == "%Y%m%d_%H%M%S" else real(fmt, *args)

    monkeypatch.setattr(patcher_module.time, "strftime", fake)


# ─── Patch strategies ─────────────────────────────────────────────────────────

class TestEvalStrategy:

    def test_replaces_eval_call(self):
        assert _patch_eval("x = eval(user_input)\n") == "x = ast.literal_eval(user_input)\n"

    def test_handles_space_before_paren(self):
        assert _patch_eval("x = eval (s)") == "x = ast.literal_eval(s)"

    def test_leaves_literal_eval_alone(self):
        assert _patch_eval("x = ast.literal_eval(s)") is None

    def test_returns_none_without_eval(self):
        assert _patch_eval("x = evaluate(s)") is None


class TestBareExceptStrategy:

    def test_replaces_bare_except(self):
        assert _patch_bare_except("    except:\n") == "    except Exception as e:\n"

    @pytest.mark.parametrize("line", ["    except ValueError:\n", "    except Exception:\n",
                                      "x = 1\n", "    except: pass\n"])
    def test_ignores_other_lines(self, line):
        assert _patch_bare_except(line) is None


class TestSwallowedExceptionStrategy:

    def test_replaces_except_pass_with_logging_keeping_indent(self):
        patched = _patch_swallowed_exception("        except Exception: pass\n")
        first, second = patched.split("\n")
        assert first == "        except Exception as e:"
        assert second.startswith("            import logging;")
        assert "logging.warning" in second

    def test_patched_result_is_valid_python(self):
        patched = _patch_swallowed_exception("except Exception: pass\n")
        compile("try:\n    pass\n" + patched + "\n", "<patched>", "exec")

    @pytest.mark.parametrize("line", ["    except Exception:\n", "    except ValueError: pass\n"])
    def test_ignores_other_lines(self, line):
        assert _patch_swallowed_exception(line) is None


class TestMissingTimeoutStrategy:

    @pytest.mark.parametrize("method", ["get", "post", "put", "delete", "patch"])
    def test_adds_timeout_to_requests_calls(self, method):
        line = f"r = requests.{method}(url)\n"
        assert _patch_missing_timeout(line) == f"r = requests.{method}(url, timeout=30)"

    def test_existing_timeout_is_kept(self):
        assert _patch_missing_timeout("r = requests.get(url, timeout=5)") is None

    def test_multiline_call_is_not_patched(self):
        assert _patch_missing_timeout("r = requests.get(url,") is None

    def test_non_requests_line_is_ignored(self):
        assert _patch_missing_timeout("r = session.get(url)") is None


class TestInsecureRandomStrategy:

    def test_only_patched_in_security_context(self):
        assert _patch_insecure_random("pick = random.randint(1, 6)") is None

    def test_random_in_token_context_uses_secrets(self):
        patched = _patch_insecure_random("token = random.random()")
        assert "secrets." in patched and "random.random" not in patched

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py _patch_insecure_random rewrites random.randint(a, b) to "
               "secrets.randbelow(a, b) but randbelow takes exactly one argument, so the "
               "patched line raises TypeError at runtime (the syntax check cannot catch it)",
    )
    def test_randint_replacement_is_callable(self):
        import secrets

        patched = _patch_insecure_random("session_id = random.randint(1000, 9999)")
        namespace = {"secrets": secrets}
        exec(patched, namespace)
        assert 1000 <= namespace["session_id"] <= 9999


class TestTlsAndShellStrategies:

    def test_removes_verify_false(self):
        assert _patch_tls_disabled("requests.get(url, verify=False)") == "requests.get(url)"
        assert _patch_tls_disabled("requests.get(url, timeout=3, verify=False)") \
            == "requests.get(url, timeout=3)"

    def test_leaves_verify_true_alone(self):
        assert _patch_tls_disabled("requests.get(url, verify=True)") is None

    def test_shell_true_becomes_shell_false_with_marker(self):
        patched = _patch_shell_injection("subprocess.run(cmd, shell=True)")
        assert patched.startswith("subprocess.run(cmd, shell=False")
        assert "PATCHED" in patched

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py _patch_shell_injection appends a '# PATCHED' comment in the "
               "middle of the call, which comments out the closing parenthesis, so the patched "
               "single-line call is a syntax error and every shell=True patch is rolled back",
    )
    def test_shell_patch_result_is_valid_python(self):
        patched = _patch_shell_injection("subprocess.run(cmd, shell=True)")
        compile(patched + "\n", "<patched>", "exec")

    def test_no_shell_true_no_patch(self):
        assert _patch_shell_injection("subprocess.run(cmd, shell=False)") is None


def test_every_strategy_is_registered_for_a_category():
    assert set(PATCH_STRATEGIES) == {
        "arbitrary_code_execution", "bare_exception", "exception_swallowing",
        "missing_timeout", "insecure_random", "tls_disabled", "shell_injection",
    }


# ─── PatchRecord ──────────────────────────────────────────────────────────────

class TestPatchRecord:

    def test_ids_are_unique_and_short(self):
        ids = {PatchRecord().patch_id for _ in range(50)}
        assert len(ids) == 50 and all(len(i) == 12 for i in ids)

    def test_to_dict_contains_audit_fields(self):
        record = PatchRecord(file_path="/x/y.py", line_number=3, original_line="a",
                             patched_line="b", user_approved="alice", success=True)
        data = record.to_dict()
        assert data["file"] == "/x/y.py" and data["line"] == 3
        assert data["original"] == "a" and data["patched"] == "b"
        assert data["user_approved"] == "alice"
        assert data["success"] is True and data["rollback_done"] is False
        assert data["patch_id"] == record.patch_id
        json.dumps(data)   # must be serialisable


# ─── Construction ─────────────────────────────────────────────────────────────

class TestConstruction:

    def test_defaults_live_under_the_root_and_are_created_lazily(self, root):
        p = SelfPatcher(str(root))
        assert p.backup_dir == root / ".patch_backups"
        assert p.patch_log == root / ".patch_log.json"
        assert not p.backup_dir.exists()
        assert not p.patch_log.exists()
        assert p.get_patch_history() == []

    def test_custom_backup_dir_and_log_path(self, root, tmp_path, target, engine):
        p = SelfPatcher(str(root), backup_dir=str(tmp_path / "bk"),
                        patch_log_path=str(tmp_path / "log.json"))
        record = apply(p, engine, vuln(target, 4))
        assert (tmp_path / "bk").is_dir()
        assert os.path.dirname(record.backup_path) == str(tmp_path / "bk")
        assert (tmp_path / "log.json").exists()

    def test_protected_file_list_covers_the_security_modules(self):
        assert {"permissions/engine.py", "security/patcher.py", "security/scanner.py"} \
            <= PROTECTED_FILES


# ─── apply_patch: success ─────────────────────────────────────────────────────

class TestApplyPatchSuccess:

    def test_patches_only_the_target_line(self, patcher, engine, target):
        record = apply(patcher, engine, vuln(target, 4))

        lines = target.read_text(encoding="utf-8").splitlines()
        original = SOURCE.splitlines()
        assert "ast.literal_eval(s)" in lines[3]
        assert "caf\u00e9" in lines[3]                       # non-ascii survives
        assert lines[:3] == original[:3] and lines[4:] == original[4:]
        compile(target.read_text(encoding="utf-8"), str(target), "exec")
        assert record.success is True and record.rollback_done is False

    def test_record_describes_the_change(self, patcher, engine, target):
        v = vuln(target, 7)
        record = apply(patcher, engine, v)

        assert record.vulnerability is v
        assert record.file_path == str(target)
        assert record.line_number == 7
        assert record.original_line == "return eval(s)"
        assert record.patched_line == "return ast.literal_eval(s)"
        assert record.user_approved == USER
        assert record.applied_by == "aria_self_patcher"

    def test_backup_holds_the_original_content(self, patcher, engine, target):
        record = apply(patcher, engine, vuln(target, 4))

        backup = open(record.backup_path, encoding="utf-8").read()
        assert backup == SOURCE
        assert os.path.dirname(record.backup_path) == str(patcher.backup_dir)
        assert os.path.basename(record.backup_path).startswith("mod_")

    def test_history_and_log_file_are_updated(self, patcher, engine, target, root):
        record = apply(patcher, engine, vuln(target, 4))

        history = patcher.get_patch_history()
        assert [h["patch_id"] for h in history] == [record.patch_id]
        assert history[0]["success"] is True
        on_disk = json.loads((root / ".patch_log.json").read_text())
        assert on_disk[0]["patch_id"] == record.patch_id
        assert on_disk[0]["user_approved"] == USER

    def test_final_line_without_trailing_newline(self, patcher, engine, root):
        path = root / "core" / "nonl.py"
        path.write_text("x = 1\nresult = eval(data)", encoding="utf-8")
        apply(patcher, engine, vuln(path, 2))
        assert path.read_text(encoding="utf-8") == "x = 1\nresult = ast.literal_eval(data)\n"

    def test_other_strategies_end_to_end(self, patcher, engine, root):
        path = root / "core" / "net.py"
        path.write_text(
            "import requests\n"
            "def f(url):\n"
            "    return requests.get(url)\n"
            "def g():\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n",
            encoding="utf-8",
        )
        apply(patcher, engine, vuln(path, 3, "missing_timeout"))
        apply(patcher, engine, vuln(path, 7, "bare_exception"))
        text = path.read_text(encoding="utf-8")
        assert "requests.get(url, timeout=30)" in text
        assert "except Exception as e:" in text
        compile(text, str(path), "exec")

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py replaces eval( with ast.literal_eval( but never adds "
               "'import ast', so patching a file that does not already import ast produces "
               "code that raises NameError (ast.parse cannot detect that)",
    )
    def test_patched_eval_is_runnable_without_prior_ast_import(self, patcher, engine, target):
        apply(patcher, engine, vuln(target, 4))

        namespace = {}
        exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), namespace)
        assert namespace["parse"]("[1, 2]") == [1, 2]


# ─── apply_patch: refusals ────────────────────────────────────────────────────

class TestApplyPatchRefusals:

    def assert_untouched(self, patcher, target):
        assert target.read_text(encoding="utf-8") == SOURCE
        assert patcher.get_patch_history() == []

    def test_permission_is_required(self, patcher, engine, target):
        with pytest.raises(PermissionDeniedError) as exc:
            patcher.apply_patch(vuln(target, 4), USER, engine)

        assert exc.value.capability == Capability.SELF_MODIFY
        self.assert_untouched(patcher, target)
        assert not patcher.backup_dir.exists()          # nothing was even backed up

    def test_grant_for_another_vulnerability_does_not_apply(self, patcher, engine, target):
        engine.grant(USER, Capability.SELF_MODIFY, scope="some-other-vuln", hours=1)
        with pytest.raises(PermissionDeniedError):
            patcher.apply_patch(vuln(target, 4), USER, engine)
        self.assert_untouched(patcher, target)

    def test_grant_for_a_different_user_does_not_apply(self, patcher, engine, target):
        v = vuln(target, 4)
        engine.grant("mallory", Capability.SELF_MODIFY, scope=v.vuln_id, hours=1)
        with pytest.raises(PermissionDeniedError):
            patcher.apply_patch(v, USER, engine)
        self.assert_untouched(patcher, target)

    def test_other_capabilities_do_not_imply_self_modify(self, patcher, engine, target):
        engine.grant(USER, Capability.FILE_WRITE, hours=1)
        with pytest.raises(PermissionDeniedError):
            patcher.apply_patch(vuln(target, 4), USER, engine)
        self.assert_untouched(patcher, target)

    def test_expired_grant_is_rejected(self, patcher, engine, target):
        v = vuln(target, 4)
        grant = engine.grant(USER, Capability.SELF_MODIFY, scope=v.vuln_id, hours=1)
        grant.expires_at = time.time() - 1
        with pytest.raises(PermissionDeniedError):
            patcher.apply_patch(v, USER, engine)
        self.assert_untouched(patcher, target)

    def test_single_use_grant_cannot_be_replayed(self, patcher, engine, target):
        v = vuln(target, 4)
        allow(engine, v)
        patcher.apply_patch(v, USER, engine)
        with pytest.raises(PermissionDeniedError):
            patcher.apply_patch(v, USER, engine)
        assert len(patcher.get_patch_history()) == 1

    def test_non_auto_patchable_vulnerability_is_refused(self, patcher, engine, target):
        v = vuln(target, 4, patch_safe=False)
        allow(engine, v)
        with pytest.raises(ValueError, match="not auto-patchable"):
            patcher.apply_patch(v, USER, engine)
        self.assert_untouched(patcher, target)
        assert not patcher.backup_dir.exists()

    @pytest.mark.parametrize("relative", [
        "permissions/engine.py", "security/scanner.py", "security/patcher.py",
    ])
    def test_protected_files_are_never_patched_even_with_permission(
        self, patcher, engine, root, relative
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SOURCE, encoding="utf-8")
        v = vuln(path, 4)
        allow(engine, v)

        with pytest.raises(ValueError, match="protected"):
            patcher.apply_patch(v, USER, engine)

        assert path.read_text(encoding="utf-8") == SOURCE
        assert not patcher.backup_dir.exists()

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py checks the protected-file list against the path string only, "
               "so a symlink with an innocent name that points at permissions/engine.py is "
               "patched through the link",
    )
    def test_symlink_to_protected_file_is_refused(self, patcher, engine, root):
        protected = root / "permissions" / "engine.py"
        protected.parent.mkdir(parents=True)
        protected.write_text(SOURCE, encoding="utf-8")
        link = root / "core" / "innocent.py"
        try:
            link.symlink_to(protected)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not available on this platform")
        v = vuln(link, 4)
        allow(engine, v)

        with pytest.raises(ValueError):
            patcher.apply_patch(v, USER, engine)

        assert protected.read_text(encoding="utf-8") == SOURCE

    def test_line_beyond_end_of_file_is_rejected_and_file_restored(self, patcher, engine, target):
        with pytest.raises(ValueError, match="not found"):
            apply(patcher, engine, vuln(target, 99))
        self.assert_untouched(patcher, target)

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py computes line_idx = line_number - 1 without a lower bound, so "
               "line_number 0 (the Vulnerability default) or a negative number indexes from "
               "the end of the file and patches the last line",
    )
    @pytest.mark.parametrize("line_number", [0, -3])   # both resolve to an "eval" line
    def test_line_number_below_one_is_rejected(self, patcher, engine, target, line_number):
        v = vuln(target, 4)
        v.line_number = line_number
        allow(engine, v)
        with pytest.raises(ValueError):
            patcher.apply_patch(v, USER, engine)
        assert target.read_text(encoding="utf-8") == SOURCE

    def test_strategy_not_applicable_to_the_line_leaves_file_unchanged(
        self, patcher, engine, target
    ):
        with pytest.raises(ValueError, match="not applicable"):
            apply(patcher, engine, vuln(target, 1))       # "import os" has no eval
        self.assert_untouched(patcher, target)

    def test_category_without_strategy_is_rejected(self, patcher, engine, target):
        with pytest.raises(ValueError, match="No patch strategy"):
            apply(patcher, engine, vuln(target, 4, category="hardcoded_secret"))
        self.assert_untouched(patcher, target)

    def test_missing_target_file_raises_and_records_nothing(self, patcher, engine, root):
        ghost = root / "core" / "ghost.py"
        with pytest.raises(FileNotFoundError):
            apply(patcher, engine, vuln(ghost, 1))
        assert patcher.get_patch_history() == []


# ─── apply_patch: automatic rollback ──────────────────────────────────────────

class TestAutomaticRollback:

    def test_syntax_error_after_patch_restores_the_original_file(self, patcher, engine, root):
        path = root / "core" / "chain.py"
        original = "import requests\nresp = requests.get(url).json()\n"
        path.write_text(original, encoding="utf-8")

        # The missing_timeout strategy turns this into "requests.get(url).json(, timeout=30)".
        with pytest.raises(RuntimeError, match="syntax error"):
            apply(patcher, engine, vuln(path, 2, "missing_timeout"))

        assert path.read_text(encoding="utf-8") == original
        assert patcher.get_patch_history() == []
        assert json.loads(patcher.patch_log.read_text()) == []

    def test_patcher_remains_usable_after_a_rollback(self, patcher, engine, root, target):
        broken = root / "core" / "chain.py"
        broken.write_text("resp = requests.get(url).json()\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            apply(patcher, engine, vuln(broken, 1, "missing_timeout"))

        record = apply(patcher, engine, vuln(target, 4))

        assert record.success is True
        assert len(patcher.get_patch_history()) == 1

    def test_failed_write_is_rolled_back(self, patcher, engine, target, monkeypatch):
        original_write = pathlib.Path.write_text

        def flaky_write(self, data, *args, **kwargs):
            original_write(self, data[: len(data) // 2], *args, **kwargs)   # truncated write
            raise OSError("disk full")

        monkeypatch.setattr(pathlib.Path, "write_text", flaky_write)

        with pytest.raises(OSError, match="disk full"):
            apply(patcher, engine, vuln(target, 4))

        assert target.read_text(encoding="utf-8") == SOURCE
        assert patcher.get_patch_history() == []


# ─── Manual rollback ──────────────────────────────────────────────────────────

class TestRollback:

    def test_rollback_restores_original_content(self, patcher, engine, target):
        record = apply(patcher, engine, vuln(target, 4))
        assert target.read_text(encoding="utf-8") != SOURCE

        assert patcher.rollback(record.patch_id) is True

        assert target.read_text(encoding="utf-8") == SOURCE
        assert record.rollback_done is True and record.success is False
        assert patcher.get_patch_history()[0]["rollback_done"] is True

    def test_rolling_back_twice_is_harmless(self, patcher, engine, target):
        record = apply(patcher, engine, vuln(target, 4))
        assert patcher.rollback(record.patch_id) is True
        assert patcher.rollback(record.patch_id) is True
        assert target.read_text(encoding="utf-8") == SOURCE

    def test_unknown_patch_id(self, patcher):
        assert patcher.rollback("nonexistent") is False

    def test_missing_backup_makes_rollback_fail_without_crashing(self, patcher, engine, target):
        record = apply(patcher, engine, vuln(target, 4))
        patched_text = target.read_text(encoding="utf-8")
        os.remove(record.backup_path)

        assert patcher.rollback(record.patch_id) is False

        assert target.read_text(encoding="utf-8") == patched_text
        assert record.rollback_done is False

    def test_rollback_all_restores_every_file_and_counts(
        self, patcher, engine, root, target, unique_backup_names
    ):
        second = root / "core" / "second.py"
        second.write_text(SOURCE, encoding="utf-8")
        apply(patcher, engine, vuln(target, 4))
        apply(patcher, engine, vuln(second, 7))

        assert patcher.rollback_all() == 2

        assert target.read_text(encoding="utf-8") == SOURCE
        assert second.read_text(encoding="utf-8") == SOURCE
        assert patcher.rollback_all() == 0

    def test_rollback_all_undoes_several_patches_in_one_file(
        self, patcher, engine, target, unique_backup_names
    ):
        apply(patcher, engine, vuln(target, 4))
        apply(patcher, engine, vuln(target, 7))

        assert patcher.rollback_all() == 2
        assert target.read_text(encoding="utf-8") == SOURCE

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py names backups <stem>_<YYYYmmdd_HHMMSS><suffix>, so two patches "
               "to the same file within one second overwrite each other's backup and "
               "rollback_all can no longer restore the original (apply_report patches many "
               "lines of one file within the same second)",
    )
    def test_rollback_all_after_two_patches_in_the_same_second(
        self, patcher, engine, target, frozen_backup_clock
    ):
        apply(patcher, engine, vuln(target, 4))
        apply(patcher, engine, vuln(target, 7))

        patcher.rollback_all()

        assert target.read_text(encoding="utf-8") == SOURCE

    def test_rollback_all_ignores_patches_already_rolled_back(
        self, patcher, engine, root, target, unique_backup_names
    ):
        second = root / "core" / "second.py"
        second.write_text(SOURCE, encoding="utf-8")
        first_record = apply(patcher, engine, vuln(target, 4))
        apply(patcher, engine, vuln(second, 4))
        patcher.rollback(first_record.patch_id)

        assert patcher.rollback_all() == 1


# ─── apply_report ─────────────────────────────────────────────────────────────

class TestApplyReport:

    @pytest.fixture
    def open_engine(self):
        engine = PermissionEngine()
        engine.grant(USER, Capability.SELF_MODIFY, hours=1)       # wildcard scope
        return engine

    def test_applies_patchable_and_skips_the_rest(
        self, patcher, open_engine, root, target, unique_backup_names, capsys
    ):
        ghost = root / "core" / "ghost.py"
        report = ScanReport(aria_root=str(root), vulnerabilities=[
            vuln(target, 4),
            vuln(target, 7),
            vuln(target, 1),                                       # strategy not applicable
            vuln(ghost, 1),                                        # file missing
            vuln(target, 4, patch_safe=False),                     # needs human review
        ])

        records = patcher.apply_report(report, USER, open_engine)

        assert [r.line_number for r in records] == [4, 7]
        assert all(r.success for r in records)
        assert "Applied 2/4 patches" in capsys.readouterr().out
        text = target.read_text(encoding="utf-8")
        assert text.count("ast.literal_eval(s)") == 2

    def test_only_severity_filters_the_work(self, patcher, open_engine, target, root,
                                            unique_backup_names):
        report = ScanReport(aria_root=str(root), vulnerabilities=[
            vuln(target, 4, severity="critical"),
            vuln(target, 7, severity="low"),
        ])

        records = patcher.apply_report(report, USER, open_engine, only_severity=["critical"])

        assert [r.line_number for r in records] == [4]
        assert "return eval(s)" in target.read_text(encoding="utf-8").splitlines()[6]

    def test_without_permission_nothing_is_changed_and_nothing_raises(
        self, patcher, engine, target, root
    ):
        report = ScanReport(aria_root=str(root), vulnerabilities=[vuln(target, 4), vuln(target, 7)])

        assert patcher.apply_report(report, USER, engine) == []

        assert target.read_text(encoding="utf-8") == SOURCE

    def test_empty_report(self, patcher, open_engine, root):
        assert patcher.apply_report(ScanReport(aria_root=str(root)), USER, open_engine) == []

    def test_scanner_findings_flow_into_the_patcher(
        self, patcher, open_engine, root, unique_backup_names
    ):
        path = root / "core" / "scanned.py"
        path.write_text("import ast\n\ndef run(s):\n    return eval(s)\n", encoding="utf-8")
        report = SecurityScanner(str(root)).scan(include_dirs=["core"])
        assert any(v.patch_safe and v.line_number == 4 for v in report.vulnerabilities)

        records = patcher.apply_report(report, USER, open_engine)

        assert records
        text = path.read_text(encoding="utf-8")
        assert "ast.literal_eval(s)" in text
        assert "return eval(s)" not in text
        compile(text, str(path), "exec")


# ─── History and audit log ────────────────────────────────────────────────────

class TestHistory:

    def test_print_history_when_empty(self, patcher, capsys):
        patcher.print_patch_history()
        assert "No patches applied yet" in capsys.readouterr().out

    def test_print_history_lists_patches(self, patcher, engine, target, capsys):
        record = apply(patcher, engine, vuln(target, 4))
        patcher.print_patch_history()
        out = capsys.readouterr().out
        assert record.patch_id in out
        assert "mod.py:4" in out
        assert "arbitrary_code_execution" in out
        assert f"approved by {USER}" in out

    def test_corrupt_patch_log_does_not_prevent_startup(self, root):
        (root / ".patch_log.json").write_text("{ not json", encoding="utf-8")
        p = SelfPatcher(str(root))
        assert p.get_patch_history() == []

    def test_get_patch_history_returns_plain_dicts(self, patcher, engine, target):
        apply(patcher, engine, vuln(target, 4))
        entry = patcher.get_patch_history()[0]
        assert isinstance(entry, dict)
        json.dumps(entry)

    @pytest.mark.xfail(
        strict=True,
        reason="bug: patcher.py _load_history parses the patch log and prints the record count "
               "but never populates _patch_history, so after a restart the history is empty, "
               "rollback(patch_id) cannot find earlier patches and the next save overwrites "
               "the log",
    )
    def test_history_survives_a_restart(self, root, engine, target):
        first = SelfPatcher(str(root))
        record = apply(first, engine, vuln(target, 4))

        second = SelfPatcher(str(root))

        assert [h["patch_id"] for h in second.get_patch_history()] == [record.patch_id]

