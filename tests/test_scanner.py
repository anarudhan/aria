"""
Tests for ARIA Security Scanner
"""
import pytest
import tempfile
from pathlib import Path
from aria.security.scanner import SecurityScanner, ScanReport, Vulnerability


@pytest.fixture
def scanner(tmp_path):
    return SecurityScanner(str(tmp_path))


def _all(report): return report.vulnerabilities


def test_detects_eval(tmp_path):
    (tmp_path / "test_eval.py").write_text("result = eval(user_input)\n")
    report = SecurityScanner(str(tmp_path)).scan()
    assert any(v.category == "arbitrary_code_execution" for v in _all(report))


def test_detects_hardcoded_secret(tmp_path):
    (tmp_path / "config.py").write_text('API_KEY = "sk-prod-abc123xyz456"\npassword = "hunter2"\n')
    report = SecurityScanner(str(tmp_path)).scan()
    assert any(v.category == "hardcoded_secret" for v in _all(report))


def test_detects_shell_true(tmp_path):
    (tmp_path / "runner.py").write_text("import subprocess\nsubprocess.run(cmd, shell=True)\n")
    report = SecurityScanner(str(tmp_path)).scan()
    assert any(v.category == "shell_injection" for v in _all(report))


def test_detects_missing_timeout(tmp_path):
    (tmp_path / "client.py").write_text("import requests\nrequests.get('https://example.com')\n")
    report = SecurityScanner(str(tmp_path)).scan()
    assert any(v.category == "missing_timeout" for v in _all(report))


def test_detects_tls_disabled(tmp_path):
    (tmp_path / "client.py").write_text("import requests\nrequests.get(url, verify=False)\n")
    report = SecurityScanner(str(tmp_path)).scan()
    assert any(v.category == "tls_disabled" for v in _all(report))


def test_clean_file_has_no_findings(tmp_path):
    (tmp_path / "clean.py").write_text(
        'def add(a: int, b: int) -> int:\n'
        '    """Add two numbers."""\n'
        '    return a + b\n'
    )
    report = SecurityScanner(str(tmp_path)).scan()
    assert len(_all(report)) == 0


def test_scan_report_severity_filters(tmp_path):
    (tmp_path / "mixed.py").write_text(
        'result = eval(x)\nimport requests\nrequests.get(url)\n'
    )
    report = SecurityScanner(str(tmp_path)).scan()
    all_f = report.critical + report.high + report.medium + report.low
    assert len(all_f) == len(_all(report))


def test_auto_patchable_vs_requires_human(tmp_path):
    (tmp_path / "test.py").write_text('eval(x)\nAPI_KEY = "sk-prod-secret"\n')
    report = SecurityScanner(str(tmp_path)).scan()
    patchable_ids = {v.vuln_id for v in report.auto_patchable}
    human_ids     = {v.vuln_id for v in report.requires_human}
    all_ids       = {v.vuln_id for v in _all(report)}
    assert patchable_ids | human_ids == all_ids
    assert patchable_ids & human_ids == set()


def test_scan_skips_non_python_files(tmp_path):
    (tmp_path / "readme.md").write_text("eval(x)\n")
    (tmp_path / "config.json").write_text('{"key": "secret_password_123"}\n')
    report = SecurityScanner(str(tmp_path)).scan()
    assert report.files_scanned == 0


def test_vulnerability_has_required_fields(tmp_path):
    (tmp_path / "bad.py").write_text("eval(user_input)\n")
    report = SecurityScanner(str(tmp_path)).scan()
    assert len(_all(report)) > 0
    vuln = _all(report)[0]
    assert vuln.vuln_id
    assert vuln.severity in ("critical", "high", "medium", "low")
    assert vuln.category
    assert vuln.file_path
    assert vuln.line_number > 0
    assert vuln.description
    assert vuln.suggestion
    assert isinstance(vuln.patch_safe, bool)


def test_print_report_does_not_crash(tmp_path, capsys):
    (tmp_path / "test.py").write_text("eval(x)\n")
    report = SecurityScanner(str(tmp_path)).scan()
    report.print_report()
    assert len(capsys.readouterr().out) > 0
