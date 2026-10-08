"""The ARIA version must come from one place and agree everywhere."""
import re
from pathlib import Path

import aria
from aria._version import __version__
from aria.core.models import ARIA_ORIGIN
from aria.core.self_aware.consciousness import HealthSnapshot

ROOT = Path(__file__).resolve().parent.parent


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_package_exposes_the_single_version():
    assert aria.__version__ == __version__


def test_pyproject_reads_version_from_the_single_source():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "aria._version.__version__" }' in text
    # No hard coded version may remain in the [project] table.
    project_table = text.split("[project.optional-dependencies]")[0]
    assert not re.search(r'^version\s*=\s*"', project_table, re.MULTILINE)


def test_origin_record_uses_the_version():
    assert ARIA_ORIGIN["version"] == __version__


def test_health_snapshot_defaults_to_the_version():
    assert HealthSnapshot.__dataclass_fields__["aria_version"].default == __version__


def test_readme_header_and_changelog_match():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Version {__version__} |" in readme.splitlines()[2]
    assert f"**v{__version__}**" in readme
