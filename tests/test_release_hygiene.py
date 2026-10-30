from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = sorted((ROOT / "scripts" / "paper_runs").glob("*.sh"))
PUBLIC_TEXT = [*ROOT.glob("*.md"), *ROOT.glob("docs/*.md"), *ROOT.glob("scripts/**/*.py"),
               *ROOT.glob("scripts/**/*.sh"), *ROOT.glob("*.py"), *ROOT.glob("*.cff"),
               *ROOT.glob("*.bib"), *ROOT.glob(".github/**/*.yml")]


@pytest.mark.parametrize("script", DRIVERS, ids=lambda p: p.name)
def test_driver_is_valid_bash(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


RUN_DRIVERS = [p for p in DRIVERS if not p.name.startswith("setup_")]


@pytest.mark.parametrize("script", RUN_DRIVERS, ids=lambda p: p.name)
def test_driver_changes_to_the_repository_root(script):
    """Drivers live two levels below the root and must `cd` there, not into scripts/."""
    cd_lines = [ln for ln in script.read_text().splitlines() if re.match(r"\s*cd\s", ln)]
    root_cd = [ln for ln in cd_lines if '$(dirname "$0")/../..' in ln or '"$REPO"' in ln]
    assert root_cd, f"{script.name}: no `cd` to the repository root ({cd_lines})"
    assert not any(re.search(r'dirname "\$0"\)/\.\.["\s]', ln) for ln in cd_lines)


@pytest.mark.parametrize("script", DRIVERS, ids=lambda p: p.name)
def test_driver_scripts_exist(script):
    for rel in set(re.findall(r"\bscripts/[\w./-]+\.(?:py|sh)\b", script.read_text())):
        assert (ROOT / rel).exists(), f"{script.name} references missing {rel}"


def test_no_author_machine_paths_in_released_files():
    offenders = []
    for path in PUBLIC_TEXT:
        if path.name == "test_release_hygiene.py":
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if "/home/mikeslin" in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, offenders
