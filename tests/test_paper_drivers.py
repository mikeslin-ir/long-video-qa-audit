from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "run_wave10_matched_selectors.sh",
        "run_videotree_n900.sh",
        "run_cascade_ksweep.sh",
        "run_xbackbone_ksweep.sh",
        "run_eventqa_anomaly.sh",
        "run_music_avqa_videollama2av.sh",
    ],
)
def test_driver_uses_release_root_and_propagates_failure(tmp_path, name):
    workspace = tmp_path / "release"
    drivers = workspace / "scripts" / "paper_runs"
    drivers.mkdir(parents=True)
    script = drivers / name
    shutil.copy2(ROOT / "scripts" / "paper_runs" / name, script)
    (workspace / "scripts" / "run_qa_benchmark.py").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "python"
    interpreter.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$PWD" > "$DRIVER_CWD"\n'
        'printf "%s\\n" "$@" > "$DRIVER_ARGS"\n'
        'test -f "$1" || exit 43\n'
        'exit 42\n'
    )
    interpreter.chmod(0o755)
    (bin_dir / "python3").symlink_to(interpreter)
    cwd_record = tmp_path / "cwd.txt"
    args_record = tmp_path / "args.txt"
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "PY": str(interpreter),
            "DRIVER_CWD": str(cwd_record),
            "DRIVER_ARGS": str(args_record),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    assert cwd_record.read_text().strip() == str(workspace)
    assert args_record.read_text().splitlines()[0] == "scripts/run_qa_benchmark.py"
    assert not (workspace / "outputs" / ".cascade_ksweep_done").exists()
    assert not (workspace / "outputs" / ".eventqa_done").exists()
