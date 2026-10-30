"""Figure 1 entry point: builds the three-panel motivation design."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_fig1_motivation.py")),
                   run_name="__main__")
