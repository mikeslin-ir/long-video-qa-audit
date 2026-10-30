"""Load skill cost / use-case hints from the ``skills/*.skill.md`` cards.

The Markdown cards in ``<repo>/skills/`` are the single, human-editable source
of truth for the planner's per-skill ``cost`` and one-line ``hint`` that the LLM
reads when choosing tools. This module parses the YAML-style front-matter of
each card and returns a ``{skill_name: "cost  (hint)"}`` mapping in exactly the
format the planner prompt expects.

The parser is intentionally tiny (no PyYAML dependency): it only reads the
``name``, ``cost`` and ``hint`` keys from the leading ``---`` front-matter
block, splitting on the first colon and stripping surrounding quotes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _dequote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_front_matter(text: str) -> dict[str, str]:
    """Return the key/value pairs of the leading ``---`` front-matter block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        fm[key.strip()] = _dequote(raw)
    return fm


@lru_cache(maxsize=1)
def load_skill_hints(skills_dir: str | None = None) -> dict[str, str]:
    """Map each skill name to its ``"cost  (hint)"`` prompt string from cards.

    Missing cards or a missing ``skills/`` directory yield an empty dict, so the
    planner can fall back to its inline defaults and never hard-fail.
    """
    root = Path(skills_dir) if skills_dir else _SKILLS_DIR
    hints: dict[str, str] = {}
    if not root.is_dir():
        return hints
    for card in sorted(root.glob("*.skill.md")):
        try:
            fm = _parse_front_matter(card.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = fm.get("name")
        cost = fm.get("cost")
        hint = fm.get("hint")
        if name and cost is not None and hint:
            hints[name] = f"{cost}  ({hint})"
    return hints
