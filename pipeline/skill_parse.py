"""Skill structure parser for the proximal (backward) shrinkage step.

SkillGrad's forward step only ever *adds* to the skill, so the skill package
S = (H, B, Q) accretes over iterations. The proximal step needs to reason about
the skill as a set of discrete, individually-removable *units* so it can
measure each unit's marginal utility and shrink low-utility ones.

This module segments a skill package into coarse units and can materialize an
ablated copy S \\ {unit} (used by ``pipeline.utility`` for leave-one-out
utility estimation). Granularity (M1, coarse):

  - Each ``## `` section of ``SKILL.md`` is one L2 unit (id ``L2:<slug>``).
    The frontmatter (H) and the pre-section preamble (title + intro) are
    structural and are never treated as ablatable units.
  - Each file under ``references/`` is one L3 group (id ``L3:<filename>``),
    together with *all* pointer lines in ``SKILL.md`` that mention it
    (``references/<filename>``). Ablating an L3 group removes both the file
    and its pointer lines, so we never leave an orphan L3 file or a dangling
    pointer — this is the "group" in the group-sparsity analogy.

Nothing here calls a model or an evaluator; it is pure text surgery.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def count_tokens(text: str) -> int:
    """Best-effort token count.

    Uses tiktoken (cl100k_base) when available; otherwise falls back to a
    whitespace word count, which is a coarse but monotone proxy. Character
    length is reported separately by callers and is always exact.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


@dataclass
class Unit:
    """One coarse, ablatable skill unit (an L2 section or an L3 group)."""

    id: str
    kind: str  # "l2_section" | "l3_group"
    header: str
    chars: int
    tokens: int
    # L2 only: line span [line_start, line_end) in SKILL.md
    line_start: int | None = None
    line_end: int | None = None
    # L3 only: reference filename + pointer line indices in SKILL.md
    filename: str | None = None
    pointer_lines: list[int] = field(default_factory=list)


@dataclass
class SkillStructure:
    """Parsed view of a skill package."""

    skill_dir: Path
    skill_md_path: Path
    frontmatter_lines: int
    preamble_chars: int
    l2_units: list[Unit]
    l3_units: list[Unit]

    @property
    def units(self) -> list[Unit]:
        return list(self.l2_units) + list(self.l3_units)

    def unit_by_id(self, unit_id: str) -> Unit:
        for u in self.units:
            if u.id == unit_id:
                return u
        raise KeyError(f"Unit id not found: {unit_id}")

    def totals(self) -> dict:
        l2_chars = sum(u.chars for u in self.l2_units)
        l3_chars = sum(u.chars for u in self.l3_units)
        skill_md_chars = len(self.skill_md_path.read_text(encoding="utf-8"))
        return {
            "n_l2": len(self.l2_units),
            "n_l3": len(self.l3_units),
            "n_units": len(self.units),
            "l2_chars": l2_chars,
            "l3_chars": l3_chars,
            "preamble_chars": self.preamble_chars,
            "skill_md_chars": skill_md_chars,
        }


_SLUG_RE = re.compile("[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def _xlsx_dir(skill_dir: str | Path) -> Path:
    return Path(skill_dir) / "xlsx"


def _split_frontmatter(lines: list[str]) -> tuple[int, int]:
    """Return (frontmatter_end_idx, body_start_idx).

    frontmatter_end_idx is the index of the closing ``---`` (exclusive body
    starts after it). If there is no frontmatter, both are 0.
    """
    if not lines or lines[0].strip() != "---":
        return (0, 0)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return (i, i + 1)
    return (0, 0)


def _section_spans(lines: list[str], body_start: int) -> list[tuple[str, int, int]]:
    """Find ``## `` (level-2) section spans in the body.

    Returns a list of (header_text, start_idx, end_idx) with end exclusive.
    Only lines beginning with exactly ``## `` (not ``#`` or ``###``) start a
    section. Anything between body_start and the first section is preamble.
    """
    heads: list[tuple[str, int]] = []
    for i in range(body_start, len(lines)):
        if re.match(r"^##\s+\S", lines[i]):
            header = lines[i].lstrip("#").strip()
            heads.append((header, i))
    spans: list[tuple[str, int, int]] = []
    for k, (header, start) in enumerate(heads):
        end = heads[k + 1][1] if k + 1 < len(heads) else len(lines)
        spans.append((header, start, end))
    return spans


def parse_skill(skill_dir: str | Path) -> SkillStructure:
    """Parse a skill package into coarse ablatable units."""
    skill_dir = Path(skill_dir)
    skill_md = _xlsx_dir(skill_dir) / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found at {skill_md}")
    text = skill_md.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    fm_end, body_start = _split_frontmatter(lines)
    spans = _section_spans(lines, body_start)
    first_section_start = spans[0][1] if spans else len(lines)
    preamble_chars = sum(len(lines[i]) for i in range(body_start, first_section_start))

    l2_units: list[Unit] = []
    for header, start, end in spans:
        seg = "".join(lines[start:end])
        l2_units.append(
            Unit(
                id=f"L2:{_slug(header)}",
                kind="l2_section",
                header=header,
                chars=len(seg),
                tokens=count_tokens(seg),
                line_start=start,
                line_end=end,
            )
        )

    l3_units: list[Unit] = []
    refs_dir = _xlsx_dir(skill_dir) / "references"
    if refs_dir.is_dir():  # a model may clobber references/ into a file; skip if so
        for f in sorted(refs_dir.iterdir()):
            if not f.is_file():
                continue
            file_text = f.read_text(encoding="utf-8")
            fchars = len(file_text)
            pointer_lines = [
                i for i, ln in enumerate(lines) if f"references/{f.name}" in ln
            ]
            pointer_chars = sum(len(lines[i]) for i in pointer_lines)
            l3_units.append(
                Unit(
                    id=f"L3:{f.name}",
                    kind="l3_group",
                    header=f.name,
                    chars=fchars + pointer_chars,
                    tokens=count_tokens(file_text),
                    filename=f.name,
                    pointer_lines=pointer_lines,
                )
            )

    return SkillStructure(
        skill_dir=skill_dir,
        skill_md_path=skill_md,
        frontmatter_lines=fm_end,
        preamble_chars=preamble_chars,
        l2_units=l2_units,
        l3_units=l3_units,
    )


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    """Collapse 3+ consecutive blank lines into a single blank line."""
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if ln.strip() == "":
            blanks += 1
            if blanks >= 2:
                continue
        else:
            blanks = 0
        out.append(ln)
    return out


def materialize_without(
    skill_dir: str | Path, unit_id: str, dest_dir: str | Path
) -> Path:
    """Copy the skill package to ``dest_dir`` with ``unit_id`` removed.

    Returns the destination skill dir (parent of xlsx/), ready to hand to the
    evaluator as ``--skill-dir``.
      - L2 section: the ``## `` block is dropped from SKILL.md.
      - L3 group: the ``references/<file>`` is deleted and every pointer line
        mentioning it is dropped from SKILL.md (no orphan / no dangling link).
    """
    skill_dir = Path(skill_dir)
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(skill_dir, dest_dir)

    struct = parse_skill(skill_dir)
    unit = struct.unit_by_id(unit_id)

    dest_md = _xlsx_dir(dest_dir) / "SKILL.md"
    lines = dest_md.read_text(encoding="utf-8").splitlines(keepends=True)

    if unit.kind == "l2_section":
        drop = set(range(unit.line_start, unit.line_end))
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        dest_md.write_text("".join(_collapse_blank_runs(kept)), encoding="utf-8")
        return dest_dir

    if unit.kind == "l3_group":
        drop = set(unit.pointer_lines)
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        dest_md.write_text("".join(_collapse_blank_runs(kept)), encoding="utf-8")
        ref_path = _xlsx_dir(dest_dir) / "references" / unit.filename
        if ref_path.exists():
            ref_path.unlink()
        return dest_dir

    raise ValueError(f"Unknown unit kind: {unit.kind}")


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Inspect a skill's ablatable units")
    ap.add_argument("--skill-dir", required=True, help="Parent of xlsx/")
    args = ap.parse_args()

    s = parse_skill(args.skill_dir)
    print(json.dumps(s.totals(), indent=2))
    print("\nUnits:")
    for u in s.units:
        extra = f" pointers={u.pointer_lines}" if u.kind == "l3_group" else ""
        print(f"  {u.id:40s} chars={u.chars:6d} tokens={u.tokens:5d}{extra}")
