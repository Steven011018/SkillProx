"""Proximal (backward) shrinkage step for a trained skill.

SkillGrad's forward loop (execute -> diagnose -> momentum -> patch) only adds,
so the trained skill accretes redundant, over-specific, or mildly harmful
content. This module runs the missing backward half: a greedy, validation-gated
sequence of shrinkage edits driven by the measured marginal utility u_i from
`pipeline.utility`.

For each candidate unit (lowest utility first; positive-utility units are
protected), a Shrinker agent consolidates / demotes / removes it, orphaned L3
files are garbage-collected, and the trial skill is re-scored on the val split.
The edit is accepted only if val accuracy does not drop (conservative, because
20-task val is noisy); otherwise it is rolled back. The final skill S_prox is
the composition of all accepted edits, to be compared against the baseline on
the held-out test split.

This is the "A" operator (post-hoc consolidation). Interleaving with forward
steps ("C") builds on the same primitives.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from agents import Agent, Runner

from data.layout import normalized_dir_for, splits_dir_for
from data.split import load_split
from pipeline.execution import load_dataset
from pipeline.helpers import _build_file_tools, _resolve_model
from pipeline.skill_parse import parse_skill
from pipeline.utility import eval_skill, _safe_label
from prompts.shrinkage import SHRINKAGE_PROMPT
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_settings
from runners.trajectory_logger import TrajectoryLogger, stream_with_logging


def _skill_chars(skill_dir: Path) -> int:
    xlsx = Path(skill_dir) / "xlsx"
    total = len((xlsx / "SKILL.md").read_text(encoding="utf-8"))
    refs = xlsx / "references"
    if refs.is_dir():  # a model may clobber references/ into a file; skip if so
        for f in refs.iterdir():
            if f.is_file():
                total += len(f.read_text(encoding="utf-8"))
    return total


def _gc_orphans(skill_dir: Path) -> list[str]:
    """Delete reference files that no longer have any pointer in SKILL.md.

    This is how the Shrinker "removes" an L3 chapter: it deletes the pointer
    line, and we collect the now-unreferenced file. Returns deleted filenames.
    """
    struct = parse_skill(skill_dir)
    md_text = struct.skill_md_path.read_text(encoding="utf-8")
    refs = Path(skill_dir) / "xlsx" / "references"
    deleted: list[str] = []
    if refs.is_dir():  # a model may clobber references/ into a file; skip if so
        for f in sorted(refs.iterdir()):
            if f.is_file() and f"references/{f.name}" not in md_text:
                f.unlink()
                deleted.append(f.name)
    return deleted


def _find_target(skill_dir: Path, unit_id: str):
    """Locate a unit in the *current* skill by its audit id (header/filename).

    Returns the matching Unit or None if it has already been shrunk away.
    """
    struct = parse_skill(skill_dir)
    for u in struct.units:
        if u.id == unit_id:
            return u
    return None


def _describe_target(unit) -> str:
    if unit.kind == "l2_section":
        return f"the L2 section headed `## {unit.header}` in SKILL.md ({unit.chars} chars)"
    return (
        f"the L3 reference file `references/{unit.filename}` "
        f"({unit.chars} chars incl. its pointer text) and its pointer line(s) in SKILL.md"
    )


async def run_shrink_agent(
    skill_dir: Path,
    unit,
    model: str,
    project_root: Path,
    cost_tracker: CostTracker,
    log_path: Path,
) -> str:
    """Run the Shrinker agent in-place on `skill_dir` for one target unit."""
    read_file, write_file = _build_file_tools(project_root)
    agent = Agent(
        name="Shrinker",
        instructions=SHRINKAGE_PROMPT,
        model=_resolve_model(model),
        model_settings=get_model_settings(model),
        tools=[read_file, write_file],
    )

    xlsx = Path(skill_dir) / "xlsx"
    skill_files = [str(xlsx / "SKILL.md")]
    refs = xlsx / "references"
    if refs.is_dir():  # a model may clobber references/ into a file; skip if so
        for f in sorted(refs.iterdir()):
            if f.is_file():
                skill_files.append(str(f))

    query = (
        "Shrink the skill by processing exactly one target unit.\n\n"
        f"Target unit (flagged low/negative marginal value): {_describe_target(unit)}\n\n"
        "Skill files:\n"
        + "\n".join(f"  - {p}" for p in skill_files)
        + "\n\nSalvage any genuinely useful, generalizable content from the target "
        "into the most-overlapping retained section, then remove the redundant / "
        "contradictory / verbose remainder. Leave all other sections unchanged. "
        "Make the skill strictly smaller."
    )

    logger = TrajectoryLogger(log_path)
    output = ""
    try:
        result = Runner.run_streamed(agent, query, max_turns=20)
        await stream_with_logging(result, logger)
        output = result.final_output or ""
        delta = cost_tracker.update(result)
        cost_tracker.print_step("SHRINK", delta)
    except Exception as e:
        print(f"  [shrink] error: {e}")
        output = f"[SHRINK ERROR] {e}"
    logger.flush()
    return output


def prox(
    skill_dir: Path,
    audit_path: Path,
    split: dict,
    dataset: list[dict],
    data_dir: Path,
    model: str,
    out_dir: Path,
    *,
    val_key: str = "val_indices",
    hard_tol: float = 0.0,
    cell_tol: float = 0.02,
    executor_concurrency: int = 4,
    test_cases: int = 1,
    resume: bool = True,
    gate_metric: str = "hard",
    require_improve: bool = False,
    accept_margin: float = 0.0,
    sat_guard: float | None = None,
    max_compression: float | None = None,
    min_neg_utility: float | None = None,
) -> dict:
    """Greedy, val-gated proximal shrinkage. Returns the run report.

    Extra knobs (all default to the original val-gated tie-accept behavior):
      gate_metric      primary val metric for the accept gate ("hard"|"cell").
      require_improve  accept a cut only if the primary metric strictly improves
                       by > accept_margin (kills "tie accepted" over-shrinkage).
      accept_margin    margin for require_improve.
      sat_guard        if start val (primary metric) >= this, skip prox entirely
                       (val is saturated -> gate is uninformative -> ship input).
      max_compression  stop accepting cuts once cumulative compression >= this.
      min_neg_utility  only consider units with u_cell < this as candidates
                       (protect zero-utility units); None keeps negative+zero.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(".").resolve()

    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    units = audit["units"]
    # Accept both audit schemas: current utility.py writes "class"; older
    # audit files use "cls". Normalize to a single key so candidate selection
    # works regardless of which produced the file.
    def _cls(u: dict) -> str | None:
        return u.get("class") or u.get("cls")
    for u in units:
        u["cls"] = _cls(u)
    if min_neg_utility is not None:
        # Method D: only cut units with a clearly negative cell-utility; treat
        # zero-utility units as protected (they may help on test even if val
        # cannot measure it).
        cand_units = [u for u in units if u.get("u_cell", 0.0) < min_neg_utility]
        protected = [u["id"] for u in units if u.get("u_cell", 0.0) >= min_neg_utility]
    else:
        cand_units = [u for u in units if u.get("cls") in ("negative", "zero")]
        protected = [u["id"] for u in units if u.get("cls") == "positive"]
    candidates = sorted(
        cand_units,
        key=lambda u: (u.get("u_cell", 0), u.get("u_hard", 0)),
    )
    print(f"[prox] {len(candidates)} candidate units (protected: {protected})")

    accepted = out_dir / "skill_prox"
    if accepted.exists():
        shutil.rmtree(accepted)
    shutil.copytree(skill_dir, accepted)

    cost = CostTracker(model)
    val_indices = [int(i) for i in split[val_key]]
    acc_val = eval_skill(
        accepted, val_indices, dataset, data_dir, model, out_dir / "evals" / "start",
        executor_concurrency=executor_concurrency, test_cases=test_cases, resume=resume,
    )
    start_val = dict(acc_val)
    start_chars = _skill_chars(accepted)
    print(
        f"[prox] start val: hard={acc_val['hard']:.4f} cell={acc_val['cell']:.4f} "
        f"chars={start_chars}"
    )

    gate_key = "cell" if gate_metric == "cell" else "hard"

    # Method B saturation guard: when the val gate is already maxed out it can no
    # longer tell a harmful cut from a harmless one, so prox degenerates into pure
    # length minimization. Skip entirely and ship the input skill unchanged.
    if sat_guard is not None and start_val[gate_key] >= sat_guard:
        print(
            f"[prox] SAT-GUARD: start val {gate_key}={start_val[gate_key]:.4f} "
            f">= {sat_guard} -> skip shrinkage, ship input skill as-is."
        )
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "skill_dir": str(skill_dir),
            "out_dir": str(out_dir),
            "model": model,
            "val_key": val_key,
            "gate_metric": gate_key,
            "sat_guard": sat_guard,
            "skipped_saturated": True,
            "start_val": start_val,
            "final_val": acc_val,
            "start_chars": start_chars,
            "final_chars": start_chars,
            "compression": 0.0,
            "n_candidates": len(candidates),
            "n_accepted": 0,
            "protected": protected,
            "log": [],
        }
        (out_dir / "shrinkage_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    log: list[dict] = []
    for k, cu in enumerate(candidates, 1):
        uid = cu["id"]
        target = _find_target(accepted, uid)
        if target is None:
            print(f"[prox] ({k}/{len(candidates)}) {uid}: already shrunk away, skip")
            log.append({"unit": uid, "action": "skip_gone"})
            continue

        trial = out_dir / "trial"
        if trial.exists():
            shutil.rmtree(trial)
        shutil.copytree(accepted, trial)
        before_chars = _skill_chars(trial)
        print(
            f"[prox] ({k}/{len(candidates)}) shrinking {uid} "
            f"(cls={cu.get('cls')}, u_cell={cu.get('u_cell')}) ..."
        )

        asyncio.run(
            run_shrink_agent(
                trial, target, model, project_root, cost,
                out_dir / "logs" / f"{_safe_label(uid)}.jsonl",
            )
        )
        deleted = _gc_orphans(trial)

        try:
            parse_skill(trial)
            ok_struct = True
            struct_err = ""
        except Exception as e:
            ok_struct = False
            struct_err = str(e)
            print(f"[prox]   structure broke: {e}")

        after_chars = _skill_chars(trial)
        if not ok_struct or after_chars >= before_chars:
            reason = "bad_structure" if not ok_struct else "not_smaller"
            print(f"[prox]   reject ({reason}) | chars {before_chars}->{after_chars}")
            log.append({
                "unit": uid, "action": f"reject_{reason}",
                "chars_before": before_chars, "chars_after": after_chars,
                "ok_struct": ok_struct, "struct_err": struct_err,
                "accepted": False,
            })
            shutil.rmtree(trial)
            continue

        trial_val = eval_skill(
            trial, val_indices, dataset, data_dir, model,
            out_dir / "evals" / _safe_label(uid),
            executor_concurrency=executor_concurrency, test_cases=test_cases, resume=resume,
        )
        d_hard = round(trial_val["hard"] - acc_val["hard"], 4)
        d_cell = round(trial_val["cell"] - acc_val["cell"], 4)

        # Primary gate: either strictly improve (method B) or don't-drop (tol).
        prim_tol = hard_tol if gate_key == "hard" else cell_tol
        d_prim = trial_val[gate_key] - acc_val[gate_key]
        if require_improve:
            prim_ok = d_prim > accept_margin
        else:
            prim_ok = trial_val[gate_key] >= acc_val[gate_key] - prim_tol
        # Secondary gate: the other metric must not drop beyond its own tol
        # (method C tightens cell_tol to 0 so cell can never regress).
        sec_key = "cell" if gate_key == "hard" else "hard"
        sec_tol = cell_tol if sec_key == "cell" else hard_tol
        sec_ok = trial_val[sec_key] >= acc_val[sec_key] - sec_tol
        # Method D compression cap: stop cutting once we've compressed enough.
        if max_compression is not None and start_chars:
            cur_comp = 1 - _skill_chars(accepted) / start_chars
            comp_ok = cur_comp < max_compression
        else:
            comp_ok = True
        accept = prim_ok and sec_ok and comp_ok
        entry = {
            "unit": uid, "cls": cu.get("cls"), "u_cell": cu.get("u_cell"),
            "chars_before": before_chars, "chars_after": after_chars,
            "gc_deleted": deleted,
            "val_before": {"hard": acc_val["hard"], "cell": acc_val["cell"]},
            "val_after": {"hard": trial_val["hard"], "cell": trial_val["cell"]},
            "d_hard": d_hard, "d_cell": d_cell, "accepted": accept,
        }
        if accept:
            shutil.rmtree(accepted)
            shutil.move(str(trial), str(accepted))
            acc_val = dict(trial_val)
            print(
                f"[prox]   ACCEPT | chars {before_chars}->{after_chars} | "
                f"val hard={acc_val['hard']:.4f} cell={acc_val['cell']:.4f}"
            )
        else:
            shutil.rmtree(trial)
            reason = "cap" if not comp_ok else ("no-improve" if require_improve else "val drop")
            entry["reject_reason"] = reason
            print(
                f"[prox]   reject ({reason}) | d_hard={d_hard:+.4f} d_cell={d_cell:+.4f}"
            )
        log.append(entry)

    final_chars = _skill_chars(accepted)
    compression = round(1 - final_chars / start_chars, 4) if start_chars else 0.0
    n_accepted = sum(1 for e in log if e.get("accepted"))
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "skill_dir": str(skill_dir),
        "out_dir": str(out_dir),
        "model": model,
        "val_key": val_key,
        "hard_tol": hard_tol,
        "cell_tol": cell_tol,
        "gate_metric": gate_key,
        "require_improve": require_improve,
        "accept_margin": accept_margin,
        "sat_guard": sat_guard,
        "max_compression": max_compression,
        "min_neg_utility": min_neg_utility,
        "start_val": start_val,
        "final_val": acc_val,
        "start_chars": start_chars,
        "final_chars": final_chars,
        "compression": compression,
        "n_candidates": len(candidates),
        "n_accepted": n_accepted,
        "protected": protected,
        "log": log,
    }
    (out_dir / "shrinkage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"[prox] DONE | chars {start_chars}->{final_chars} "
        f"(-{compression * 100:.1f}%) | accepted {n_accepted}/{len(candidates)} | "
        f"val hard {start_val['hard']:.4f}->{acc_val['hard']:.4f} "
        f"cell {start_val['cell']:.4f}->{acc_val['cell']:.4f}"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Proximal (backward) shrinkage of a trained skill."
    )
    ap.add_argument("--skill-dir", required=True, help="Parent of xlsx/.")
    ap.add_argument("--audit", required=True,
                    help="utility_audit_clean.json from the utility audit.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=42)
    ap.add_argument("--val-key", default="val_indices")
    ap.add_argument("--hard-tol", type=float, default=0.0)
    ap.add_argument("--cell-tol", type=float, default=0.02)
    ap.add_argument("--executor-concurrency", type=int, default=4)
    ap.add_argument("--test-cases", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--gate-metric", choices=["hard", "cell"], default="hard",
                    help="Primary val metric for the accept gate.")
    ap.add_argument("--require-improve", action="store_true",
                    help="Accept a cut only if the gate metric strictly improves "
                         "by > --accept-margin (method B).")
    ap.add_argument("--accept-margin", type=float, default=0.0)
    ap.add_argument("--sat-guard", type=float, default=None,
                    help="If start val (gate metric) >= this, skip prox entirely "
                         "(method B saturation guard).")
    ap.add_argument("--max-compression", type=float, default=None,
                    help="Stop accepting cuts once compression >= this (method D).")
    ap.add_argument("--min-neg-utility", type=float, default=None,
                    help="Only cut units with u_cell < this; protect the rest "
                         "(method D).")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_dir = Path(args.data_dir) if args.data_dir else normalized_dir_for(results_root)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset dir {data_dir} not found.")
    split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
    split = load_split(split_dir)
    dataset = load_dataset(data_dir)

    prox(
        Path(args.skill_dir), Path(args.audit), split, dataset, data_dir, args.model,
        Path(args.out_dir),
        val_key=args.val_key,
        hard_tol=args.hard_tol,
        cell_tol=args.cell_tol,
        executor_concurrency=args.executor_concurrency,
        test_cases=args.test_cases,
        resume=not args.no_resume,
        gate_metric=args.gate_metric,
        require_improve=args.require_improve,
        accept_margin=args.accept_margin,
        sat_guard=args.sat_guard,
        max_compression=args.max_compression,
        min_neg_utility=args.min_neg_utility,
    )


if __name__ == "__main__":
    main()
