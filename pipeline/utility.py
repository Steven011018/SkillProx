"""Rule-utility audit — the "coordinate magnitude" of the proximal step.

In proximal gradient descent the backward (prox) step shrinks coordinates by
their magnitude |v_i|. On a discrete skill there is no analytic magnitude, so
we *measure* it: the marginal utility of unit i is the drop in validation
score when that unit is ablated (leave-one-out):

    u_i = Val(S) - Val(S \\ {i})

    u_i > 0   the unit helps         (removing it hurts)   -> keep
    u_i ≈ 0   dead weight            (removing it no-ops)   -> shrink/prune
    u_i < 0   actively harmful       (removing it helps)    -> prune (best win)

This module runs the audit by reusing the existing evaluator
(``runners.stream_runner.run_stream``) on the held-out ``val`` split. It does
NOT perform any shrinkage — it only produces the utility signal that a later
proximal (shrinkage) step will threshold. This is the M1 go/no-go check for
whether SkillGrad's accretive bias has actually left dead/harmful units in the
trained skill.

Val(S) is reported on two metrics:
  - cell accuracy  -> primary u_i signal (continuous, less quantized on 20 val)
  - hard accuracy  -> gate / headline metric (matches the paper's口径)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from data.layout import normalized_dir_for, splits_dir_for
from data.split import load_split
from pipeline.execution import load_dataset
from pipeline.skill_parse import Unit, parse_skill, materialize_without
from runners.stream_runner import run_stream


def _safe_label(unit_id: str) -> str:
    return re.sub("[^A-Za-z0-9._-]+", "_", unit_id)


def _read_summary(workdir: Path) -> dict | None:
    """Return a cached, completed eval_summary.json in workdir, else None."""
    p = workdir / "eval_summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") == "completed":
        return d
    return None


def eval_skill(
    skill_dir: Path,
    indices: list[int],
    dataset: list[dict],
    data_dir: Path,
    model: str,
    workdir: Path,
    *,
    executor_concurrency: int = 4,
    grader_concurrency: int = 1,
    max_turns: int = 30,
    test_cases: int = 1,
    recalc_timeout: int = 180,
    resume: bool = True,
) -> dict:
    """Evaluate one skill folder on ``indices``; return the score dict.

    Reuses a cached ``eval_summary.json`` in ``workdir`` when ``resume`` is set
    and the prior run completed — ablation audits are expensive, so a mid-audit
    crash should not re-pay for finished evals.
    """
    summary = _read_summary(workdir) if resume else None
    if summary is None:
        summary = asyncio.run(
            run_stream(
                indices=indices,
                dataset=dataset,
                skill_dir=skill_dir,
                model=model,
                workdir=workdir,
                data_dir=data_dir,
                project_root=Path(".").resolve(),
                max_turns=max_turns,
                executor_concurrency=executor_concurrency,
                grader_concurrency=grader_concurrency,
                recalc_timeout=recalc_timeout,
                output_format="eval",
                test_cases=test_cases,
            )
        )
    agg = summary["aggregate"]
    return {
        "hard": agg.get("mean_hard_graded", 0),
        "cell": agg.get("mean_cell_graded", 0),
        "n_graded": agg.get("n_graded", 0),
        "n_total": summary.get("total", len(indices)),
        "retry_needed": agg.get("retry_needed", 0),
    }


def _classify(u_cell: float, u_hard: float, eps_cell: float) -> str:
    """Primary classification on cell utility; hard is reported alongside."""
    if u_cell < -eps_cell:
        return "negative"
    if u_cell <= eps_cell:
        return "zero"
    return "positive"


def audit(
    skill_dir: Path,
    split: dict,
    dataset: list[dict],
    data_dir: Path,
    model: str,
    out_dir: Path,
    *,
    val_key: str = "val_indices",
    executor_concurrency: int = 4,
    grader_concurrency: int = 1,
    max_turns: int = 30,
    test_cases: int = 1,
    eps_cell: float = 0.005,
    resume: bool = True,
    ablation_parallel: int = 1,
) -> dict:
    """Leave-one-out utility audit of every unit on the val split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evals_dir = out_dir / "evals"
    cand_dir = out_dir / "candidates"

    val_indices = [int(i) for i in split[val_key]]
    struct = parse_skill(skill_dir)
    units = struct.units
    print(
        f"[audit] {len(units)} units ({len(struct.l2_units)} L2 sections + "
        f"{len(struct.l3_units)} L3 groups) | val={len(val_indices)} tasks | model={model}"
    )

    print("[audit] evaluating baseline (full skill) ...")
    base = eval_skill(
        skill_dir, val_indices, dataset, data_dir, model, evals_dir / "baseline",
        executor_concurrency=executor_concurrency,
        grader_concurrency=grader_concurrency,
        max_turns=max_turns, test_cases=test_cases, resume=resume,
    )
    print(f"[audit] baseline val: hard={base['hard']:.4f} cell={base['cell']:.4f}")

    def _ablate_one(k: int, u: Unit) -> dict:
        label = _safe_label(u.id)
        print(f"[audit] ({k}/{len(units)}) ablating {u.id} ...")
        cand = materialize_without(skill_dir, u.id, cand_dir / label)
        res = eval_skill(
            cand, val_indices, dataset, data_dir, model, evals_dir / label,
            executor_concurrency=executor_concurrency,
            grader_concurrency=grader_concurrency,
            max_turns=max_turns, test_cases=test_cases, resume=resume,
        )
        u_cell = round(base["cell"] - res["cell"], 4)
        u_hard = round(base["hard"] - res["hard"], 4)
        cls = _classify(u_cell, u_hard, eps_cell)
        row = {
            "id": u.id,
            "kind": u.kind,
            "header": u.header,
            "chars": u.chars,
            "tokens": u.tokens,
            "ablated_hard": res["hard"],
            "ablated_cell": res["cell"],
            "u_hard": u_hard,
            "u_cell": u_cell,
            "class": cls,
            "safe_to_shrink": bool(u_hard <= 0),
            "n_graded": res["n_graded"],
        }
        flag = " [SAFE-SHRINK]" if row["safe_to_shrink"] else ""
        print(f"        {u.id}: u_cell={u_cell:+.4f} u_hard={u_hard:+.4f} -> {cls}{flag}")
        return row

    unit_rows: list[dict] = []
    workers = max(1, min(ablation_parallel, len(units)))
    if workers == 1:
        for k, u in enumerate(units, 1):
            unit_rows.append(_ablate_one(k, u))
    else:
        print(f"[audit] parallel ablations: {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_ablate_one, k, u): u.id
                for k, u in enumerate(units, 1)
            }
            for fut in as_completed(futures):
                unit_rows.append(fut.result())

    unit_rows.sort(key=lambda r: (r["u_cell"], r["u_hard"]))
    counts = {c: sum(1 for r in unit_rows if r["class"] == c)
              for c in ("negative", "zero", "positive")}
    n_safe = sum(1 for r in unit_rows if r["safe_to_shrink"])
    shrinkable_chars = sum(r["chars"] for r in unit_rows if r["safe_to_shrink"])

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "skill_dir": str(skill_dir),
        "model": model,
        "val_key": val_key,
        "n_val": len(val_indices),
        "test_cases": test_cases,
        "eps_cell": eps_cell,
        "baseline": base,
        "skill_totals": struct.totals(),
        "class_counts": counts,
        "n_safe_to_shrink": n_safe,
        "shrinkable_chars": shrinkable_chars,
        "units": unit_rows,
    }

    (out_dir / "utility_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_md(out_dir / "utility_audit.md", report)
    print(
        f"[audit] done | negative={counts['negative']} zero={counts['zero']} "
        f"positive={counts['positive']} | safe_to_shrink={n_safe} "
        f"({shrinkable_chars} chars)"
    )
    return report


def _write_md(path: Path, report: dict) -> None:
    """Human-readable audit table, sorted most-shrinkable first."""
    b = report["baseline"]
    lines = [
        "# Utility audit (leave-one-out)",
        "",
        f"- model: `{report['model']}`  |  val: `{report['val_key']}` "
        f"({report['n_val']} tasks)  |  eps_cell={report['eps_cell']}",
        f"- baseline val: hard={b['hard']:.4f}  cell={b['cell']:.4f}",
        f"- class counts: {report['class_counts']}  |  "
        f"safe_to_shrink: {report['n_safe_to_shrink']} "
        f"({report['shrinkable_chars']} chars)",
        "",
        "| # | unit | kind | chars | u_cell | u_hard | class | safe |",
        "|---|------|------|------:|-------:|-------:|-------|------|",
    ]
    for i, r in enumerate(report["units"], 1):
        lines.append(
            f"| {i} | `{r['id']}` | {r['kind']} | {r['chars']} | "
            f"{r['u_cell']:+.4f} | {r['u_hard']:+.4f} | {r['class']} | "
            f"{'yes' if r['safe_to_shrink'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Leave-one-out utility audit of a trained skill."
    )
    ap.add_argument("--skill-dir", required=True, help="Parent of xlsx/.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=42)
    ap.add_argument("--val-key", default="val_indices")
    ap.add_argument("--executor-concurrency", type=int, default=4)
    ap.add_argument("--grader-concurrency", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--test-cases", type=int, default=1)
    ap.add_argument("--eps-cell", type=float, default=0.005)
    ap.add_argument("--ablation-parallel", type=int, default=1,
                    help="Run this many LOO ablation val-evals concurrently.")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    data_dir = Path(args.data_dir) if args.data_dir else normalized_dir_for(results_root)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset dir {data_dir} not found.")
    split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
    split = load_split(split_dir)
    dataset = load_dataset(data_dir)

    audit(
        Path(args.skill_dir), split, dataset, data_dir, args.model,
        Path(args.out_dir),
        val_key=args.val_key,
        executor_concurrency=args.executor_concurrency,
        grader_concurrency=args.grader_concurrency,
        max_turns=args.max_turns,
        test_cases=args.test_cases,
        eps_cell=args.eps_cell,
        resume=not args.no_resume,
        ablation_parallel=args.ablation_parallel,
    )


if __name__ == "__main__":
    main()
