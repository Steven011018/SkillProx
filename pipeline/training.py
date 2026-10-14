"""SkillGrad training loop.

One iteration:

    EXECUTE  →  CLASSIFY (failed vs contrastive successes)
             →  DIAGNOSE (per task, in parallel)
             →  MOMENTUM (cross-iteration pattern record + overlay)
             →  PATCH    (layer-aware in-place edit of the skill)

The optimization analogy:

    parameter         = the structured skill package S = (H, B, Q)
    loss evidence     = task outcomes + trajectory traces
    gradient          = per-task textual diagnoses
    momentum          = pattern memory M_t and overlay O_t
    parameter update  = layer-aware skill patch
"""

import argparse
import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from data.layout import (
    base_trajectories_dir_for,
    normalized_dir_for,
    run_dir_for,
    run_id_for,
    splits_dir_for,
)
from data.split import (
    build_fixed_update_training_set,
    build_training_set,
    create_split,
    identify_failures,
    load_split,
)
from pipeline.diagnoser import assemble_diagnoses, classify_batch, run_diagnose
from pipeline.execution import (
    _write_workspace,
    load_dataset,
    prepare_seed_data,
    run_execute,
    assess_seed,
)
from pipeline.momentum import run_momentum
from pipeline.patcher import run_patch
from runners.cost_tracker import CostTracker
from runners.model_dispatch import get_client_for_model
from scripts.manifest_update import upsert as manifest_upsert


# ═══════════════════════════════════════════════════════════════════════════
# Skill snapshot (orchestrator-private, kept for forensic inspection)
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_skill(skills_dir: Path, dest: Path):
    """Copy the skill folder to dest (per-iteration snapshot)."""
    skill_dir = skills_dir / "xlsx"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "xlsx").exists():
        shutil.rmtree(dest / "xlsx")
    shutil.copytree(skill_dir, dest / "xlsx")


def _restore_skill(snapshot_dir: Path, skills_dir: Path):
    """Restore the in-place skill from a snapshot produced by _snapshot_skill.

    Used by the closed-loop retry loop: each patch attempt starts from the
    same pre-patch base, and the winning attempt (or the pristine base, if no
    attempt cleared the gate) is restored at the end.
    """
    src = snapshot_dir / "xlsx"
    dst = skills_dir / "xlsx"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

async def run_training(
    data_dir: Path,
    skills_dir: Path,
    model: str,
    project_root: Path,
    output_dir: Path,
    base_trajectories_dir: Path,
    training_config: dict,
    max_turns: int = 30,
    concurrency: int = 4,
    interactive: bool = False,
    start_iteration: int = 1,
    closed_loop: dict | None = None,
):
    """Run the SkillGrad training loop.

    Args:
        data_dir: Path to normalized dataset.
        skills_dir: Path to skill directory (modified in-place).
        model: Model name (e.g., "gpt-5.4").
        project_root: Project root for file path resolution.
        output_dir: Where to save iteration artifacts.
        base_trajectories_dir: Directory with base skill trajectories.
        training_config: From build_training_set() — contains batches.
        max_turns: Max agent turns per execution.
        concurrency: Max parallel executions.
        interactive: If True, pause after each iteration for human review.
        start_iteration: Resume from this iteration (1-indexed).
    """
    dataset = load_dataset(data_dir)
    batches = training_config["batches"]
    n_iterations = len(batches)

    cost_tracker = CostTracker(model)
    semaphore = asyncio.Semaphore(concurrency)
    openai_client = get_client_for_model(model)

    # ── Closed-loop config ────────────────────────────────────────────────
    cl = dict(closed_loop or {})
    cl_enabled = bool(cl.get("enabled", False))
    cl_max_attempts = int(cl.get("max_attempts", 3))   # DIAGNOSE→PATCH tries
    cl_hard_tol = int(cl.get("hard_tol", 0))           # allowed hard regression
    cl_prior_k = int(cl.get("prior_k", 6))             # memory entries injected
    # Accept/reject memory: compact per-iteration records injected into the
    # next iteration's diagnosis prompt. Persisted for resume + forensics.
    prior_path = output_dir / "diagnoser_prior.jsonl"
    prior_entries: list[dict] = []
    if cl_enabled and prior_path.exists():
        for line in prior_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    prior_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Map task IDs to dataset indices
    id_to_idx = {str(s["id"]): i for i, s in enumerate(dataset)}

    # Training state — load prior results if resuming
    results_path = output_dir / "training_results.json"
    if start_iteration > 1 and results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        print(f"  Loaded {len(results.get('iterations', []))} prior iterations")
    else:
        results = {
            "config": training_config,
            "model": model,
            "skill_name": "xlsx",
            "early_stop": {"all_correct": 4},
            "status": "running",
            "iterations": [],
        }

    consecutive_all_correct = 0

    # Reconstruct streak counter from prior iterations
    for prior in results.get("iterations", []):
        if prior.get("result") == "all_correct":
            consecutive_all_correct += 1
        else:
            consecutive_all_correct = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Empty pattern-record placeholder for iter_1's bootstrap momentum call.
    # iter_2+ momentum reads the previous iter's momentum_memory.md instead.
    momentum_initial_path = output_dir / "momentum_memory_initial.md"
    if not momentum_initial_path.exists():
        momentum_initial_path.write_text(
            "# Pattern record (initial)\n\n",
            encoding="utf-8",
        )

    print(f"\n{'=' * 60}")
    print(f"  SkillGrad: {n_iterations} iterations, batch_size={training_config['batch_size']}, "
          f"schedule={training_config.get('batch_schedule', 'epoch')}")
    print(f"  Model: {model}")
    print(f"  Skills: {skills_dir}")
    if start_iteration > 1:
        print(f"  Resuming from iteration {start_iteration}")
    if interactive:
        print(f"  Interactive mode: pause after each iteration")
    print(f"{'=' * 60}\n")

    def _cost_snapshot(ct: CostTracker) -> dict:
        """Capture current cumulative cost state for computing deltas."""
        return {
            "input_tokens": ct.input_tokens,
            "cached_tokens": ct.cached_tokens,
            "output_tokens": ct.output_tokens,
            "reasoning_tokens": ct.reasoning_tokens,
            "requests": ct.requests,
            "cost": ct.total_cost,
        }

    def _cost_delta(before: dict, after: dict) -> dict:
        """Compute the difference between two cost snapshots."""
        return {
            k: round(after[k] - before[k], 6) if isinstance(after[k], float)
               else after[k] - before[k]
            for k in before
        }

    # ── Closed-loop helpers (only exercised when cl_enabled) ──────────────
    async def _execute_assess(batch_ids: list[str], work_dir: Path) -> list[dict]:
        """Execute the current in-place skill on a batch and assess outcomes.

        Mirrors the pre-patch EXECUTE/ASSESS block; reused by the forward gate to
        re-measure the patched skill on the same batch.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        sd_list = [
            prepare_seed_data(dataset, id_to_idx[t], data_dir, work_dir)
            for t in batch_ids
        ]
        er = await asyncio.gather(*[
            run_execute(sd, semaphore, skills_dir, model, project_root,
                        max_turns, 0, cost_tracker, openai_client)
            for sd in sd_list
        ])
        return await asyncio.gather(*[
            asyncio.to_thread(assess_seed, sd, e, 0)
            for sd, e in zip(sd_list, er)
        ])

    def _batch_stats(assessments: list[dict]) -> tuple[int, float]:
        """(hard correct count, mean soft accuracy) for a batch of assessments."""
        n = max(len(assessments), 1)
        correct = sum(1 for a in assessments if a["is_correct"])
        soft = sum(a["accuracy"]["accuracy"] for a in assessments) / n
        return correct, soft

    def _render_prior() -> str | None:
        """Render the last cl_prior_k accept/reject records as a compact bullet list."""
        if not prior_entries:
            return None
        rows = []
        for e in prior_entries[-cl_prior_k:]:
            verdict = "ACCEPTED" if e.get("accepted") else "rejected"
            labels = ", ".join(e.get("labels", [])) or "(no labels)"
            rows.append(
                f"- iter {e.get('iteration')}: {verdict} "
                f"(hard {e.get('pre_correct')}->{e.get('post_correct')} of "
                f"{e.get('batch_size')}); directions: {labels}"
            )
        return "\n".join(rows)

    def _build_reject_context(
        pre_correct: int, post_correct: int,
        pre_soft: float, post_soft: float,
        diagnoses: list[dict], batch_size: int,
    ) -> str:
        """Summarize a rejected attempt so the next diagnose changes direction."""
        tried = "; ".join(
            f"{d['id']}({d['type']}): {_extract_label(d['diagnosis'])}"
            for d in diagnoses
        ) or "(no labels extracted)"
        return (
            f"Re-execution result: hard {pre_correct}->{post_correct} of "
            f"{batch_size}, soft {pre_soft:.1%}->{post_soft:.1%} "
            f"(no net gain).\n"
            f"Edit directions that were just tried and did NOT help:\n{tried}"
        )

    for iter_num in range(start_iteration, n_iterations + 1):
        batch_ids = batches[iter_num - 1]
        iter_dir = output_dir / f"iter_{iter_num}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        t_iter = time.time()
        iter_cost_before = _cost_snapshot(cost_tracker)
        stage_timing = {}
        stage_costs = {}

        print(f"\n{'━' * 60}")
        print(f"  ITERATION {iter_num}/{n_iterations} — batch: {batch_ids}")
        print(f"{'━' * 60}")

        _write_workspace(iter_dir / "batch_seeds.json", batch_ids)

        # ── Snapshot skill before patching ──
        snapshot_dir = output_dir / f"snapshot_iter_{iter_num}"
        _snapshot_skill(skills_dir, snapshot_dir)

        # Also save SKILL.md text snapshot
        skill_md_path = skills_dir / "xlsx" / "SKILL.md"
        if skill_md_path.exists():
            shutil.copy2(skill_md_path, output_dir / f"SKILL.md.iter_{iter_num - 1}")

        # ── EXECUTE ──
        print(f"\n  ── EXECUTE ──")
        t_stage = time.time()
        cost_before_stage = _cost_snapshot(cost_tracker)

        assessments = await _execute_assess(batch_ids, iter_dir)

        stage_timing["execute"] = round(time.time() - t_stage, 2)
        stage_costs["execute"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))

        # ── CLASSIFY ──
        failed, contrastive = classify_batch(assessments)
        n_failed = len(failed)
        n_contrastive = len(contrastive)
        print(f"\n  ── CLASSIFY: {n_failed} failed, {n_contrastive} contrastive ──")

        # Check early stopping: all correct
        if n_failed == 0:
            consecutive_all_correct += 1
            print(f"  All seeds correct (streak: {consecutive_all_correct})")
            results["iterations"].append({
                "iteration": iter_num,
                "batch": batch_ids,
                "result": "all_correct",
                "n_failed": 0,
                "n_contrastive": n_contrastive,
                "duration_s": round(time.time() - t_iter, 2),
                "cost": _cost_delta(iter_cost_before, _cost_snapshot(cost_tracker)),
                "stage_timing": stage_timing,
                "stage_costs": stage_costs,
            })
            _write_workspace(output_dir / "training_results.json", results)
            if consecutive_all_correct >= 4:
                print(f"\n  EARLY STOP: 4 consecutive all-correct iterations")
                break
            # All tasks correct and no contrastive evidence to learn from.
            if n_contrastive == 0:
                continue

        consecutive_all_correct = 0 if n_failed > 0 else consecutive_all_correct

        cl_meta = None
        if not cl_enabled:
            # ═══════════════ Baseline forward (unchanged) ═══════════════
            # ── DIAGNOSE (parallel) ──
            print(f"\n  ── DIAGNOSE ──")
            t_stage = time.time()
            cost_before_stage = _cost_snapshot(cost_tracker)

            diag_tasks = []
            for a in failed:
                diag_tasks.append(run_diagnose(
                    a, "failure", iter_dir, skills_dir,
                    base_trajectories_dir, model, project_root,
                    semaphore, cost_tracker,
                ))
            for a in contrastive:
                diag_tasks.append(run_diagnose(
                    a, "contrastive", iter_dir, skills_dir,
                    base_trajectories_dir, model, project_root,
                    semaphore, cost_tracker,
                ))

            diagnoses = await asyncio.gather(*diag_tasks)

            stage_timing["diagnose"] = round(time.time() - t_stage, 2)
            stage_costs["diagnose"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))

            # ── ASSEMBLE ──
            batch_diagnoses_path = iter_dir / "batch_diagnoses.md"
            assemble_diagnoses(diagnoses, assessments, batch_diagnoses_path)

            # ── MOMENTUM (pre-patch, iter_num >= 2 only) ──
            # iter_1 has no prior pattern record; we run momentum AFTER iter_1's
            # patch as a bootstrap call so iter_2 has a non-empty record to read.
            patcher_overlay_path = None
            patcher_memory_path = None
            if iter_num >= 2:
                print(f"\n  ── MOMENTUM ──")
                t_stage = time.time()
                cost_before_stage = _cost_snapshot(cost_tracker)

                previous_record = output_dir / f"iter_{iter_num - 1}" / "momentum_memory.md"
                momentum_record_path = iter_dir / "momentum_memory.md"
                momentum_overlay_path = iter_dir / "momentum_overlay.md"
                await run_momentum(
                    diagnoses_path=batch_diagnoses_path,
                    previous_record_path=previous_record,
                    skills_dir=skills_dir,
                    record_output_path=momentum_record_path,
                    overlay_output_path=momentum_overlay_path,
                    model=model,
                    project_root=project_root,
                    cost_tracker=cost_tracker,
                    iter_dir=iter_dir,
                    iter_num=iter_num,
                )
                patcher_overlay_path = momentum_overlay_path
                patcher_memory_path = momentum_record_path

                stage_timing["momentum"] = round(time.time() - t_stage, 2)
                stage_costs["momentum"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))

            # ── PATCH ──
            print(f"\n  ── PATCH ──")
            t_stage = time.time()
            cost_before_stage = _cost_snapshot(cost_tracker)

            patch_output = await run_patch(
                batch_diagnoses_path, skills_dir,
                model, project_root, cost_tracker, iter_dir,
                overlay_path=patcher_overlay_path,
                momentum_memory_path=patcher_memory_path,
            )

            stage_timing["patch"] = round(time.time() - t_stage, 2)
            stage_costs["patch"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))

            # ── MOMENTUM bootstrap (post-patch, iter_1 only) ──
            # Initializes momentum_memory.md so iter_2's pre-patch momentum has
            # a non-empty pattern record to read. The agent reads the post-patch
            # skill so it can label skill coverage for iter_1's patches.
            if iter_num == 1:
                print(f"\n  ── MOMENTUM bootstrap ──")
                t_stage = time.time()
                cost_before_stage = _cost_snapshot(cost_tracker)

                momentum_record_path = iter_dir / "momentum_memory.md"
                momentum_overlay_path = iter_dir / "momentum_overlay.md"
                await run_momentum(
                    diagnoses_path=batch_diagnoses_path,
                    previous_record_path=momentum_initial_path,
                    skills_dir=skills_dir,
                    record_output_path=momentum_record_path,
                    overlay_output_path=momentum_overlay_path,
                    model=model,
                    project_root=project_root,
                    cost_tracker=cost_tracker,
                    iter_dir=iter_dir,
                    iter_num=iter_num,
                )

                stage_timing["momentum_bootstrap"] = round(time.time() - t_stage, 2)
                stage_costs["momentum_bootstrap"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))
        else:
            # ═════ Closed-loop forward ═════
            # Within an iteration: patch, re-execute on the same batch, and
            #     accept only when the patched skill does not regress; otherwise
            #     roll back, feed the rejection reason to the diagnoser, and
            #     retry in a different direction.
            # Across iterations: inject a compact accept/reject memory.
            print(f"\n  ── CLOSED-LOOP (up to {cl_max_attempts} attempts) ──")
            t_stage = time.time()
            cost_before_stage = _cost_snapshot(cost_tracker)

            pre_correct, pre_soft = _batch_stats(assessments)
            prior_text = _render_prior()
            reject_context = None
            attempts_meta: list[dict] = []
            best: dict | None = None

            for attempt in range(1, cl_max_attempts + 1):
                # Every attempt patches from the same pre-patch base.
                _restore_skill(snapshot_dir, skills_dir)
                attempt_dir = iter_dir / f"attempt_{attempt}"
                attempt_dir.mkdir(parents=True, exist_ok=True)

                diag_tasks = []
                for a in failed:
                    diag_tasks.append(run_diagnose(
                        a, "failure", attempt_dir, skills_dir,
                        base_trajectories_dir, model, project_root,
                        semaphore, cost_tracker,
                        prior=prior_text, reject_context=reject_context,
                    ))
                for a in contrastive:
                    diag_tasks.append(run_diagnose(
                        a, "contrastive", attempt_dir, skills_dir,
                        base_trajectories_dir, model, project_root,
                        semaphore, cost_tracker,
                        prior=prior_text, reject_context=reject_context,
                    ))
                diagnoses = await asyncio.gather(*diag_tasks)

                batch_diagnoses_path = attempt_dir / "batch_diagnoses.md"
                assemble_diagnoses(diagnoses, assessments, batch_diagnoses_path)

                overlay_path = None
                memory_path = None
                if iter_num >= 2:
                    previous_record = output_dir / f"iter_{iter_num - 1}" / "momentum_memory.md"
                    memory_path = attempt_dir / "momentum_memory.md"
                    overlay_path = attempt_dir / "momentum_overlay.md"
                    await run_momentum(
                        diagnoses_path=batch_diagnoses_path,
                        previous_record_path=previous_record,
                        skills_dir=skills_dir,
                        record_output_path=memory_path,
                        overlay_output_path=overlay_path,
                        model=model, project_root=project_root,
                        cost_tracker=cost_tracker, iter_dir=attempt_dir,
                        iter_num=iter_num,
                    )

                await run_patch(
                    batch_diagnoses_path, skills_dir,
                    model, project_root, cost_tracker, attempt_dir,
                    overlay_path=overlay_path, momentum_memory_path=memory_path,
                )
                # Keep this attempt's patched skill so the winner can be restored.
                _snapshot_skill(skills_dir, attempt_dir / "patched_skill")

                # ── Forward gate: re-execute the patched skill on the same batch ──
                post_assessments = await _execute_assess(
                    batch_ids, attempt_dir / "recheck")
                post_correct, post_soft = _batch_stats(post_assessments)
                accept = (post_correct >= pre_correct - cl_hard_tol
                          and post_soft >= pre_soft - 1e-9)
                print(f"  [gate] attempt {attempt}: hard {pre_correct}->{post_correct}"
                      f", soft {pre_soft:.1%}->{post_soft:.1%} "
                      f"=> {'ACCEPT' if accept else 'reject'}")

                cand = {
                    "attempt": attempt,
                    "correct": post_correct, "soft": round(post_soft, 4),
                    "dir": attempt_dir / "patched_skill",
                    "diagnoses": diagnoses,
                    "batch_diag_path": batch_diagnoses_path,
                    "overlay": overlay_path, "memory": memory_path,
                    "labels": [_extract_label(d["diagnosis"]) for d in diagnoses],
                }
                attempts_meta.append({
                    "attempt": attempt, "post_correct": post_correct,
                    "post_soft": round(post_soft, 4), "accept": accept,
                })
                if best is None or (post_correct, post_soft) > (best["correct"], best["soft"]):
                    best = cand
                if post_correct > pre_correct:  # strict hard gain — take it
                    best = cand
                    break
                reject_context = _build_reject_context(
                    pre_correct, post_correct, pre_soft, post_soft,
                    diagnoses, len(assessments))

            # Accept the winner only if it does not regress vs the pre-patch base.
            accepted = (best is not None
                        and best["correct"] >= pre_correct - cl_hard_tol
                        and best["soft"] >= round(pre_soft, 4) - 1e-9)
            if accepted:
                _restore_skill(best["dir"], skills_dir)
            else:
                _restore_skill(snapshot_dir, skills_dir)  # revert — no net gain

            diagnoses = best["diagnoses"] if best else []
            batch_diagnoses_path = (best["batch_diag_path"] if best
                                    else iter_dir / "batch_diagnoses.md")

            # Carry momentum forward so the next iteration has a pattern record.
            if iter_num >= 2:
                if best and best["memory"] and best["memory"].exists():
                    shutil.copy2(best["memory"], iter_dir / "momentum_memory.md")
                if best and best["overlay"] and best["overlay"].exists():
                    shutil.copy2(best["overlay"], iter_dir / "momentum_overlay.md")
            else:
                # iter_1 bootstrap on the winner's diagnoses (post-decision skill).
                await run_momentum(
                    diagnoses_path=batch_diagnoses_path,
                    previous_record_path=momentum_initial_path,
                    skills_dir=skills_dir,
                    record_output_path=iter_dir / "momentum_memory.md",
                    overlay_output_path=iter_dir / "momentum_overlay.md",
                    model=model, project_root=project_root,
                    cost_tracker=cost_tracker, iter_dir=iter_dir,
                    iter_num=iter_num,
                )

            stage_timing["closed_loop"] = round(time.time() - t_stage, 2)
            stage_costs["closed_loop"] = _cost_delta(cost_before_stage, _cost_snapshot(cost_tracker))

            cl_meta = {
                "n_attempts": len(attempts_meta),
                "attempts": attempts_meta,
                "accepted": accepted,
                "pre_correct": pre_correct,
                "post_correct": best["correct"] if best else pre_correct,
                "post_soft_accuracy": best["soft"] if best else round(pre_soft, 4),
            }

            # ── Append this iteration's accept/reject record to memory ──
            entry = {
                "iteration": iter_num,
                "accepted": accepted,
                "pre_correct": pre_correct,
                "post_correct": cl_meta["post_correct"],
                "batch_size": len(assessments),
                "labels": best["labels"] if best else [],
            }
            prior_entries.append(entry)
            with open(prior_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        # ── Record the patched outcome (pre-execution accuracy) ──
        pre_correct = sum(1 for a in assessments if a["is_correct"])
        pre_soft = sum(a["accuracy"]["accuracy"] for a in assessments) / len(assessments)
        print(f"\n  ── PATCHED (pre correct: {pre_correct}/{len(assessments)}, "
              f"soft: {pre_soft:.1%}) ──")

        # Record iteration result with full cost/timing details
        iter_duration = round(time.time() - t_iter, 2)
        iter_cost = _cost_delta(iter_cost_before, _cost_snapshot(cost_tracker))

        iter_result = {
            "iteration": iter_num,
            "batch": batch_ids,
            "result": ("patched" if (cl_meta is None or cl_meta["accepted"])
                       else "reverted"),
            "n_failed": n_failed,
            "n_contrastive": n_contrastive,
            "pre_correct": pre_correct,
            "pre_soft_accuracy": round(pre_soft, 4),
            "seed_outcomes": {
                a["id"]: {
                    "pre_correct": a["is_correct"],
                    "pre_accuracy": a["accuracy"]["accuracy"],
                    "pre_match_count": a["accuracy"]["match_count"],
                    "pre_total_count": a["accuracy"]["total_count"],
                }
                for a in assessments
            },
            "duration_s": iter_duration,
            "cost": iter_cost,
            "stage_timing": stage_timing,
            "stage_costs": stage_costs,
        }
        if cl_meta is not None:
            iter_result["closed_loop"] = cl_meta
        results["iterations"].append(iter_result)
        results["cumulative_cost"] = _cost_snapshot(cost_tracker)
        _write_workspace(output_dir / "training_results.json", results)

        # Log iteration summary
        _log_iteration(output_dir / "iteration_log.md", iter_num, iter_result, assessments, diagnoses)

        # Per-iteration markdown one-pager for quick eyeballing
        _write_iter_summary(iter_dir / "summary.md", iter_num, iter_result, diagnoses)

        # Interactive pause
        if interactive:
            try:
                user_input = input(
                    "\n  Press Enter to continue, or type 'stop' to stop training: "
                ).strip().lower()
                if user_input == "stop":
                    print("  Stopping training (user requested).")
                    break
            except EOFError:
                break

    # Save final skill
    results["status"] = "completed"
    results["cost"] = cost_tracker.to_dict()
    _write_workspace(output_dir / "training_results.json", results)

    # Copy final skill to a clean directory
    final_dir = output_dir / "final_skill" / "xlsx"
    final_dir.mkdir(parents=True, exist_ok=True)
    skill_xlsx = skills_dir / "xlsx"
    if skill_xlsx.exists():
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(skill_xlsx, final_dir)

    print(f"\n{'=' * 60}")
    print(f"  Training complete")
    print(f"  Final skill: {final_dir}")
    print(f"  Cost:\n{cost_tracker.summary()}")
    print(f"{'=' * 60}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Iteration log
# ═══════════════════════════════════════════════════════════════════════════

def _log_iteration(
    log_path: Path,
    iter_num: int,
    iter_result: dict,
    assessments: list[dict],
    diagnoses: list[dict],
):
    """Append an iteration summary to the human-readable iteration log."""
    lines = []
    if iter_num == 1:
        lines.append("# Iteration Log\n")

    lines.append(f"\n## Iteration {iter_num}\n")
    lines.append(f"**Batch**: {iter_result['batch']}")
    lines.append(f"**Result**: {iter_result['result']} "
                 f"(correct: {iter_result.get('pre_correct', '?')}, "
                 f"soft: {iter_result.get('pre_soft_accuracy', '?')})\n")

    if diagnoses:
        lines.append("### Diagnoses")
        for d in diagnoses:
            label = ""
            for line in d["diagnosis"].split("\n"):
                if line.strip().startswith("LABEL:"):
                    label = line.strip().replace("LABEL:", "").strip()
                    break
            lines.append(f"- **{d['id']}** ({d['type']}): {label}")
        lines.append("")

    lines.append("### Seed Outcomes")
    for a in assessments:
        acc = a["accuracy"]
        lines.append(
            f"- {a['id']}: "
            f"{acc['accuracy']:.1%} ({acc['match_count']}/{acc['total_count']}) "
            f"[{iter_result['result'].lower()}]"
        )
    lines.append("")

    mode = "a" if iter_num > 1 else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Per-iteration summary (markdown one-pager)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_label(diagnosis_text: str) -> str:
    for line in diagnosis_text.split("\n"):
        if line.strip().startswith("LABEL:"):
            return line.strip().replace("LABEL:", "").strip()
    return ""


def _write_iter_summary(
    summary_path: Path,
    iter_num: int,
    iter_result: dict,
    diagnoses: list[dict],
) -> None:
    """Per-iteration one-pager for fast eyeballing."""
    timings = iter_result.get("stage_timing", {})
    costs = iter_result.get("stage_costs", {})
    lines = [
        f"# Iteration {iter_num}\n",
        f"- Batch: `{iter_result['batch']}`",
        f"- Result: **{iter_result['result']}**",
        f"- Correct: {iter_result.get('pre_correct', '?')}",
        f"- Soft acc: {iter_result.get('pre_soft_accuracy', '?')}",
        f"- Duration: {iter_result.get('duration_s', '?')}s",
        "",
        "## Stage timing (s)",
    ]
    for k, v in timings.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Stage cost")
    for k, v in costs.items():
        c = v.get("cost") if isinstance(v, dict) else None
        lines.append(f"- {k}: ${c}")
    lines.append("")
    if diagnoses:
        lines.append("## Diagnosis labels")
        for d in diagnoses:
            lines.append(f"- {d['id']} ({d['type']}): {_extract_label(d['diagnosis'])}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillGrad training loop: executor -> diagnoser -> momentum -> patcher.")
    parser.add_argument("--data-dir", required=True, help="Path to SpreadsheetBench dataset")
    parser.add_argument("--skills-dir", default="seeds", help="Directory containing the bootstrap skill (expects <skills-dir>/xlsx/SKILL.md).")
    parser.add_argument("--results-root", default="results",
                        help="Root for splits/, base_trajectories/, runs/, manifest.json")
    parser.add_argument("--base-trajectories-dir", default=None,
                        help="Override for the base-trajectories cache directory. "
                             "Default: layout-derived "
                             "results/base_trajectories/master_<M>_heldout_<H>/<model>/. "
                             "Use this when the same --model was used to collect "
                             "multiple base runs (e.g. different initial skills) "
                             "and you need to point training at a specific one.")
    # Identity
    parser.add_argument("--method", default="skillgrad",
                        help="Tag used as the run_id prefix. Default: skillgrad.")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--config-tag", default=None,
                        help="Optional suffix appended to run_id so multiple "
                             "runs with the same method+model do not collide.")
    # Seeds
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=0)
    # Hyperparams
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-schedule",
                        choices=["epoch", "fixed-updates"],
                        default="epoch",
                        help="'epoch' (default) chunks the selected train set "
                             "once. 'fixed-updates' keeps the selected "
                             "n_train pool fixed, cycles through reshuffled "
                             "epochs as needed, and emits exactly "
                             "--n-iterations full batches.")
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="Required with --batch-schedule fixed-updates. "
                             "Useful for fixed update-budget ablations.")
    parser.add_argument("--batch-seed", type=int, default=None,
                        help="Optional seed for reshuffling later epochs in "
                             "fixed-updates. Defaults to --training-seed.")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--start-iteration", type=int, default=1)
    # Closed-loop forward
    parser.add_argument("--closed-loop", action="store_true",
                        help="Enable the closed-loop forward: re-execute each "
                             "patch on its batch and roll back + retry with "
                             "reject feedback, injecting a cross-iteration "
                             "accept/reject memory into the diagnoser.")
    parser.add_argument("--cl-max-attempts", type=int, default=3,
                        help="Max DIAGNOSE→PATCH attempts per iteration.")
    parser.add_argument("--cl-hard-tol", type=int, default=0,
                        help="Allowed hard-accuracy regression (batch count) "
                             "before a patch is rejected. 0 = no regression.")
    parser.add_argument("--cl-prior-k", type=int, default=6,
                        help="Number of prior-iteration accept/reject records injected.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_dir = Path(args.data_dir)

    # ── Resolve canonical paths via data/layout.py ────────────────────────
    split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
    normalized_dir = normalized_dir_for(results_root)
    base_traj_dir = base_trajectories_dir_for(
        results_root, args.master_seed, args.heldout_seed, args.model,
    )
    if args.base_trajectories_dir:
        base_traj_dir = Path(args.base_trajectories_dir).resolve()
        print(f"Base-trajectories override: {base_traj_dir}")

    # ── Ensure split exists (master+heldout-seed-keyed, shared) ───────────
    split_path = split_dir / "split.json"
    if not split_path.exists():
        print(f"Creating canonical split at {split_dir}...")
        create_split(
            data_dir, split_dir,
            master_seed=args.master_seed,
            heldout_seed=args.heldout_seed,
            normalized_dir=normalized_dir,
        )
    split = load_split(split_dir)
    print(f"Split (master={args.master_seed}, heldout={args.heldout_seed}): "
          f"{split['n_evolution']} evolution, {split['n_test']} test")

    # ── Failure ids (depends on master+heldout+model — base trajectories) ─
    failures_path = base_traj_dir / "failure_ids.json"
    if not failures_path.exists():
        if not base_traj_dir.exists():
            raise FileNotFoundError(
                f"Base trajectories not found at {base_traj_dir}. "
                "Run scripts/base_traj.sh first with the same "
                f"--master-seed {args.master_seed} --heldout-seed "
                f"{args.heldout_seed} --model {args.model}."
            )
        failure_ids, success_ids = identify_failures(base_traj_dir, split["evolution_ids"])
        _write_workspace(failures_path, {"failure_ids": failure_ids, "success_ids": success_ids})
    else:
        failure_ids = json.loads(failures_path.read_text(encoding="utf-8"))["failure_ids"]
    print(f"Loaded {len(failure_ids)} failure IDs from {base_traj_dir}")

    # ── Build training set (training-seed varies; selects 40 of failures) ─
    if args.batch_schedule == "fixed-updates":
        if args.n_iterations is None:
            parser.error("--n-iterations is required for --batch-schedule fixed-updates")
        training_config = build_fixed_update_training_set(
            failure_ids=failure_ids,
            training_seed=args.training_seed,
            n_train=args.n_train,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            batch_seed=args.batch_seed,
        )
    else:
        if args.n_iterations is not None:
            parser.error("--n-iterations requires --batch-schedule fixed-updates")
        if args.batch_seed is not None:
            parser.error("--batch-seed requires --batch-schedule fixed-updates")
        training_config = build_training_set(
            failure_ids, args.training_seed, args.n_train, args.batch_size,
        )
    training_config.update({
        "model": args.model,
        "max_turns": args.max_turns,
        "concurrency": args.concurrency,
    })

    # ── Construct run identity & run directory ────────────────────────────
    run_id = run_id_for(
        method=args.method,
        model=args.model,
        config_tag=args.config_tag,
    )
    run_dir = run_dir_for(results_root, run_id)
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    _write_workspace(train_dir / "train_set.json", training_config)
    print(f"Run id: {run_id}")
    print(f"Run dir: {run_dir}")

    # ── Write run config.json (full source of truth) ──────────────────────
    started_at = datetime.now(timezone.utc).isoformat()
    config_payload = {
        "run_id": run_id,
        "method": args.method,
        "model": args.model,
        "config_tag": args.config_tag,
        "seeds": {
            "master_seed": args.master_seed,
            "heldout_seed": args.heldout_seed,
            "training_seed": args.training_seed,
        },
        "split_path": str(split_path),
        "base_trajectories_path": str(base_traj_dir),
        "training_config": {
            "n_train": args.n_train,
            "batch_size": args.batch_size,
            "batch_schedule": args.batch_schedule,
            "n_iterations": args.n_iterations,
            "batch_seed": args.batch_seed,
            "max_turns": args.max_turns,
            "concurrency": args.concurrency,
        },
        "closed_loop": {
            "enabled": args.closed_loop,
            "max_attempts": args.cl_max_attempts,
            "hard_tol": args.cl_hard_tol,
            "prior_k": args.cl_prior_k,
        },
        "started": started_at,
        "status": "running",
    }
    _write_workspace(run_dir / "config.json", config_payload)
    manifest_upsert(results_root, run_dir)

    # ── Stage the bootstrap skill into run_dir/skills/xlsx/ ───────────────
    skills_work_dir = run_dir / "skills"
    if not (skills_work_dir / "xlsx" / "SKILL.md").exists():
        base_skill_dir = Path(args.skills_dir)
        skill_dest = skills_work_dir / "xlsx"
        skill_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_skill_dir / "xlsx" / "SKILL.md", skill_dest / "SKILL.md")
        refs_src = base_skill_dir / "xlsx" / "references"
        if refs_src.exists():
            shutil.copytree(refs_src, skill_dest / "references")
        print(f"Copied base skill to {skills_work_dir}")

    # ── Run training; write artifacts under run_dir/train/ ────────────────
    t_start = time.time()
    try:
        results = asyncio.run(run_training(
            data_dir=normalized_dir,
            skills_dir=skills_work_dir,
            model=args.model,
            project_root=Path(".").resolve(),
            output_dir=train_dir,
            base_trajectories_dir=base_traj_dir,
            training_config=training_config,
            max_turns=args.max_turns,
            concurrency=args.concurrency,
            interactive=args.interactive,
            start_iteration=args.start_iteration,
            closed_loop={
                "enabled": args.closed_loop,
                "max_attempts": args.cl_max_attempts,
                "hard_tol": args.cl_hard_tol,
                "prior_k": args.cl_prior_k,
            },
        ))
        final_status = "completed"
    except Exception as exc:
        results = None
        final_status = f"failed: {exc}"
        raise
    finally:
        # ── Update config.json + manifest with metrics ────────────────────
        config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config_payload["status"] = final_status
        config_payload["completed"] = datetime.now(timezone.utc).isoformat()
        config_payload["elapsed_s"] = round(time.time() - t_start, 1)

        if results is not None:
            iters = results.get("iterations", [])
            n_patched = sum(1 for it in iters if it.get("result") == "patched")
            n_all_correct = sum(1 for it in iters if it.get("result") == "all_correct")
            config_payload["metrics"] = {
                "n_iterations": len(iters),
                "n_patched": n_patched,
                "n_all_correct": n_all_correct,
                # test_hard / test_cell_acc filled in later by the eval step
                "test_hard": None,
                "test_cell_acc": None,
            }
            config_payload["cost"] = results.get("cost") or results.get("cumulative_cost")
        _write_workspace(run_dir / "config.json", config_payload)
        manifest_upsert(results_root, run_dir)

    print(f"\nTraining results: {train_dir / 'training_results.json'}")
    print(f"Final skill:      {train_dir / 'final_skill' / 'xlsx' / 'SKILL.md'}")
    print(f"Run config:       {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()
