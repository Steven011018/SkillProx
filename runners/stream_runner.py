"""Shared executor+grader stream runner for evaluation and base trajectories."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv
from agents import Runner

from data.layout import (
    base_trajectories_dir_for,
    normalized_dir_for,
    run_dir_for,
    run_id_for,
    splits_dir_for,
)
from data.load import SpreadsheetBenchDataset
from data.split import identify_failures, load_split
from pipeline.execution import build_task_string, build_task_string_oj, load_dataset
from pipeline.executor import SkillAgent
from prompts.executor import EXECUTOR_PROMPT
from runners.model_dispatch import get_client_for_model
from scripts.manifest_update import upsert as manifest_upsert
from runners.soffice import find_soffice, install_soffice_wrapper
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_kwargs
from runners.trajectory_logger import (
    TrajectoryLogger,
    save_merged_trace,
    set_phase,
    stream_with_logging,
)
from evaluators.xlsx_compare import compute_accuracy_on_copy, format_comparison

load_dotenv()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def _seed_cost(tc_results: list[dict]) -> dict:
    return {
        "input_tokens": sum(r.get("cost", {}).get("input_tokens", 0) for r in tc_results),
        "cached_tokens": sum(r.get("cost", {}).get("cached_tokens", 0) for r in tc_results),
        "output_tokens": sum(r.get("cost", {}).get("output_tokens", 0) for r in tc_results),
        "reasoning_tokens": sum(r.get("cost", {}).get("reasoning_tokens", 0) for r in tc_results),
        "requests": sum(r.get("cost", {}).get("requests", 0) for r in tc_results),
        "cost": round(sum(r.get("cost", {}).get("cost", 0) for r in tc_results), 6),
    }


def _is_context_overflow(errors: list | str | None) -> bool:
    """Permanent model-limit failures; retrying the same prompt will not help."""
    if not errors:
        return False
    blob = errors if isinstance(errors, str) else " ".join(str(x) for x in errors)
    low = blob.lower()
    return "maximum context length" in low or (
        "input_tokens" in low and "context" in low
    )


def _no_output_status(errors: list | None, *, oj: bool = False) -> tuple[str, str]:
    if _is_context_overflow(errors):
        return "failed", "context_overflow"
    if oj:
        return "retry_needed", "oj_no_outputs"
    return "retry_needed", ("exec_error_no_output" if errors else "no_output")


def _eval_aggregate(records: dict[int, dict], total: int) -> dict:
    graded = [r for r in records.values() if r.get("status") == "graded"]
    all_scored = [r for r in records.values() if "hard_score" in r]
    pass_count = sum(1 for r in graded if r.get("hard_score") == 1.0)
    pending = sum(1 for r in records.values() if r.get("status") in {"grade_pending", "grading"})
    executing = sum(1 for r in records.values() if r.get("status") == "executing")
    retry_needed = sum(1 for r in records.values() if r.get("status") == "retry_needed")
    failed = sum(1 for r in records.values() if r.get("status") == "failed")
    stopped = sum(1 for r in records.values() if r.get("status") == "stopped_early")
    not_started = total - len(records)
    best_possible = pass_count + pending + executing + not_started
    n_scored = len(all_scored)
    return {
        "n_seeds": n_scored,
        "mean_cell_accuracy": round(sum(r.get("cell_accuracy", 0.0) for r in all_scored) / n_scored, 4) if n_scored else 0.0,
        "mean_hard": round(sum(r.get("hard_score", 0.0) for r in all_scored) / n_scored, 4) if n_scored else 0.0,
        "n_perfect": sum(1 for r in all_scored if r.get("hard_score") == 1.0),
        "n_total": total,
        "n_records": len(records),
        "n_scored": n_scored,
        "n_graded": len(graded),
        "n_pass_graded": pass_count,
        "mean_cell_graded": round(sum(r.get("cell_accuracy", 0.0) for r in graded) / len(graded), 4) if graded else 0.0,
        "mean_hard_graded": round(pass_count / len(graded), 4) if graded else 0.0,
        "conservative_hard_if_pending_zero": round(pass_count / total, 4) if total else 0.0,
        "best_possible_hard_count": best_possible,
        "best_possible_hard": round(best_possible / total, 4) if total else 0.0,
        "pending_outputs": pending,
        "executing": executing,
        "retry_needed": retry_needed,
        "failed": failed,
        "stopped_early": stopped,
        "not_started": not_started,
    }


def _base_aggregate(results: list[dict], total: int) -> dict:
    done = len(results)
    passed = sum(1 for r in results if r.get("is_correct"))
    return {
        "n_done": done,
        "n_seeds": total,
        "n_perfect": passed,
        "n_failed": done - passed,
        "hard_score": round(passed / done, 4) if done else 0.0,
        "mean_cell_accuracy": round(sum(r.get("cell_accuracy", 0.0) for r in results) / done, 4) if done else 0.0,
    }


def _summary(output_format: str, records: dict[int, dict], config: dict, cost: CostTracker, t0: float, status: str) -> dict:
    total = config["total"]
    if output_format == "base_trajectories":
        results = sorted(records.values(), key=lambda r: str(r.get("id", "")))
        return {
            "config": config,
            "aggregate": _base_aggregate(results, total),
            "status": status,
            "completed": len(results),
            "total": total,
            "elapsed": round(time.time() - t0, 1),
            "cost": cost.to_dict(),
            "results": results,
        }
    results = sorted(records.values(), key=lambda r: r.get("index", 999999))
    return {
        "config": config,
        "status": status,
        "completed": len(records),
        "completed_records": len(records),
        "total": total,
        "aggregate": _eval_aggregate(records, total),
        "cost": cost.to_dict(),
        "elapsed": round(time.time() - t0, 1),
        "results": results,
    }


async def _execute_record(
    dataset: list[dict],
    idx: int,
    data_dir: Path,
    skill_dir: Path,
    model: str,
    project_root: Path,
    workdir: Path,
    max_turns: int,
    cost: CostTracker,
    output_format: str,
    test_cases: int,
    test_case_start: int,
    openai_client,
    oj: bool = False,
) -> dict:
    example = dataset[idx]
    ex_id = str(example["id"])
    spreadsheet_dir = data_dir / "spreadsheet" / ex_id
    model_kwargs = get_model_kwargs(model, openai_client=openai_client)
    tc_jobs = []
    tc_exec_info = []
    errors = []

    if oj and output_format == "eval":
        infer_dir = workdir / f"eval_{ex_id}_oj"
        infer_dir.mkdir(parents=True, exist_ok=True)
        first_input = spreadsheet_dir / f"{test_case_start}_{ex_id}_input.xlsx"
        task_input = infer_dir / "input.xlsx"
        shutil.copy2(first_input, task_input)
        task_output = infer_dir / "output.xlsx"
        solution_path = infer_dir / "solution.py"
        task_str = build_task_string_oj(
            example, task_input, task_output, solution_path
        )
        logger = TrajectoryLogger(infer_dir / "trajectory.jsonl")
        set_phase("EVALUATE", logger)
        executor = SkillAgent(
            skills_dir=skill_dir, model=model, max_turns=max_turns,
            system_prompt=EXECUTOR_PROMPT, model_kwargs=model_kwargs,
            include_skills=None, project_root=project_root,
        )
        tc_cost = {}
        error_msg = ""
        started = time.time()
        try:
            result = Runner.run_streamed(
                executor.agent, task_str, max_turns=max_turns
            )
            await stream_with_logging(result, logger)
            tc_cost = cost.update(result)
        except Exception as exc:
            error_msg = str(exc)
            errors.append(error_msg)
        finally:
            logger.flush()

        for tc in range(test_case_start, test_cases + 1):
            input_path = spreadsheet_dir / f"{tc}_{ex_id}_input.xlsx"
            answer_path = spreadsheet_dir / f"{tc}_{ex_id}_answer.xlsx"
            task_dir = workdir / f"eval_{ex_id}_tc{tc}"
            task_dir.mkdir(parents=True, exist_ok=True)
            output_path = task_dir / "output.xlsx"
            run_error = error_msg
            if solution_path.exists() and input_path.exists():
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, str(solution_path), str(input_path),
                        str(output_path), cwd=str(infer_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=180
                    )
                    if proc.returncode:
                        run_error = stderr.decode(errors="replace")[-2000:]
                except Exception as exc:
                    run_error = str(exc)
            elif not solution_path.exists():
                run_error = "solution.py not produced"
            tc_exec_info.append({
                "tc": tc, "elapsed": round(time.time() - started, 1),
                "trajectory_path": str(infer_dir / "trajectory.jsonl"),
                "cost": tc_cost if tc == test_case_start else {},
                "output_exists": output_path.exists(), "error": run_error,
                "task_dir": str(task_dir),
            })
            tc_jobs.append({
                "tc": tc, "task_output": str(output_path),
                "answer_path": str(answer_path),
                "answer_position": example["answer_position"],
                "task_dir": str(task_dir),
            })
        grade_jobs = [
            job for job, info in zip(tc_jobs, tc_exec_info)
            if info["output_exists"]
        ]
        status, reason = (
            ("grade_pending", "") if grade_jobs else _no_output_status(errors, oj=True)
        )
        record = {
            "index": idx, "id": ex_id,
            "instruction": example["instruction"][:300],
            "instruction_type": example.get("instruction_type", ""),
            "answer_position": example.get("answer_position", ""),
            "status": status,
            "retry_reason": reason,
            "exec_errors": errors, "elapsed": round(time.time() - started, 1),
            "cost": _seed_cost(tc_exec_info),
            "n_test_cases": len(tc_jobs),
            "test_case_exec": tc_exec_info, "grade_jobs": grade_jobs,
        }
        if not grade_jobs:
            record.update({
                "cell_accuracy": 0.0, "hard_score": 0.0,
                "match_count": 0, "total_count": 0,
            })
        return record

    tc_end = test_case_start if output_format == "base_trajectories" else test_cases
    for tc in range(test_case_start, tc_end + 1):
        input_path = spreadsheet_dir / f"{tc}_{ex_id}_input.xlsx"
        answer_path = spreadsheet_dir / f"{tc}_{ex_id}_answer.xlsx"
        if not input_path.exists() or not answer_path.exists():
            errors.append(f"missing_input_or_answer_tc{tc}")
            continue

        if output_format == "base_trajectories":
            task_workdir = workdir / ex_id
            log_name = "exec_r0.jsonl"
            phase = "EXECUTE"
            include_skills = ["xlsx"]
        elif output_format == "training":
            task_workdir = workdir / f"evolve_{ex_id}"
            log_name = "exec_r0.jsonl"
            phase = "EXECUTE"
            include_skills = ["xlsx"]
        else:
            task_workdir = workdir / f"eval_{ex_id}_tc{tc}"
            log_name = "trajectory.jsonl"
            phase = "EVALUATE"
            include_skills = None

        task_workdir.mkdir(parents=True, exist_ok=True)
        task_input = task_workdir / "input.xlsx"
        if not task_input.exists():
            shutil.copy2(input_path, task_input)
        task_output = task_workdir / "output.xlsx"
        task_str = build_task_string(example, task_input, task_output)

        logger = TrajectoryLogger(task_workdir / log_name)
        set_phase(phase, logger)
        logger.log_meta("config", {"index": idx, "id": ex_id, "test_case": tc, "model": model, "task": task_str[:500]})

        executor = SkillAgent(
            skills_dir=skill_dir,
            model=model,
            max_turns=max_turns,
            system_prompt=EXECUTOR_PROMPT,
            model_kwargs=model_kwargs,
            include_skills=include_skills,
            project_root=project_root,
        )

        start = time.time()
        tc_cost = {}
        error_msg = ""
        try:
            result = Runner.run_streamed(executor.agent, task_str, max_turns=max_turns)
            await stream_with_logging(result, logger)
            tc_cost = cost.update(result)
            cost.print_step(f"{phase} {ex_id} TC{tc}", tc_cost)
        except Exception as exc:
            error_msg = str(exc)
            errors.append(error_msg)
            logger.log_meta("error", {"message": error_msg})
            print(f"  [{phase} {ex_id} TC{tc}] ERROR: {exc}")
        logger.flush()

        tc_exec_info.append({
            "tc": tc,
            "elapsed": round(time.time() - start, 1),
            "trajectory_path": str(task_workdir / log_name),
            "cost": tc_cost,
            "output_exists": task_output.exists(),
            "error": error_msg,
            "task_dir": str(task_workdir),
        })
        tc_jobs.append({
            "tc": tc,
            "task_output": str(task_output),
            "answer_path": str(answer_path),
            "answer_position": example["answer_position"],
            "task_dir": str(task_workdir),
        })

    grade_jobs = [job for job, info in zip(tc_jobs, tc_exec_info) if info["output_exists"]]
    status, reason = (
        ("grade_pending", "") if grade_jobs else _no_output_status(errors, oj=False)
    )
    record = {
        "index": idx,
        "id": ex_id,
        "instruction": example["instruction"][:300],
        "instruction_type": example.get("instruction_type", ""),
        "answer_position": example.get("answer_position", ""),
        "status": status,
        "retry_reason": reason,
        "exec_errors": errors,
        "elapsed": round(sum(i["elapsed"] for i in tc_exec_info), 1),
        "cost": _seed_cost(tc_exec_info),
        "n_test_cases": len(tc_jobs),
        "test_case_exec": tc_exec_info,
        "grade_jobs": grade_jobs,
    }
    if output_format == "base_trajectories":
        record["task_dir"] = str(workdir / ex_id)
    if not grade_jobs:
        record.update({"cell_accuracy": 0.0, "hard_score": 0.0, "match_count": 0, "total_count": 0})
    return record


async def _grade_record(record: dict, recalc_timeout: int, lock_path: Path | None, output_format: str) -> dict:
    tc_results = []
    exec_by_tc = {i["tc"]: i for i in record.get("test_case_exec", [])}
    for job in record.get("grade_jobs", []):
        acc = await asyncio.to_thread(
            compute_accuracy_on_copy,
            job["task_output"],
            job["answer_path"],
            job["answer_position"],
            recalc_timeout,
            lock_path,
        )
        info = exec_by_tc.get(job["tc"], {})
        passed = acc["accuracy"] == 1.0
        tc_results.append({
            "test_case": job["tc"],
            "cell_accuracy": acc["accuracy"],
            "match_count": acc["match_count"],
            "total_count": acc["total_count"],
            "passed": passed,
            "elapsed": info.get("elapsed", 0),
            "trajectory_path": info.get("trajectory_path", ""),
            "cost": info.get("cost", {}),
            "status": "graded",
        })
        print(f"  [GRADE idx={record['index']} id={record['id']} TC{job['tc']}] {acc['match_count']}/{acc['total_count']} ({acc['accuracy']:.1%})")

        if output_format == "base_trajectories":
            task_dir = Path(job["task_dir"])
            assessment = {
                "id": record["id"],
                "cell_accuracy": acc["accuracy"],
                "match_count": acc["match_count"],
                "total_count": acc["total_count"],
                "is_correct": passed,
            }
            _save_json(task_dir / "assessment.json", assessment)
            try:
                comp = await asyncio.to_thread(format_comparison, job["task_output"], job["answer_path"], job["answer_position"])
                (task_dir / "cell_comparison.txt").write_text(comp, encoding="utf-8")
            except Exception as exc:
                (task_dir / "cell_comparison.txt").write_text(f"(cell_comparison failed: {exc})", encoding="utf-8")
            raw_trace = task_dir / "exec_r0.jsonl"
            if raw_trace.exists():
                try:
                    save_merged_trace(raw_trace, task_dir / "trace.jsonl")
                except Exception:
                    pass

    n_tc = len(tc_results)
    n_passed = sum(1 for r in tc_results if r["passed"])
    graded = dict(record)
    graded.pop("grade_jobs", None)
    graded["status"] = "graded"
    graded["test_case_results"] = tc_results
    graded["cell_accuracy"] = sum(r["cell_accuracy"] for r in tc_results) / n_tc if n_tc else 0.0
    graded["hard_score"] = 1.0 if n_tc > 0 and n_passed == n_tc else 0.0
    graded["match_count"] = sum(r["match_count"] for r in tc_results)
    graded["total_count"] = sum(r["total_count"] for r in tc_results)
    if output_format == "base_trajectories":
        graded["is_correct"] = graded["hard_score"] == 1.0
    return graded


async def run_stream(
    indices: list[int],
    dataset: list[dict],
    skill_dir: Path,
    model: str,
    workdir: Path,
    *,
    data_dir: Path,
    project_root: Path | None = None,
    max_turns: int = 30,
    executor_concurrency: int = 3,
    grader_concurrency: int = 1,
    grade_queue_max: int = 20,
    recalc_timeout: int = 180,
    on_record_complete: Callable[[dict], None] | None = None,
    output_format: Literal["eval", "base_trajectories", "training"] = "eval",
    test_cases: int = 1,
    test_case_start: int = 1,
    soffice_lock_path: Path | None = None,
    use_soffice_wrapper: bool = True,
    stop_if_best_below: int | None = None,
    failure_ids_dir: Path | None = None,
    oj: bool = False,
) -> dict:
    """Run executor workers and grader workers over a list of dataset indices.

    `failure_ids_dir` controls where `failure_ids.json` is written in
    base_trajectories mode. None (default) preserves the historical
    location `workdir.parent`. Set explicitly to keep the rerun's
    failure_ids file inside an isolated workdir (used by base-trajectories
    `--output-dir`).
    """
    project_root = project_root or Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    summary_name = "summary.json" if output_format == "base_trajectories" else "eval_summary.json"
    summary_path = workdir / summary_name
    total = len(indices)

    if soffice_lock_path is None:
        soffice_lock_path = Path("/tmp/skillgrad_soffice.lock")
    if use_soffice_wrapper:
        tools_dir = install_soffice_wrapper(workdir, soffice_lock_path)
        if tools_dir:
            print(f"  Installed soffice wrapper in {tools_dir}")
        else:
            print("  WARNING: LibreOffice/soffice not found; grading may fail.")
    else:
        real_soffice = find_soffice()
        if real_soffice:
            import os
            os.environ["SE_PIPELINE_REAL_SOFFICE"] = real_soffice
            os.environ["SE_PIPELINE_SOFFICE_LOCK"] = str(soffice_lock_path)

    config = {
        "mode": output_format,
        "model": model,
        "skills_dir": str(skill_dir),
        "data_dir": str(data_dir),
        "indices": indices,
        "max_turns": max_turns,
        "executor_concurrency": executor_concurrency,
        "grader_concurrency": grader_concurrency,
        "grade_queue_max": grade_queue_max,
        "test_cases": test_cases,
        "test_case_start": test_case_start,
        "oj": oj,
        "recalc_timeout": recalc_timeout,
        "soffice_lock_path": str(soffice_lock_path),
        "use_soffice_wrapper": use_soffice_wrapper,
        "stop_if_best_below": stop_if_best_below,
        "total": total,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    records: dict[int, dict] = {}
    if output_format == "base_trajectories":
        for idx in indices:
            ex_id = str(dataset[idx]["id"])
            ap = workdir / ex_id / "assessment.json"
            if ap.exists():
                records[idx] = json.loads(ap.read_text(encoding="utf-8")) | {"index": idx}
    elif summary_path.exists():
        # Per-task eval resume: retain only terminal records. Interrupted
        # executing/grading/retry-needed tasks are safely re-queued.
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            wanted = set(indices)
            for record in previous.get("results", []):
                idx = record.get("index")
                if idx in wanted and record.get("status") in {
                    "graded", "stopped_early", "failed"
                }:
                    records[idx] = record
            if records:
                print(f"  Resuming eval with {len(records)}/{total} tasks done")
        except (OSError, ValueError, TypeError):
            records = {}

    cost = CostTracker(model)
    openai_client = get_client_for_model(model)
    t0 = time.time()
    _save_json(summary_path, _summary(output_format, records, config, cost, t0, "running"))

    pending = [idx for idx in indices if idx not in records]
    exec_queue: asyncio.Queue[int | None] = asyncio.Queue()
    grade_queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=grade_queue_max)
    stop_event = asyncio.Event()
    for idx in pending:
        exec_queue.put_nowait(idx)

    def save(status: str = "running") -> dict:
        summary = _summary(output_format, records, config, cost, t0, status)
        _save_json(summary_path, summary)
        agg = summary["aggregate"]
        if output_format == "base_trajectories":
            print(f"  [{summary['completed']}/{total}] hard={agg.get('hard_score', 0):.1%} cell={agg.get('mean_cell_accuracy', 0):.1%}")
        else:
            target = f" target>{stop_if_best_below - 1}" if stop_if_best_below else ""
            print(
                "  [MONITOR] "
                f"graded={agg['n_graded']}/{total} "
                f"pass={agg['n_pass_graded']} "
                f"graded_hard={agg.get('mean_hard_graded', 0):.1%} "
                f"cell={agg.get('mean_cell_graded', 0):.1%} "
                f"pending={agg.get('pending_outputs', 0)} "
                f"executing={agg.get('executing', 0)} "
                f"retry_needed={agg.get('retry_needed', 0)} "
                f"best={agg['best_possible_hard_count']}/{total}"
                f"{target} cost=${cost.total_cost:.4f}"
            )
            if stop_if_best_below is not None and agg["best_possible_hard_count"] < stop_if_best_below:
                stop_event.set()
        return summary

    async def executor_worker(worker_id: int) -> None:
        while True:
            idx = await exec_queue.get()
            if idx is None:
                exec_queue.task_done()
                return
            ex_id = str(dataset[idx]["id"])
            if stop_event.is_set() and output_format != "base_trajectories":
                records[idx] = {"index": idx, "id": ex_id, "status": "stopped_early", "cell_accuracy": 0.0, "hard_score": 0.0, "match_count": 0, "total_count": 0, "elapsed": 0, "cost": {}}
                exec_queue.task_done()
                save()
                continue
            records[idx] = {"index": idx, "id": ex_id, "status": "executing"}
            save()
            print(f"  [EXEC worker={worker_id}] idx={idx} id={ex_id}")
            try:
                record = await _execute_record(
                    dataset, idx, data_dir, skill_dir, model, project_root,
                    workdir, max_turns, cost, output_format, test_cases,
                    test_case_start, openai_client, oj,
                )
            except Exception as exc:
                record = {"index": idx, "id": ex_id, "status": "retry_needed", "retry_reason": f"executor_exception: {exc}", "cell_accuracy": 0.0, "hard_score": 0.0, "match_count": 0, "total_count": 0, "elapsed": 0, "cost": {}}
            records[idx] = record
            save()
            if record.get("status") == "grade_pending":
                await grade_queue.put(record)
            exec_queue.task_done()

    async def grader_worker(worker_id: int) -> None:
        while True:
            record = await grade_queue.get()
            if record is None:
                grade_queue.task_done()
                return
            idx = record["index"]
            records[idx]["status"] = "grading"
            save()
            print(f"  [GRADE worker={worker_id}] idx={idx} id={record['id']}")
            try:
                graded = await _grade_record(record, recalc_timeout, soffice_lock_path, output_format)
            except Exception as exc:
                graded = dict(record)
                graded.update({"status": "retry_needed", "retry_reason": f"grading_exception: {exc}", "cell_accuracy": 0.0, "hard_score": 0.0, "match_count": 0, "total_count": 0})
            records[idx] = graded
            if on_record_complete:
                on_record_complete(graded)
            grade_queue.task_done()
            save()

    exec_workers = [asyncio.create_task(executor_worker(i + 1)) for i in range(executor_concurrency)]
    grade_workers = [asyncio.create_task(grader_worker(i + 1)) for i in range(grader_concurrency)]
    await exec_queue.join()
    for _ in exec_workers:
        exec_queue.put_nowait(None)
    await asyncio.gather(*exec_workers)
    await grade_queue.join()
    for _ in grade_workers:
        await grade_queue.put(None)
    await asyncio.gather(*grade_workers)

    final_status = "stopped_early" if stop_event.is_set() else "completed"
    summary = save(final_status)
    agg = summary["aggregate"]
    elapsed = summary["elapsed"]

    # Final pretty summary block (mirrors v4 evaluate.py:962-976)
    w = 78
    print(f"\n{'=' * w}")
    if output_format == "base_trajectories":
        print(f"  BASE TRAJECTORIES ({final_status}, {elapsed:.0f}s)")
        print(f"{'=' * w}")
        print(
            f"  Done: {summary['completed']}/{total}  |  "
            f"hard: {agg.get('hard_score', 0):.1%}  |  "
            f"cell: {agg.get('mean_cell_accuracy', 0):.1%}  |  "
            f"perfect: {agg.get('n_perfect', 0)}/{summary['completed']}"
        )
    else:
        print(f"  STREAM EVALUATION RESULTS ({final_status}, {elapsed:.0f}s)")
        print(f"{'=' * w}")
        print(
            f"  Graded: {agg['n_graded']}/{total} | "
            f"Pass: {agg['n_pass_graded']} | "
            f"Graded hard: {agg.get('mean_hard_graded', 0):.1%} | "
            f"Cell (graded): {agg.get('mean_cell_graded', 0):.1%}"
        )
        print(
            f"  Best possible: {agg['best_possible_hard_count']}/{total} | "
            f"Pending: {agg.get('pending_outputs', 0)} | "
            f"Retry needed: {agg.get('retry_needed', 0)} | "
            f"Stopped early: {agg.get('stopped_early', 0)}"
        )
    print(f"  Cost: ${cost.total_cost:.4f}")
    print(f"{'=' * w}")

    if output_format == "base_trajectories":
        ids = [str(dataset[idx]["id"]) for idx in indices]
        failure_ids, success_ids = identify_failures(workdir, ids)
        fid_dir = failure_ids_dir if failure_ids_dir is not None else workdir
        _save_json(fid_dir / "failure_ids.json", {"failure_ids": failure_ids, "success_ids": success_ids, "summary": summary["aggregate"]})
        print(f"  Failure IDs saved: {fid_dir / 'failure_ids.json'}")
    print(f"  Results saved to {summary_path}")
    return summary


def _indices_from_split(split_path: Path, split_name: str) -> list[int]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if split_name in data:
        return [int(i) for i in data[split_name]]
    key = f"{split_name}_indices"
    if key in data:
        return [int(i) for i in data[key]]
    if split_name == "test100" and "test_pool_indices" in data:
        return [int(i) for i in data["test_pool_indices"][:100]]
    raise KeyError(f"Split '{split_name}' not found in {split_path}")


def _maybe_update_run_metrics(workdir: Path, summary: dict, results_root: Path) -> None:
    """If `workdir` is `<run_dir>/eval` and a sibling `config.json` exists,
    patch metrics into it and upsert the manifest. Silent no-op otherwise.
    """
    run_dir = workdir.parent
    config_path = run_dir / "config.json"
    if workdir.name != "eval" or not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = config.get("metrics") or {}
    agg = summary.get("aggregate", {})
    metrics["test_hard"] = agg.get("mean_hard_graded") or agg.get("hard_score")
    metrics["test_cell_acc"] = agg.get("mean_cell_graded") or agg.get("mean_cell_accuracy")
    metrics["eval_n_graded"] = agg.get("n_graded")
    metrics["eval_n_total"] = summary.get("total")
    config["metrics"] = metrics
    if config.get("status") == "running":
        config["status"] = "completed"
    config["eval_completed"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_upsert(results_root, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared stream runner")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_runtime(p):
        p.add_argument("--max-turns", type=int, default=30)
        p.add_argument("--executor-concurrency", type=int, default=3)
        p.add_argument("--grader-concurrency", type=int, default=1)
        p.add_argument("--grade-queue-max", type=int, default=20)
        p.add_argument("--recalc-timeout", type=int, default=180)
        p.add_argument("--soffice-lock-path", default="/tmp/skillgrad_soffice.lock")

    # ── eval: evaluate a skill folder against a split ─────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate any skill folder on a held-out split.",
    )
    add_runtime(p_eval)
    p_eval.add_argument("--skill-dir", required=True,
                        help="Path to the skill folder (the parent of xlsx/).")
    p_eval.add_argument("--output-dir", required=True,
                        help="Where eval_summary.json + per-task dirs go.")
    p_eval.add_argument("--data-dir", default=None,
                        help="Normalized dataset dir (with spreadsheet/<id>/"
                             "1_<id>_{input,answer}.xlsx). Defaults to "
                             "<results-root>/normalized.")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--master-seed", type=int, default=0,
                        help="Picks which split to evaluate against.")
    p_eval.add_argument("--heldout-seed", type=int, default=42)
    p_eval.add_argument("--results-root", default="results",
                        help="Used to locate the canonical split file.")
    p_eval.add_argument("--split", default="test_indices",
                        help='Split key from split.json: "test_indices" '
                             '(default), "val_indices", "test_pool_indices".')
    p_eval.add_argument("--test-cases", type=int, default=1)
    p_eval.add_argument("--test-case-start", type=int, default=1)
    p_eval.add_argument("--stop-if-best-below", type=int)
    p_eval.add_argument("--oj", action="store_true")

    # ── base-trajectories: collect trajectories on the 200-task pool ──────
    p_base = sub.add_parser(
        "base-trajectories",
        help="Collect base-skill trajectories on the evolution pool.",
    )
    add_runtime(p_base)
    p_base.add_argument("--results-root", default="results")
    p_base.add_argument("--master-seed", type=int, default=0)
    p_base.add_argument("--heldout-seed", type=int, default=42)
    p_base.add_argument("--model", default="gpt-5.4")
    p_base.add_argument("--data-dir", default=None,
                        help="Normalized dataset dir. Defaults to "
                             "<results-root>/normalized.")
    p_base.add_argument("--skills-dir", default="seeds",
                        help="Directory containing the bootstrap skill (expects <skills-dir>/xlsx/SKILL.md).")
    p_base.add_argument("--output-dir", default=None,
                        help="Override the default model-keyed workdir "
                             "(results/base_trajectories/master_<M>_heldout_<H>/<model>/). "
                             "Useful for isolated re-runs that should not "
                             "overwrite existing model results. When set, "
                             "the rerun's failure_ids.json is written inside "
                             "this folder rather than at the shared parent.")
    p_base.add_argument("--ids", nargs="+", default=None,
                        help="Subset of evolution-pool task IDs to process. "
                             "Default: every ID in split['evolution_ids']. "
                             "Unknown IDs raise an error before any work runs.")

    args = parser.parse_args()

    results_root = Path(args.results_root)
    data_dir = Path(args.data_dir) if args.data_dir else normalized_dir_for(results_root)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset dir {data_dir} not found. Run "
            f"`python -m data.split split --master-seed {args.master_seed} "
            f"--heldout-seed {args.heldout_seed} --data-dir <raw>` first to "
            f"create the normalized symlinks, or pass --data-dir explicitly."
        )
    split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
    if not (split_dir / "split.json").exists():
        raise FileNotFoundError(
            f"Canonical split not found at {split_dir}. Run "
            f"`python -m data.split split --master-seed {args.master_seed} "
            f"--heldout-seed {args.heldout_seed}` first."
        )
    split = load_split(split_dir)
    dataset = load_dataset(data_dir)

    if args.command == "base-trajectories":
        id_to_idx = {str(s["id"]): i for i, s in enumerate(dataset)}
        indices = [id_to_idx[str(tid)] for tid in split["evolution_ids"]]
        if args.ids:
            requested = {str(t) for t in args.ids}
            pool = {str(tid) for tid in split["evolution_ids"]}
            unknown = requested - pool
            if unknown:
                raise ValueError(
                    f"--ids contains task IDs not in the evolution pool: "
                    f"{sorted(unknown)}"
                )
            indices = [i for i in indices if str(dataset[i]["id"]) in requested]
        if args.output_dir:
            workdir = Path(args.output_dir)
        else:
            workdir = base_trajectories_dir_for(
                results_root, args.master_seed, args.heldout_seed, args.model,
            )
        output_format = "base_trajectories"
        skill_dir = Path(args.skills_dir)
        test_cases = 1
        test_case_start = 1
        stop_if_best_below = None
    else:
        # eval mode — paths are explicit
        skill_dir = Path(args.skill_dir)
        if not (skill_dir / "xlsx").exists():
            raise FileNotFoundError(
                f"No xlsx skill found under {skill_dir} "
                f"(expected {skill_dir}/xlsx/SKILL.md)."
            )
        try:
            indices = [int(i) for i in split[args.split]]
        except KeyError:
            raise KeyError(
                f"Split key '{args.split}' not found in {split_dir / 'split.json'}. "
                f"Available: {sorted(k for k in split if k.endswith('_indices'))}"
            )
        workdir = Path(args.output_dir)
        output_format = "eval"
        test_cases = args.test_cases
        test_case_start = args.test_case_start
        stop_if_best_below = args.stop_if_best_below

    # When --output-dir is used on base-trajectories, keep the rerun's
    # failure_ids.json inside that isolated folder rather than writing to
    # the shared parent. None preserves the historical behavior for every
    # other invocation.
    failure_ids_dir = (
        workdir
        if args.command == "base-trajectories" and getattr(args, "output_dir", None)
        else None
    )

    summary = asyncio.run(run_stream(
        indices=indices,
        dataset=dataset,
        skill_dir=skill_dir,
        model=args.model,
        workdir=workdir,
        data_dir=data_dir,
        project_root=Path(".").resolve(),
        max_turns=args.max_turns,
        executor_concurrency=args.executor_concurrency,
        grader_concurrency=args.grader_concurrency,
        grade_queue_max=args.grade_queue_max,
        recalc_timeout=args.recalc_timeout,
        output_format=output_format,
        test_cases=test_cases,
        test_case_start=test_case_start,
        soffice_lock_path=Path(args.soffice_lock_path),
        stop_if_best_below=stop_if_best_below,
        failure_ids_dir=failure_ids_dir,
        oj=getattr(args, "oj", False),
    ))

    if args.command == "eval":
        # Auto-update run config.json + manifest only when the output is
        # under runs/<run_id>/eval/. Otherwise eval is a one-off and we
        # leave the manifest alone.
        _maybe_update_run_metrics(workdir, summary, results_root)


if __name__ == "__main__":
    main()
