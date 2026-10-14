"""Executor-side data preparation and assessment helpers."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from agents import Runner

from evaluators.xlsx_compare import (
    cells_to_text,
    compute_accuracy,
    extract_cells,
    format_comparison,
)
from pipeline.executor import SkillAgent
from prompts.executor import EXECUTOR_PROMPT
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_kwargs
from runners.trajectory_logger import (
    TrajectoryLogger,
    build_execution_trace,
    set_phase,
    stream_with_logging,
)

def _write_workspace(path: Path, data: dict | str) -> None:
    """Write a JSON or text artifact to the shared workspace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_dataset(data_dir: Path) -> list[dict]:
    """Load SpreadSheetBench dataset.json."""
    dataset_path = data_dir / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def build_task_string(
    example: dict,
    input_xlsx_path: Path,
    output_xlsx_path: Path,
) -> str:
    """Construct the task string for the executor agent."""
    return (
        f"A user needs help with a spreadsheet task.\n\n"
        f"## User's Request\n"
        f"{example['instruction']}\n\n"
        f"## Input File\n"
        f"The spreadsheet is at: {input_xlsx_path}\n\n"
        f"## Instructions\n"
        f"1. Read the input spreadsheet\n"
        f"2. Write Python code to accomplish the user's request\n"
        f"3. Save the result to: {output_xlsx_path}\n"
        f"4. The answer should appear in cells: {example['answer_position']}\n\n"
        f"Use openpyxl to read and write the Excel file. "
        f"Make sure to save the workbook after making changes."
    )


def build_task_string_oj(
    example: dict,
    input_xlsx_path: Path,
    output_xlsx_path: Path,
    solution_path: Path,
) -> str:
    """Build an OJ task that requires a reusable path-parameterized script."""
    return (
        f"A user needs help with a spreadsheet task.\n\n"
        f"## User's Request\n{example['instruction']}\n\n"
        f"## Input File\n{input_xlsx_path}\n\n"
        f"Create {solution_path} as a reusable Python program accepting exactly "
        f"two positional arguments: INPUT_XLSX and OUTPUT_XLSX. Run it now on "
        f"the input above to create {output_xlsx_path}. The evaluator will run "
        f"the same script unchanged on hidden workbooks. Use openpyxl and save "
        f"the result. The answer belongs in {example['answer_position']}."
    )


def build_ground_truth_text(
    example: dict,
    answer_xlsx_path: Path,
) -> str:
    """Pre-extract ground truth cells as text."""
    cells = extract_cells(answer_xlsx_path, example["answer_position"])
    return (
        f"Expected cell values at {example['answer_position']}:\n"
        + cells_to_text(cells)
    )


def prepare_seed_data(
    dataset: list[dict],
    idx: int,
    data_dir: Path,
    workdir: Path,
) -> dict:
    """Set up per-seed working directory and build task/ground-truth strings."""
    example = dataset[idx]
    ex_id = example["id"]

    spreadsheet_dir = data_dir / "spreadsheet" / str(ex_id)
    input_path = spreadsheet_dir / f"1_{ex_id}_input.xlsx"
    answer_path = spreadsheet_dir / f"1_{ex_id}_answer.xlsx"

    task_workdir = workdir / f"evolve_{ex_id}"
    task_workdir.mkdir(parents=True, exist_ok=True)
    task_input = task_workdir / "input.xlsx"
    shutil.copy2(input_path, task_input)
    task_output = task_workdir / "output.xlsx"

    task_str = build_task_string(example, task_input, task_output)
    ground_truth = build_ground_truth_text(example, answer_path)

    return {
        "index": idx,
        "id": ex_id,
        "example": example,
        "task_str": task_str,
        "ground_truth": ground_truth,
        "task_output": task_output,
        "answer_path": answer_path,
        "task_workdir": task_workdir,
    }


async def run_execute(
    seed_data: dict,
    semaphore: asyncio.Semaphore,
    skills_dir: Path,
    model: str,
    project_root: Path,
    max_turns: int,
    round_num: int,
    cost_tracker: CostTracker,
    openai_client=None,
) -> dict:
    """Execute ONE seed with SkillAgent."""
    async with semaphore:
        idx = seed_data["index"]
        ex_id = seed_data["id"]
        task_str = seed_data["task_str"]
        task_workdir = seed_data["task_workdir"]

        print(f"\n  [exec] Starting seed {idx} (id={ex_id}) round={round_num}")
        model_kwargs = get_model_kwargs(model, openai_client=openai_client)

        log_path = task_workdir / f"exec_r{round_num}.jsonl"
        t0 = time.time()
        executor_output = ""
        logger = None
        retries = max(0, int(os.getenv("SKILLGRAD_EXEC_RETRIES", "0")))
        task_output = Path(seed_data["task_output"])
        for attempt in range(retries + 1):
            if task_output.exists():
                task_output.unlink()
            logger = TrajectoryLogger(log_path)
            set_phase("EXECUTE", logger)
            executor = SkillAgent(
                skills_dir=skills_dir,
                model=model,
                max_turns=max_turns,
                system_prompt=EXECUTOR_PROMPT,
                model_kwargs=model_kwargs,
                include_skills=["xlsx"],
                project_root=project_root,
            )
            try:
                result = Runner.run_streamed(
                    executor.agent, task_str, max_turns=max_turns,
                )
                await stream_with_logging(result, logger)
                executor_output = result.final_output or ""
                delta = cost_tracker.update(result)
                cost_tracker.print_step(f"EXEC seed={idx} r{round_num}", delta)
            except Exception as e:
                print(f"  [exec] Seed {idx} EXECUTE error: {e}")
                executor_output = f"[EXECUTION ERROR] {e}"
            finally:
                logger.flush()
            if task_output.exists():
                break
            if attempt < retries:
                print(
                    f"  [exec] Seed {idx} produced no output.xlsx; "
                    f"retry {attempt + 1}/{retries}"
                )
        elapsed = round(time.time() - t0, 2)

        return {
            "index": idx,
            "id": ex_id,
            "executor_output": executor_output,
            "elapsed": elapsed,
            "trajectory_path": str(log_path),
            "logger": logger,
            "task_workdir": str(task_workdir),
        }


def assess_seed(seed_data: dict, exec_result: dict, round_num: int = 0) -> dict:
    """Programmatic evaluation — no LLM. Computes accuracy + cell diff.

    Also writes execution trace and assessment to workspace, named by round
    to avoid overwrites across rounds.
    """
    idx = seed_data["index"]
    ex_id = seed_data["id"]
    example = seed_data["example"]
    task_output = seed_data["task_output"]
    answer_path = seed_data["answer_path"]

    # format_comparison recalculates the output file (soffice) once,
    # then compute_accuracy skips recalculation to avoid double work.
    try:
        cell_comp = format_comparison(
            task_output, answer_path, example["answer_position"],
        )
        acc = compute_accuracy(
            task_output, answer_path, example["answer_position"],
            recalculate=False,
        )
    except Exception as exc:
        # A truncated/non-XLSX model artifact must become a scored failure,
        # not abort the entire training iteration.
        cell_comp = f"(assessment failed: {type(exc).__name__}: {exc})"
        acc = {"accuracy": 0.0, "match_count": 0, "total_count": 0}
    is_correct = acc["accuracy"] == 1.0

    print(
        f"  [assess] Seed {idx} (id={ex_id}): "
        f"acc={acc['match_count']}/{acc['total_count']} "
        f"({acc['accuracy']:.1%}){' PASS' if is_correct else ''}"
    )

    # Write execution trace to workspace (round-numbered to avoid overwrites)
    trace_path = Path(exec_result["task_workdir"]) / f"execution_trace_r{round_num}.md"
    try:
        trace_text = build_execution_trace(exec_result["trajectory_path"])
        _write_workspace(trace_path, trace_text)
    except Exception as e:
        print(f"  [assess] Seed {idx}: trace extraction failed: {e}")
        _write_workspace(trace_path, "(trace extraction failed)")

    # Write assessment to workspace (round-numbered to avoid overwrites)
    assessment_path = Path(exec_result["task_workdir"]) / f"assessment_r{round_num}.json"
    _write_workspace(assessment_path, {
        "accuracy": acc["accuracy"],
        "match_count": acc["match_count"],
        "total_count": acc["total_count"],
        "is_correct": is_correct,
    })

    return {
        "index": idx,
        "id": ex_id,
        "example": example,
        "task_str": seed_data["task_str"],
        "ground_truth": seed_data["ground_truth"],
        "executor_output": exec_result["executor_output"],
        "is_correct": is_correct,
        "accuracy": acc,
        "cell_comparison": cell_comp,
        "elapsed": exec_result["elapsed"],
        "trajectory_path": exec_result["trajectory_path"],
        "execution_trace_path": str(trace_path),
        "logger": exec_result["logger"],
        "task_workdir": exec_result["task_workdir"],
    }
