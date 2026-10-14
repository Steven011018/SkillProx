"""Momentum stage.

Reads the current batch's diagnoses, the previous iteration's pattern record,
and the current guidance (skill), then writes:

  - an updated pattern record (`momentum_memory.md`) summarizing reusable
    failure-risk and successful-fix patterns across iterations, and
  - a per-attempt overlay (`momentum_overlay.md`) with one annotation per
    current task labeling the signal, current skill coverage, and an editor
    direction (`add`, `strengthen`, or `preserve`).

The patcher consumes the overlay; the persistent record stays internal to
this stage and is read by the next iteration's momentum call.
"""

import time
from pathlib import Path
from typing import Optional

from agents import Agent, Runner

from pipeline.helpers import _build_file_tools, _resolve_model
from prompts.momentum import MOMENTUM_PROMPT
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_settings
from runners.trajectory_logger import TrajectoryLogger, stream_with_logging


async def run_momentum(
    diagnoses_path: Path,
    previous_record_path: Path,
    skills_dir: Path,
    record_output_path: Path,
    overlay_output_path: Path,
    model: str,
    project_root: Path,
    cost_tracker: CostTracker,
    iter_dir: Path,
    iter_num: int,
) -> None:
    """Run the pattern-record writer for one iteration."""
    read_file, write_file = _build_file_tools(project_root)

    agent = Agent(
        name="PatternRecordWriter",
        instructions=MOMENTUM_PROMPT,
        model=_resolve_model(model),
        model_settings=get_model_settings(model),
        tools=[read_file, write_file],
    )

    # List skill files so the agent doesn't guess filenames.
    skill_dir = skills_dir / "xlsx"
    skill_files = [str(skill_dir / "SKILL.md")]
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for f in sorted(refs_dir.iterdir()):
            if f.is_file():
                skill_files.append(str(f))

    query = (
        f"This is iteration {iter_num}.\n\n"
        f"Read the input files and write your two outputs to the paths below.\n\n"
        f"Inputs:\n"
        f"  Current iteration's task attempts: {diagnoses_path}\n"
        f"  Previous iteration's pattern record: {previous_record_path}\n"
        f"  Current guidance files:\n"
        + "\n".join(f"    - {p}" for p in skill_files)
        + "\n\n"
        f"Outputs:\n"
        f"  Updated pattern record: {record_output_path}\n"
        f"  Per-attempt overlay: {overlay_output_path}\n"
    )

    logger = TrajectoryLogger(iter_dir / "momentum.jsonl")
    print(f"  [momentum] iter {iter_num}: synthesizing pattern record + overlay")
    t0 = time.time()
    try:
        result = Runner.run_streamed(agent, query, max_turns=15)
        await stream_with_logging(result, logger)
        delta = cost_tracker.update(result)
        cost_tracker.print_step("MOMENTUM", delta)
    except Exception as e:
        print(f"  [momentum] error: {e}")
        # Fallback: write minimal record + overlay so downstream doesn't crash.
        if not record_output_path.exists():
            record_output_path.write_text(
                f"# Pattern record (after iter {iter_num})\n\n"
                f"(Pattern-record writer failed: {e})\n",
                encoding="utf-8",
            )
        if not overlay_output_path.exists():
            overlay_output_path.write_text(
                f"# Per-attempt overlay (iter {iter_num})\n\n"
                f"(Pattern-record writer failed: {e}; patcher should read the diagnoses directly.)\n",
                encoding="utf-8",
            )
    logger.flush()
    elapsed = round(time.time() - t0, 2)
    print(f"  [momentum] Done in {elapsed}s")
