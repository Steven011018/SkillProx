"""Per-task diagnose stage.

Two diagnoser flavors run in parallel inside the training loop:
  - "failure": a task in the current batch failed. Diagnose what the agent
    got wrong by reading the trace + cell comparison + answer.
  - "contrastive": a task that previously failed (in the base trajectory)
    now succeeds. Diagnose what changed — what knowledge in the evolved
    skill produced the success.

`classify_batch` is the trivial post-execute splitter that tells the
training loop which assessments go to which diagnoser flavor.
"""

import asyncio
import re
import time
from pathlib import Path

from agents import Agent, Runner

from pipeline.helpers import _build_file_tools, _resolve_model
from pipeline.execution import _write_workspace
from prompts.diagnoser import (
    CONTRASTIVE_DIAGNOSER_PROMPT,
    FAILURE_DIAGNOSER_PROMPT,
)
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_settings
from runners.trajectory_logger import (
    TrajectoryLogger,
    save_merged_trace,
    stream_with_logging,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFY — split batch results into failed vs contrastive
# ═══════════════════════════════════════════════════════════════════════════

def classify_batch(assessments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split assessed seeds into complete failures and contrastive (now correct).

    All seeds in the training set failed with the base skill. If a seed
    now succeeds with the evolved skill, it provides a contrastive signal.

    Returns:
        (failed, contrastive) — two lists of assessment dicts.
    """
    failed = [a for a in assessments if not a["is_correct"]]
    contrastive = [a for a in assessments if a["is_correct"]]
    return failed, contrastive


# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSE — per-task diagnosis (parallel)
# ═══════════════════════════════════════════════════════════════════════════

def _build_failure_diagnosis_query(
    assessment: dict,
    merged_trace_path: Path,
    skill_dir: Path,
) -> str:
    """Build the query for the complete failure diagnoser."""
    task_desc = assessment["example"]["instruction"]
    cell_comp = assessment["cell_comparison"]
    acc = assessment["accuracy"]

    return (
        f"## Task\n{task_desc}\n\n"
        f"## Cell Comparison (expected vs actual)\n"
        f"Accuracy: {acc['match_count']}/{acc['total_count']} "
        f"({acc['accuracy']:.1%})\n\n"
        f"{cell_comp}\n\n"
        f"## Files (read with read_file as needed)\n"
        f"- Execution trace: {merged_trace_path}\n"
        f"- Agent output: {assessment['task_workdir']}/output.xlsx\n"
        f"- Skill directory: {skill_dir}\n"
    )


def _build_contrastive_diagnosis_query(
    assessment: dict,
    merged_trace_path: Path,
    base_trace_path: Path,
    base_cell_comparison: str,
    skill_dir: Path,
) -> str:
    """Build the query for the contrastive diagnoser."""
    task_desc = assessment["example"]["instruction"]

    return (
        f"## Task\n{task_desc}\n\n"
        f"## What was wrong in the first attempt\n"
        f"{base_cell_comparison}\n\n"
        f"## Files (read with read_file as needed)\n"
        f"- First attempt trace (failed): {base_trace_path}\n"
        f"- Second attempt trace (succeeded): {merged_trace_path}\n"
        f"- Skill directory: {skill_dir}\n"
    )


def _extract_diagnosis(agent_output: str) -> str:
    """Extract the <diagnosis> block from agent output."""
    match = re.search(r"<diagnosis>(.*?)</diagnosis>", agent_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return agent_output.strip()


def _build_closed_loop_preamble(
    prior: str | None,
    reject_context: str | None,
) -> str:
    """Assemble the optional closed-loop context prepended to a diagnosis query.

    `prior` is the cross-iteration memory (which diagnosis directions have
    historically led to accepted vs rejected patches). `reject_context` is the
    within-iteration signal: the previous attempt's patch was re-executed on
    the same batch and did not help (or regressed), so the diagnoser is asked to
    change direction rather than repeat the rejected edit.
    """
    blocks = []
    if reject_context:
        blocks.append(
            "## IMPORTANT — the previous patch attempt this iteration was "
            "REJECTED\n"
            "Re-executing the patched skill on this same batch did not improve "
            "outcomes (see below). Do NOT repeat the same edit direction — "
            "diagnose a *different* root cause or a more targeted fix.\n\n"
            f"{reject_context}\n"
        )
    if prior:
        blocks.append(
            "## Prior-iteration feedback (which diagnosis directions helped)\n"
            "Use this as a soft prior — favor directions that were accepted, "
            "be cautious with directions that were repeatedly rejected.\n\n"
            f"{prior}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


async def run_diagnose(
    assessment: dict,
    diagnosis_type: str,  # "failure" or "contrastive"
    iter_dir: Path,
    skills_dir: Path,
    base_trajectories_dir: Path,
    model: str,
    project_root: Path,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
    prior: str | None = None,
    reject_context: str | None = None,
) -> dict:
    """Run diagnosis for one seed.

    Closed-loop extras (both optional, no-op when None so the baseline path is
    unchanged):
      prior:          cross-iteration accept/reject memory injected into the prompt.
      reject_context: within-iteration rejection feedback for a retry.
    """
    async with semaphore:
        ex_id = assessment["id"]
        task_workdir = Path(assessment["task_workdir"])

        # Build merged trace from raw JSONL
        raw_trace = task_workdir / "exec_r0.jsonl"
        merged_trace = task_workdir / "trace.jsonl"
        if raw_trace.exists():
            save_merged_trace(raw_trace, merged_trace)

        # Select prompt and build query
        if diagnosis_type == "failure":
            system_prompt = FAILURE_DIAGNOSER_PROMPT
            query = _build_failure_diagnosis_query(
                assessment, merged_trace, skills_dir / "xlsx",
            )
        else:
            # Contrastive: need base trajectory and base cell comparison
            base_trace = base_trajectories_dir / str(ex_id) / "trace.jsonl"
            base_assessment_path = base_trajectories_dir / str(ex_id) / "assessment.json"
            base_cell_comp_path = base_trajectories_dir / str(ex_id) / "cell_comparison.txt"

            base_cell_comp = "(base cell comparison not available)"
            if base_cell_comp_path.exists():
                base_cell_comp = base_cell_comp_path.read_text(encoding="utf-8")

            system_prompt = CONTRASTIVE_DIAGNOSER_PROMPT
            query = _build_contrastive_diagnosis_query(
                assessment, merged_trace, base_trace,
                base_cell_comp, skills_dir / "xlsx",
            )

        # Closed-loop context (reject feedback + prior memory); no-op if both None.
        preamble = _build_closed_loop_preamble(prior, reject_context)
        if preamble:
            query = preamble + "\n" + query

        # Build diagnoser agent
        read_file, _ = _build_file_tools(project_root)
        agent = Agent(
            name=f"Diagnoser-{ex_id}",
            instructions=system_prompt,
            model=_resolve_model(model),
            model_settings=get_model_settings(model),
            tools=[read_file],
        )

        # Run
        diag_dir = iter_dir / f"diagnose_{ex_id}"
        diag_dir.mkdir(parents=True, exist_ok=True)
        logger = TrajectoryLogger(diag_dir / "diagnosis.jsonl")

        print(f"  [diagnose] {diagnosis_type} for {ex_id}")
        t0 = time.time()
        try:
            result = Runner.run_streamed(agent, query, max_turns=15)
            await stream_with_logging(result, logger)
            output = result.final_output or ""
            delta = cost_tracker.update(result)
            cost_tracker.print_step(f"DIAGNOSE {ex_id}", delta)
        except Exception as e:
            print(f"  [diagnose] {ex_id} error: {e}")
            output = f"[DIAGNOSIS ERROR] {e}"

        logger.flush()

        # Extract and save diagnosis text
        diagnosis_text = _extract_diagnosis(output)
        _write_workspace(diag_dir / "diagnosis.txt", diagnosis_text)

        return {
            "id": ex_id,
            "type": diagnosis_type,
            "diagnosis": diagnosis_text,
            "elapsed": round(time.time() - t0, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLE — collect all diagnoses into one file
# ═══════════════════════════════════════════════════════════════════════════

def assemble_diagnoses(
    diagnoses: list[dict],
    assessments: list[dict],
    output_path: Path,
) -> Path:
    """Write all diagnoses into a single batch_diagnoses.md file."""
    acc_by_id = {a["id"]: a for a in assessments}
    lines = [f"# Batch Diagnoses\n"]

    for diag in diagnoses:
        ex_id = diag["id"]
        dtype = diag["type"]
        assessment = acc_by_id.get(ex_id, {})
        acc = assessment.get("accuracy", {})

        if dtype == "failure":
            acc_str = f"{acc.get('accuracy', 0):.1%}" if acc else "N/A"
            header = f"## Task {ex_id} (failed, accuracy: {acc_str})"
        else:
            header = f"## Task {ex_id} (previously failed, now succeeds)"

        lines.append(f"\n{header}\n")
        lines.append(diag["diagnosis"])
        lines.append("")

        # Include evidence paths
        task_workdir = assessment.get("task_workdir", "")
        lines.append(f"Evidence:")
        lines.append(f"  - trace: {task_workdir}/trace.jsonl")
        if dtype == "contrastive":
            lines.append(f"  - base trace: (see base_trajectories/{ex_id}/trace.jsonl)")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [assemble] Wrote {len(diagnoses)} diagnoses to {output_path}")
    return output_path
