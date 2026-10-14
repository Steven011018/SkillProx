"""Skill patch stage.

The patcher reads the current task evidence, optional momentum overlay, and
current skill files. It evolves the skill in place by writing new content via
the write_file tool. In the current pipeline every patch is accepted; skill
snapshots are kept for forensics.
"""

import hashlib
import time
from pathlib import Path
from typing import Optional

from agents import Agent, Runner

from pipeline.helpers import _build_file_tools, _resolve_model
from prompts.patcher import PATCHER_PROMPT
from runners.cost_tracker import CostTracker
from runners.model_settings import get_model_settings
from runners.trajectory_logger import TrajectoryLogger, stream_with_logging


def _skill_fingerprint(skill_dir: Path) -> str:
    """Hash SKILL.md + every references/*.md so we can detect whether the
    patcher actually persisted an edit (some models narrate but never call
    write_file, leaving the skill frozen at the base template)."""
    h = hashlib.md5()
    paths = [skill_dir / "SKILL.md"]
    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():
        paths += [f for f in sorted(refs_dir.iterdir()) if f.is_file()]
    for p in paths:
        try:
            h.update(p.name.encode())
            h.update(p.read_bytes())
        except OSError:
            continue
    return h.hexdigest()


async def run_patch(
    batch_diagnoses_path: Path,
    skills_dir: Path,
    model: str,
    project_root: Path,
    cost_tracker: CostTracker,
    iter_dir: Path,
    overlay_path: Optional[Path] = None,
    momentum_memory_path: Optional[Path] = None,
) -> str:
    """Run the patcher agent to evolve the skill."""
    read_file, write_file = _build_file_tools(project_root)

    agent = Agent(
        name="Patcher",
        instructions=PATCHER_PROMPT,
        model=_resolve_model(model),
        model_settings=get_model_settings(model),
        tools=[read_file, write_file],
    )

    # List actual files so the agent doesn't guess filenames
    skill_dir = skills_dir / "xlsx"
    skill_files = [str(skill_dir / "SKILL.md")]
    refs_dir = skill_dir / "references"
    if refs_dir.is_dir():  # a model may clobber references/ into a file; skip if so
        for f in sorted(refs_dir.iterdir()):
            if f.is_file():
                skill_files.append(str(f))

    query = (
        f"Evolve the skill based on the analysis below.\n\n"
        f"Original diagnoses: {batch_diagnoses_path}\n"
    )
    if overlay_path is not None:
        query += f"Per-attempt overlay: {overlay_path}\n"
    if momentum_memory_path is not None:
        query += f"Cross-iteration pattern record: {momentum_memory_path}\n"
    query += (
        f"Skill files:\n"
        + "\n".join(f"  - {p}" for p in skill_files)
        + "\n"
        f"Reference directory for new L3 files: {refs_dir}\n"
        f"(Create files there when a reusable procedure should live outside "
        f"`SKILL.md`; it may not exist yet.)\n"
    )

    logger = TrajectoryLogger(iter_dir / "patcher.jsonl")
    print(f"  [patch] Evolving skill")
    t0 = time.time()
    output = ""

    # Some models (notably Qwen A3B) read the inputs, narrate a plan, then end
    # the turn WITHOUT ever calling write_file, leaving the skill frozen. Detect
    # that by fingerprinting the skill before/after and re-invoking with an
    # explicit write mandate until an edit is persisted (bounded).
    before = _skill_fingerprint(skill_dir)
    max_attempts = 3
    attempt_query = query
    for attempt in range(1, max_attempts + 1):
        try:
            result = Runner.run_streamed(agent, attempt_query, max_turns=20)
            await stream_with_logging(result, logger)
            output = result.final_output or ""
            delta = cost_tracker.update(result)
            cost_tracker.print_step("PATCH", delta)
        except Exception as e:
            print(f"  [patch] error: {e}")
            output = f"[PATCH ERROR] {e}"
        after = _skill_fingerprint(skill_dir)
        if after != before:
            if attempt > 1:
                print(f"  [patch] skill persisted on attempt {attempt}")
            break
        if attempt < max_attempts:
            print(
                f"  [patch] WARNING: no write_file persisted (attempt {attempt}"
                f"/{max_attempts}); retrying with explicit write mandate"
            )
            attempt_query = (
                "Your previous response did not save anything: you did not call "
                "the write_file tool, so the skill is unchanged. This is a "
                "failure. Read the current SKILL.md now, then you MUST call "
                "write_file to save the full updated SKILL.md (and any new "
                "references/*.md) before ending your turn. Do not merely "
                "describe changes.\n\n" + query
            )
        else:
            print(
                f"  [patch] WARNING: skill still unchanged after {max_attempts} "
                f"attempts — leaving skill as-is"
            )
    logger.flush()
    elapsed = round(time.time() - t0, 2)
    print(f"  [patch] Done in {elapsed}s")

    return output
