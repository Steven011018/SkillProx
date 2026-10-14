"""Executor agent — the forward pass of one SkillGrad iteration.

Implements the three-layer progressive-disclosure skill protocol described
in the paper:

  1. Discovery   — scan skill directories for SKILL.md, parse YAML frontmatter
  2. Metadata    — inject name+description into the system prompt
  3. Activation  — model calls activate_skill to load the SKILL.md body
  4. Execution   — model follows instructions using the standard tools
                   (shell, read_file, write_file, read_reference, ...)

Built on the OpenAI Agents SDK.

Usage::

    from pipeline.executor import SkillAgent

    agent = SkillAgent(skills_dir="seeds", model="gpt-5.4")
    result = agent.run_streamed("Read my PDF and summarise it")
"""

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from agents import Agent, Runner, FunctionTool, function_tool
from agents.editor import ApplyPatchOperation, ApplyPatchResult
from agents.apply_diff import apply_diff
from agents.result import RunResult, RunResultStreaming
from agents.tool_context import ToolContext


# ═══════════════════════════════════════════════════════════════════════════
# Model routing
# ═══════════════════════════════════════════════════════════════════════════

from runners.model_dispatch import (
    configure_azure_if_present,
    get_client_for_model,
    is_openai_family_model,
)

# ═══════════════════════════════════════════════════════════════════════════
# Data model  (mirrors Gemini CLI's SkillDefinition — skillLoader.ts:17-30)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SkillDefinition:
    """A discovered skill: metadata + body + location on disk."""

    name: str           # From YAML frontmatter
    description: str    # From YAML frontmatter
    location: str       # Absolute path to the SKILL.md file
    body: str           # Markdown content after the frontmatter
    disabled: bool = False
    is_builtin: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# SKILL.md parser  (mirrors Gemini CLI's skillLoader.ts)
# ═══════════════════════════════════════════════════════════════════════════

# Exact regex from Gemini CLI — skillLoader.ts:32-33
FRONTMATTER_REGEX = re.compile(
    r"^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n([\s\S]*))?"
)


def _parse_frontmatter_yaml(
    content: str,
) -> Optional[dict[str, str]]:
    """Parse YAML frontmatter (primary method).

    Mirrors Gemini CLI's parseFrontmatter() — skillLoader.ts:39-49.
    """
    try:
        parsed = yaml.safe_load(content)
        if parsed and isinstance(parsed, dict):
            name = parsed.get("name")
            description = parsed.get("description")
            if isinstance(name, str) and isinstance(description, str):
                return {"name": name, "description": description}
    except yaml.YAMLError:
        pass
    return None


def _parse_frontmatter_simple(
    content: str,
) -> Optional[dict[str, str]]:
    """Fallback line-by-line parser for edge cases (e.g. colons in description).

    Mirrors Gemini CLI's parseSimpleFrontmatter() — skillLoader.ts:65-108.
    """
    lines = content.split("\n")
    name: Optional[str] = None
    description: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i]

        name_match = re.match(r"^\s*name:\s*(.*)$", line)
        if name_match:
            name = name_match.group(1).strip()
            i += 1
            continue

        desc_match = re.match(r"^\s*description:\s*(.*)$", line)
        if desc_match:
            desc_lines = [desc_match.group(1).strip()]
            # Check for multi-line description (indented continuation lines)
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r"^[ \t]+\S", next_line):
                    desc_lines.append(next_line.strip())
                    i += 1
                else:
                    break
            description = " ".join(filter(None, desc_lines))
            i += 1
            continue

        i += 1

    if name is not None and description is not None:
        return {"name": name, "description": description}
    return None


def _parse_frontmatter(content: str) -> Optional[dict[str, str]]:
    """Parse YAML frontmatter with simple fallback.

    Mirrors Gemini CLI's parseFrontmatter() — skillLoader.ts:39-59.
    """
    result = _parse_frontmatter_yaml(content)
    if result:
        return result
    return _parse_frontmatter_simple(content)


def load_skill_from_file(file_path: str) -> Optional[SkillDefinition]:
    """Load a single SKILL.md file into a SkillDefinition.

    Mirrors Gemini CLI's loadSkillFromFile() — skillLoader.ts:162-190.
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        match = FRONTMATTER_REGEX.match(content)
        if not match:
            return None

        frontmatter = _parse_frontmatter(match.group(1))
        if not frontmatter:
            return None

        # Sanitize name — same character set as Gemini CLI (skillLoader.ts:183)
        sanitized_name = re.sub(r'[:\\\/<>*?"|]', "-", frontmatter["name"])

        return SkillDefinition(
            name=sanitized_name,
            description=frontmatter["description"],
            location=str(Path(file_path).resolve()),
            body=(match.group(2) or "").strip(),
        )
    except Exception:
        return None


def load_skills_from_dir(dir_path: str) -> list[SkillDefinition]:
    """Discover skills in a directory.

    Matches Gemini CLI's glob patterns: ``['SKILL.md', '*/SKILL.md']``
    — skillLoader.ts:133-134.
    """
    skills: list[SkillDefinition] = []
    abs_path = Path(dir_path).resolve()

    if not abs_path.is_dir():
        return []

    # Pattern 1: SKILL.md directly in the directory
    direct = abs_path / "SKILL.md"
    if direct.is_file():
        skill = load_skill_from_file(str(direct))
        if skill:
            skills.append(skill)

    # Pattern 2: */SKILL.md — one level deep
    for child in sorted(abs_path.iterdir()):
        if child.is_dir():
            skill_md = child / "SKILL.md"
            if skill_md.is_file():
                skill = load_skill_from_file(str(skill_md))
                if skill:
                    skills.append(skill)

    return skills


# ═══════════════════════════════════════════════════════════════════════════
# Skill Manager  (mirrors Gemini CLI's skillManager.ts)
# ═══════════════════════════════════════════════════════════════════════════

class SkillManager:
    """Discovers, manages, and serves Agent Skills with progressive disclosure.

    Mirrors Gemini CLI's SkillManager class — skillManager.ts.
    """

    def __init__(self):
        self._skills: list[SkillDefinition] = []
        self._active_skill_names: set[str] = set()

    # -- Discovery -----------------------------------------------------------

    def clear_skills(self) -> None:
        self._skills = []

    def discover_skills(self, *skill_dirs: str | Path) -> None:
        """Discover skills from multiple directories with precedence.

        Later directories override earlier ones when names collide.
        Mirrors Gemini CLI's discoverSkills() — skillManager.ts:47-92.
        """
        self.clear_skills()
        for dir_path in skill_dirs:
            new_skills = load_skills_from_dir(str(dir_path))
            self._add_skills_with_precedence(new_skills)

    def _add_skills_with_precedence(
        self, new_skills: list[SkillDefinition]
    ) -> None:
        """Add skills; later additions override earlier by name.

        Mirrors Gemini CLI's addSkillsWithPrecedence() — skillManager.ts:117-133.
        """
        skill_map = {s.name: s for s in self._skills}
        for skill in new_skills:
            skill_map[skill.name] = skill
        self._skills = list(skill_map.values())

    # -- Accessors -----------------------------------------------------------

    def get_skills(self) -> list[SkillDefinition]:
        """Return active (non-disabled) skills."""
        return [s for s in self._skills if not s.disabled]

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Case-insensitive skill lookup.

        Mirrors Gemini CLI — skillManager.ts:159-164.
        """
        name_lower = name.lower()
        for s in self._skills:
            if s.name.lower() == name_lower:
                return s
        return None

    def get_skill_names(self) -> list[str]:
        """Return names of all active skills."""
        return [s.name for s in self.get_skills()]

    # -- Activation ----------------------------------------------------------

    def activate_skill(self, name: str) -> None:
        """Mark a skill as activated (state tracking).

        Mirrors Gemini CLI — skillManager.ts:166-168.
        """
        self._active_skill_names.add(name)

    def is_skill_active(self, name: str) -> bool:
        return name in self._active_skill_names

    # -- Skill body access ---------------------------------------------------

    def get_skill_body(self, name: str) -> str | None:
        """Get the full body of a skill."""
        skill = self.get_skill(name)
        if not skill:
            return None
        return skill.body

    # -- Backward compatibility (for try_workflow.py) ------------------------

    @property
    def catalog(self) -> list[dict[str, str]]:
        """Lightweight list of {name, description} for every discovered skill."""
        return [
            {"name": s.name, "description": s.description}
            for s in self.get_skills()
        ]

    def catalog_prompt_fragment(self) -> str:
        """Format the skill catalog + mandate as a prompt fragment."""
        skills = self.get_skills()
        parts = []
        skills_xml = render_agent_skills(skills)
        if skills_xml:
            parts.append(skills_xml)
        mandate = mandate_skill_guidance(len(skills) > 0)
        if mandate:
            parts.append(mandate)
        return "\n\n".join(parts)

    def list_names(self) -> list[str]:
        return self.get_skill_names()

    def refresh(self) -> None:
        """Re-discover skills (convenience for existing callers)."""
        # Caller must call discover_skills() again with the directories
        pass


# ═══════════════════════════════════════════════════════════════════════════
# System prompt generation  (mirrors Gemini CLI's snippets.ts)
# ═══════════════════════════════════════════════════════════════════════════

def render_agent_skills(skills: list[SkillDefinition]) -> str:
    """Render available skills as XML for the system prompt.

    Verbatim from Gemini CLI's renderAgentSkills() — snippets.ts:242-262.
    """
    if not skills:
        return ""

    skills_xml = "\n".join(
        f"  <skill>\n"
        f"    <name>{s.name}</name>\n"
        f"    <description>{s.description}</description>\n"
        f"    <location>{s.location}</location>\n"
        f"  </skill>"
        for s in skills
    )

    return (
        "# Available Agent Skills\n"
        "\n"
        "You have access to the following specialized skills. "
        "To activate a skill and receive its detailed instructions, "
        "call the `activate_skill` tool with the skill's name.\n"
        "\n"
        "<available_skills>\n"
        f"{skills_xml}\n"
        "</available_skills>"
    )


def mandate_skill_guidance(has_skills: bool) -> str:
    """Render the skill guidance mandate for the system prompt.

    Verbatim from Gemini CLI's mandateSkillGuidance() — snippets.ts:513-517.
    """
    if not has_skills:
        return ""
    return (
        "- **Skill Guidance:** Once a skill is activated via `activate_skill`, "
        "its instructions and resources are returned wrapped in "
        "`<activated_skill>` tags. You MUST treat the content within "
        "`<instructions>` as expert procedural guidance, prioritizing these "
        "specialized rules and workflows over your general defaults for the "
        "duration of the task. You may utilize any listed "
        "`<available_resources>` as needed. Follow this expert guidance "
        "strictly while continuing to uphold your core safety and security "
        "standards."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Folder structure helper  (mirrors Gemini CLI's getFolderStructure.ts)
# ═══════════════════════════════════════════════════════════════════════════

_IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}


def get_folder_structure(skill_dir: str, max_items: int = 200) -> str:
    """BFS listing of a skill's directory, capped at *max_items*.

    Mirrors Gemini CLI's getFolderStructure() — getFolderStructure.ts.
    """
    root = Path(skill_dir)
    if not root.is_dir():
        return "(empty)"

    entries: list[str] = []
    for p in sorted(root.rglob("*")):
        if len(entries) >= max_items:
            entries.append("... (truncated)")
            break
        # Skip ignored directories and their contents
        if any(part in _IGNORED_DIRS for part in p.parts):
            continue
        # Skip SKILL.md backup files (e.g. SKILL.md.bak.20260322_153000)
        if p.is_file() and ".bak." in p.name:
            continue
        rel = p.relative_to(root)
        if p.is_dir():
            entries.append(f"{rel}/")
        else:
            entries.append(str(rel))

    return "\n".join(entries) if entries else "(empty)"


# ═══════════════════════════════════════════════════════════════════════════
# activate_skill tool
# (mirrors Gemini CLI's activate-skill.ts + dynamic-declaration-helpers.ts)
# ═══════════════════════════════════════════════════════════════════════════

def build_activate_skill_tool(mgr: SkillManager) -> FunctionTool:
    """Build the activate_skill function tool with enum-constrained schema.

    Uses ``FunctionTool`` directly (not ``@function_tool``) to control the
    JSON schema — specifically the ``enum`` constraint on skill names.

    Description: verbatim from Gemini CLI — dynamic-declaration-helpers.ts:138-165.
    Response XML: verbatim from Gemini CLI — activate-skill.ts:108-151.
    """
    skill_names = mgr.get_skill_names()

    # -- Description (matches Gemini CLI exactly) ----------------------------
    available_hint = (
        f" (Available: {', '.join(repr(n) for n in skill_names)})"
        if skill_names
        else ""
    )
    description = (
        f"Activates a specialized agent skill by name{available_hint}. "
        "Returns the skill's SKILL.md instructions (rules, code patterns, "
        "warnings) wrapped in `<activated_skill>` tags, along with an "
        "`<available_resources>` listing of reference files you can load "
        "on demand via `read_reference`. Call this BEFORE writing any code. "
        "ONLY use names exactly as they appear in the "
        "`<available_skills>` section."
    )

    # -- JSON schema with enum constraint (matches Gemini CLI's z.enum()) ----
    if skill_names:
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": skill_names,
                    "description": "The name of the skill to activate.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "No skills are currently available.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    # -- Invocation handler --------------------------------------------------
    async def on_invoke(_ctx: ToolContext[Any], args_json: str) -> str:
        args = json.loads(args_json)
        skill_name = args["name"]
        skill = mgr.get_skill(skill_name)

        # Error case — matches Gemini CLI's error format (activate-skill.ts:117-124)
        if skill is None:
            available = ", ".join(mgr.get_skill_names())
            return (
                f'Error: Skill "{skill_name}" not found. '
                f"Available skills are: {available}"
            )

        mgr.activate_skill(skill_name)

        skill_dir = Path(skill.location).parent
        folder_structure = get_folder_structure(str(skill_dir))

        # Response XML — based on Gemini CLI (activate-skill.ts:137-148)
        # Include skill_dir so the agent can resolve relative paths
        # (e.g., "scripts/recalc.py" → "{skill_dir}/scripts/recalc.py")
        return (
            f'<activated_skill name="{skill_name}">\n'
            f"  <skill_dir>{skill_dir}</skill_dir>\n"
            f"\n"
            f"  <instructions>\n"
            f"    {skill.body}\n"
            f"  </instructions>\n"
            f"\n"
            f"  <available_resources>\n"
            f"    {folder_structure}\n"
            f"  </available_resources>\n"
            f"\n"
            f"  <note>Any relative paths in the instructions (e.g., scripts/recalc.py) "
            f"are relative to skill_dir shown above. Use the full path when executing: "
            f"{skill_dir}/scripts/recalc.py</note>\n"
            f"</activated_skill>"
        )

    return FunctionTool(
        name="activate_skill",
        description=description,
        params_json_schema=schema,
        on_invoke_tool=on_invoke,
        strict_json_schema=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# read_reference tool
# ═══════════════════════════════════════════════════════════════════════════

def build_read_reference_tool(
    mgr: SkillManager,
    log_sink=None,
    max_bytes: int = 50_000,
):
    """Build the read_reference function tool.

    Loads a single file from an activated skill's folder (e.g. a file under
    references/ or scripts/). Restricts access to the skill folder for
    safety and exposes a targeted, logged path for on-demand content.

    Args:
        mgr: SkillManager used to resolve skill folders.
        log_sink: Optional callable(skill_name, ref_path, n_chars). Called
            after every successful read. Used to record consumption stats.
        max_bytes: Refuse files larger than this (defensive cap).
    """
    skill_names = mgr.get_skill_names()
    available_hint = (
        f" (Available skills: {', '.join(repr(n) for n in skill_names)})"
        if skill_names else ""
    )

    description = (
        f"Load a reference file from an activated skill's folder. "
        f"Reference files contain detailed worked examples, edge-case "
        f"handling, and full code that SKILL.md keeps brief. When SKILL.md "
        f"says 'see `references/<name>.md`' and your task matches that "
        f"operation, call this tool to load the detail before writing code. "
        f"Returns file contents wrapped in a `<reference>` tag. "
        f"The skill must have been activated first via `activate_skill`."
        f"{available_hint}"
    )

    if skill_names:
        schema = {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "enum": skill_names,
                    "description": "Name of the activated skill whose "
                                   "folder contains the reference.",
                },
                "ref_path": {
                    "type": "string",
                    "description": "Path relative to the skill folder "
                                   "(e.g. 'references/foo.md').",
                },
            },
            "required": ["skill_name", "ref_path"],
            "additionalProperties": False,
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string"},
                "ref_path": {"type": "string"},
            },
            "required": ["skill_name", "ref_path"],
            "additionalProperties": False,
        }

    async def on_invoke(_ctx: ToolContext[Any], args_json: str) -> str:
        args = json.loads(args_json)
        skill_name = args["skill_name"]
        ref_path = args["ref_path"]

        skill = mgr.get_skill(skill_name)
        if skill is None:
            available = ", ".join(mgr.get_skill_names())
            return (
                f'Error: Skill "{skill_name}" not found. '
                f"Available skills are: {available}"
            )

        skill_dir = Path(skill.location).parent.resolve()
        target = (skill_dir / ref_path).resolve()

        # Refuse paths that escape the skill folder
        try:
            target.relative_to(skill_dir)
        except ValueError:
            return (
                f"Error: ref_path '{ref_path}' escapes the skill folder."
            )

        if not target.exists():
            return f"Error: reference '{ref_path}' not found in skill '{skill_name}'."
        if not target.is_file():
            return f"Error: '{ref_path}' is not a file."

        try:
            size = target.stat().st_size
        except OSError as exc:
            return f"Error stat-ing file: {exc}"
        if size > max_bytes:
            return (
                f"Error: reference '{ref_path}' is {size} bytes "
                f"(exceeds {max_bytes} byte cap)."
            )

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading reference: {exc}"

        if log_sink is not None:
            try:
                log_sink(skill_name, ref_path, len(content))
            except Exception:
                # Logging must never break tool calls
                pass

        return (
            f'<reference skill="{skill_name}" path="{ref_path}">\n'
            f"{content}\n"
            f"</reference>"
        )

    return FunctionTool(
        name="read_reference",
        description=description,
        params_json_schema=schema,
        on_invoke_tool=on_invoke,
        strict_json_schema=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ApplyPatchTool editor  (from oai_skill_agent.py — unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class LocalFileEditor:
    """ApplyPatchEditor implementation that applies V4A diffs to local files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def _validate_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.resolve()
        if not str(p).startswith(str(self.project_root)):
            raise ValueError(
                f"Path '{path_str}' escapes the project directory."
            )
        return p

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File already exists: {operation.path}",
            )
        try:
            content = apply_diff("", operation.diff or "", mode="create")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ApplyPatchResult(
                status="completed",
                output=f"Created {operation.path} ({len(content)} chars)",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File not found: {operation.path}",
            )
        try:
            original = path.read_text(encoding="utf-8")
            updated = apply_diff(original, operation.diff or "")
            path.write_text(updated, encoding="utf-8")
            return ApplyPatchResult(
                status="completed",
                output=f"Updated {operation.path}",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        try:
            path = self._validate_path(operation.path)
        except ValueError as exc:
            return ApplyPatchResult(status="failed", output=str(exc))
        if not path.exists():
            return ApplyPatchResult(
                status="failed",
                output=f"File not found: {operation.path}",
            )
        try:
            path.unlink()
            return ApplyPatchResult(
                status="completed",
                output=f"Deleted {operation.path}",
            )
        except Exception as exc:
            return ApplyPatchResult(status="failed", output=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# SkillAgent — the main public API
# ═══════════════════════════════════════════════════════════════════════════

class SkillAgent:
    """An OpenAI SDK Agent with Agent Skills support mirroring Gemini CLI.

    The implementation follows Gemini CLI's exact pattern:
      1. Skills metadata (name+description) injected as XML into system prompt
      2. Model calls ``activate_skill`` tool to load full instructions on demand
      3. Instructions returned as ``<activated_skill>`` XML in tool response
      4. Model follows instructions using standard tools (shell, read, write, etc.)

    Args:
        skills_dir:    Path(s) to skill directories. A single path or a list
                       of paths.  When multiple paths are given, later paths
                       have higher precedence (can override earlier skills
                       with the same name).
        model:         OpenAI model identifier (e.g. ``"gpt-4o"``, ``"gpt-5-mini"``).
        system_prompt: Optional base system prompt.  The skill catalog and
                       mandate are always appended automatically.
        project_root:  Working directory for shell and file tools.
                       Defaults to the current working directory.
        max_turns:     Maximum agent-loop iterations per run.
        model_kwargs:  Extra keyword arguments forwarded to the underlying
                       ``agents.Agent`` constructor.
    """

    def __init__(
        self,
        skills_dir: str | Path | list[str | Path],
        model: str = "gpt-5.4-mini",
        system_prompt: str | None = None,
        project_root: str | Path | None = None,
        max_turns: int = 10,
        model_kwargs: dict | None = None,
        include_skills: list[str] | None = None,
        read_reference_log_sink=None,
    ):
        # Route through Azure Foundry if AZURE_OPENAI_API_KEY is set.
        # Idempotent — only fires the first time.
        configure_azure_if_present()

        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.max_turns = max_turns
        self.model = model
        self.read_reference_log_sink = read_reference_log_sink

        # -- Phase 1: Discovery ------------------------------------------------
        self.skill_manager = SkillManager()
        dirs = skills_dir if isinstance(skills_dir, list) else [skills_dir]
        self.skill_manager.discover_skills(*dirs)

        # -- Optional filter: only keep named skills ---------------------------
        if include_skills is not None:
            allowed = {n.lower() for n in include_skills}
            self.skill_manager._skills = [
                s for s in self.skill_manager._skills
                if s.name.lower() in allowed
            ]

        # -- Build tools -------------------------------------------------------
        self.tools = self._build_tools()
        self.tool_names = [
            t.name if hasattr(t, "name") else type(t).__name__
            for t in self.tools
        ]

        # -- Build system prompt (mirrors Gemini CLI's promptProvider.ts) ------
        skills = self.skill_manager.get_skills()

        base = system_prompt or (
            "You are a helpful assistant. You have access to agent skills "
            "that extend your capabilities beyond your training data."
        )

        skills_section = render_agent_skills(skills)
        mandate = mandate_skill_guidance(len(skills) > 0)

        parts = [base]
        if skills_section:
            parts.append(skills_section)
        if mandate:
            parts.append(mandate)

        self.system_prompt = "\n\n".join(parts)

        # -- Build the underlying OpenAI Agent ---------------------------------
        kwargs = dict(model_kwargs or {})
        openai_client = kwargs.pop("openai_client", None)

        # If caller didn't pass an explicit client AND Azure (Foundry) is
        # configured via env vars, route through the right Azure client
        # for this model family.
        if openai_client is None:
            openai_client = get_client_for_model(self.model)

        if openai_client is not None:
            # OpenAI families (gpt-*, o-series) ride on Azure's
            # /openai/v1/responses path → use OpenAIResponsesModel.
            # Non-OpenAI families (grok, kimi, deepseek, gemini, …) speak
            # only chat-completions on Azure's inference endpoint → use
            # OpenAIChatCompletionsModel.
            if is_openai_family_model(self.model):
                from agents.models.openai_responses import OpenAIResponsesModel
                resolved_model = OpenAIResponsesModel(
                    model=self.model, openai_client=openai_client,
                )
            else:
                from agents.models.openai_chatcompletions import (
                    OpenAIChatCompletionsModel,
                )
                resolved_model = OpenAIChatCompletionsModel(
                    model=self.model, openai_client=openai_client,
                )
        else:
            resolved_model = self.model

        self.agent = Agent(
            name="SkillAgent",
            instructions=self.system_prompt,
            tools=self.tools,
            model=resolved_model,
            **kwargs,
        )

    # -- Tool factory -------------------------------------------------------

    def _build_tools(self) -> list:
        """Create all agent tools.

        - ``activate_skill``: built via ``FunctionTool`` for enum schema control
        - Standard tools (read, write, glob, grep): built via ``@function_tool`` closures
        - ``shell`` + ``apply_patch``: SDK built-in tools
        """
        mgr = self.skill_manager
        root = self.project_root

        # --- activate_skill (Gemini CLI pattern, enum-constrained) ---
        activate_tool = build_activate_skill_tool(mgr)

        # --- read_reference (V5 addition: on-demand L3 fetch with logging) ---
        read_reference_tool = build_read_reference_tool(
            mgr, log_sink=self.read_reference_log_sink,
        )

        # --- Standard tools (closures over `root`) ---

        # File extensions that require skill activation (binary/structured formats)
        _skill_only_extensions = {
            ".pdf", ".xlsx", ".xls", ".xlsm",
            ".pptx", ".ppt", ".docx", ".doc",
        }

        @function_tool
        def read_file(file_path: str) -> str:
            """Read the contents of a TEXT file (e.g., .py, .txt, .json, .csv, .md).

            This tool is for plain-text files ONLY. It CANNOT read binary or
            structured files like PDF, XLSX, PPTX, or DOCX. For those file
            types, you MUST activate the corresponding skill first and use
            the code examples it provides.

            Args:
                file_path: Absolute or project-relative path to the file.
            """
            p = Path(file_path)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            if not p.exists():
                return f"Error: file '{file_path}' not found."
            if p.suffix.lower() in _skill_only_extensions:
                return (
                    f"Error: '{p.suffix}' files cannot be read as plain text. "
                    f"Activate the appropriate skill (e.g., pdf, xlsx, pptx) "
                    f"and use its code examples to process this file."
                )
            try:
                return p.read_text(encoding="utf-8")
            except Exception as exc:
                return f"Error reading file: {exc}"

        @function_tool
        def write_file(file_path: str, content: str) -> str:
            """Write content to a file, creating parent directories if needed.

            Use this to create or update files.

            Args:
                file_path: Absolute or project-relative path to write to.
                content: The full text content to write.
            """
            p = Path(file_path)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
                return f"Successfully wrote {len(content)} characters to {p}"
            except Exception as exc:
                return f"Error writing file: {exc}"

        @function_tool
        def glob_files(pattern: str, path: str = "") -> str:
            """Find files matching a glob pattern within the project.

            Args:
                pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts").
                path: Subdirectory to search in, relative to project root.
            """
            search_root = (root / path).resolve() if path else root.resolve()
            try:
                matches = sorted(search_root.glob(pattern))
                results = [
                    str(m.relative_to(root.resolve()))
                    for m in matches
                    if m.is_file()
                ]
                if not results:
                    return "No files matched the pattern."
                return "\n".join(results)
            except Exception as exc:
                return f"Error in glob: {exc}"

        @function_tool
        async def grep_files(
            pattern: str, path: str = "", glob_filter: str = ""
        ) -> str:
            """Search file contents for a regex pattern.

            Args:
                pattern: Regular expression pattern to search for.
                path: Subdirectory to search in, relative to project root.
                glob_filter: Optional glob to filter files (e.g. "*.py").
            """
            search_path = (root / path).resolve() if path else root.resolve()
            cmd_str = f"grep -rn {pattern!r} {search_path}"
            if glob_filter:
                cmd_str = f"grep -rn --include {glob_filter!r} {pattern!r} {search_path}"
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(root),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=30,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return "Error: grep timed out after 30s."
                output = stdout.decode().strip()
                if not output:
                    return "No matches found."
                if len(output) > 10000:
                    output = output[:10000] + "\n...(truncated)"
                return output
            except Exception as exc:
                return f"Error in grep: {exc}"

        # --- Shell tool (wrapped as FunctionTool for visibility) ---
        # Uses subprocess.run() directly instead of SDK ShellTool types

        async def shell_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            args = json.loads(args_json)
            commands = args["commands"]
            timeout_ms = args.get("timeout_ms")
            timeout_sec = (timeout_ms / 1000.0) if timeout_ms else 120

            outputs = []
            for cmd in commands:
                try:
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(root),
                        env={**os.environ},
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        outputs.append(
                            f"Command: {cmd}\nError: timed out after {timeout_sec}s"
                        )
                        break
                    cmd_output = f"Command: {cmd}\n"
                    if stdout:
                        cmd_output += f"stdout:\n{stdout.decode()}\n"
                    if stderr:
                        cmd_output += f"stderr:\n{stderr.decode()}\n"
                    cmd_output += f"exit_code: {proc.returncode}"
                    outputs.append(cmd_output)
                except Exception as e:
                    outputs.append(f"Command: {cmd}\nError: {e}")
                    break

            return "\n\n".join(outputs) if outputs else "No output"

        shell_tool = FunctionTool(
            name="shell",
            description=(
                "Execute shell commands in the project directory. "
                "Use this to run terminal commands, scripts, tests, build tools, etc. "
                "Each command runs in the project root with the user's environment."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "commands": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of shell commands to execute sequentially.",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds (default: 120000).",
                    },
                },
                "required": ["commands"],
            },
            on_invoke_tool=shell_handler,
            strict_json_schema=False,
        )

        # --- Apply patch tool (wrapped as FunctionTool for visibility) ---
        editor = LocalFileEditor(root)

        async def apply_patch_handler(_ctx: ToolContext[Any], args_json: str) -> str:
            args = json.loads(args_json)
            operations = args["operations"]

            from agents.editor import ApplyPatchOperation

            results = []
            for op_dict in operations:
                operation = ApplyPatchOperation(
                    operation_type=op_dict.get("operation_type", "update"),
                    path=op_dict["path"],
                    diff=op_dict.get("diff"),
                )

                if operation.operation_type == "create":
                    result = editor.create_file(operation)
                elif operation.operation_type == "update":
                    result = editor.update_file(operation)
                elif operation.operation_type == "delete":
                    result = editor.delete_file(operation)
                else:
                    results.append(
                        f"Error: Unknown operation type '{operation.operation_type}' for {operation.path}"
                    )
                    continue

                results.append(f"[{result.status}] {result.output}")

            return "\n".join(results) if results else "No operations performed"

        apply_patch_tool = FunctionTool(
            name="apply_patch",
            description=(
                "Apply file operations using unified diff format (create/update/delete). "
                "Use this to create new files, update existing files with diffs, or delete files. "
                "For updates, provide a unified diff (V4A format)."
            ),
            params_json_schema={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation_type": {
                                    "type": "string",
                                    "enum": ["create", "update", "delete"],
                                    "description": "Type of operation to perform.",
                                },
                                "path": {
                                    "type": "string",
                                    "description": "File path (absolute or project-relative).",
                                },
                                "diff": {
                                    "type": "string",
                                    "description": "Unified diff string for create/update operations.",
                                },
                            },
                            "required": ["operation_type", "path"],
                        },
                        "description": "List of file operations to perform.",
                    },
                },
                "required": ["operations"],
            },
            on_invoke_tool=apply_patch_handler,
            strict_json_schema=False,
        )

        return [
            activate_tool,
            read_reference_tool,
            read_file,
            write_file,
            # glob_files and grep_files removed: all file paths are provided
            # via progressive disclosure in queries. These tools had 0% hit
            # rate across 13 training iterations and wasted tokens.
            shell_tool,
            #apply_patch_tool,
        ]

    # -- Rebuild (re-purpose for a different phase) -------------------------

    def rebuild(self, system_prompt: str, name: str = "SkillAgent") -> None:
        """Re-create the underlying Agent with new instructions.

        Shares the same SkillManager and tools — only the Agent object and
        system prompt are replaced.  Use this to switch the agent to a
        different workflow phase (e.g. from task execution to skill evolution).

        Args:
            system_prompt: New base system prompt for the agent.
            name: Display name for the new Agent instance.
        """
        skills = self.skill_manager.get_skills()
        skills_section = render_agent_skills(skills)
        mandate = mandate_skill_guidance(len(skills) > 0)

        parts = [system_prompt]
        if skills_section:
            parts.append(skills_section)
        if mandate:
            parts.append(mandate)

        self.system_prompt = "\n\n".join(parts)
        self.agent = Agent(
            name=name,
            instructions=self.system_prompt,
            tools=self.tools,
            model=self.model,
        )

    # -- Run methods --------------------------------------------------------

    async def run(self, query: str, **kwargs) -> RunResult:
        """Run the agent to completion (async).

        Args:
            query: The user's input query.
            **kwargs: Extra keyword args forwarded to ``Runner.run``.
        """
        return await Runner.run(
            self.agent,
            query,
            max_turns=kwargs.pop("max_turns", self.max_turns),
            **kwargs,
        )

    def run_streamed(self, query: str, **kwargs) -> RunResultStreaming:
        """Start a streamed agent run.

        Returns a ``RunResultStreaming`` whose ``.stream_events()`` async
        iterator yields events in real time.

        Args:
            query: The user's input query.
            **kwargs: Extra keyword args forwarded to ``Runner.run_streamed``.
        """
        return Runner.run_streamed(
            self.agent,
            query,
            max_turns=kwargs.pop("max_turns", self.max_turns),
            **kwargs,
        )
