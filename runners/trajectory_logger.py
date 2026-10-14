"""Trajectory logging, stream draining, and trace post-processing.

`TrajectoryLogger` writes per-event JSONL plus a colored terminal echo.
`stream_with_logging` drains a `RunResultStreaming` and routes each item to
the logger. `merge_trace_events`, `save_merged_trace`, `format_trajectory`,
`build_execution_trace`, and `summarize_iterations` post-process saved
trajectories into formats suitable for human inspection or for feeding to
downstream agents (diagnoser / momentum / patcher / evaluator queries).
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.items import (
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
)
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent


# ---------------------------------------------------------------------------
# Trajectory Logger
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """Logs every step of an agent run to both terminal and a JSONL file.

    Each event is recorded as a JSON object with:
      - step:      sequential step number
      - timestamp: ISO-8601 UTC timestamp
      - event:     semantic event name
      - agent:     which agent produced this event
      - content:   event-specific payload
    """

    def __init__(self, log_path: Path | str, append: bool = False):
        self.log_path = Path(log_path)
        self._step = 0
        self._events: list[dict[str, Any]] = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.log_path.exists():
            # Load existing events so flush() preserves them
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._events.append(entry)
                        self._step = max(self._step, entry.get("step", 0))
        elif not append:
            # Truncate stale file from a prior failed run so the per-event
            # appends in _record() start from a clean slate. Without this,
            # if a previous run was killed mid-execution and left events on
            # disk, the new run's appended events would mix with stale ones.
            try:
                self.log_path.write_text("", encoding="utf-8")
            except OSError:
                pass

    # -- Core recording -----------------------------------------------------

    def _record(self, event: str, agent: str, content: dict[str, Any]):
        self._step += 1
        entry = {
            "step": self._step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "agent": agent,
            "content": content,
        }
        self._events.append(entry)

        # Append to disk immediately so the trace is recoverable if the
        # process is killed mid-run (e.g., during an API stream stall). Old
        # behavior was to buffer in memory until flush() at the end of the
        # run; that lost the trace on any kill. Best-effort write — failures
        # here must not interrupt the agent loop.
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

        # Terminal output
        print(f"\n{'─' * 60}")
        print(f"  Step {self._step} | {event} | agent={agent}")
        print(f"{'─' * 60}")
        for key, val in content.items():
            display = str(val)
            if len(display) > 500:
                display = display[:500] + "…(truncated)"
            print(f"  {key}: {display}")

    # -- Typed helpers ------------------------------------------------------

    def log_tool_call(self, agent_name: str, tool_name: str, arguments: str):
        self._record("tool_call", agent_name, {
            "tool": tool_name,
            "arguments": arguments,
        })

    def log_tool_output(self, agent_name: str, tool_name: str, output: str):
        self._record("tool_output", agent_name, {
            "tool": tool_name,
            "output": output,
        })

    def log_message(self, agent_name: str, text: str):
        self._record("message", agent_name, {"text": text})

    def log_reasoning(self, agent_name: str, text: str):
        self._record("reasoning", agent_name, {"text": text})

    def log_meta(self, event: str, content: dict[str, Any]):
        """Log non-agent events (discovery, config, etc.)."""
        self._record(event, "system", content)

    # -- Persistence --------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def flush(self):
        """Write all accumulated events to the JSONL file."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            for entry in self._events:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n[Trajectory saved to {self.log_path}  ({len(self._events)} events)]")


# ---------------------------------------------------------------------------
# Message extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_message(raw_item) -> str:
    """Pull plain text out of a ResponseOutputMessage."""
    parts = []
    for block in getattr(raw_item, "content", []):
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else str(raw_item)


def _extract_reasoning_text(
    raw_item: Any,
    summary_parts: dict[str, list[str]],
) -> str:
    """Extract readable text from a ResponseReasoningItem.

    Tries, in order:
      1. ``raw_item.summary`` — list of Summary objects (each has ``.text``).
         Populated when ``ModelSettings.reasoning`` includes ``summary="concise"``
         (or ``"auto"`` / ``"detailed"``).
      2. ``raw_item.content`` — list of Content objects (each has ``.text``).
         Populated with the actual reasoning text (if returned by the API).
      3. Accumulated raw-stream summary parts keyed by ``raw_item.id``.
      4. Falls back to ``str(raw_item)`` so nothing is silently lost.
    """
    parts: list[str] = []

    # 1. Summaries
    for s in getattr(raw_item, "summary", None) or []:
        if hasattr(s, "text") and s.text:
            parts.append(s.text)

    # 2. Content (actual reasoning text)
    for c in getattr(raw_item, "content", None) or []:
        if hasattr(c, "text") and c.text:
            parts.append(c.text)

    # 3. Fallback: raw stream accumulation
    if not parts:
        item_id = getattr(raw_item, "id", "")
        if item_id and item_id in summary_parts:
            parts.extend(summary_parts.pop(item_id))

    # Return empty string when no text is available (caller decides whether to log)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stream-with-logging helper
# ---------------------------------------------------------------------------

async def stream_with_logging(result, logger: TrajectoryLogger) -> None:
    """Drain a ``RunResultStreaming`` object, logging every event.

    This encapsulates the event-loop pattern so callers don't need to
    duplicate the tool_call / tool_output / message / reasoning mapping.

    Args:
        result: A ``RunResultStreaming`` returned by ``Runner.run_streamed``.
        logger: The ``TrajectoryLogger`` that receives events.
    """
    # Map call_id → tool name so tool outputs get labelled correctly
    call_id_to_name: dict[str, str] = {}
    # Accumulate reasoning summary text from raw stream events (fallback)
    _reasoning_summary_parts: dict[str, list[str]] = {}  # item_id → [text, ...]

    import time as _time
    _last_event_time = _time.time()
    _event_count = 0

    async for event in result.stream_events():
        _now = _time.time()
        _gap = _now - _last_event_time
        _event_count += 1
        if _gap > 5.0:
            print(f"    [stream] event #{_event_count} after {_gap:.1f}s gap — type={type(event).__name__}")
        _last_event_time = _now

        # --- Raw stream events: capture reasoning summary text deltas ---
        if isinstance(event, RawResponsesStreamEvent):
            raw_data = event.data
            ev_type = getattr(raw_data, "type", None)
            if ev_type == "response.reasoning_summary_text.done":
                item_id = getattr(raw_data, "item_id", "")
                text = getattr(raw_data, "text", "")
                if text:
                    _reasoning_summary_parts.setdefault(item_id, []).append(text)
            continue

        if not isinstance(event, RunItemStreamEvent):
            continue

        item = event.item
        agent_name = item.agent.name if item.agent else "unknown"

        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            if isinstance(raw, dict):
                raw_type = raw.get("type", "unknown")
                call_id = raw.get("call_id")
                if raw_type == "shell_call":
                    tool_name = "shell"
                    action = raw.get("action", {})
                    cmds = action.get("commands", []) if isinstance(action, dict) else []
                    arguments = json.dumps(cmds)
                elif raw_type == "apply_patch_call":
                    tool_name = "apply_patch"
                    op = raw.get("operation", {})
                    arguments = json.dumps(op) if op else ""
                else:
                    tool_name = raw_type
                    arguments = str(raw)
            else:
                # Hosted tools (WebSearchTool, FileSearchTool) produce
                # Pydantic models with .type/.id instead of .name/.call_id.
                raw_type = getattr(raw, "type", None)

                if raw_type == "web_search_call":
                    tool_name = "web_search"
                    call_id = getattr(raw, "id", None)
                    action = getattr(raw, "action", None)
                    if action:
                        action_type = getattr(action, "type", "search")
                        if action_type == "search":
                            arguments = json.dumps({
                                "query": getattr(action, "query", None),
                                "queries": getattr(action, "queries", None),
                            })
                        elif action_type == "open_page":
                            arguments = json.dumps({
                                "url": getattr(action, "url", ""),
                            })
                        elif action_type == "find_in_page":
                            arguments = json.dumps({
                                "pattern": getattr(action, "pattern", ""),
                                "url": getattr(action, "url", ""),
                            })
                        else:
                            arguments = str(action)
                    else:
                        arguments = ""

                elif raw_type == "file_search_call":
                    tool_name = "file_search"
                    call_id = getattr(raw, "id", None)
                    arguments = json.dumps({
                        "queries": getattr(raw, "queries", []),
                    })

                else:
                    # Standard ResponseFunctionToolCall
                    tool_name = getattr(raw, "name", "unknown")
                    call_id = getattr(raw, "call_id", None)
                    arguments = getattr(raw, "arguments", "")

            if call_id:
                call_id_to_name[call_id] = tool_name
            logger.log_tool_call(agent_name, tool_name, arguments)

            # Hosted tools don't emit ToolCallOutputItem events — results
            # are consumed server-side.  Log available sources/results as
            # a synthetic tool_output so the trajectory is complete.
            if raw_type == "web_search_call":
                action = getattr(raw, "action", None)
                sources = getattr(action, "sources", None) if action else None
                if sources:
                    output = json.dumps([
                        {"url": getattr(s, "url", ""),
                         "title": getattr(s, "title", "")}
                        for s in sources
                    ], ensure_ascii=False)
                    logger.log_tool_output(agent_name, "web_search", output)
            elif raw_type == "file_search_call":
                results = getattr(raw, "results", None)
                if results:
                    output = json.dumps([
                        {"filename": getattr(r, "filename", ""),
                         "score": getattr(r, "score", None),
                         "text": (getattr(r, "text", "") or "")[:500]}
                        for r in results
                    ], ensure_ascii=False)
                    logger.log_tool_output(agent_name, "file_search", output)

        elif isinstance(item, ToolCallOutputItem):
            raw = item.raw_item
            if isinstance(raw, dict):
                call_id = raw.get("call_id")
            else:
                call_id = getattr(raw, "call_id", None)
            tool_name = call_id_to_name.pop(call_id, call_id or "unknown")
            logger.log_tool_output(agent_name, tool_name, str(item.output))

        elif isinstance(item, MessageOutputItem):
            text = extract_text_from_message(item.raw_item)
            # Chat-completions tool-only turns wrap the assistant message as a
            # MessageOutputItem with empty content text (the model produced
            # only `tool_calls`, no preamble). Skip those — they produce
            # noise like `{"text": ""}` in the trace. Responses-API tool-only
            # turns don't emit a MessageOutputItem at all, so this filter is
            # a no-op on that code path.
            if text and text.strip():
                logger.log_message(agent_name, text)

        elif isinstance(item, ReasoningItem):
            reasoning_text = _extract_reasoning_text(
                item.raw_item, _reasoning_summary_parts
            )
            # Only log reasoning when there's actual text (e.g. summary was
            # requested via ModelSettings reasoning.summary="concise").
            # The model's visible output is captured separately as "message".
            if reasoning_text:
                logger.log_reasoning(agent_name, reasoning_text)


# ---------------------------------------------------------------------------
# Trajectory formatting (for passing to evaluator / evolution agents)
# ---------------------------------------------------------------------------

def format_trajectory(logger: TrajectoryLogger) -> str:
    """Format logged events into a readable text block for agent consumption.

    Produces a structured summary of tool calls, outputs, messages, and
    reasoning — suitable for injecting into an evaluator's query as context.
    """
    lines: list[str] = []
    for entry in logger.events:
        event = entry["event"]
        agent = entry["agent"]
        content = entry["content"]
        step = entry["step"]

        if event == "tool_call":
            tool = content.get("tool", "?")
            args = content.get("arguments", "")
            # Truncate long arguments for readability
            if len(args) > 300:
                args = args[:300] + "…"
            lines.append(f"[Step {step}] {agent} called tool `{tool}`({args})")

        elif event == "tool_output":
            tool = content.get("tool", "?")
            output = content.get("output", "")
            if len(output) > 500:
                output = output[:500] + "…"
            lines.append(f"[Step {step}] Tool `{tool}` returned: {output}")

        elif event == "message":
            text = content.get("text", "")
            if len(text) > 500:
                text = text[:500] + "…"
            lines.append(f"[Step {step}] {agent} said: {text}")

        elif event == "reasoning":
            text = content.get("text", "")
            if len(text) > 300:
                text = text[:300] + "…"
            lines.append(f"[Step {step}] {agent} reasoning: {text}")

        elif event == "phase":
            lines.append(f"\n--- Phase: {content.get('phase', '?')} ---")

        elif event in ("query", "run_complete", "skills_discovered", "agent_created"):
            lines.append(f"[Step {step}] [{event}] {json.dumps(content, default=str)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Merged Trace Format (V6)
# ---------------------------------------------------------------------------

def merge_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge raw trajectory events into one-entry-per-action format.

    Transforms the raw JSONL (separate tool_call / tool_output lines, phase
    markers, etc.) into a compact representation:
      - One entry per agent action (tool_call + tool_output merged)
      - Phase markers and metadata events skipped
      - Shell commands presented as raw text, not escaped JSON
      - write_file output dropped (redundant with input)
      - Full content preserved — no truncation
    """
    merged: list[dict[str, Any]] = []
    step = 0
    i = 0

    while i < len(events):
        ev = events[i]
        etype = ev.get("event", "")
        content = ev.get("content", {})
        ts = ev.get("timestamp", "")

        # Skip non-informative events
        if etype in ("phase", "query", "run_complete", "skills_discovered",
                      "agent_created", "iteration_start", "feedback"):
            i += 1
            continue

        if etype == "tool_call":
            tool = content.get("tool", "unknown")
            args_raw = content.get("arguments", "")

            # Parse arguments from JSON string
            try:
                parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                parsed = args_raw

            # Look ahead for matching tool_output
            output = None
            next_i = i + 1
            if next_i < len(events) and events[next_i].get("event") == "tool_output":
                output = events[next_i].get("content", {}).get("output", "")
                i = next_i + 1
            else:
                i += 1

            step += 1
            entry: dict[str, Any] = {"step": step, "tool": tool, "ts": ts}

            # Format input by tool type
            if tool == "shell":
                if isinstance(parsed, dict):
                    cmds = parsed.get("commands", [])
                elif isinstance(parsed, list):
                    cmds = parsed
                else:
                    cmds = [str(parsed)]
                entry["input"] = "\n".join(str(c) for c in cmds)
                if output is not None:
                    entry["output"] = output
            elif tool == "write_file":
                entry["input"] = parsed  # {file_path, content}
                # output is just "Successfully wrote N chars" — skip
            elif tool == "activate_skill":
                entry["input"] = parsed
                if output is not None:
                    entry["output"] = output
            elif tool == "read_reference":
                entry["input"] = parsed
                if output is not None:
                    entry["output"] = output
            else:
                entry["input"] = parsed
                if output is not None:
                    entry["output"] = output

            merged.append(entry)

        elif etype == "tool_output":
            # Orphaned tool_output — skip
            i += 1

        elif etype == "message":
            text = content.get("text", "")
            if text.strip():
                step += 1
                merged.append({
                    "step": step, "tool": None,
                    "response": text, "ts": ts,
                })
            i += 1

        elif etype == "reasoning":
            text = content.get("text", "")
            if text.strip():
                step += 1
                merged.append({
                    "step": step, "tool": None,
                    "reasoning": text, "ts": ts,
                })
            i += 1

        else:
            i += 1

    return merged


def save_merged_trace(
    raw_jsonl_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a raw trajectory JSONL to merged one-action-per-line format.

    Args:
        raw_jsonl_path: Path to the raw JSONL (existing TrajectoryLogger output).
        output_path: Where to save the merged JSONL. Defaults to
                     ``<same_dir>/trace.jsonl``.

    Returns:
        Path to the saved merged trace file.
    """
    raw_jsonl_path = Path(raw_jsonl_path)
    if output_path is None:
        output_path = raw_jsonl_path.parent / "trace.jsonl"
    else:
        output_path = Path(output_path)

    events: list[dict[str, Any]] = []
    with open(raw_jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    merged = merge_trace_events(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in merged:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return output_path


# ---------------------------------------------------------------------------
# Workflow phase / iteration helpers
# ---------------------------------------------------------------------------

def set_phase(phase: str, logger: TrajectoryLogger) -> None:
    """Log a workflow phase marker and print a banner to the terminal."""
    logger.log_meta("phase", {"phase": phase})
    print(f"\n{'=' * 60}")
    print(f"  PHASE: {phase}")
    print(f"{'=' * 60}")


def set_iteration(iteration: int, logger: TrajectoryLogger) -> None:
    """Log an iteration boundary marker."""
    logger.log_meta("iteration_start", {"iteration": iteration})
    print(f"\n{'─' * 60}")
    print(f"  ITERATION {iteration}")
    print(f"{'─' * 60}")


def summarize_iterations(logger: TrajectoryLogger) -> str:
    """Build a cumulative summary of all past iterations for reflection.

    For each iteration, extracts:
    - Which skills were activated
    - Key tool calls (abbreviated)
    - User feedback

    Returns a compact text summary suitable for injecting into a reflection
    prompt without blowing up context size.
    """
    iterations: dict[int, dict] = {}  # iter_num -> info
    current_iter = -1

    for entry in logger.events:
        event = entry["event"]
        content = entry["content"]

        if event == "iteration_start":
            current_iter = content["iteration"]
            iterations[current_iter] = {
                "skills": [],
                "tools": [],
                "feedback": "",
                "output": "",
            }

        elif current_iter >= 0 and current_iter in iterations:
            if event == "tool_call":
                tool = content.get("tool", "?")
                if tool == "activate_skill":
                    try:
                        args = json.loads(content.get("arguments", "{}"))
                        iterations[current_iter]["skills"].append(
                            args.get("name", "?")
                        )
                    except Exception:
                        pass
                elif tool == "shell":
                    args_str = content.get("arguments", "")
                    # Brief description: first 80 chars
                    iterations[current_iter]["tools"].append(args_str[:80])

            elif event == "message":
                # Capture last message as output summary
                text = content.get("text", "")
                if text:
                    iterations[current_iter]["output"] = text[:200]

        # Feedback events are logged outside iteration blocks
        if event == "feedback":
            iter_num = content.get("iteration", current_iter)
            if iter_num in iterations:
                iterations[iter_num]["feedback"] = content.get("feedback", "")

    # Format summary
    lines: list[str] = []
    for i in sorted(iterations.keys()):
        info = iterations[i]
        lines.append(f"### Iteration {i}")
        if info["skills"]:
            lines.append(f"- Skills: {', '.join(info['skills'])}")
        if info["tools"]:
            lines.append(f"- Approach: {len(info['tools'])} code executions")
            for t in info["tools"][:3]:  # Show at most 3
                lines.append(f"  - `{t}...`")
        if info["output"]:
            lines.append(f"- Output excerpt: {info['output'][:150]}...")
        if info["feedback"]:
            lines.append(f'- User feedback: "{info["feedback"]}"')
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution Trace (ReAct format)
# ---------------------------------------------------------------------------

def build_execution_trace(
    jsonl_path: str | Path,
    max_output_chars: int = 500,
) -> str:
    """Build a sequential ReAct execution trace from trajectory JSONL.

    Produces a human-readable markdown summary preserving causal order:
      THOUGHT (reasoning or pre-action message) ->
      ACTION (tool call) ->
      OBSERVATION (tool output, truncated)

    Returns markdown string. No LLM needed — pure parsing.
    """
    import re as _re

    events = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        return "(empty trajectory)\n"

    parts: list[str] = []
    step = 0
    pending_thought: str | None = None
    n_errors = 0

    for ev in events:
        etype = ev.get("event", "")
        content = ev.get("content", {})

        # Buffer thoughts (reasoning or pre-tool-call messages)
        if etype == "reasoning":
            text = content.get("text", "").strip()
            if text:
                pending_thought = text[:500]

        elif etype == "message":
            text = content.get("text", "").strip()
            if text:
                pending_thought = text[:500]

        elif etype == "tool_call":
            step += 1
            tool = content.get("tool", "?")
            args_raw = content.get("arguments", "")

            entry = [f"### Step {step}"]

            # Flush buffered thought
            if pending_thought:
                entry.append(f"THOUGHT: {pending_thought}")
                pending_thought = None

            # Format action by tool type
            if tool == "shell":
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    cmds = args.get("commands", args) if isinstance(args, dict) else args
                    if isinstance(cmds, str):
                        cmds = [cmds]
                    cmd_text = "\n".join(str(c) for c in cmds)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    cmd_text = str(args_raw)[:500]
                entry.append("ACTION: shell")
                entry.append(f"```\n{cmd_text}\n```")
            else:
                args_brief = str(args_raw)[:200]
                entry.append(f"ACTION: {tool}({args_brief})")

            parts.append("\n".join(entry))

        elif etype == "tool_output":
            output = content.get("output", "")
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + f"\n...(truncated, {len(output)} total chars)"

            has_error = bool(_re.search(
                r"Traceback|Error:|Exception:|exit code [1-9]",
                output,
            ))
            if has_error:
                n_errors += 1

            marker = " **[ERROR]**" if has_error else ""
            parts.append(f"OBSERVATION:{marker}\n```\n{output}\n```\n")

            pending_thought = None

    header = f"## Execution Trace ({step} actions, {n_errors} errors)\n\n"
    return header + "\n".join(parts)
