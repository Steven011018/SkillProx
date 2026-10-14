"""Shared helpers for the per-stage pipeline agents.

`_resolve_model` wraps a bare model name with the configured Azure
(Foundry) client when one is available, returning either a string (for the
plain-OpenAI path) or an OpenAI{Responses,ChatCompletions}Model wrapper.

`_build_file_tools` constructs the read_file / write_file `function_tool`
closures used by the diagnoser, momentum, and patcher agents (the
executor's tool set lives in `pipeline.executor.SkillAgent` and is
richer).
"""

from pathlib import Path

from agents import function_tool

from runners.model_dispatch import get_client_for_model, get_model_class


def _resolve_model(model: str):
    """Wrap with the configured OpenAI-compatible client when available."""
    client = get_client_for_model(model)
    if client is None:
        return model
    model_cls = get_model_class(model)
    return model_cls(model=model, openai_client=client)


def _build_file_tools(project_root: Path):
    """Build read_file and write_file tools for plain (non-skill) agents."""
    root = project_root

    @function_tool
    def read_file(file_path: str) -> str:
        """Read the contents of a text file.

        Args:
            file_path: Absolute or project-relative path to the file.
        """
        p = Path(file_path)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if not p.exists():
            return f"Error: file '{file_path}' not found."
        try:
            return p.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading file: {exc}"

    @function_tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file, creating parent directories if needed.

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

    return read_file, write_file
