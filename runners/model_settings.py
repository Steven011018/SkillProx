"""Model compatibility helpers — reasoning-effort detection and kwargs builders."""

import os

from agents import ModelSettings


_REASONING_MODELS = {"gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano"}


def supports_reasoning(model: str) -> bool:
    """Return True if the model supports the reasoning.effort parameter."""
    return model in _REASONING_MODELS or model.startswith("gpt-5")


def get_model_kwargs(model: str, openai_client=None) -> dict:
    """Return model_kwargs dict with optional custom client.

    Reasoning effort is intentionally NOT set here. The API default applies
    (for gpt-5.4 that means no reasoning, i.e. "none"). To re-enable, add
    `kwargs["model_settings"] = ModelSettings(reasoning={"effort": "..."})`
    where effort ∈ {"minimal", "low", "medium", "high"}.

    Always includes a `model_settings` entry that forces
    `include_usage=True`, so chat-completions providers (Foundry kimi/grok,
    Gemini compat) emit a usage chunk on the streaming tail and the global
    ledger in `runners.model_dispatch` can capture it. No-op on the
    responses-API path used by OpenAI gpt-*/o-series.

    Args:
        model: Model identifier string (kept for signature stability).
        openai_client: Optional AsyncOpenAI client for local/custom API endpoints.
                       When provided, SkillAgent uses OpenAIChatCompletionsModel.
    """
    kwargs = {"model_settings": get_model_settings(model)}
    if openai_client is not None:
        kwargs["openai_client"] = openai_client
    return kwargs


def get_model_settings(model: str) -> ModelSettings:
    """Return ModelSettings with `include_usage=True`.

    The agents-SDK only defaults `stream_options={"include_usage": True}`
    when the client's base URL is api.openai.com (see
    `ChatCmplHelpers.get_stream_options_param`). For Azure/Foundry/Gemini
    clients that default is None, which means providers emit no usage on
    streamed calls and our chat-completions usage ledger stays at zero.
    Forcing `include_usage=True` here makes the SDK send the option
    explicitly, restoring correct cost tracking. Harmless on the
    responses-API path (gpt-5.4 / o-series), which ignores the field.
    """
    kwargs = {"include_usage": True}
    temperature = os.getenv("SKILLGRAD_TEMPERATURE")
    if temperature:
        kwargs["temperature"] = float(temperature)
    elif "qwen" in (model or "").lower():
        kwargs["temperature"] = 0.7

    if "qwen" in (model or "").lower() and os.getenv(
        "SKILLGRAD_QWEN_THINKING", ""
    ).strip().lower() not in {"1", "true", "yes"}:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    return ModelSettings(**kwargs)
