"""Model dispatch for OpenAI, Azure (Foundry / Azure OpenAI), and Gemini clients.

The pipeline targets a single backend per model name. Set the relevant env
pair(s) below to point at whichever resource(s) host the models you want.

  AZURE_OPENAI_API_KEY    — Azure / Foundry key
  AZURE_OPENAI_ENDPOINT   — full base URL, e.g.
      https://<rsrc>.openai.azure.com/openai/v1/      (Azure OpenAI)
      https://<rsrc>.services.ai.azure.com/openai/v1/ (Foundry, OpenAI-compat)
      https://<rsrc>.services.ai.azure.com/models/    (Foundry, inference)

  GEMINI_API_KEY          — Google AI Studio key (used when model name
                            starts with "gemini-")
  GEMINI_BASE_URL         — optional override; defaults to Google's
                            OpenAI-compat endpoint
                            https://generativelanguage.googleapis.com/v1beta/openai/

`get_client_for_model(model)` dispatches per call:
  - "gemini-*" → Gemini client (requires GEMINI_API_KEY)
  - anything else → Azure/Foundry client if configured, else plain OpenAI

If exactly one of the Azure pair is set, fails loudly (silent fall-through
to OpenAI bit us before — confusing 401s far from the root cause). Same
half-config check is applied to the Gemini pair when GEMINI_BASE_URL is
set without GEMINI_API_KEY.

Wrapper-class selection still depends on model family (see
`get_model_class`): OpenAI families (gpt-*, o-series) speak the responses
API, everything else (kimi, grok, deepseek, gemini, …) uses
chat-completions.
"""

import os
import asyncio
import threading
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()


class _RoundRobinCompletions:
    """Round-robin chat-completions proxy with endpoint failover."""

    def __init__(self, clients: list):
        self._clients = clients
        self._lock = threading.Lock()
        self._idx = 0

    def _ordered_clients(self):
        with self._lock:
            start = self._idx
            self._idx = (self._idx + 1) % len(self._clients)
        return [
            self._clients[(start + offset) % len(self._clients)]
            for offset in range(len(self._clients))
        ]

    async def create(self, *args, **kwargs):
        attempts = max(
            1, int(os.getenv("SKILLGRAD_ENDPOINT_RETRIES", "1")) + 1
        )
        errors = []
        ordered = self._ordered_clients()
        for attempt in range(attempts):
            client = ordered[attempt % len(ordered)]
            try:
                return await client.chat.completions.create(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2 ** attempt, 5))
        raise errors[-1]

    def __getattr__(self, name):
        return getattr(self._clients[0].chat.completions, name)


class _RoundRobinChat:
    def __init__(self, clients: list):
        self._clients = clients
        self.completions = _RoundRobinCompletions(clients)

    def __getattr__(self, name):
        return getattr(self._clients[0].chat, name)


class _RoundRobinClient:
    """AsyncOpenAI-shaped facade over multiple vLLM replicas."""

    def __init__(self, clients: list):
        self._clients = clients
        self.chat = _RoundRobinChat(clients)

    def __getattr__(self, name):
        return getattr(self._clients[0], name)


def _parse_vllm_endpoints(endpoint: str) -> list[str]:
    raw = os.getenv("SKILLGRAD_VLLM_ENDPOINTS") or endpoint
    urls = []
    for part in raw.split(","):
        url = part.strip()
        if url:
            url = url if url.endswith("/") else url + "/"
            if url not in urls:
                urls.append(url)
    return urls

# Cumulative usage ledger populated by the chat-completions wrapper. Read
# by CostTracker.update() as a fallback when the SDK's own usage is empty
# (the chat-completions / Foundry-inference common case — the SDK only
# aggregates usage cleanly on the responses path).
_AZURE_USAGE_TOTALS: dict = {
    "input_tokens":     0,
    "cached_tokens":    0,
    "output_tokens":    0,
    "reasoning_tokens": 0,
    "requests":         0,
}

_AZURE_CLIENT: Optional[Any] = None
_AZURE_CHECKED: bool = False

_GEMINI_CLIENT: Optional[Any] = None
_GEMINI_CHECKED: bool = False

_GEMINI_DEFAULT_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


def get_azure_usage_totals() -> dict:
    """Return a copy of the current cumulative Azure usage totals.

    Returns zero-valued dict if Azure routing is not active. Safe to call
    from any code path; never raises.
    """
    return dict(_AZURE_USAGE_TOTALS)


def _record_chat_usage(u) -> None:
    """Append a chat-completions `CompletionUsage` to the global ledger.

    Tolerant to missing detail fields — Foundry/Gemini-compat providers
    don't always populate prompt_tokens_details / completion_tokens_details.
    """
    if u is None:
        return
    _AZURE_USAGE_TOTALS["input_tokens"]  += getattr(u, "prompt_tokens",     0) or 0
    _AZURE_USAGE_TOTALS["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
    _AZURE_USAGE_TOTALS["requests"]      += 1
    ptd = getattr(u, "prompt_tokens_details", None)
    if ptd is not None:
        _AZURE_USAGE_TOTALS["cached_tokens"] += (
            getattr(ptd, "cached_tokens", 0) or 0
        )
    ctd = getattr(u, "completion_tokens_details", None)
    if ctd is not None:
        _AZURE_USAGE_TOTALS["reasoning_tokens"] += (
            getattr(ctd, "reasoning_tokens", 0) or 0
        )


class _UsageRecordingStream:
    """Transparent proxy around an AsyncStream[ChatCompletionChunk] that
    captures usage from the final chunk into `_AZURE_USAGE_TOTALS`.

    The agents-SDK requests `stream_options={"include_usage": True}`, so
    chat-completions providers (Foundry kimi/grok, Gemini compat) emit a
    final chunk carrying `usage`. The non-streaming wrapper path can't see
    this because `chat.completions.create(stream=True)` returns the stream
    *before* any chunk has been produced — so we instrument iteration.

    Records exactly once per request. Delegates every non-iteration
    attribute (close, response, ...) to the inner stream via __getattr__.
    """

    __slots__ = ("_inner", "_recorded")

    def __init__(self, inner):
        self._inner = inner
        self._recorded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self._inner.__anext__()
        if not self._recorded:
            u = getattr(chunk, "usage", None)
            if u is not None:
                _record_chat_usage(u)
                self._recorded = True
        return chunk

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name):
        # Only called when normal lookup misses — i.e. for any attribute
        # not in __slots__ and not defined on this class. Delegates to
        # the wrapped AsyncStream so .close() / .response / etc. still work.
        return getattr(self._inner, name)


def _wrap_chat_completions_create(client) -> None:
    """Replace `client.chat.completions.create` with a wrapper that records
    usage into `_AZURE_USAGE_TOTALS` for both streaming and non-streaming
    calls.

    Non-streaming: `usage` is on the returned `ChatCompletion` directly —
    record immediately, return response unchanged.
    Streaming: SDK passes `stream_options={"include_usage": True}`, so the
    final chunk carries usage. We wrap the returned AsyncStream in
    `_UsageRecordingStream` which snoops chunks during iteration.

    Relies on AsyncOpenAI's `chat.completions` being cached_property — the
    wrapper persists across all subsequent uses of this client. Harmless
    when the active model rides the responses path instead (the wrapper
    simply never fires).
    """
    completions = client.chat.completions
    orig_create = completions.create

    async def wrapped_create(*args, **kwargs):
        resp = await orig_create(*args, **kwargs)
        # Non-streaming → ChatCompletion with .usage populated.
        u = getattr(resp, "usage", None)
        if u is not None:
            _record_chat_usage(u)
            return resp
        # Streaming → AsyncStream has no .usage; capture from the final
        # chunk during iteration. Detected by presence of __aiter__
        # (ChatCompletion does not have __aiter__).
        if hasattr(resp, "__aiter__"):
            return _UsageRecordingStream(resp)
        return resp

    completions.create = wrapped_create


def get_azure_client_if_configured():
    """Return a singleton AsyncOpenAI client pointed at the configured Azure
    resource, or None if neither env var is set.

    The endpoint is used as-is — paste the full base URL including the path
    suffix (`/openai/v1/` for OpenAI-compat, `/models/` for Foundry
    inference). Code does not interpret which suffix is active; it's just
    an OpenAI-compatible base URL.

    Raises:
      RuntimeError if exactly one of the two env vars is set.
    """
    global _AZURE_CLIENT, _AZURE_CHECKED
    if _AZURE_CHECKED:
        return _AZURE_CLIENT
    _AZURE_CHECKED = True

    api_key  = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if bool(api_key) ^ bool(endpoint):
        present = "AZURE_OPENAI_API_KEY"  if api_key else "AZURE_OPENAI_ENDPOINT"
        missing = "AZURE_OPENAI_ENDPOINT" if api_key else "AZURE_OPENAI_API_KEY"
        raise RuntimeError(
            "Partial Azure configuration detected.\n"
            f"  Set:     [{present!r}]\n"
            f"  Missing: [{missing!r}]\n"
            "Set BOTH AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to use "
            "Azure, or unset both to fall back to OpenAI."
        )

    if not (api_key and endpoint):
        return None  # plain OpenAI mode

    # Lazy imports so this module's import-time cost stays small when Azure
    # isn't in use.
    from openai import AsyncOpenAI
    from agents import set_tracing_export_api_key
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(True)

    # Belt-and-braces: even with tracing disabled, some SDK code paths
    # (e.g. internal default-client construction during validation) still
    # check OPENAI_API_KEY. Real calls go through the explicit Azure client
    # below; this dummy just satisfies the env-var check.
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-azure-routing")
    try:
        set_tracing_export_api_key("sk-dummy-azure-routing")
    except Exception:
        pass  # older SDK versions may not have this; not fatal

    endpoints = _parse_vllm_endpoints(endpoint)
    clients = []
    for base_url in endpoints:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            _wrap_chat_completions_create(client)
        except Exception as exc:
            print(f"[model_dispatch] WARNING: usage wrapper install failed: {exc}")
        clients.append(client)

    _AZURE_CLIENT = (
        _RoundRobinClient(clients) if len(clients) > 1 else clients[0]
    )
    print(
        f"[model_dispatch] routing enabled | endpoints={','.join(endpoints)}"
    )
    return _AZURE_CLIENT


def configure_azure_if_present() -> bool:
    """Backwards-compat alias. Returns True if Azure is in use."""
    return get_azure_client_if_configured() is not None


def get_gemini_client_if_configured():
    """Return a singleton AsyncOpenAI client pointed at Google's
    OpenAI-compatible Gemini endpoint, or None if GEMINI_API_KEY is unset.

    Google exposes a chat-completions-compatible API at
    https://generativelanguage.googleapis.com/v1beta/openai/ — so the same
    AsyncOpenAI client + OpenAIChatCompletionsModel wrapper used for
    Foundry kimi/grok works here with a different base URL and key.

    The chat-completions usage wrapper is installed so the global
    `_AZURE_USAGE_TOTALS` ledger (read by CostTracker) captures Gemini
    requests on the same code path. The name is kept for back-compat;
    in practice it's a provider-agnostic chat-completions ledger.

    Raises:
      RuntimeError if GEMINI_BASE_URL is set without GEMINI_API_KEY
      (parallels the Azure half-config check).
    """
    global _GEMINI_CLIENT, _GEMINI_CHECKED
    if _GEMINI_CHECKED:
        return _GEMINI_CLIENT
    _GEMINI_CHECKED = True

    api_key  = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL") or _GEMINI_DEFAULT_BASE_URL

    if not api_key:
        # If only the override URL is set without a key, treat as misconfig.
        if os.getenv("GEMINI_BASE_URL"):
            raise RuntimeError(
                "Partial Gemini configuration detected.\n"
                "  Set:     ['GEMINI_BASE_URL']\n"
                "  Missing: ['GEMINI_API_KEY']\n"
                "Set GEMINI_API_KEY to use Gemini, or unset GEMINI_BASE_URL."
            )
        return None  # Gemini not configured

    if not base_url.endswith("/"):
        base_url = base_url + "/"

    from openai import AsyncOpenAI
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(True)
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-azure-routing")

    _GEMINI_CLIENT = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        _wrap_chat_completions_create(_GEMINI_CLIENT)
    except Exception as exc:
        print(f"[model_dispatch] WARNING: usage wrapper install failed (gemini): {exc}")

    print(f"[model_dispatch] Gemini routing enabled | base_url={base_url}")
    return _GEMINI_CLIENT


def _is_gemini_model(model: str | None) -> bool:
    """True if the model name belongs to Google's Gemini family."""
    if not model:
        return False
    return model.lower().startswith("gemini-")


def _is_openai_family_model(model: str) -> bool:
    """True if the model name belongs to OpenAI's gpt-* / o-series families.

    Used to pick the agents-SDK wrapper:
      OpenAIResponsesModel        — OpenAI families (responses API).
      OpenAIChatCompletionsModel  — everything else (Foundry kimi/grok/...
                                    speak chat-completions only).
    """
    if not model:
        return True  # unknown → default to OpenAI path for back-compat
    m = model.lower()
    return (
        m.startswith("gpt-")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m == "chatgpt-4o-latest"
    )


def is_openai_family_model(model: str) -> bool:
    """Public wrapper for OpenAI-family routing checks."""
    return _is_openai_family_model(model)


def get_client_for_model(model: str | None):
    """Return the configured OpenAI-compatible client for this model.

      - "gemini-*" → Gemini client (raises if GEMINI_API_KEY is missing).
      - anything else → Azure/Foundry client if configured, else None
        (caller falls back to plain OpenAI).

    Wrapper-class selection (responses vs chat-completions) is independent
    of provider and handled by `get_model_class`.
    """
    if _is_gemini_model(model):
        client = get_gemini_client_if_configured()
        if client is None:
            raise RuntimeError(
                f"Model '{model}' requires GEMINI_API_KEY to be set in the "
                "environment. Add it to .env (and optionally set GEMINI_BASE_URL)."
            )
        return client
    return get_azure_client_if_configured()


def get_model_class(model: str):
    """Return the agents-SDK model wrapper class for a model family."""
    if is_openai_family_model(model):
        from agents.models.openai_responses import OpenAIResponsesModel
        return OpenAIResponsesModel
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    return OpenAIChatCompletionsModel
