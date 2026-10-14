"""Cost tracking for OpenAI / Azure runs.

When Azure (Foundry) routing is active, the agents-SDK's own
`result.context_wrapper.usage` returns all zeros (the SDK does not aggregate
usage across the OpenAIChatCompletionsModel + custom-client code path).
CostTracker falls back to a process-wide ledger populated by the chat-
completions wrapper installed in ``runners.model_dispatch``.

Per-call attribution under that fallback is approximate when concurrent Runner
calls share the same CostTracker, but the cumulative running total is exact —
each tracker takes a snapshot at construction time and computes deltas
against it.
"""


# Pricing: dollars per 1M tokens — (input, cached_input, output)
# Azure Foundry pay-as-you-go matches the model providers' official API prices,
# so the same table covers OpenAI direct and Foundry-routed deployments.
# For non-OpenAI Foundry models, cached_input is set equal to input unless a
# cache-hit price is explicitly published — Foundry's chat-completions usage
# payload generally does not report cached_tokens for these families anyway.
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    # OpenAI
    "gpt-5.4":       (2.50,  0.25,  15.00),
    "gpt-5.3-codex": (1.75,  0.175, 14.00),
    "gpt-5.2":       (1.75,  0.175, 14.00),
    "gpt-5.1":       (1.25,  0.125, 10.00),
    "gpt-5":         (1.25,  0.125, 10.00),
    "gpt-5.4-mini":  (0.75,  0.075,  4.50),
    "gpt-5.4-nano":  (0.20,  0.02,   1.25),
    "gpt-5-mini":    (0.25,  0.025,  2.00),
    "gpt-4.1":       (2.00,  0.50,   8.00),
    "gpt-4o":        (2.50,  1.25,  10.00),
    "gpt-4.1-mini":  (0.40,  0.10,   1.60),
    "gpt-4.1-nano":  (0.10,  0.025,  0.40),
    # Azure Foundry — DeepSeek (Microsoft launch-blog price)
    "DeepSeek-V3.2":            (0.58,  0.58,   1.68),
    # Azure Foundry — xAI Grok (Azure Global verified)
    "grok-4-1-fast-reasoning":  (0.20,  0.20,   0.50),
    "grok-4-20-reasoning":      (2.00,  2.00,   6.00),
    "grok-4-20-non-reasoning":  (2.00,  2.00,   6.00),
    # Azure Foundry — Moonshot Kimi (Azure Global verified; k2.6 cache from Moonshot direct)
    "Kimi-K2.5":         (0.60,  0.60,   3.00),
    "Kimi-K2.6":         (0.95,  0.16,   4.00),
    # Google Gemini — direct Google AI Studio API (text I/O).
    # Prices are best-effort estimates from public announcements; verify
    # against ai.google.dev/pricing before any large run, since Google has
    # adjusted these rates multiple times.
    "gemini-2.5-flash":       (0.30,  0.075,  2.50),
    "gemini-2.5-flash-lite":  (0.10,  0.025,  0.40),
    "gemini-2.5-pro":         (1.25,  0.31,  10.00),
    "gemini-3-pro":           (2.00,  0.25,  12.00),
    "gemini-3-flash":         (0.50,  0.13,   4.00),
    "gemini-3.1-flash-lite":  (0.15,  0.04,   0.60),
}

# Case-insensitive lookup index. Foundry / OpenAI deployment names vary in
# capitalisation across resources (e.g. "Kimi-K2.6" vs "kimi-k2.6"); this
# lets the price table match either form.
_MODEL_PRICING_LOWER: dict[str, tuple[float, float, float]] = {
    k.lower(): v for k, v in MODEL_PRICING.items()
}


class CostTracker:
    """Accumulate token usage across multiple Runner calls and compute cost."""

    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.requests = 0
        self.steps: list[tuple[str, dict]] = []

        # Snapshot of the global Azure usage ledger at construction. Used
        # to compute per-call deltas when SDK usage is empty.
        self._azure_snapshot = self._read_azure_totals()

    @staticmethod
    def _read_azure_totals() -> dict:
        """Best-effort read of the Azure usage ledger. Returns zeros if the
        Azure path is unused or model_dispatch is not importable for any reason.
        """
        try:
            from runners.model_dispatch import get_azure_usage_totals
            return get_azure_usage_totals()
        except Exception:
            return {
                "input_tokens": 0, "cached_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0, "requests": 0,
            }

    def update(self, result) -> dict:
        """Extract usage from a completed RunResultStreaming and accumulate.

        Returns a dict with the delta for this call.
        """
        usage = result.context_wrapper.usage
        inp = usage.input_tokens
        out = usage.output_tokens
        cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        reasoning = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        reqs = usage.requests

        # Fallback: SDK reported zeros but our Azure ledger advanced.
        # This is the normal path when Azure (Foundry) routing is active.
        if inp == 0 and out == 0 and reqs == 0:
            current = self._read_azure_totals()
            d_in        = current["input_tokens"]     - self._azure_snapshot["input_tokens"]
            d_cached    = current["cached_tokens"]    - self._azure_snapshot["cached_tokens"]
            d_out       = current["output_tokens"]    - self._azure_snapshot["output_tokens"]
            d_reasoning = current["reasoning_tokens"] - self._azure_snapshot["reasoning_tokens"]
            d_reqs      = current["requests"]         - self._azure_snapshot["requests"]
            if d_in > 0 or d_out > 0 or d_reqs > 0:
                inp, cached, out, reasoning, reqs = (
                    d_in, d_cached, d_out, d_reasoning, d_reqs,
                )
                self._azure_snapshot = current

        self.input_tokens += inp
        self.cached_tokens += cached
        self.output_tokens += out
        self.reasoning_tokens += reasoning
        self.requests += reqs

        delta = {
            "input_tokens": inp,
            "cached_tokens": cached,
            "output_tokens": out,
            "reasoning_tokens": reasoning,
            "requests": reqs,
            "cost": self._compute_cost(inp, cached, out),
        }
        return delta

    def _compute_cost(self, inp: int, cached: int, out: int) -> float:
        prices = (
            MODEL_PRICING.get(self.model)
            or _MODEL_PRICING_LOWER.get(self.model.lower())
        )
        if not prices:
            return 0.0
        inp_price, cached_price, out_price = prices
        # Cached tokens are a subset of input tokens — charge at cached rate
        non_cached = inp - cached
        return (
            non_cached * inp_price / 1_000_000
            + cached * cached_price / 1_000_000
            + out * out_price / 1_000_000
        )

    @property
    def total_cost(self) -> float:
        return self._compute_cost(
            self.input_tokens, self.cached_tokens, self.output_tokens,
        )

    def summary(self) -> str:
        cost = self.total_cost
        lines = [
            f"  Tokens: {self.input_tokens:,} in "
            f"({self.cached_tokens:,} cached) / "
            f"{self.output_tokens:,} out "
            f"({self.reasoning_tokens:,} reasoning)",
            f"  Requests: {self.requests}",
        ]
        if cost > 0:
            lines.append(f"  Cost: ${cost:.4f}")
        else:
            lines.append(f"  Cost: unknown model '{self.model}' — add to MODEL_PRICING")
        return "\n".join(lines)

    def print_step(self, label: str, delta: dict):
        """Print a one-line cost update after a single runner call."""
        self.steps.append((label, delta))
        cost_str = f"${delta['cost']:.4f}" if delta["cost"] > 0 else "?"
        print(
            f"  [{label}] {delta['input_tokens']:,} in / "
            f"{delta['output_tokens']:,} out | "
            f"cost: {cost_str}  (cumulative: ${self.total_cost:.4f})"
        )

    def cost_by_component(self) -> dict:
        """Group recorded steps by component and return per-component totals."""
        components: dict[str, dict] = {}
        for label, delta in self.steps:
            if label.startswith("EXEC"):
                component = "execute"
            elif label.startswith("DIAGNOSE"):
                component = "diagnose"
            elif label.startswith("REFLECT"):
                component = "reflect"
            elif label.startswith("AGGREGATE"):
                component = "aggregate"
            elif label.startswith("PATCH"):
                component = "patch"
            else:
                component = "other"

            if component not in components:
                components[component] = {
                    "input_tokens": 0, "output_tokens": 0,
                    "reasoning_tokens": 0, "cached_tokens": 0,
                    "requests": 0, "cost": 0.0,
                }
            for key in ("input_tokens", "output_tokens", "reasoning_tokens",
                        "cached_tokens", "requests", "cost"):
                components[component][key] += delta.get(key, 0)

        # Round cost values
        for comp in components.values():
            comp["cost"] = round(comp["cost"], 6)

        return components

    def to_dict(self) -> dict:
        """Return full cost summary as a serializable dict."""
        return {
            "total": {
                "input_tokens": self.input_tokens,
                "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "requests": self.requests,
                "cost": round(self.total_cost, 6),
            },
            "by_component": self.cost_by_component(),
        }
