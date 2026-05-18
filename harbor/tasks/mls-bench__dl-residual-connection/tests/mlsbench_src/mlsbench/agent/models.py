"""Model client abstraction for multiple LLM providers.

Thinking/reasoning support:
  Claude:           Extended thinking via thinking={"type":"enabled","budget_tokens":N} +
                    betas=["interleaved-thinking-2025-05-14"]. Thinking blocks MUST be
                    preserved verbatim in subsequent assistant messages.
  DeepSeek-Reasoner: Uses deepseek-reasoner model (or deepseek-chat + thinking enabled).
                    Returns reasoning_content in response. Tool calls ARE supported.
                    reasoning_content must be sent back within a turn (multi-step tool
                    calling) but cleared between user turns. Does NOT support tool_choice.
  Qwen:             Enable via extra_body={"enable_thinking": True}. Returns reasoning_content.
                    reasoning_content must NOT be sent back in subsequent messages.
  Kimi:             OpenAI-compatible API at api.moonshot.ai. Kimi K2 thinking models
                    (kimi-k2-*) return reasoning_content. reasoning_content MUST be
                    preserved and sent back (like DeepSeek). max_tokens >= 16000.
                    Does NOT support tool_choice for thinking models.
  OpenAI o-models:  Built-in reasoning; pass reasoning_effort via extra_body if desired.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any


# Env-var fallbacks per provider key. When a provider's `api_key` is empty in
# the YAML config (the new default for committed configs), fall back to the
# corresponding host env var. Keeps secrets out of files we ship.
_PROVIDER_ENV_FALLBACKS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY_NEW",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "vertex_ai": "VERTEX_AI_API_KEY",
    "litellm": "LITELLM_API_KEY",
    "kimi": "KIMI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "glm": "GLM_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def _resolve_api_key(cfg: dict, provider: str | None = None) -> str:
    """Return the api_key for a provider, falling back to env vars."""
    raw = (cfg or {}).get("api_key") or ""
    if raw:
        # Allow ${VAR} interpolation so users can put `${OPENAI_API_KEY}` in
        # tracked configs without leaking the literal value.
        if raw.startswith("${") and raw.endswith("}"):
            return os.environ.get(raw[2:-1], "")
        return raw
    if provider:
        env_name = _PROVIDER_ENV_FALLBACKS.get(provider.lower())
        if env_name:
            return os.environ.get(env_name, "")
    return ""


def _extract_usage(response: Any, model: str) -> dict:
    """Normalize token usage across Anthropic, OpenAI Chat Completions,
    OpenAI Responses, OpenRouter, DeepSeek, Qwen, and Gemini-via-OpenAI
    responses into a single dict.

    Fields that aren't reported come back as 0 (not None) so the caller can
    sum across calls without dropping into Optional math.
    """
    record: dict = {
        "model": model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
    }
    usage = getattr(response, "usage", None)
    if usage is None:
        return record

    # Anthropic-native (messages.create): input_tokens/output_tokens +
    # cache_read_input_tokens/cache_creation_input_tokens.
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None or output_tokens is not None:
        prompt = int(input_tokens or 0)
        completion = int(output_tokens or 0)
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        record.update(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=cached,
            cache_creation_tokens=cache_created,
        )
        return record

    # OpenAI Responses API reports input_tokens/output_tokens under different
    # attributes on newer SDKs; fall through if that flavor ever lands here.

    # OpenAI Chat Completions (and compatible): prompt_tokens/completion_tokens
    # + prompt_tokens_details.cached_tokens (OpenAI caching).
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or prompt + completion)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    # Anthropic-via-OpenRouter: usage has cache_read_input_tokens on the root.
    if not cached:
        cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    record.update(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        cache_creation_tokens=cache_created,
    )
    return record


class ModelClient(ABC):
    """Abstract LLM client that returns a tool_use action dict."""

    # Subclasses MUST assign this after each call() (None or the usage dict
    # from _extract_usage). Agents read it to capture tokens even when call()
    # returns None (no tool_use) so we don't lose accounting on retries/nudges.
    _last_usage: dict | None = None

    @abstractmethod
    def call(self, messages: list, tools: list, system: str = "") -> dict | None:
        """Call the model and return an action dict:
            {
              "name": str,                  # tool name
              "input": dict,               # tool input
              "assistant_message": dict,   # Anthropic-format assistant message to append to history
              "thinking": str | None,      # reasoning/thinking text for logging
              "usage": dict,                # per-call token usage (see _extract_usage)
            }
        Returns None if no tool call was made. ``self._last_usage`` is still
        populated in that case so the caller can log tokens.
        """


# ---------------------------------------------------------------------------
# Anthropic (Claude) client
# ---------------------------------------------------------------------------

class AnthropicClient(ModelClient):
    """Client for Claude models (or Anthropic-compatible APIs like Kimi coding)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 thinking_config: dict | None = None):
        import anthropic
        self.model = model
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        # thinking_config: {"enabled": bool, "budget_tokens": int}
        self.thinking_config = thinking_config or {}

    def call(self, messages: list, tools: list, system: str = "") -> dict | None:
        thinking_enabled = self.thinking_config.get("enabled", False)
        budget_tokens = self.thinking_config.get("budget_tokens", 8000)

        # When thinking is enabled, max_tokens must be > budget_tokens
        max_tokens = budget_tokens + 16384 if thinking_enabled else 16384

        # Enable prompt caching: wrap system prompt and tools with cache_control
        # so that the stable prefix (system + tools + initial user message) is cached.
        # Anthropic caches everything up to a cache_control breakpoint.
        cached_tools = list(tools)
        if cached_tools:
            # Add cache_control to the last tool — caches system + all tools
            last_tool = dict(cached_tools[-1])
            last_tool["cache_control"] = {"type": "ephemeral"}
            cached_tools[-1] = last_tool

        # Add cache_control to the first user message (the large initial prompt).
        # This ensures the stable prefix is cached across all subsequent turns.
        cached_messages = list(messages)
        if cached_messages and cached_messages[0].get("role") == "user":
            first = dict(cached_messages[0])
            content = first.get("content", "")
            if isinstance(content, str):
                first["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            cached_messages[0] = first

        # Sliding cache breakpoint: place cache_control on the most recent
        # user message (usually the latest tool_result batch). Without this,
        # cached_tokens plateaus at the initial prefix across turns 2+ even
        # though tool_use/tool_result content is stable. Anthropic reads the
        # RIGHTMOST breakpoint as "cache up to here" and allows up to 4
        # breakpoints total, so combining static (first user) + rolling
        # (latest user) is within budget.
        if len(cached_messages) > 1:
            for i in range(len(cached_messages) - 1, 0, -1):
                m = cached_messages[i]
                if m.get("role") != "user":
                    continue
                m = dict(m)
                content = m.get("content", "")
                if isinstance(content, str):
                    m["content"] = [
                        {"type": "text", "text": content,
                         "cache_control": {"type": "ephemeral"}}
                    ]
                elif isinstance(content, list) and content:
                    new_content = [dict(b) if isinstance(b, dict) else b for b in content]
                    for j in range(len(new_content) - 1, -1, -1):
                        blk = new_content[j]
                        btype = blk.get("type") if isinstance(blk, dict) else None
                        if btype in ("text", "tool_result"):
                            blk["cache_control"] = {"type": "ephemeral"}
                            break
                    m["content"] = new_content
                cached_messages[i] = m
                break

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": cached_messages,
            "tools": cached_tools,
        }
        if system:
            # Wrap system as content block with cache_control
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if thinking_enabled:
            # SDK 0.43 doesn't accept `thinking` as a top-level kwarg on
            # messages.stream(); pass it via extra_body so the param still
            # reaches the API.
            # Newer Claude 4.7+ models (e.g. claude-opus-4-7 on Vertex) require
            # the adaptive thinking format with output_config.effort instead of
            # enabled+budget_tokens. Detect by version suffix.
            model_lower = self.model.lower()
            uses_adaptive = ("opus-4-7" in model_lower or "opus-4.7" in model_lower
                             or "sonnet-4-7" in model_lower or "sonnet-4.7" in model_lower)
            if uses_adaptive:
                effort = (self.thinking_config.get("reasoning_effort") or "high").lower()
                kwargs.setdefault("extra_body", {})["thinking"] = {"type": "adaptive"}
                kwargs["extra_body"]["output_config"] = {"effort": effort}
            else:
                kwargs.setdefault("extra_body", {})["thinking"] = {
                    "type": "enabled", "budget_tokens": budget_tokens,
                }
            # temperature must be 1 when extended thinking is enabled
            kwargs["temperature"] = 1
            # Interleaved-thinking beta header for thinking + tool use.
            # Claude 4+ supports this natively, but the header is still
            # accepted (and required for older Claude 3.5 models).
            kwargs["extra_headers"] = {
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            }

        # Use streaming: with thinking enabled, max_tokens (= budget_tokens + 16384)
        # can exceed the SDK's 10-minute non-streaming threshold. Streaming
        # preserves prompt caching (cache_control is request-side; cache_creation/
        # cache_read still come back in the final usage).
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        self._last_usage = _extract_usage(response, self.model)

        # Parse response blocks.
        # We only dispatch ONE tool call per turn (matching base.py which creates
        # a single tool_result).  Keep the first tool_use block; additional ones
        # are dropped from content_blocks to avoid Anthropic requiring a
        # tool_result for every tool_use in the conversation.
        tool_use_block = None
        thinking_parts: list[str] = []
        content_blocks: list[dict] = []

        for block in response.content:
            btype = block.type
            if btype == "thinking":
                thinking_parts.append(block.thinking)
                # The Anthropic API REQUIRES `signature` on thinking blocks in
                # subsequent turns. Some proxies (LiteLLM/Vertex) strip the
                # signature on response — sending such a block back triggers
                # 400 invalid_request_error. Skip storing it in that case so
                # the conversation history stays valid.
                sig = getattr(block, "signature", None)
                if sig:
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": sig,
                    })
            elif btype == "redacted_thinking":
                redacted_dict: dict = {"type": "redacted_thinking", "data": block.data}
                content_blocks.append(redacted_dict)
            elif btype == "tool_use":
                if tool_use_block is None:
                    # Keep only the first tool_use
                    tool_use_block = {"name": block.name, "input": block.input, "id": block.id}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                # else: silently drop extra tool calls
            elif btype == "text":
                content_blocks.append({"type": "text", "text": block.text})

        if tool_use_block is None:
            return None

        thinking_text = "\n\n".join(thinking_parts) if thinking_parts else ""

        return {
            "name": tool_use_block["name"],
            "input": tool_use_block["input"],
            "assistant_message": {"role": "assistant", "content": content_blocks},
            "thinking": thinking_text or None,
            "usage": _extract_usage(response, self.model),
        }


# ---------------------------------------------------------------------------
# OpenAI-compatible client (OpenAI, DeepSeek, Qwen, Vertex via LiteLLM, etc.)
# ---------------------------------------------------------------------------

class OpenAIClient(ModelClient):
    """Client for OpenAI-compatible APIs (OpenAI, DeepSeek, Qwen, LiteLLM proxy, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        thinking_config: dict | None = None,
        provider_config: dict | None = None,
    ):
        from openai import OpenAI
        self.model = model
        self.thinking_config = thinking_config or {}
        self.base_url = base_url or ""
        # Optional per-provider tuning (e.g. openrouter pin behavior) passed
        # down from build_client. Shape: {"pin": bool, "order_overrides": {...}}
        self.provider_config = provider_config or {}
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        # Bump request timeout for thinking/agentic calls. GLM-5.x batches its
        # tool-call output instead of streaming, which can stall non-stream
        # responses past the SDK's 600s default for long contexts. 1200s is
        # generous enough to absorb that without inflating cost on snappy
        # models (which return well under the window anyway).
        # Ref: https://aarongxa.com/posts/fixing-glm-5-1-timeouts-in-hermes-agent/
        kwargs["timeout"] = 1200.0
        kwargs["max_retries"] = 1
        self.client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    # OpenRouter provider-routing helpers
    # ------------------------------------------------------------------
    # When going through OpenRouter, the same model id (e.g. anthropic/
    # claude-opus-4.6) can be served by multiple upstream providers (Anthropic
    # direct, Bedrock, Vertex). Prompt caches are per-upstream, so drifting
    # between them makes `cached_tokens` bounce to zero. Pinning to one
    # upstream keeps the cache warm. See:
    # https://openrouter.ai/docs/guides/best-practices/prompt-caching
    _OPENROUTER_PROVIDER_ORDER_BY_PREFIX = {
        "anthropic/": ["anthropic"],
        "openai/": ["openai"],
        "google/": ["google-ai-studio"],
        "deepseek/": ["deepseek"],
        "qwen/": ["alibaba"],
        "moonshot/": ["moonshot"],
        "meta-llama/": ["groq", "togetherai"],
    }

    def _openrouter_pin_enabled(self) -> bool:
        val = self.provider_config.get("pin", True)
        return bool(val)

    def _openrouter_provider_order(self, model: str) -> list[str] | None:
        overrides = self.provider_config.get("order_overrides") or {}
        for prefix, order in overrides.items():
            if model.startswith(prefix):
                return list(order)
        for prefix, order in self._OPENROUTER_PROVIDER_ORDER_BY_PREFIX.items():
            if model.startswith(prefix):
                return list(order)
        return None

    def call(self, messages: list, tools: list, system: str = "") -> dict | None:
        thinking_enabled = self.thinking_config.get("enabled", False)
        budget_tokens = self.thinking_config.get("budget_tokens", 8000)

        # OpenRouter speaks a slightly different protocol (reasoning_details,
        # provider pinning) — detect via base_url, since model-name prefixes
        # like "openai/", "anthropic/", "vertex_ai/" are also used for plain
        # LiteLLM routing and don't imply OpenRouter.
        is_openrouter = "openrouter" in self.base_url
        # `bare_model` is used for downstream provider detection
        # (startswith("gpt-"), startswith("claude"), …). Any prefix-routed
        # model name (`<provider>/<model>` or `<router>/<vendor>/<model>`)
        # needs the prefix stripped, regardless of which proxy it goes
        # through. Take the trailing segment.
        bare_model = self.model.rsplit("/", 1)[-1] if "/" in self.model else self.model

        is_anthropic_model = bare_model.startswith("claude")
        is_qwen = bare_model.lower().startswith("qwen")
        # Anthropic prompt caching requires explicit cache_control markers on
        # content blocks (system / first user / sliding latest). Add them
        # whenever we talk to a Claude model through any OpenAI-compatible
        # path (OpenRouter or a LiteLLM proxy fronting Anthropic/Bedrock).
        # Without these markers Claude never caches — `cached_tokens` stays 0.
        needs_cache_control = is_anthropic_model
        is_o_model = bare_model.startswith(("o1", "o3", "o4"))
        is_deepseek = bare_model.startswith("deepseek")
        is_deepseek_reasoner = bare_model == "deepseek-reasoner"
        # Vertex AI MaaS-hosted models (e.g. deepseek-v3.2-maas) are
        # identified by the "-maas" suffix in the model name itself — no
        # need to look at the routing prefix. They expose thinking through
        # chat_template_kwargs.thinking, not the DeepSeek-native toggle.
        # Ref: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/thinking
        is_vertex_deepseek = is_deepseek and bare_model.endswith("-maas")
        # DeepSeek thinking: deepseek-reasoner, deepseek-chat w/ thinking enabled,
        # or Vertex DeepSeek MaaS with thinking enabled.
        is_deepseek_thinking = (
            is_deepseek_reasoner
            or (is_deepseek and thinking_enabled)
            or (is_vertex_deepseek and thinking_enabled)
        )
        is_kimi = bare_model.startswith(("kimi", "moonshot"))
        is_kimi_thinking = is_kimi and bare_model.startswith("kimi-k2")

        # Convert Anthropic-style tool schemas to OpenAI function format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

        # Convert Anthropic-style messages to OpenAI format
        openai_msgs: list[dict] = []
        if system:
            if needs_cache_control:
                # Anthropic prompt caching: wrap system as content block array
                openai_msgs.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                    ],
                })
            else:
                openai_msgs.append({"role": "system", "content": system})

        first_user_seen = False
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                # Add cache_control to the first user message (the large initial prompt)
                if needs_cache_control and role == "user" and not first_user_seen:
                    first_user_seen = True
                    openai_msgs.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                        ],
                    })
                else:
                    openai_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                deepseek_reasoning: str = ""
                openrouter_reasoning_details: list[dict] = []
                tool_calls_for_msg: list[dict] = []
                tool_results_for_msg: list[dict] = []

                for block in content:
                    btype = block.get("type")
                    if btype in ("thinking", "redacted_thinking"):
                        if is_openrouter:
                            # OpenRouter: preserve reasoning_details for multi-turn
                            openrouter_reasoning_details.extend(
                                block.get("reasoning_details", [])
                            )
                        elif is_deepseek_thinking or is_kimi_thinking:
                            # Preserve reasoning_content for DeepSeek/Kimi within a turn
                            deepseek_reasoning = block.get("thinking", "")
                        # For other models: drop (must not be sent back per API rules)
                    elif btype == "tool_use":
                        tool_calls_for_msg.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })
                    elif btype == "tool_result":
                        tool_results_for_msg.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block["content"]),
                        })
                    elif btype == "text":
                        openai_msgs.append({"role": role, "content": block["text"]})

                if tool_calls_for_msg:
                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_for_msg,
                    }
                    # Preserve reasoning_content for DeepSeek/Kimi thinking
                    if (is_deepseek_thinking or is_kimi_thinking) and deepseek_reasoning:
                        assistant_msg["reasoning_content"] = deepseek_reasoning
                    # Preserve reasoning_details for OpenRouter multi-turn
                    if is_openrouter and openrouter_reasoning_details:
                        assistant_msg["reasoning_details"] = openrouter_reasoning_details
                    openai_msgs.append(assistant_msg)

                for tr in tool_results_for_msg:
                    openai_msgs.append(tr)

        # Anthropic prompt caching with a *sliding* breakpoint: in addition to
        # the static cache_control on the system prompt + initial user message
        # (set above), also place one on the most recent user/tool message each
        # call. Without this the cache never grows past the initial prefix —
        # turns 2+ keep adding tool_use/tool_result blocks but none of them get
        # cached, so cached_tokens plateaus at the first-message size.
        # Anthropic allows up to 4 breakpoints; we use ≤3 (system + initial
        # user + rolling latest), leaving headroom.
        if needs_cache_control and openai_msgs:
            for i in range(len(openai_msgs) - 1, -1, -1):
                m = openai_msgs[i]
                if m.get("role") not in ("user", "tool"):
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = [
                        {"type": "text", "text": content,
                         "cache_control": {"type": "ephemeral"}}
                    ]
                elif isinstance(content, list) and content:
                    # Tag only the last text block to advance the breakpoint.
                    for blk in reversed(content):
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            blk["cache_control"] = {"type": "ephemeral"}
                            break
                    else:
                        # No text block (e.g. pure tool_result with stringified
                        # content): convert to block form and tag it.
                        m["content"] = [
                            {"type": "text", "text": str(content),
                             "cache_control": {"type": "ephemeral"}}
                        ]
                break

        # Build extra_body for provider-specific thinking options
        extra_body: dict = {}
        if thinking_enabled:
            if is_openrouter:
                # OpenRouter unified reasoning API
                reasoning_effort = self.thinking_config.get("reasoning_effort", "high")
                extra_body["reasoning"] = {"effort": reasoning_effort}
            elif is_qwen:
                extra_body["enable_thinking"] = True
            elif is_o_model:
                extra_body["reasoning_effort"] = self.thinking_config.get("reasoning_effort", "medium")
            elif is_vertex_deepseek:
                # Vertex AI MaaS hosts DeepSeek V3.2 and exposes thinking via
                # chat_template_kwargs.thinking (default off). The DeepSeek-
                # native {"thinking": {"type": "enabled"}} toggle is NOT
                # accepted here — passing it yields 400.
                # Ref: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/thinking
                extra_body.setdefault("chat_template_kwargs", {})["thinking"] = True
            elif is_deepseek and not is_deepseek_reasoner:
                # Enable thinking mode for deepseek-chat via extra_body
                extra_body["thinking"] = {"type": "enabled"}

        # OpenRouter-specific: pin upstream provider so sticky-routing doesn't
        # rotate us onto a cold cache, and turn on usage accounting so
        # prompt_tokens_details.cached_tokens comes back in the response.
        # Mapping is by our model-name prefix; covers our three primary
        # experiment models (claude-opus-4.6, gpt-5.4, gemini-3.1-pro) plus
        # DeepSeek / Qwen / Kimi routes. Users who need an unpinned run can
        # disable by setting providers.openrouter.pin=false in config.
        if is_openrouter and self._openrouter_pin_enabled():
            pinned = self._openrouter_provider_order(self.model)
            if pinned:
                extra_body.setdefault("provider", {})
                extra_body["provider"].setdefault("order", pinned)
                extra_body["provider"].setdefault("allow_fallbacks", False)
            extra_body.setdefault("usage", {"include": True})

        create_kwargs: dict = {
            "model": self.model,
            "messages": openai_msgs,
            "tools": openai_tools,
        }
        # Kimi thinking models need max_tokens >= 16000 for reasoning + content
        if is_kimi_thinking:
            create_kwargs["max_tokens"] = max(16000, budget_tokens + 4096)
        # Several providers reject tool_choice=required with thinking/reasoning:
        # - All deepseek-thinking variants (direct reasoner, deepseek-chat w/
        #   thinking, Vertex MaaS w/ thinking): the upstream model rejects
        #   forced tool calls while reasoning.
        # - Kimi k2 thinking: no tool_choice support at all.
        # - Claude via OpenRouter + thinking: only "auto" allowed.
        # - OpenRouter + thinking (Qwen, etc.): upstream rejects required in thinking mode.
        skip_tool_choice = (
            is_deepseek_thinking
            or is_kimi_thinking
            or (is_openrouter and thinking_enabled)
            or (is_qwen and thinking_enabled)
        )
        if not skip_tool_choice:
            create_kwargs["tool_choice"] = "required"
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**create_kwargs)
        self._last_usage = _extract_usage(response, self.model)

        # Extract the response
        response_dict = response.model_dump()
        if not response_dict or not response_dict.get("choices"):
            raise RuntimeError(
                "Malformed API response: no choices returned "
                f"(response_dict={type(response_dict).__name__})"
            )
        choice_dict = response_dict["choices"][0]
        msg_dict = choice_dict.get("message", {})

        tool_calls = msg_dict.get("tool_calls") or []
        if not tool_calls:
            return None

        tc = tool_calls[0]
        tool_name = tc["function"]["name"]
        try:
            tool_input = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            tool_input = {"error": f"Failed to parse tool arguments: {e}"}
        tool_id = tc.get("id", "")

        # Extract thinking/reasoning text
        thinking_text: str = msg_dict.get("reasoning_content") or msg_dict.get("reasoning") or ""

        # OpenRouter reasoning_details (structured reasoning blocks)
        reasoning_details: list[dict] = msg_dict.get("reasoning_details") or []

        # Build the Anthropic-format assistant_message content.
        content_blocks: list[dict] = []

        # For OpenRouter: store reasoning as a thinking block with reasoning_details
        if is_openrouter and (thinking_text or reasoning_details):
            if not thinking_text and reasoning_details:
                # Extract text from reasoning_details
                thinking_text = "\n\n".join(
                    d.get("text", "") for d in reasoning_details
                    if d.get("type") in ("reasoning.text", "reasoning.summary") and d.get("text")
                )
            content_blocks.append({
                "type": "thinking",
                "thinking": thinking_text,
                "reasoning_details": reasoning_details,
            })
        # For DeepSeek/Kimi thinking (including Vertex MaaS): store
        # reasoning_content as a thinking block so it's preserved and sent back
        # in subsequent calls within the turn.
        elif (is_deepseek_thinking or is_kimi_thinking) and thinking_text:
            content_blocks.append({
                "type": "thinking",
                "thinking": thinking_text,
            })

        content_blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": tool_input,
        })

        return {
            "name": tool_name,
            "input": tool_input,
            "assistant_message": {"role": "assistant", "content": content_blocks},
            "thinking": thinking_text or None,
            "usage": _extract_usage(response, self.model),
        }


# ---------------------------------------------------------------------------
# OpenAI Responses API client (for models like gpt-5.4-pro that only
# support the Responses API, not Chat Completions)
# ---------------------------------------------------------------------------

class OpenAIResponsesClient(ModelClient):
    """Client using OpenAI Responses API (client.responses.create)."""

    # Models that require the Responses API (not chat completions)
    RESPONSES_ONLY_MODELS = {"gpt-5.4-pro", "gpt-5.4-pro-2026-03-05"}

    def __init__(self, model: str, api_key: str, thinking_config: dict | None = None):
        from openai import OpenAI
        self.model = model
        self.thinking_config = thinking_config or {}
        # gpt-5.4-pro can take 10+ minutes per request
        self.client = OpenAI(api_key=api_key, timeout=1800.0)

    def call(self, messages: list, tools: list, system: str = "") -> dict | None:
        # Convert Anthropic-style tool schemas to Responses API format
        responses_tools = [
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
            for t in tools
        ]

        # Build input from messages (Responses API uses 'input' not 'messages')
        input_list: list[dict] = []
        if system:
            input_list.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                input_list.append({"role": role, "content": content})
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        input_list.append({"role": role, "content": block["text"]})
                    elif btype == "tool_use":
                        # Function call from assistant — add as function_call item
                        input_list.append({
                            "type": "function_call",
                            "call_id": block.get("id", ""),
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        })
                    elif btype == "tool_result":
                        # Function result from user — add as function_call_output
                        input_list.append({
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": str(block["content"]),
                        })
                    # Skip thinking blocks

        create_kwargs: dict = {
            "model": self.model,
            "input": input_list,
            "tools": responses_tools,
            "max_output_tokens": 128000,
        }

        # Reasoning effort for thinking models
        thinking_enabled = self.thinking_config.get("enabled", False)
        if thinking_enabled:
            effort = self.thinking_config.get("reasoning_effort", "high")
            create_kwargs["reasoning"] = {"effort": effort}

        response = self.client.responses.create(**create_kwargs)
        self._last_usage = _extract_usage(response, self.model)

        # Extract function call from response output
        tool_call = None
        thinking_text = ""
        for item in response.output:
            if item.type == "function_call":
                tool_call = item
            elif item.type == "reasoning":
                # Extract reasoning/thinking text if available
                if hasattr(item, "summary") and item.summary:
                    thinking_text = "\n".join(
                        s.get("text", "") for s in item.summary
                        if isinstance(s, dict)
                    ) if isinstance(item.summary, list) else str(item.summary)

        if not tool_call:
            return None

        tool_name = tool_call.name
        try:
            tool_input = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            tool_input = {"error": f"Failed to parse tool arguments: {e}"}
        call_id = tool_call.call_id

        # Build Anthropic-format assistant_message content for history
        content_blocks: list[dict] = []
        content_blocks.append({
            "type": "tool_use",
            "id": call_id,
            "name": tool_name,
            "input": tool_input,
        })

        return {
            "name": tool_name,
            "input": tool_input,
            "assistant_message": {"role": "assistant", "content": content_blocks},
            "thinking": thinking_text or None,
            "usage": _extract_usage(response, self.model),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_client(global_config: dict) -> ModelClient:
    """Instantiate the correct ModelClient based on model name and config."""
    model: str = global_config.get("model", "")
    if not model:
        raise ValueError("No model specified. Pass --model when running the agent.")

    providers = global_config.get("providers", {})
    thinking_config = global_config.get("thinking", {})

    # Generic prefix dispatch: `<prefix>/<name>` routes to providers.<prefix>.
    # This covers vertex_ai/, openai/, openrouter/, anthropic/, qwen/, etc.
    # uniformly — point providers.<prefix>.base_url at whichever upstream you
    # want (direct API, LiteLLM proxy, OpenRouter, …). Extra fields under the
    # provider entry (e.g. pin / order_overrides for OpenRouter) flow through
    # as provider_config so URL-keyed runtime adaptations still kick in.
    if "/" in model:
        prefix = model.split("/", 1)[0]
        cfg = providers.get(prefix)
        if not cfg:
            raise ValueError(
                f"Model '{model}' uses prefix '{prefix}/' but providers.{prefix} "
                "is not configured. Add an entry under `providers:` in your config."
            )
        api_key = _resolve_api_key(cfg, prefix)
        base_url = cfg.get("base_url", "").rstrip("/")
        if not base_url:
            raise ValueError(
                f"providers.{prefix}.base_url is required for model '{model}'."
            )
        # OpenAI SDK expects /v1 suffix.
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        provider_config = {k: v for k, v in cfg.items() if k not in {"api_key", "base_url"}}
        return OpenAIClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            thinking_config=thinking_config,
            provider_config=provider_config,
        )

    elif model.startswith("claude"):
        cfg = providers.get("anthropic", {})
        api_key = _resolve_api_key(cfg, "anthropic")
        # Always use the native Anthropic SDK for Claude, even when a
        # base_url is set (e.g. a LiteLLM proxy). LiteLLM exposes Anthropic's
        # /v1/messages passthrough, so the SDK talks to the proxy directly.
        # Going through OpenAIClient would force an OpenAI→Anthropic format
        # translation at the proxy; the native client uses Anthropic-format
        # cache_control as a first-class API field, so prompt caching is
        # reliable and reported correctly (cache_creation/cache_read).
        # The Anthropic SDK appends "/v1/messages" itself, so strip a
        # trailing "/v1" if the config supplied one for the OpenAI SDK.
        base_url = (cfg.get("base_url") or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return AnthropicClient(model=model, api_key=api_key,
                               base_url=base_url or None,
                               thinking_config=thinking_config)

    elif model.startswith(("gpt-", "o1", "o3", "o4")):
        cfg = providers.get("openai", {})
        api_key = _resolve_api_key(cfg, "openai")
        base_url = cfg.get("base_url", "")
        # Models that only support the Responses API
        if model in OpenAIResponsesClient.RESPONSES_ONLY_MODELS:
            return OpenAIResponsesClient(model=model, api_key=api_key, thinking_config=thinking_config)
        if base_url:
            return OpenAIClient(model=model, api_key=api_key, base_url=base_url, thinking_config=thinking_config)
        return OpenAIClient(model=model, api_key=api_key, thinking_config=thinking_config)

    elif model.startswith("deepseek"):
        cfg = providers.get("deepseek", {})
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "deepseek"),
            base_url=cfg.get("base_url", "https://api.deepseek.com/v1"),
            thinking_config=thinking_config,
        )

    elif model.startswith("qwen"):
        cfg = providers.get("qwen", {})
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "qwen"),
            base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            thinking_config=thinking_config,
        )

    elif model.startswith("glm"):
        cfg = providers.get("glm", {})
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "glm"),
            base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            thinking_config=thinking_config,
        )

    elif model.lower().startswith("minimax"):
        cfg = providers.get("minimax", {}) or providers.get("MiniMax", {})
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "minimax"),
            base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            thinking_config=thinking_config,
        )

    elif model.startswith("gemini"):
        cfg = providers.get("gemini", {})
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "gemini"),
            base_url=cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            thinking_config=thinking_config,
        )

    elif model.startswith(("kimi", "moonshot")):
        cfg = providers.get("kimi", {})
        base_url = cfg.get("base_url", "https://api.moonshot.ai/v1")
        # Kimi coding endpoint (api.kimi.com/coding/) speaks Anthropic protocol
        if "api.kimi.com" in base_url:
            return AnthropicClient(
                model=model,
                api_key=_resolve_api_key(cfg, "kimi"),
                base_url=base_url,
                thinking_config=thinking_config,
            )
        return OpenAIClient(
            model=model,
            api_key=_resolve_api_key(cfg, "kimi"),
            base_url=base_url,
            thinking_config=thinking_config,
        )

    else:
        raise ValueError(
            f"Unknown model prefix for '{model}'. "
            "Expected: claude-*, gpt-*, o1/o3/o4*, deepseek-*, qwen-*, glm-*, gemini-*, kimi-*/moonshot-*, "
            "or provider/model for OpenRouter"
        )
