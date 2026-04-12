"""
LLM客户端封装
统一使用OpenAI格式调用 — но прозрачно роутит в Anthropic SDK когда
base_url указывает на Anthropic или Anunnak proxy.

Anthropic /v1/messages API отличается от OpenAI /v1/chat/completions:
- endpoint: /v1/messages (Anthropic) vs /v1/chat/completions (OpenAI)
- system message: top-level param `system` (Anthropic) vs role in messages list (OpenAI)
- response shape: content[0].text (Anthropic) vs choices[0].message.content (OpenAI)
- max_tokens: required (Anthropic) vs optional (OpenAI)
- response_format: not supported (Anthropic) vs supported (OpenAI)

Этот wrapper скрывает различия: callers пишут OpenAI-стиль, а под капотом
запрос идёт через anthropic SDK когда base_url — Anthropic. Зачем: наш
Anunnak billing proxy — drop-in replacement для Anthropic /v1/messages, и
openai SDK туда бить не может (он по умолчанию POST'ит в /chat/completions
что 401/404).
"""

import json
import re
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI

from ..config import Config


def _is_anthropic_base_url(base_url: Optional[str]) -> bool:
    """True if base_url points at Anthropic or the Anunnak proxy.

    Anunnak proxy is a 1:1 Anthropic-compatible replacement exposing
    /v1/messages with ×3 markup billing + per-user attribution. It must
    be treated exactly like real Anthropic for SDK selection purposes.
    """
    if not base_url:
        return False
    lowered = base_url.lower()
    return "anthropic" in lowered or "anunnak.com" in lowered


def _normalize_anthropic_base_url(base_url: str) -> str:
    """Strip trailing /v1 (and any trailing slashes) from base_url.

    anthropic SDK adds /v1/messages itself — if we keep /v1 in base_url
    we get /v1/v1/messages which 404s. Accept both with- and without-v1
    forms so existing env configs don't break.
    """
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[:-3]
    return stripped


def _get_request_model() -> Optional[str]:
    """Read X-LLM-Model header from current Flask request context (if any).

    Allows Anunnak callers to override the env-level LLM_MODEL_NAME on a
    per-request basis so the model chosen in the Gosha UI flows through to
    every internal MiroFish LLM call (ontology extraction, persona
    generation, report sections, etc.).
    """
    try:
        from flask import request as _req
        return _req.headers.get('X-LLM-Model')
    except (ImportError, RuntimeError):
        return None


def _split_system_from_messages(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """Extract the system message (if any) from an OpenAI-style messages list.

    Returns (system_text, remaining_messages). Anthropic expects `system` as
    a top-level param on messages.create() and rejects role=system inside
    the messages array.
    """
    system_parts: List[str] = []
    other: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content", "")
            if isinstance(content, str) and content:
                system_parts.append(content)
        else:
            other.append(m)
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, other


def _strip_think_tags(text: str) -> str:
    """Some models (MiniMax M2.5, GLM, etc.) wrap reasoning in <think>...</think>.

    Strip before returning content. Preserves existing upstream behavior.
    """
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


class LLMClient:
    """Unified OpenAI/Anthropic LLM client.

    Same public API as upstream (chat / chat_json), but transparently routes
    to anthropic.Anthropic().messages.create() when base_url is Anthropic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        # Priority: explicit arg > X-LLM-Model header > env default.
        self.model = model or _get_request_model() or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.is_anthropic = _is_anthropic_base_url(self.base_url)

        if self.is_anthropic:
            # Use anthropic SDK for Anthropic/Anunnak-proxy endpoints.
            # Imported lazily so envs that don't use Anthropic don't need
            # the dep installed.
            import anthropic  # type: ignore
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=_normalize_anthropic_base_url(self.base_url),
            )
            self.openai_client = None
        else:
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.anthropic_client = None

    # Back-compat alias: upstream and in-tree code reference self.client.
    # Keep it pointing at whichever backend we actually use so naive
    # attribute access doesn't crash.
    @property
    def client(self):  # pragma: no cover - trivial delegation
        return self.anthropic_client if self.is_anthropic else self.openai_client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Send a chat request and return plain text.

        `response_format` is a hint for OpenAI-compat backends and is
        silently ignored for Anthropic (which doesn't support it — the
        caller's prompts already request JSON where needed).
        """
        if self.is_anthropic:
            system_text, other_messages = _split_system_from_messages(messages)
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": other_messages,
            }
            if system_text:
                kwargs["system"] = system_text
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = self.anthropic_client.messages.create(**kwargs)  # type: ignore[union-attr]
            # Anthropic returns content as a list of blocks. Concatenate
            # text blocks in order and discard non-text blocks.
            parts: List[str] = []
            for block in (response.content or []):
                text_attr = getattr(block, "text", None)
                if text_attr:
                    parts.append(text_attr)
            content = "".join(parts)
        else:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            response = self.openai_client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            content = response.choices[0].message.content or ""

        return _strip_think_tags(content)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat request and parse the response as JSON.

        For OpenAI backends we pass response_format={"type":"json_object"}
        as a hint; for Anthropic we rely on prompt-level instructions and
        robust post-processing (strip markdown fences, parse).
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        cleaned = response.strip()
        # Strip leading/trailing markdown fences if present.
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned}")
