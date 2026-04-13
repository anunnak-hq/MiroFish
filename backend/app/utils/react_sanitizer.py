"""
Strip ReACT internal traces from LLM output.

Two-pass design:
  Pass 1 — exact anchors (aggressive, safe): Thought/Action/Observation lines,
            <tool_call> blocks, ```json blocks, standalone ``` fences.
  Pass 2 — multilingual thinking heuristic (conservative): only strips lines
            that start with a known thinking prefix AND contain a tool reference.
            Covers Chinese, Russian, and English thinking patterns.
"""

import re

# ── Pass 1 patterns (exact, always safe to strip) ────────────────────────────

_REACT_TRACE_LINE = re.compile(
    r'^\s*(Thought|Action|Observation|Action Input)\s*:.*$',
    re.MULTILINE,
)
_TOOL_CALL_BLOCK = re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL)
_JSON_FENCED_BLOCK = re.compile(r'```(?:json)?\s*\{.*?\}\s*```', re.DOTALL)
_STANDALONE_FENCE = re.compile(r'^```\s*$', re.MULTILINE)

# ── Pass 2 patterns (multilingual thinking heuristic) ────────────────────────
# Lines matching a thinking prefix AND a tool reference are stripped.
# Thinking prefixes WITHOUT tool refs are kept (could be legitimate content).

_THINKING_PREFIX = re.compile(
    r'^\s*('
    # Chinese
    r'让我|我需要|我来|接下来|我尝试|我将'
    r'|'
    # Russian
    r'Позвольте мне|Давайте я|Мне нужно|Попробую|Я попробую|Я использую|Давайте'
    r'|'
    # English (informal thinking, not the formal Thought:/Action: caught in Pass 1)
    r'Let me try|Let me use|I need to|I will try|I\'ll use|Let me search|Let me perform'
    r')',
)
_TOOL_REFERENCE = re.compile(
    r'quick_search|panorama_search|insight_forge|tool_call'
    r'|search|分析'
    # Russian tool references
    r'|поиск|быстрый поиск|инструмент|данных о|данные о|получения',
    re.IGNORECASE,
)

# ── Cleanup ──────────────────────────────────────────────────────────────────

_EXCESS_BLANK_LINES = re.compile(r'\n{3,}')


def sanitize_react_output(text: str) -> str:
    """Strip ReACT internal traces from LLM output, language-agnostic."""
    if not text:
        return text

    # Pass 1: exact anchors
    text = _REACT_TRACE_LINE.sub('', text)
    text = _TOOL_CALL_BLOCK.sub('', text)
    text = _JSON_FENCED_BLOCK.sub('', text)
    text = _STANDALONE_FENCE.sub('', text)

    # Pass 2: multilingual thinking lines (only if ALSO contain tool reference)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if _THINKING_PREFIX.match(line) and _TOOL_REFERENCE.search(line):
            continue  # strip this line
        cleaned.append(line)
    text = '\n'.join(cleaned)

    # Cleanup
    text = _EXCESS_BLANK_LINES.sub('\n\n', text)
    return text.strip()
