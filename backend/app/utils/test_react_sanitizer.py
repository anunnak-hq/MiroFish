"""Unit tests for react_sanitizer — run before deploying to MiroFish container."""

import pytest
from react_sanitizer import sanitize_react_output


class TestPass1ExactAnchors:
    """Pass 1: aggressive stripping of exact ReACT markers."""

    def test_strip_thought_line(self):
        text = "Some intro.\nThought: I need to analyze the market data.\nThe market is growing."
        result = sanitize_react_output(text)
        assert "Thought:" not in result
        assert "Some intro." in result
        assert "The market is growing." in result

    def test_strip_action_and_observation(self):
        text = (
            "Introduction.\n"
            "Action: quick_search\n"
            "Observation: Found 15 results about food trucks.\n"
            "The food truck industry is booming."
        )
        result = sanitize_react_output(text)
        assert "Action:" not in result
        assert "Observation:" not in result
        assert "Introduction." in result
        assert "The food truck industry is booming." in result

    def test_strip_action_input(self):
        text = "Start.\n  Action Input: {\"query\": \"test\"}\nEnd."
        result = sanitize_react_output(text)
        assert "Action Input:" not in result
        assert "Start." in result
        assert "End." in result

    def test_strip_tool_call_block(self):
        text = (
            "Before.\n"
            '<tool_call>\n{"name": "insight_forge", "parameters": {"query": "test"}}\n</tool_call>\n'
            "After."
        )
        result = sanitize_react_output(text)
        assert "<tool_call>" not in result
        assert "</tool_call>" not in result
        assert "insight_forge" not in result
        assert "Before." in result
        assert "After." in result

    def test_strip_json_fenced_block(self):
        text = 'Before.\n```json\n{"query": "food truck profitability"}\n```\nAfter.'
        result = sanitize_react_output(text)
        assert "```json" not in result
        assert "food truck profitability" not in result
        assert "Before." in result
        assert "After." in result

    def test_strip_standalone_fence(self):
        text = "Before.\n```\nAfter."
        result = sanitize_react_output(text)
        # The standalone ``` should be gone
        assert "```" not in result
        assert "Before." in result
        assert "After." in result


class TestPass2ChineseThinking:
    """Pass 2: conservative Chinese thinking heuristic (prefix + tool ref)."""

    def test_strip_chinese_thinking_with_tool_ref(self):
        text = "市场分析结果。\n让我尝试quick_search来获取更多数据。\n市场前景良好。"
        result = sanitize_react_output(text)
        assert "让我尝试quick_search" not in result
        assert "市场分析结果。" in result
        assert "市场前景良好。" in result

    def test_strip_chinese_thinking_with_search(self):
        text = "开始。\n我需要search相关信息来分析。\n结束。"
        result = sanitize_react_output(text)
        assert "我需要search" not in result
        assert "开始。" in result

    def test_strip_chinese_thinking_with_analysis_ref(self):
        text = "数据。\n我来分析这个市场的竞争格局。\n结论。"
        result = sanitize_react_output(text)
        assert "我来分析" not in result
        assert "数据。" in result


class TestPass2RussianThinking:
    """Pass 2: Russian thinking traces from today's actual MiroFish output."""

    def test_strip_russian_quick_search_trace(self):
        text = "Заголовок секции.\nПозвольте мне провести быстрый поиск для получения конкретных данных о финансовых показателях.\nРынок растёт."
        result = sanitize_react_output(text)
        assert "Позвольте мне" not in result
        assert "Заголовок секции." in result
        assert "Рынок растёт." in result

    def test_strip_russian_quick_search_with_tool_name(self):
        text = "Начало.\nПозвольте мне использовать quick_search для получения конкретных данных.\nКонец."
        result = sanitize_react_output(text)
        assert "Позвольте мне использовать quick_search" not in result
        assert "Начало." in result
        assert "Конец." in result

    def test_strip_russian_try_search(self):
        text = "Данные.\nПопробую найти данные о конкурентных реакциях.\nВывод."
        result = sanitize_react_output(text)
        assert "Попробую" not in result
        assert "Вывод." in result

    def test_preserve_russian_legitimate_with_prefix(self):
        """Russian sentence starting with 'Позвольте' but no tool reference."""
        text = "Позвольте мне обратить внимание на ключевые тенденции рынка."
        assert sanitize_react_output(text) == text

    def test_strip_english_let_me_search(self):
        text = "Intro.\nLet me search for the relevant market data.\nConclusion."
        result = sanitize_react_output(text)
        assert "Let me search" not in result
        assert "Intro." in result
        assert "Conclusion." in result


class TestPreserveLegitimateContent:
    """Content in any language without ReACT traces MUST pass through unchanged."""

    def test_preserve_english_paragraph(self):
        text = (
            "The Georgian food truck market in Austin shows strong potential. "
            "With a growing tech community and established food truck culture, "
            "the business model appears viable for first-year profitability."
        )
        assert sanitize_react_output(text) == text

    def test_preserve_russian_paragraph(self):
        text = (
            "Рынок грузинских фудтраков в Остине демонстрирует сильный потенциал. "
            "Растущее техническое сообщество и устоявшаяся культура фудтраков "
            "делают бизнес-модель перспективной для достижения прибыльности в первый год."
        )
        assert sanitize_react_output(text) == text

    def test_preserve_chinese_market_analysis(self):
        """Chinese paragraph about market analysis without thinking prefixes + tool refs."""
        text = (
            "奥斯汀的格鲁吉亚美食卡车市场显示出强劲的潜力。"
            "随着科技社区的发展和成熟的美食卡车文化，"
            "该商业模式在第一年实现盈利的前景看好。"
            "市场需求稳步增长，消费者对新口味的接受度较高。"
        )
        assert sanitize_react_output(text) == text

    def test_preserve_chinese_with_common_words(self):
        """现在 and 首先 are common words that should NOT be stripped."""
        text = "现在市场竞争激烈。首先需要考虑定价策略。"
        assert sanitize_react_output(text) == text

    def test_chinese_thinking_prefix_without_tool_ref_preserved(self):
        """Chinese prefix WITHOUT tool reference → keep (could be legitimate)."""
        text = "让我们回顾一下过去一年的市场变化和消费者趋势。"
        assert sanitize_react_output(text) == text


class TestRealWorldExample:
    """Simulated real fragment from yesterday's MiroFish report."""

    def test_mixed_content_with_traces(self):
        text = (
            "## 市场需求的强势确认\n\n"
            "Thought: I need to analyze the market demand data from the simulation.\n"
            "Action: insight_forge\n"
            '<tool_call>\n{"name": "insight_forge", "parameters": {"query": "格鲁吉亚美食卡车消费者需求"}}\n</tool_call>\n'
            "Observation: The simulation shows strong consumer interest.\n\n"
            "让我尝试quick_search来获取消费者反馈数据。\n\n"
            "The Austin food truck market demonstrates robust demand for Georgian cuisine. "
            "Survey data from the simulation indicates 73% of tech workers expressed "
            "willingness to try Georgian food, with khachapuri and khinkali ranking "
            "highest in appeal.\n\n"
            "Consumer education costs remain a concern, but the simulation suggests "
            "that sampling events could convert 40% of first-time triers into repeat customers."
        )
        result = sanitize_react_output(text)
        # All traces gone
        assert "Thought:" not in result
        assert "Action:" not in result
        assert "Observation:" not in result
        assert "<tool_call>" not in result
        assert "让我尝试quick_search" not in result
        # Legitimate content preserved
        assert "市场需求的强势确认" in result
        assert "The Austin food truck market" in result
        assert "khachapuri and khinkali" in result
        assert "sampling events could convert 40%" in result


class TestEdgeCases:
    """Edge cases and cleanup behavior."""

    def test_empty_string(self):
        assert sanitize_react_output("") == ""

    def test_none_input(self):
        assert sanitize_react_output(None) is None

    def test_collapse_blank_lines(self):
        text = "Line 1.\n\n\n\n\nLine 2."
        result = sanitize_react_output(text)
        assert result == "Line 1.\n\nLine 2."

    def test_strips_leading_trailing_whitespace(self):
        text = "\n\n  Content here.  \n\n"
        result = sanitize_react_output(text)
        assert result == "Content here."
