"""Prompt templates for LLM-generated test-boundary samples."""

from __future__ import annotations

BOUNDARY_SCENARIOS: dict[str, str] = {
    "extreme_force": "超长纯汉字无句读，仅在末尾句号，触发参考force硬切",
    "extreme_multi_force": "两段纯汉字块，中间一个句号，触发多次force",
    "extreme_l2_gap": "前半纯汉字force/L1路径，后半密集逗号",
    "extreme_l1_edge": "L1句号刚好出现在L1阈值附近",
    "long_l1_pair": "两段足够长的中文，每段末尾必须是句号，中间仅逗号/分号",
    "long_no_punct": "超长无强句末标点段落，用于触发force硬切",
    "long_dense_l2": "大量逗号/分号，但只在句号处作为强边界",
    "mixed_lang": "数字、英文缩写、专名与中英文混排",
    "quote_suffix": "引号/括号后缀，如：他说：“……。”",
    "question_exclaim": "问号、感叹号与陈述句交替",
    "short_burst": "多个极短句连续出现",
    "numbers": "订单号、电话、日期等数字密集文本",
    "ellipsis_l3": "省略号、换行等弱边界",
    "long_news": "新闻播报体，两段以上长句",
}

SYSTEM_PROMPT = """你是中文语音合成在线切分评测语料编写专家。
任务：编写用于 test-boundary 的完整中文文本（不是 segments 列表）。

硬性要求：
1. 只输出 JSON，不要 markdown，不要解释。
2. full_text 必须是自然中文，适合朗读，避免敏感内容。
3. 根据 scenario 刻意设计切分风险（长无标点、密集逗号、混排等）。
4. 文本长度建议 80-220 个汉字；long_no_punct 场景可更长。
5. 不要复用常见新闻稿原句。

JSON schema:
{
  "sample_id": "boundary_0001",
  "scenario_tags": "long_l1_pair",
  "full_text": "完整文本……"
}
"""


def build_user_prompt(*, sample_index: int, scenario: str) -> str:
    hint = BOUNDARY_SCENARIOS.get(scenario, scenario)
    return (
        f"请生成 1 条 test-boundary 样本。\n"
        f"- sample_id: boundary_{sample_index:04d}\n"
        f"- scenario_tags: {scenario}\n"
        f"- scenario_hint: {hint}\n"
        f"- language: Chinese\n"
        f"直接返回 JSON 对象，字段仅 sample_id / scenario_tags / full_text。"
    )
