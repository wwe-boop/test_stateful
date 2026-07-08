"""Prompt templates for test-prosody text synthesis."""

from __future__ import annotations

SCENARIOS: dict[str, str] = {
    "question_to_statement": "疑问句后接平叙句，制造语调转折边界",
    "exclamation_to_narrative": "感叹句后接平静叙述，制造情绪落差边界",
    "long_short_alternate": "长短句交替，制造节奏突变边界",
    "number_english_mix": "包含数字、英文缩写或专名，制造发音切换边界",
    "emotion_progression": "情绪逐步递进，最后收束到陈述句",
    "mixed_punctuation": "逗号、分号、问号、感叹号、句号混合出现",
}

DEFAULT_INSTRUCTS = [
    "用平静温和的叙述语气说",
    "用专业新闻播报的语气说",
    "用轻松自然的聊天语气说",
    "用略带悬疑感的语气说",
    "用正式庄重的语气说",
]

DEFAULT_SPEAKERS = ["Vivian", "Serena", "Ethan"]


SYSTEM_PROMPT = """你是中文语音合成评测语料编写专家。
任务：为“模拟流式 TTS 跨片段韵律稳定性”实验编写测试文本。

硬性要求：
1. 只输出 JSON，不要 markdown，不要解释。
2. 文本必须是自然中文，适合朗读，避免网络梗和敏感内容。
3. 每条样本拆成 6-12 个 segments（子句/短句），每个 segment 自带结尾标点。
4. 刻意设计“高跳变风险边界”，例如：疑问→陈述、感叹→平叙、长短句交替、数字/英文夹杂、情绪递进。
5. 每个 segment 必须是完整可读片段，长度建议 8-40 个汉字。
6. punct_class 只能是：comma, period, question, exclamation, semicolon, colon。
7. 不要复用常见有声书/新闻稿原句，避免与训练语料重叠。

JSON schema:
{
  "sample_id": "prosody_mini_001",
  "scenario": "question_to_statement",
  "language": "Chinese",
  "speaker": "Vivian",
  "instruct": "用平静温和的叙述语气说",
  "segments": [
    {"text": "你真的相信这件事吗？", "punct_class": "question"},
    {"text": "我原本以为也是如此，", "punct_class": "comma"},
    {"text": "直到看见第三份报告才改变看法。", "punct_class": "period"}
  ]
}
"""


def build_user_prompt(
    *,
    sample_index: int,
    scenario: str,
    speaker: str,
    instruct: str,
    min_segments: int = 6,
    max_segments: int = 12,
) -> str:
    scenario_hint = SCENARIOS.get(scenario, scenario)
    return (
        f"请生成 1 条 test-prosody 样本。\n"
        f"- sample_id: prosody_mini_{sample_index:03d}\n"
        f"- scenario: {scenario}\n"
        f"- scenario_hint: {scenario_hint}\n"
        f"- speaker: {speaker}\n"
        f"- instruct: {instruct}\n"
        f"- segments 数量: {min_segments}-{max_segments}\n"
        f"- language: Chinese\n"
        f"直接返回 JSON 对象。"
    )
