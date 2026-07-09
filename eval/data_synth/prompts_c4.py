"""Prompt templates for SteadyStream C4 synthetic full-passage text."""

from __future__ import annotations

SCENARIOS: dict[str, str] = {
    "service_dialogue": "客服/机器人对话式说明，语气自然但不口水化",
    "daily_narrative": "生活叙事，有轻微情绪推进和回落",
    "product_briefing": "产品/设备说明，夹少量数字或英文缩写",
    "news_explainer": "新闻播报式解释，信息密度中等",
    "process_instruction": "流程指引或操作说明，包含先后顺序",
    "reflective_story": "平静回忆或观察，跨句韵律应连贯",
    "question_answer": "先提出疑问，再用几句解释回答",
    "mixed_punctuation": "逗号、分号、冒号、问号、感叹号混合",
}

DEFAULT_INSTRUCTS = [
    "用平静温和的叙述语气说",
    "用轻松自然的聊天语气说",
    "用专业但不生硬的播报语气说",
    "用耐心清楚的讲解语气说",
    "用略带关切的服务语气说",
]

SYSTEM_PROMPT = """你是中文语音合成训练语料编写专家。
任务：为 SteadyStream C4 续写微调生成“整段一次合成”的多子句段落文本。

硬性要求：
1. 只输出 JSON，不要 markdown，不要解释。
2. 每条 sample 是一个自然中文段落，适合 TTS 朗读。
3. 每条 sample 拆成 4-12 个 segments，每个 segment 自带结尾标点。
4. 整段中文汉字数控制在 100-220 之间；宁可写得稍充分，不要短小摘要。
5. punct_class 只能是：comma, period, question, exclamation, semicolon, colon。
6. 需要多样标点，至少包含两类 punct_class；不要全是逗号或全是句号。
7. 避免真实新闻、书籍、有声书、歌词、影视台词、名人讲话原句；必须原创。
8. 不要出现敏感政治、色情、暴力、自伤、医疗诊断、投资承诺等高风险内容。
9. 不要复用“test-prosody-mini”风格的高跳变评测句；这是训练文本，不是边界压力测试。

JSON schema:
{
  "samples": [
    {
      "scenario": "daily_narrative",
      "language": "Chinese",
      "instruct": "用平静温和的叙述语气说",
      "segments": [
        {"text": "清晨的社区广场还没有完全热闹起来，", "punct_class": "comma"},
        {"text": "保洁车沿着花坛慢慢驶过，留下很轻的水声。", "punct_class": "period"},
        {"text": "值班人员看了看预约表，", "punct_class": "comma"},
        {"text": "又把临时通行码的提示牌往门口挪了半步。", "punct_class": "period"},
        {"text": "如果有人忘记带证件怎么办？", "punct_class": "question"},
        {"text": "他指了指旁边的登记台，说那里可以人工核验。", "punct_class": "period"}
      ]
    }
  ]
}
"""


def build_user_prompt(
    *,
    start_index: int,
    batch_size: int,
    scenario: str,
    instruct: str,
    min_segments: int,
    max_segments: int,
    min_chars: int,
    max_chars: int,
) -> str:
    scenario_hint = SCENARIOS.get(scenario, scenario)
    return (
        f"请生成 {batch_size} 条 C4 synthetic full-passage samples。\n"
        f"- sample_id 将由程序分配，从 c4_synth_{start_index:05d} 开始；你不要输出 sample_id。\n"
        f"- scenario: {scenario}\n"
        f"- scenario_hint: {scenario_hint}\n"
        f"- instruct: {instruct}\n"
        f"- segments 数量: {min_segments}-{max_segments}\n"
        f"- 每条整段中文汉字数硬性范围: {min_chars}-{max_chars}\n"
        f"- 实际写作时请优先控制在 100-220 个汉字，不要低于 90 个汉字。\n"
        f"- language: Chinese\n"
        f"- 每条都必须原创，并且同一批内部不要复用句式。\n"
        f"直接返回 JSON 对象，顶层字段只能包含 samples。"
    )
