"""Programmatic builders for test-boundary samples (no LLM required)."""

from __future__ import annotations

import random
from typing import Callable

# Reusable Chinese filler clauses (~20-35 chars each).
_FILLERS: tuple[str, ...] = (
    "项目团队需要在有限预算内完成接口联调与性能验证",
    "工程同学正在检查Tokenizer输出与标点层级分类是否一致",
    "产品侧希望在线切分尽量贴近离线参考并控制强制切分率",
    "实验记录显示流式输入到达顺序会显著影响flush时机",
    "我们持续观察KV预算与audio步数比值对阈值缩放的影响",
    "该段落用于拉长文本长度以触发L1参考边界或force硬切",
    "播报员会以较快的语速读完数字英文混排的技术名词",
    "读者在没有强句末标点时仍应感知自然的语义分段位置",
    "运维同学关注max_concurrent_segments对排队延迟的影响",
    "质检流程要求每条样本标注scenario_tags便于分桶统计",
)

# Pure clauses: no ，。！？；：… or whitespace breaks (for force-path stress).
_PURE_FRAGMENTS: tuple[str, ...] = (
    "项目团队需要在有限预算内完成接口联调与性能验证工作",
    "工程同学正在检查分词器输出与标点层级分类是否一致",
    "产品侧希望在线切分尽量贴近离线参考并控制强制切分率",
    "实验记录显示流式输入到达顺序会显著影响flush时机",
    "我们持续观察KV预算与audio步数比值对阈值缩放的影响",
    "该段落用于拉长文本长度以触发参考边界或force硬切路径",
    "播报员会以较快的语速读完数字英文混排的技术名词",
    "读者在没有强句末标点时仍应感知自然的语义分段位置",
    "运维同学关注并发分段上限对排队延迟的影响",
    "质检流程要求每条样本标注scenario_tags便于分桶统计",
    "后端服务需要在高并发条件下维持稳定的decode延迟",
    "长文本输入必须避免KV溢出同时尽量减少硬切频率",
)

# Scenario → default quota for a 200-sample dataset (sum = 200).
SCENARIO_QUOTAS: dict[str, int] = {
    "extreme_force": 25,
    "extreme_multi_force": 15,
    "extreme_l2_gap": 20,
    "long_l1_pair": 25,
    "long_no_punct": 10,
    "long_dense_l2": 20,
    "long_news": 20,
    "mixed_lang": 15,
    "quote_suffix": 10,
    "question_exclaim": 10,
    "short_burst": 10,
    "numbers": 10,
    "ellipsis_l3": 10,
    "extreme_l1_edge": 10,
}

# Curated for Seg-Table-1: KV / force-alignment stress only (no regular news/mixed/short).
SCENARIO_QUOTAS_ADVANTAGE: dict[str, int] = {
    "extreme_force": 20,
    "extreme_multi_force": 15,
    "long_no_punct": 12,
    "extreme_l1_edge": 13,
}

ADVANTAGE_SCENARIO_TAGS: frozenset[str] = frozenset(SCENARIO_QUOTAS_ADVANTAGE.keys())

QUOTA_PROFILES: dict[str, dict[str, int]] = {
    "full": SCENARIO_QUOTAS,
    "advantage": SCENARIO_QUOTAS_ADVANTAGE,
}


def _pad_clause(rng: random.Random, min_chars: int, *, trailing: str = "，") -> str:
    parts: list[str] = []
    while sum(len(p) for p in parts) < min_chars:
        parts.append(rng.choice(_FILLERS) + "，")
    text = "".join(parts)
    if trailing and not text.endswith(trailing):
        text = text.rstrip("，") + trailing
    while len(text) < min_chars:
        text += rng.choice(_FILLERS)[: max(0, min_chars - len(text))]
    return text


def _build_long_l1_pair(rng: random.Random) -> str:
    first = _pad_clause(rng, rng.randint(68, 95), trailing="。")
    second = _pad_clause(rng, rng.randint(68, 95), trailing="。")
    return first + second


def _pure_pad(rng: random.Random, min_chars: int, *, max_chars: int | None = None) -> str:
    """Concatenate punctuation-free clauses to reach a target char length."""
    upper = max_chars or min_chars + 40
    target = rng.randint(min_chars, max(min_chars, upper))
    parts: list[str] = []
    while sum(len(p) for p in parts) < target:
        parts.append(rng.choice(_PURE_FRAGMENTS))
    text = "".join(parts)
    return text[:target]


def _strip_strong_punct(text: str) -> str:
    for ch in "。！？，；：…":
        text = text.replace(ch, "")
    return text.replace("\n", "").replace("\t", "")


def _build_extreme_force(rng: random.Random) -> str:
    """Single long run with zero L1/L2/L3 until a final period (ref force at L_force)."""
    block = _pure_pad(rng, 170, max_chars=220)
    tail = _pure_pad(rng, rng.randint(40, 70))
    return block + tail + "结束。"


def _build_extreme_multi_force(rng: random.Random) -> str:
    """Two pure runs separated only by a terminal period → two ref force boundaries."""
    block1 = _pure_pad(rng, 170, max_chars=210)
    mid = _pure_pad(rng, rng.randint(45, 65)) + "第一段结束。"
    block2 = _pure_pad(rng, 165, max_chars=205)
    tail = _pure_pad(rng, rng.randint(35, 55))
    return block1 + mid + block2 + tail + "第二段结束。"


def _build_extreme_l2_gap(rng: random.Random) -> str:
    """Pure prefix (force/L1 path) + comma-dense suffix (online L2 early cut vs ref L1)."""
    prefix = _pure_pad(rng, 155, max_chars=195)
    suffix = _pad_clause(rng, rng.randint(70, 95), trailing="，")
    suffix += "随后继续补充更多细节与约束条件，"
    suffix += _pad_clause(rng, 15, trailing="。")
    return prefix + "此处为唯一强句末边界。" + suffix


def _build_extreme_l1_edge(rng: random.Random) -> str:
    """L1 period appears just after min L1 threshold on a pure prefix."""
    prefix = _pure_pad(rng, 118, max_chars=138)
    return prefix + "此处落句。" + _pad_clause(rng, rng.randint(55, 75), trailing="。")


def _build_long_no_punct(rng: random.Random) -> str:
    # Improved: prefer pure fragments; optional comma tail only in minority cases.
    body = _pure_pad(rng, rng.randint(120, 160))
    if rng.random() < 0.3:
        tail = _pad_clause(rng, 35, trailing="。")
        return body + _strip_strong_punct(tail.replace("。", "")) + "结束全文。"
    return body + "并在末尾给出句号结束全文。"


def _build_long_dense_l2(rng: random.Random) -> str:
    first = _pad_clause(rng, rng.randint(68, 90), trailing="，")
    first += "随后补充更多细节，包括风险、排期与资源，"
    first += _pad_clause(rng, 12, trailing="。")
    second = _pad_clause(rng, rng.randint(55, 80), trailing="，")
    second += "最后比较三种在线切分策略的边界质量指标，"
    second += _pad_clause(rng, 10, trailing="。")
    return first + second


def _build_long_news(rng: random.Random) -> str:
    lead = (
        "据新华社报道，某省今日发布重要民生通知，强调高温天气下户外作业安全，"
        "并要求各地完善应急物资储备与公共纳凉点服务。"
    )
    body = _pad_clause(rng, rng.randint(55, 75), trailing="，")
    tail = (
        "气象部门提醒公众关注官方滚动预报，合理安排出行，"
        "避免在午后高温时段进行高强度活动，确保人身与财产安全。"
    )
    return lead + body + tail


def _build_mixed_lang(rng: random.Random) -> str:
    prefix = (
        f"根据NASA与ECMWF联合发布的报告，GPT-4o类模型在2026-Q3的性能评测显示，"
        f"latency P95约为{rng.randint(120, 260)}ms，"
    )
    body = _pad_clause(rng, rng.randint(60, 85), trailing="。")
    return prefix + body


def _build_quote_suffix(rng: random.Random) -> str:
    inner = rng.choice(("我已经尽力了", "请再给我一点时间", "数据还需要复核"))
    head = _pad_clause(rng, rng.randint(45, 65), trailing="，")
    return head + f"他低声说：“{inner}。”然后转身离开。"


def _build_question_exclaim(rng: random.Random) -> str:
    head = _pad_clause(rng, rng.randint(50, 70), trailing="？")
    mid = rng.choice(
        (
            "这太不可思议了！",
            "现场所有人都愣住了！",
            "谁也没想到会是这个结果！",
        )
    )
    tail = _pad_clause(rng, rng.randint(35, 55), trailing="。")
    return head + mid + tail


def _build_short_burst(rng: random.Random) -> str:
    snippets = ("你好。", "谢谢。", "不客气。", "请进。", "请坐。", "再见。")
    count = rng.randint(4, 7)
    return "".join(rng.choice(snippets) for _ in range(count))


def _build_numbers(rng: random.Random) -> str:
    order = rng.randint(202607010000, 202607012359)
    phone = f"400-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
    head = f"订单号{order}已确认，预计{rng.randint(2,5)}到{rng.randint(6,9)}个工作日送达，"
    tail = _pad_clause(rng, rng.randint(35, 50), trailing="。")
    return head + f"如有疑问请拨打{phone}。" + tail


def _build_ellipsis_l3(rng: random.Random) -> str:
    head = _pad_clause(rng, rng.randint(55, 75), trailing="……")
    tail = _pad_clause(rng, rng.randint(30, 45), trailing="。")
    return head + tail


_BUILDERS: dict[str, Callable[[random.Random], str]] = {
    "extreme_force": _build_extreme_force,
    "extreme_multi_force": _build_extreme_multi_force,
    "extreme_l2_gap": _build_extreme_l2_gap,
    "extreme_l1_edge": _build_extreme_l1_edge,
    "long_l1_pair": _build_long_l1_pair,
    "long_no_punct": _build_long_no_punct,
    "long_dense_l2": _build_long_dense_l2,
    "long_news": _build_long_news,
    "mixed_lang": _build_mixed_lang,
    "quote_suffix": _build_quote_suffix,
    "question_exclaim": _build_question_exclaim,
    "short_burst": _build_short_burst,
    "numbers": _build_numbers,
    "ellipsis_l3": _build_ellipsis_l3,
}


def build_sample(scenario: str, rng: random.Random) -> dict[str, str]:
    builder = _BUILDERS.get(scenario)
    if builder is None:
        raise ValueError(f"unknown scenario: {scenario}")
    return {
        "scenario_tags": scenario,
        "full_text": builder(rng),
    }


def iter_quota_plan(
    total: int,
    *,
    seed: int = 42,
    quotas: dict[str, int] | None = None,
) -> list[str]:
    """Return scenario name per sample, quotas scaled to ``total``."""
    quota_map = quotas or SCENARIO_QUOTAS
    base = sum(quota_map.values())
    if total <= 0:
        return []

    rng = random.Random(seed)
    scenarios: list[str] = []
    for name, quota in quota_map.items():
        count = max(1, round(total * quota / base)) if total >= len(quota_map) else 0
        scenarios.extend([name] * count)

    while len(scenarios) < total:
        scenarios.append(rng.choice(list(quota_map.keys())))
    scenarios = scenarios[:total]
    rng.shuffle(scenarios)
    return scenarios


def generate_samples(
    count: int,
    *,
    seed: int = 42,
    profile: str = "full",
    sample_id_prefix: str | None = None,
) -> list[dict[str, str]]:
    if profile not in QUOTA_PROFILES:
        raise ValueError(f"unknown profile {profile!r}, expected one of {list(QUOTA_PROFILES)}")
    quotas = QUOTA_PROFILES[profile]
    prefix = sample_id_prefix or ("boundary_adv" if profile == "advantage" else "boundary")

    rng = random.Random(seed)
    plan = iter_quota_plan(count, seed=seed, quotas=quotas)
    samples: list[dict[str, str]] = []
    for idx, scenario in enumerate(plan, start=1):
        item = build_sample(scenario, rng)
        item["sample_id"] = f"{prefix}_{idx:04d}"
        item["dataset_profile"] = profile
        samples.append(item)
    return samples
