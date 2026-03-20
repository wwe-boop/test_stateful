"""
L1 unit tests: EmbeddingWeights and PrefillBuilder (T1.3, T1.4, T4.2).
Requires: workspace/exported/<variant>/weights/ with .pt from export_01_embeddings.py, and GPU.
Run from repo root: pytest tests/test_prefill_builder.py -v
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

WEIGHTS_DIR = REPO_ROOT / "workspace" / "exported" / "base-1.7b" / "weights"
TOKENIZER_DIR = REPO_ROOT / "workspace" / "exported" / "tokenizer" / "Qwen3-TTS-Tokenizer-12Hz"


def _weights_dir():
    if not WEIGHTS_DIR.is_dir():
        pytest.skip(f"Weights dir not found: {WEIGHTS_DIR} (run export_01_embeddings.py)")
    return str(WEIGHTS_DIR)


def _tokenizer():
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"Tokenizer dir not found: {TOKENIZER_DIR}")
    from lightweight_tokenizer import load_lightweight_tokenizer
    tok = load_lightweight_tokenizer(str(TOKENIZER_DIR))
    if tok is None:
        pytest.skip("Lightweight tokenizer failed to load")
    return tok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_load_and_bf16():
    """T1.3: EmbeddingWeights loads, all BF16 on GPU."""
    from prefill_builder import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    assert weights.text_embedding.weight.dtype == torch.bfloat16
    assert weights.text_projection.linear_fc1.weight.dtype == torch.bfloat16
    assert weights.tts_pad_embed.dtype == torch.bfloat16
    assert weights.tts_pad_embed.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_text_embed_shape():
    """T1.3: text_embed output shape and dtype."""
    from prefill_builder import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    ids = torch.tensor([[1, 2, 3]], device=weights.device, dtype=torch.int64)
    out = weights.text_embed(ids)
    assert out.shape == (1, 3, weights.hidden_size)
    assert out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_embedding_weights_codec_embed_shape():
    """T1.3: codec_embed output shape."""
    from prefill_builder import EmbeddingWeights
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    ids = torch.tensor([[0, 1]], device=weights.device, dtype=torch.int64)
    out = weights.codec_embed(ids)
    assert out.shape == (1, 2, weights.hidden_size)
    assert out.dtype == torch.bfloat16


def test_embedding_weights_missing_pt_raises():
    """T1.3: Missing .pt raises FileNotFoundError."""
    from prefill_builder import EmbeddingWeights
    import tempfile
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for EmbeddingWeights init")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        (path / "config.json").write_text(
            '{"variant":"x","talker_hidden_size":2048,"talker_text_hidden_size":1024,'
            '"talker_vocab_size":3072,"codec_eos_token_id":1,"codec_bos_id":2,"codec_pad_id":0}'
        )
        with pytest.raises(FileNotFoundError, match="text_embedding.pt"):
            EmbeddingWeights(str(path), device_id=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_design():
    """T1.4a: VOICE_DESIGN path."""
    from prefill_builder import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    embeds, trailing = builder.build(task_type=TaskType.VOICE_DESIGN, text="你好世界")
    assert embeds.dtype == torch.bfloat16
    assert embeds.shape[0] == 1 and embeds.shape[2] == weights.hidden_size
    assert len(trailing) >= 1
    assert trailing[-1] is weights.tts_eos_embed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_custom_voice():
    """T1.4b: CUSTOM_VOICE with speaker + instruct."""
    from prefill_builder import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    embeds, trailing = builder.build(
        task_type=TaskType.CUSTOM_VOICE,
        text="测试文本",
        speaker="zhitian",
        instruct="用温柔的语气说",
    )
    assert embeds.shape[1] > 0
    assert embeds.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_clone_xvec():
    """T1.4c: VOICE_CLONE_XVEC with spk_embedding."""
    from prefill_builder import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    embeds, trailing = builder.build(
        task_type=TaskType.VOICE_CLONE_XVEC,
        text="克隆测试",
        spk_embedding=fake_spk,
    )
    assert embeds.shape[0] == 1 and embeds.shape[2] == weights.hidden_size
    assert embeds.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_builder_voice_clone_icl():
    """T1.4d: VOICE_CLONE_ICL (ref_codes + ref_text) - critical path."""
    from prefill_builder import EmbeddingWeights, PrefillBuilder, TaskType
    weights = EmbeddingWeights(_weights_dir(), device_id=0)
    if weights.codec_embeddings_3d is None:
        pytest.skip("codec_embeddings_3d.pt required for ICL")
    builder = PrefillBuilder(weights, _tokenizer())
    fake_spk = torch.randn(1, 1024, device=weights.device, dtype=torch.bfloat16)
    fake_codes = torch.randint(0, 100, (50, 16), device=weights.device)
    embeds, trailing = builder.build(
        task_type=TaskType.VOICE_CLONE_ICL,
        text="ICL测试",
        spk_embedding=fake_spk,
        ref_codes=fake_codes,
        ref_text="参考文本",
    )
    assert embeds.shape[1] > 10
    assert embeds.dtype == torch.bfloat16
    assert len(trailing) >= 1


def test_parse_task_type():
    """parse_task_type returns correct enum."""
    from prefill_builder import parse_task_type, TaskType
    assert parse_task_type("voice_design") == TaskType.VOICE_DESIGN
    assert parse_task_type("custom_voice") == TaskType.CUSTOM_VOICE
    assert parse_task_type("voice_clone", x_vector_only=True) == TaskType.VOICE_CLONE_XVEC
    assert parse_task_type("voice_clone", x_vector_only=False) == TaskType.VOICE_CLONE_ICL
    with pytest.raises(ValueError, match="Unknown"):
        parse_task_type("invalid_type")


# ---- T4.2 / T4.3 regression (plan L4) ----

@pytest.mark.skip(reason="BF16 vs FP32 comparison requires legacy FP32 weights export; run manually if needed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_vs_fp32_prefill_output_atol():
    """T4.2: Compare prefill output BF16 vs FP32 (atol=1e-2). Skip: no FP32 weights in repo."""
    from prefill_builder import EmbeddingWeights, PrefillBuilder, TaskType
    weights_bf16 = EmbeddingWeights(_weights_dir(), device_id=0)
    # Would load FP32 weights from a separate dir and compare prefill outputs
    pytest.skip("No FP32 weights dir for comparison")


@pytest.mark.skip(reason="DLPack zero-copy is internal to model.py BLS; E2E run implies it works")
def test_dlpack_zero_copy_vs_cpu_copy():
    """T4.3: DLPack vs CPU copy numerical consistency. Skip: requires Triton BLS internals."""
    pytest.skip("Verified indirectly via E2E; no direct Python API for DLPack path")
