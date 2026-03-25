import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_1 = REPO_ROOT / "model_repository" / "tts_orchestrator" / "1"
if str(ORCH_1) not in sys.path:
    sys.path.insert(0, str(ORCH_1))

from batch_decode_scheduler import (
    FusedDecodeTicket,
    group_by_c2w_past_len,
    pad_talker_past_kv,
    padded_attention_bias,
)


def test_group_by_c2w_past_len_partitions_tickets():
    tickets = [
        FusedDecodeTicket(payload="a", talker_past_len=10, c2w_past_len=4),
        FusedDecodeTicket(payload="b", talker_past_len=12, c2w_past_len=4),
        FusedDecodeTicket(payload="c", talker_past_len=8, c2w_past_len=7),
    ]

    groups = group_by_c2w_past_len(tickets)

    assert sorted(groups.keys()) == [4, 7]
    assert [ticket.payload for ticket in groups[4]] == ["a", "b"]
    assert [ticket.payload for ticket in groups[7]] == ["c"]


def test_pad_talker_past_kv_rectangularizes_rows():
    row0 = [
        torch.ones(1, 2, 2, 4, dtype=torch.float32),
        torch.full((1, 2, 2, 4), 10.0, dtype=torch.float32),
    ]
    row1 = [
        torch.full((1, 2, 4, 4), 2.0, dtype=torch.float32),
        torch.full((1, 2, 4, 4), 20.0, dtype=torch.float32),
    ]

    batched, past_seq_lens = pad_talker_past_kv(
        [row0, row1],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert past_seq_lens.tolist() == [2, 4]
    assert len(batched) == 2
    assert batched[0].shape == (2, 2, 4, 4)
    assert batched[1].shape == (2, 2, 4, 4)
    assert torch.allclose(batched[0][0, :, :2, :], row0[0][0])
    assert torch.count_nonzero(batched[0][0, :, 2:, :]) == 0
    assert torch.allclose(batched[1][1], row1[1][0])


def test_padded_attention_bias_masks_only_padded_past_columns():
    past_seq_lens = torch.tensor([2, 4], dtype=torch.long)

    bias = padded_attention_bias(
        past_seq_lens,
        seq=1,
        padded_past_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bias.shape == (2, 1, 1, 5)
    assert torch.isneginf(bias[0, 0, 0, 2:4]).all()
    assert torch.equal(bias[0, 0, 0, :2], torch.zeros(2))
    assert torch.equal(bias[1], torch.zeros_like(bias[1]))
