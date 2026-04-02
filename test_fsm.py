import sys
sys.path.append("model_repository/tts_orchestrator/1")
from decode_fsm import DecodeSessionFSM

fsm = DecodeSessionFSM(engine_max_decode_len=512, rollover_margin=64)
fsm.compute_thresholds(past_len=20, ema_ratio=5.0)
print(f"phase_a_cap: {fsm.thresholds.d}")
print(f"a: {fsm.thresholds.a}")
print(f"b: {fsm.thresholds.b}")
print(f"c: {fsm.thresholds.c}")
