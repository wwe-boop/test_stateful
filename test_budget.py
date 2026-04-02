ema = 5.0
remaining = 492
ta_ratio = 0.70

# FSM thresholds
phase_a_cap = remaining / (ema + 1.0)
a = phase_a_cap * ta_ratio
print(f"FSM a threshold (audio frames): {a}")

# Current budget calculation
budget = ta_ratio * remaining / (ema + 1.0 + ta_ratio)
expected_audio = budget * ema
print(f"Current budget (text tokens): {budget}")
print(f"Expected audio frames: {expected_audio}")

# Correct budget calculation
correct_budget = a / ema
print(f"Correct budget (text tokens): {correct_budget}")
