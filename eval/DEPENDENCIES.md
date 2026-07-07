# SteadyStream Eval Package - Dependencies

## Required Python Packages

Install with:
```bash
pip install -r requirements.txt
```

### Core Dependencies

- **numpy >= 1.24.0** - Array operations, statistics
- **librosa >= 0.10.0** - F0 estimation (pyin), audio analysis
- **soundfile >= 0.12.0** - WAV file I/O
- **scipy >= 1.10.0** - Statistical tests (Wilcoxon)

### Optional Dependencies

- **torch >= 2.0.0** - For sim_delta speaker embedding models (can skip if not using SIM Δ)
- **torchaudio >= 2.0.0** - Audio preprocessing for speaker models

## Installation

```bash
# Core evaluation metrics
pip install numpy librosa soundfile scipy

# Full suite including speaker similarity
pip install numpy librosa soundfile scipy torch torchaudio
```

## Minimal Test (no external deps)

To test the core logic without audio processing:
```python
# Test excess metrics computation only
from eval.excess_metrics import NaturalBoundaryReference

ref_data = {
    "comma": {"f0_p50": 2.0, "f0_p75": 3.0, "energy_p50": 2.5, "energy_p75": 4.0},
}
ref = NaturalBoundaryReference(ref_data)

excess_f0 = ref.get_excess_f0(5.0, "comma")  # 5.0 - 3.0 = 2.0
print(f"Excess F0: {excess_f0} st")
```

## Docker Alternative

If running in the existing Triton container, librosa/numpy should already be available.
Check with:
```bash
docker exec -it <container> python3 -c "import librosa; print(librosa.__version__)"
```
