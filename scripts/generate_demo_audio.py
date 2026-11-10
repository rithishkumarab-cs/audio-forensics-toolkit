"""Generate a clearly synthetic signal for pipeline testing (not a speech/deepfake sample)."""
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    out = Path("demo_signal_with_discontinuity.wav")
    sr = 16000
    duration = 8
    t = np.arange(sr * duration, dtype=np.float32) / sr
    # A tone plus low-level noise, with a controlled silent interval to test the signal checker.
    x = 0.08 * np.sin(2 * np.pi * 220 * t) + 0.002 * np.random.default_rng(42).normal(size=t.size)
    x[int(3.2 * sr):int(3.55 * sr)] = 0
    sf.write(out, x, sr)
    print(f"Wrote {out.resolve()}")
    print("This is a synthetic test signal, not human speech and not a valid AI-voice detector test.")


if __name__ == "__main__":
    main()
