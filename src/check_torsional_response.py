"""Check whether the local sampling grid resolves the simulated torsional response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mvm_pinn_jera1 import Config, load_input, simulate, true_k
from utils import DEFAULT_MAT, OUTPUTS_DIR, sampling_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--outdir", type=Path, default=OUTPUTS_DIR / "data_checks")
    args = parser.parse_args()
    t, mem = load_input(args.mat)
    config = Config(str(args.mat), noise=0.0, measurements=len(t))
    duration = float(t[-1])
    reference = simulate(t, mem, config, lambda x: true_k(x, duration, config))
    dt = float(np.median(np.diff(t)))
    frequencies = np.fft.rfftfreq(len(t), dt)
    amplitude = np.abs(np.fft.rfft(reference["delta_dot"] - np.mean(reference["delta_dot"])))
    band = (frequencies >= 100.0) & (frequencies <= 400.0)
    dominant = float(frequencies[band][np.argmax(amplitude[band])])
    report = sampling_metrics(t)
    report.update({
        "dominant_torsional_frequency_Hz": dominant,
        "samples_per_dominant_period": report["effective_sample_rate_Hz"] / dominant,
        "dominant_frequency_below_Nyquist": dominant < report["Nyquist_frequency_Hz"],
    })
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "torsional_response_check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frequencies[band], amplitude[band])
    ax.axvline(dominant, color="tab:red", linestyle="--", label=f"dominant {dominant:.2f} Hz")
    ax.set(xlabel="frequency [Hz]", ylabel="FFT magnitude", title="Relative-speed torsional response")
    ax.grid(); ax.legend(); fig.tight_layout()
    fig.savefig(args.outdir / "torsional_response_check.png", dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
