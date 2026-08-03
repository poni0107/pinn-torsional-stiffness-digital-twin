"""Shared paths and lightweight data validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MAT = DATA_DIR / "jera1.mat"


def load_jera1(path: Path = DEFAULT_MAT) -> dict[str, np.ndarray]:
    """Load the public interface of jera1.mat without treating THref as a label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. See data/README.md for local dataset placement."
        )
    raw = loadmat(path)
    missing = [name for name in ("t", "Mem") if name not in raw]
    if missing:
        raise KeyError(f"Missing required MAT variables: {', '.join(missing)}")
    data = {name: np.asarray(raw[name], dtype=float).reshape(-1) for name in ("t", "Mem")}
    if "THref" in raw:
        data["THref"] = np.asarray(raw["THref"], dtype=float).reshape(-1)
    if data["t"].size != data["Mem"].size:
        raise ValueError("t and Mem must have the same number of samples")
    if data["t"].size < 20 or not np.all(np.isfinite(data["t"])) or not np.all(np.isfinite(data["Mem"])):
        raise ValueError("t and Mem must contain at least 20 finite samples")
    if np.any(np.diff(data["t"]) <= 0):
        raise ValueError("t must be strictly increasing")
    return data


def sampling_metrics(t: np.ndarray) -> dict[str, float]:
    """Return nominal sampling diagnostics for an approximately uniform grid."""
    dt = float(np.median(np.diff(np.asarray(t, dtype=float))))
    rate = 1.0 / dt
    return {
        "duration_seconds": float(t[-1] - t[0]),
        "sample_count": int(len(t)),
        "effective_sample_rate_Hz": rate,
        "Nyquist_frequency_Hz": rate / 2.0,
        "maximum_sample_gap_seconds": float(np.max(np.diff(t))),
    }
