"""Repeat the confirmed causal online benchmark without retraining.

The script records every real ``perf_counter_ns`` latency returned by the
existing benchmark implementation. Reference stiffness is loaded only after
all causal optimizer updates have completed, for evaluation and threshold
analysis.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(os.environ.get("MVM2026_EXPERIMENT_ROOT", REPO / "outputs"))
SOURCE = REPO / "src" / "mvm_pinn_jera1.py"
CHECKPOINT = Path(os.environ.get(
    "MVM2026_RELATIVE_STATE_CHECKPOINT",
    EXPERIMENT_ROOT / "first_order_weak_sigmoid_main" / "relative_state_pretrained_sigmoid.pt",
))
MAT_FILE = Path(os.environ.get("MVM2026_MAT_FILE", REPO / "data" / "jera1.mat"))
TABLES = REPO / "results" / "tables"
METRICS = REPO / "results" / "experiment_metrics"
REPETITIONS = 10
STRIDE = 50
ADAM_STEPS = 5
LEARNING_RATE = 5e-3
WARMUP_UPDATES = 20


def load_module():
    spec = importlib.util.spec_from_file_location("mvm_repeated_benchmark", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def r2(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = np.sum((reference - np.mean(reference)) ** 2)
    return float(1.0 - np.sum((reference - estimate) ** 2) / denominator)


def interpolate_crossing(t: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(values < threshold)
    if indices.size == 0:
        return None
    index = int(indices[0])
    if index == 0:
        return float(t[0])
    fraction = (threshold - values[index - 1]) / (values[index] - values[index - 1])
    return float(t[index - 1] + fraction * (t[index] - t[index - 1]))


def first_update_crossing(
    update_times: np.ndarray, values: np.ndarray, threshold: float, consecutive: int = 1
) -> float | None:
    below = values < threshold
    if consecutive == 1:
        indices = np.flatnonzero(below)
        return float(update_times[int(indices[0])]) if indices.size else None
    for end in range(consecutive - 1, len(values)):
        if np.all(below[end - consecutive + 1 : end + 1]):
            # Detection is declared only when the final confirming update arrives.
            return float(update_times[end])
    return None


def cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return name


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(10)
    module = load_module()
    module.seed_all(2026)
    t, mem = module.load_input(MAT_FILE)
    config = module.Config(mat_file=str(MAT_FILE))
    duration = float(t[-1])
    state_net = module.load_sigmoid_relative_state_checkpoint(CHECKPOINT, {"T": duration})
    for parameter in state_net.parameters():
        parameter.requires_grad_(False)
    state_net.eval()

    started = datetime.now().astimezone()
    wall_start = time.perf_counter_ns()
    runs = []
    rows = []
    for repetition in range(1, REPETITIONS + 1):
        # Re-seeding guarantees identical model/update initial conditions; measured
        # wall-clock latency is allowed to vary naturally across repetitions.
        module.seed_all(2026)
        run = module.execute_online_updates(
            state_net,
            t,
            mem,
            config,
            stride=STRIDE,
            adam_steps=ADAM_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_updates=WARMUP_UPDATES,
        )
        runs.append(run)
        for update_number, (index, latency, estimate) in enumerate(
            zip(run["update_indices"], run["latencies_ms"], run["estimated_k"]), start=1
        ):
            rows.append(
                {
                    "repetition": repetition,
                    "update_number": update_number,
                    "sample_index": int(index),
                    "update_time_s": float(t[index]),
                    "latency_ms": float(latency),
                    "estimated_k_Nm_per_rad": float(estimate),
                    "stride": STRIDE,
                    "adam_steps_per_update": ADAM_STEPS,
                    "window_length": 101,
                    "warmup_updates": WARMUP_UPDATES,
                }
            )
        print(
            f"repetition={repetition:02d}/{REPETITIONS} "
            f"updates={len(run['latencies_ms'])} mean_ms={np.mean(run['latencies_ms']):.6f} "
            f"max_ms={np.max(run['latencies_ms']):.6f}"
        )
    wall_end = time.perf_counter_ns()
    ended = datetime.now().astimezone()

    latencies = np.asarray([row["latency_ms"] for row in rows], dtype=float)
    sample_period_ms = float(np.median(np.diff(t)) * 1000.0)
    update_period_ms = STRIDE * sample_period_ms
    deadline_misses = int(np.sum(latencies > update_period_ms))
    estimate_stack = np.stack([run["estimated_k"] for run in runs])
    deterministic_spread = float(np.max(np.ptp(estimate_stack, axis=0)))
    update_indices = runs[0]["update_indices"]
    update_times = t[update_indices]
    online_estimate = estimate_stack[0]

    # Evaluation begins here, after all online update/timing runs.
    reference = module.true_k(t, duration, config)
    reference_updates = reference[update_indices]
    error = online_estimate - reference_updates
    online_rmse = float(np.sqrt(np.mean(error**2)))
    online_relative_rmse = float(100.0 * online_rmse / config.k0)
    online_r2 = r2(reference_updates, online_estimate)

    latency_csv = TABLES / "online_latency_samples.csv"
    with latency_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stiffness_csv = TABLES / "online_stiffness_series.csv"
    with stiffness_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "t", "k_true_evaluation_only", "k_online_causal"])
        writer.writerows(zip(update_indices, update_times, reference_updates, online_estimate))

    summary = {
        "benchmark_scope": "10 repetitions of the confirmed selected configuration; no retraining",
        "started_local": started.isoformat(),
        "ended_local": ended.isoformat(),
        "total_wall_time_seconds": float((wall_end - wall_start) / 1e9),
        "repetitions": REPETITIONS,
        "updates_per_repetition": int(len(update_indices)),
        "total_updates": int(latencies.size),
        "window_length": 101,
        "stride": STRIDE,
        "adam_steps_per_update": ADAM_STEPS,
        "warmup_updates_per_repetition": WARMUP_UPDATES,
        "learning_rate": LEARNING_RATE,
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "maximum_latency_ms": float(np.max(latencies)),
        "standard_deviation_latency_ms": float(np.std(latencies, ddof=1)),
        "update_period_ms": update_period_ms,
        "deadline_miss_count": deadline_misses,
        "deadline_miss_percent": float(100.0 * deadline_misses / latencies.size),
        "deterministic_estimate_max_spread": deterministic_spread,
        "online_k_RMSE": online_rmse,
        "online_k_relative_RMSE_percent": online_relative_rmse,
        "online_k_R2": online_r2,
        "timing_scope": "perf_counter_ns around Adam online update steps only; no synthetic samples",
        "hardware_software": {
            "CPU": cpu_name(),
            "OS": platform.platform(),
            "Python": platform.python_version(),
            "PyTorch": str(torch.__version__),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "logical_CPU_count": __import__("os").cpu_count(),
        },
    }
    (METRICS / "online_latency_repeated_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    robustness_rows = []
    for fraction in (0.95, 0.90, 0.85):
        threshold = config.k0 * fraction
        true_crossing = interpolate_crossing(t, reference, threshold)
        ordinary = first_update_crossing(update_times, online_estimate, threshold, consecutive=1)
        confirmed = first_update_crossing(update_times, online_estimate, threshold, consecutive=3)
        early_alarm_count = int(
            np.sum((update_times < true_crossing) & (online_estimate < threshold))
        ) if true_crossing is not None else 0
        robustness_rows.append(
            {
                "threshold_fraction": fraction,
                "threshold_Nm_per_rad": threshold,
                "true_crossing_time_s": true_crossing,
                "ordinary_crossing_time_s": ordinary,
                "ordinary_detection_delay_s": None if ordinary is None else ordinary - true_crossing,
                "three_consecutive_crossing_time_s": confirmed,
                "three_consecutive_detection_delay_s": None if confirmed is None else confirmed - true_crossing,
                "early_alarm_update_count": early_alarm_count,
                "detected_ordinary": ordinary is not None,
                "detected_three_consecutive": confirmed is not None,
            }
        )

    estimated_degradation = float(
        100.0 * (online_estimate[0] - online_estimate[-1]) / online_estimate[0]
    )
    true_degradation = float(100.0 * (config.k0 - config.k_final) / config.k0)
    detection = {
        "series_source": "results/tables/online_stiffness_series.csv",
        "series_unchanged_across_repetitions": deterministic_spread == 0.0,
        "threshold_results": robustness_rows,
        "online_initial_stiffness": float(online_estimate[0]),
        "online_final_stiffness": float(online_estimate[-1]),
        "estimated_degradation_percent": estimated_degradation,
        "true_degradation_percent": true_degradation,
        "final_degradation_error_percentage_points": abs(estimated_degradation - true_degradation),
        "final_stiffness_error_percent": float(
            100.0 * abs(online_estimate[-1] - reference_updates[-1]) / reference_updates[-1]
        ),
    }
    (METRICS / "online_detection_robustness.json").write_text(
        json.dumps(detection, indent=2), encoding="utf-8"
    )
    threshold_csv = TABLES / "online_detection_thresholds.csv"
    with threshold_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness_rows[0]))
        writer.writeheader()
        writer.writerows(robustness_rows)

    print(json.dumps(summary, indent=2))
    print(json.dumps(detection, indent=2))


if __name__ == "__main__":
    main()
