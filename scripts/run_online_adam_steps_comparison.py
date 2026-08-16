"""Compare 3, 4, and 5 Adam steps per causal online update.

RelativeStateNet is loaded once from the confirmed checkpoint and remains frozen.
Every latency value is measured with the existing perf_counter_ns instrumentation;
reference stiffness is introduced only after all causal update runs have finished.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import torch

import run_repeated_online_latency as base


ADAM_STEPS_LIST = (3, 4, 5)
ACCURACY_LIMITS = {
    "online_k_relative_RMSE_percent_max": 8.0,
    "online_k_R2_min": 0.70,
    "final_stiffness_error_percent_max": 10.0,
}


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base.TABLES.mkdir(parents=True, exist_ok=True)
    base.METRICS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(10)
    module = base.load_module()
    module.seed_all(2026)
    t, mem = module.load_input(base.MAT_FILE)
    config = module.Config(mat_file=str(base.MAT_FILE))
    duration = float(t[-1])
    state_net = module.load_sigmoid_relative_state_checkpoint(base.CHECKPOINT, {"T": duration})
    for parameter in state_net.parameters():
        parameter.requires_grad_(False)
    state_net.eval()

    started = datetime.now().astimezone()
    wall_start = time.perf_counter_ns()
    raw_runs = {}
    latency_rows = []
    for adam_steps in ADAM_STEPS_LIST:
        config_start = time.perf_counter_ns()
        runs = []
        for repetition in range(1, base.REPETITIONS + 1):
            module.seed_all(2026)
            run = module.execute_online_updates(
                state_net,
                t,
                mem,
                config,
                stride=base.STRIDE,
                adam_steps=adam_steps,
                learning_rate=base.LEARNING_RATE,
                warmup_updates=base.WARMUP_UPDATES,
            )
            runs.append(run)
            for update_number, (index, latency, estimate) in enumerate(
                zip(run["update_indices"], run["latencies_ms"], run["estimated_k"]), start=1
            ):
                latency_rows.append(
                    {
                        "adam_steps_per_update": adam_steps,
                        "repetition": repetition,
                        "update_number": update_number,
                        "sample_index": int(index),
                        "update_time_s": float(t[index]),
                        "latency_ms": float(latency),
                        "estimated_k_Nm_per_rad": float(estimate),
                        "stride": base.STRIDE,
                        "window_length": 101,
                        "warmup_updates": base.WARMUP_UPDATES,
                    }
                )
            print(
                f"steps={adam_steps} repetition={repetition:02d}/{base.REPETITIONS} "
                f"updates={len(run['latencies_ms'])} "
                f"mean_ms={np.mean(run['latencies_ms']):.6f} "
                f"max_ms={np.max(run['latencies_ms']):.6f}"
            )
        raw_runs[adam_steps] = {
            "runs": runs,
            "wall_time_seconds": float((time.perf_counter_ns() - config_start) / 1e9),
        }

    # Evaluation begins only after every causal optimizer/timing run is complete.
    reference = module.true_k(t, duration, config)
    sample_period_ms = float(np.median(np.diff(t)) * 1000.0)
    update_period_ms = base.STRIDE * sample_period_ms
    summaries = []
    series_rows = []
    per_step_detection = {}
    true_threshold = 0.90 * config.k0
    true_crossing = base.interpolate_crossing(t, reference, true_threshold)
    for adam_steps in ADAM_STEPS_LIST:
        runs = raw_runs[adam_steps]["runs"]
        update_indices = np.asarray(runs[0]["update_indices"], dtype=int)
        update_times = t[update_indices]
        estimate_stack = np.stack([run["estimated_k"] for run in runs])
        online_estimate = estimate_stack[0]
        reference_updates = reference[update_indices]
        error = online_estimate - reference_updates
        online_rmse = float(np.sqrt(np.mean(error**2)))
        relative_rmse = float(100.0 * online_rmse / config.k0)
        online_r2 = base.r2(reference_updates, online_estimate)
        final_error = float(100.0 * abs(online_estimate[-1] - reference_updates[-1]) / reference_updates[-1])
        latencies = np.asarray(
            [row["latency_ms"] for row in latency_rows if row["adam_steps_per_update"] == adam_steps],
            dtype=float,
        )
        misses = int(np.sum(latencies > update_period_ms))
        crossing = base.first_update_crossing(update_times, online_estimate, true_threshold, consecutive=1)
        offset = None if crossing is None else float(crossing - true_crossing)
        accuracy_acceptable = bool(
            relative_rmse <= ACCURACY_LIMITS["online_k_relative_RMSE_percent_max"]
            and online_r2 >= ACCURACY_LIMITS["online_k_R2_min"]
            and final_error <= ACCURACY_LIMITS["final_stiffness_error_percent_max"]
        )
        deadline_pass = bool(float(np.max(latencies)) < update_period_ms and misses == 0)
        summary = {
            "adam_steps_per_update": adam_steps,
            "repetitions": base.REPETITIONS,
            "updates_per_repetition": int(len(update_indices)),
            "total_updates": int(len(latencies)),
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "standard_deviation_latency_ms": float(np.std(latencies, ddof=1)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "maximum_latency_ms": float(np.max(latencies)),
            "update_period_ms": update_period_ms,
            "deadline_miss_count": misses,
            "deadline_miss_percent": float(100.0 * misses / len(latencies)),
            "online_k_RMSE": online_rmse,
            "online_k_relative_RMSE_percent": relative_rmse,
            "online_k_R2": online_r2,
            "final_stiffness_error_percent": final_error,
            "threshold_Nm_per_rad": true_threshold,
            "true_threshold_crossing_time_s": true_crossing,
            "estimated_threshold_crossing_time_s": crossing,
            "detection_offset_s": offset,
            "detection_offset_ms": None if offset is None else 1000.0 * offset,
            "deterministic_estimate_max_spread": float(np.max(np.ptp(estimate_stack, axis=0))),
            "accuracy_acceptable": accuracy_acceptable,
            "deadline_gate_pass": deadline_pass,
            "selection_eligible": bool(accuracy_acceptable and deadline_pass),
            "configuration_wall_time_seconds": raw_runs[adam_steps]["wall_time_seconds"],
        }
        summaries.append(summary)
        for index, time_value, truth, estimate in zip(update_indices, update_times, reference_updates, online_estimate):
            series_rows.append(
                {
                    "adam_steps_per_update": adam_steps,
                    "sample_index": int(index),
                    "t": float(time_value),
                    "k_true_evaluation_only": float(truth),
                    "k_online_causal": float(estimate),
                }
            )
        per_step_detection[adam_steps] = {
            "update_indices": update_indices,
            "update_times": update_times,
            "online_estimate": online_estimate,
            "reference_updates": reference_updates,
        }

    eligible = [item for item in summaries if item["selection_eligible"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (
                item["online_k_relative_RMSE_percent"],
                item["maximum_latency_ms"],
                item["adam_steps_per_update"],
            ),
        )
        selection_status = "deadline_and_accuracy_gate_pass"
    else:
        selected = min(
            summaries,
            key=lambda item: (
                item["deadline_miss_count"],
                item["online_k_relative_RMSE_percent"],
                item["maximum_latency_ms"],
            ),
        )
        selection_status = "no_configuration_passed_deadline_and_accuracy_gate"
    selected_steps = int(selected["adam_steps_per_update"])

    write_rows(base.TABLES / "online_latency_samples_steps345.csv", latency_rows)
    write_rows(base.TABLES / "online_adam_steps_comparison.csv", summaries)
    write_rows(base.TABLES / "online_stiffness_series_steps345.csv", series_rows)
    selected_latency_rows = [row for row in latency_rows if row["adam_steps_per_update"] == selected_steps]
    write_rows(base.TABLES / "online_latency_samples.csv", selected_latency_rows)
    selected_series = [
        {
            "sample_index": row["sample_index"],
            "t": row["t"],
            "k_true_evaluation_only": row["k_true_evaluation_only"],
            "k_online_causal": row["k_online_causal"],
        }
        for row in series_rows
        if row["adam_steps_per_update"] == selected_steps
    ]
    write_rows(base.TABLES / "online_stiffness_series.csv", selected_series)

    selected_data = per_step_detection[selected_steps]
    robustness_rows = []
    for fraction in (0.95, 0.90, 0.85):
        threshold = config.k0 * fraction
        true_cross = base.interpolate_crossing(t, reference, threshold)
        ordinary = base.first_update_crossing(selected_data["update_times"], selected_data["online_estimate"], threshold, consecutive=1)
        confirmed = base.first_update_crossing(selected_data["update_times"], selected_data["online_estimate"], threshold, consecutive=3)
        early_alarm_count = int(
            np.sum((selected_data["update_times"] < true_cross) & (selected_data["online_estimate"] < threshold))
        ) if true_cross is not None else 0
        robustness_rows.append(
            {
                "threshold_fraction": fraction,
                "threshold_Nm_per_rad": threshold,
                "true_crossing_time_s": true_cross,
                "ordinary_crossing_time_s": ordinary,
                "ordinary_detection_delay_s": None if ordinary is None else ordinary - true_cross,
                "three_consecutive_crossing_time_s": confirmed,
                "three_consecutive_detection_delay_s": None if confirmed is None else confirmed - true_cross,
                "early_alarm_update_count": early_alarm_count,
                "detected_ordinary": ordinary is not None,
                "detected_three_consecutive": confirmed is not None,
            }
        )
    selected_estimate = selected_data["online_estimate"]
    estimated_degradation = float(100.0 * (selected_estimate[0] - selected_estimate[-1]) / selected_estimate[0])
    true_degradation = float(100.0 * (config.k0 - config.k_final) / config.k0)
    detection = {
        "selected_adam_steps_per_update": selected_steps,
        "series_source": "results/tables/online_stiffness_series.csv",
        "threshold_results": robustness_rows,
        "online_initial_stiffness": float(selected_estimate[0]),
        "online_final_stiffness": float(selected_estimate[-1]),
        "estimated_degradation_percent": estimated_degradation,
        "true_degradation_percent": true_degradation,
        "final_degradation_error_percentage_points": abs(estimated_degradation - true_degradation),
        "final_stiffness_error_percent": selected["final_stiffness_error_percent"],
    }
    write_rows(base.TABLES / "online_detection_thresholds.csv", robustness_rows)
    (base.METRICS / "online_detection_robustness.json").write_text(
        json.dumps(detection, indent=2), encoding="utf-8"
    )

    ended = datetime.now().astimezone()
    total_wall = float((time.perf_counter_ns() - wall_start) / 1e9)
    comparison = {
        "benchmark_scope": "10 repetitions each for 3, 4, and 5 Adam steps per causal update; frozen RelativeStateNet; no retraining",
        "started_local": started.isoformat(),
        "ended_local": ended.isoformat(),
        "total_wall_time_seconds": total_wall,
        "window_length": 101,
        "stride": base.STRIDE,
        "warmup_updates_per_repetition": base.WARMUP_UPDATES,
        "learning_rate": base.LEARNING_RATE,
        "accuracy_limits": ACCURACY_LIMITS,
        "selection_rule": "among configurations with max latency below 25 ms, zero misses, and acceptable accuracy, choose minimum relative k-RMSE; otherwise choose the minimum-miss accuracy/latency compromise",
        "selection_status": selection_status,
        "selected_adam_steps_per_update": selected_steps,
        "selected_configuration": selected,
        "configurations": summaries,
        "claim": "causal near-real-time monitoring on the tested CPU",
        "real_time_pass_claim": False,
        "timing_scope": "perf_counter_ns around Adam online update steps only; no synthetic latency samples",
        "hardware_software": {
            "CPU": base.cpu_name(),
            "OS": platform.platform(),
            "Python": platform.python_version(),
            "PyTorch": str(torch.__version__),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "logical_CPU_count": os.cpu_count(),
        },
    }
    (base.METRICS / "online_adam_steps_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    selected_summary = dict(selected)
    selected_summary.update(
        {
            "benchmark_scope": "selected configuration from repeated 3/4/5 Adam-step comparison; frozen RelativeStateNet; no retraining",
            "started_local": started.isoformat(),
            "ended_local": ended.isoformat(),
            "total_wall_time_seconds_all_configurations": total_wall,
            "selection_status": selection_status,
            "claim": "causal near-real-time monitoring on the tested CPU",
            "real_time_pass_claim": False,
            "hardware_software": comparison["hardware_software"],
        }
    )
    (base.METRICS / "online_latency_repeated_summary.json").write_text(
        json.dumps(selected_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
