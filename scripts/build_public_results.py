"""Build portable public tables and provenance from validated result artifacts.

This script performs no simulation, optimization, checkpoint selection, or
metric recomputation from hidden data. It only normalizes already validated
JSON/CSV values into a compact publication surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "experiment_metrics"
TABLES = ROOT / "results" / "tables"
PROVENANCE = ROOT / "results" / "provenance"

SOURCE_PATHS = [
    "results/experiment_metrics/main_metrics.json",
    "results/experiment_metrics/noise003_metrics.json",
    "results/experiment_metrics/sparse751_densephysics_metrics.json",
    "results/experiment_metrics/online_adam_steps_comparison.json",
    "results/experiment_metrics/online_detection_robustness.json",
    "results/tables/constant_stiffness_results.csv",
    "results/tables/table_2_constant_stiffness_validation.csv",
    "results/tables/final_sigmoid_results.csv",
    "results/tables/table_4_ablation_study.csv",
    "results/tables/online_latency_samples_steps345.csv",
    "results/tables/online_stiffness_series_steps345.csv",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sources() -> None:
    missing = [relative for relative in SOURCE_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing validated source artifacts: " + ", ".join(missing))


def build() -> list[Path]:
    require_sources()
    main = read_json(METRICS / "main_metrics.json")
    noise = read_json(METRICS / "noise003_metrics.json")
    sparse = read_json(METRICS / "sparse751_densephysics_metrics.json")
    final_rows = {
        row["experiment"]: row
        for row in read_csv(TABLES / "final_sigmoid_results.csv")
    }
    online = read_json(METRICS / "online_adam_steps_comparison.json")
    dominant_band = noise["sampling_diagnostic"]["dominant_torsional_band_Hz"]

    time_varying_rows = []
    for scenario, labels, collocation, noise_description, metrics, source in (
        (
            "Full-rate clean",
            1501,
            1501,
            "none",
            main,
            "results/experiment_metrics/main_metrics.json",
        ),
        (
            "0.3% differential encoder noise",
            1501,
            1501,
            "0.3% of std(omega_m-omega_l), split between encoders",
            noise,
            "results/experiment_metrics/noise003_metrics.json",
        ),
        (
            "751 sensor labels + dense physics",
            751,
            1501,
            "none",
            sparse,
            "results/experiment_metrics/sparse751_densephysics_metrics.json",
        ),
    ):
        time_varying_rows.append(
            {
                "scenario": scenario,
                "sensor_labels": labels,
                "physics_collocation_points": collocation,
                "noise_description": noise_description,
                "k_relative_RMSE_percent": metrics["k_relative_RMSE_percent"],
                "k_R2": metrics["k_R2"],
                "initial_stiffness_error_percent": metrics.get(
                    "initial_stiffness_error_percent", metrics.get("initial_error_percent")
                ),
                "final_stiffness_error_percent": metrics.get(
                    "final_stiffness_error_percent", metrics.get("final_error_percent")
                ),
                "delta_R2": metrics["delta_R2"],
                "v_delta_R2": metrics["v_delta_R2"],
                "source_artifact": source,
            }
        )
    public_time_varying = TABLES / "time_varying_stiffness_summary.csv"
    write_csv(public_time_varying, time_varying_rows)

    sampling_rows = []
    for key, interpretation in (
        ("noise0_measurements121", "Aliased: Nyquist below the 228-233 Hz band"),
        ("noise0_measurements401", "Nominal Nyquist satisfied; only 2.31 samples per period"),
        (
            "noise0_measurements751_densephysics",
            "751 labels with the full 1501-point physics grid",
        ),
    ):
        row = final_rows[key]
        sampling_rows.append(
            {
                "scenario": key,
                "sensor_labels": row["measurements"],
                "physics_collocation_points": row["collocation_count"],
                "effective_sample_rate_Hz": row["effective_sample_rate_Hz"],
                "Nyquist_frequency_Hz": row["Nyquist_frequency_Hz"],
                "dominant_torsional_band_Hz": (
                    f"{dominant_band[0]:g}-{dominant_band[1]:g}"
                ),
                "samples_per_dominant_torsional_period": row[
                    "samples_per_dominant_torsional_period"
                ],
                "k_relative_RMSE_percent": row["k_relative_RMSE_percent"],
                "k_R2": row["k_R2"],
                "interpretation": interpretation,
                "source_artifact": "results/tables/final_sigmoid_results.csv",
                "dominant_band_source_artifact": (
                    "results/experiment_metrics/noise003_metrics.json"
                ),
            }
        )
    public_sampling = TABLES / "sparse_sampling_summary.csv"
    write_csv(public_sampling, sampling_rows)

    comparison_rows = []
    for row in read_csv(TABLES / "table_4_ablation_study.csv"):
        if row["Method"] in {"Second-order free profile", "Weak first-order sigmoid"}:
            comparison_rows.append(
                {
                    "method": row["Method"],
                    "physics_form": row["Physics form"],
                    "k_relative_RMSE_percent": row["Relative k-RMSE [%]"],
                    "k_R2": row["k R2"],
                    "metric_scope": row["Metric scope"],
                    "source_artifact": "results/tables/table_4_ablation_study.csv",
                }
            )
    public_comparison = TABLES / "method_comparison.csv"
    write_csv(public_comparison, comparison_rows)

    online_rows = []
    for configuration in online["configurations"]:
        online_rows.append(
            {
                "adam_steps_per_update": configuration["adam_steps_per_update"],
                "repetitions": configuration["repetitions"],
                "total_updates": configuration["total_updates"],
                "mean_latency_ms": configuration["mean_latency_ms"],
                "median_latency_ms": configuration["median_latency_ms"],
                "standard_deviation_latency_ms": configuration[
                    "standard_deviation_latency_ms"
                ],
                "p95_latency_ms": configuration["p95_latency_ms"],
                "p99_latency_ms": configuration["p99_latency_ms"],
                "maximum_latency_ms": configuration["maximum_latency_ms"],
                "deadline_miss_count": configuration["deadline_miss_count"],
                "online_k_relative_RMSE_percent": configuration[
                    "online_k_relative_RMSE_percent"
                ],
                "online_k_R2": configuration["online_k_R2"],
                "final_stiffness_error_percent": configuration[
                    "final_stiffness_error_percent"
                ],
                "selected": configuration["adam_steps_per_update"]
                == online["selected_adam_steps_per_update"],
                "source_artifact": "results/experiment_metrics/online_adam_steps_comparison.json",
            }
        )
    public_online = TABLES / "online_repeated_benchmark.csv"
    write_csv(public_online, online_rows)

    PROVENANCE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "scope": "portable extracts of validated simulation artifacts",
        "no_training_performed": True,
        "private_dataset_included": False,
        "manuscript_included": False,
        "true_stiffness_usage": "post-training simulation evaluation only",
        "source_artifacts": [
            {
                "path": relative,
                "sha256": sha256(ROOT / relative),
            }
            for relative in SOURCE_PATHS
        ],
        "published_claims": {
            "constant_stiffness_validation": [
                "results/tables/constant_stiffness_results.csv",
                "results/tables/table_2_constant_stiffness_validation.csv",
            ],
            "clean_time_varying_degradation": [
                "results/experiment_metrics/main_metrics.json"
            ],
            "noise_0p3_percent": [
                "results/experiment_metrics/noise003_metrics.json"
            ],
            "sparse751_dense_physics": [
                "results/experiment_metrics/sparse751_densephysics_metrics.json"
            ],
            "second_order_baseline": [
                "results/tables/table_4_ablation_study.csv"
            ],
            "sparse_sampling_limitations": [
                "results/tables/final_sigmoid_results.csv",
                "results/experiment_metrics/noise003_metrics.json",
            ],
            "causal_online_repeated_benchmark": [
                "results/experiment_metrics/online_adam_steps_comparison.json",
                "results/tables/online_latency_samples_steps345.csv",
                "results/tables/online_stiffness_series_steps345.csv",
            ],
        },
        "derived_public_tables": {
            "results/tables/time_varying_stiffness_summary.csv": [
                "results/experiment_metrics/main_metrics.json",
                "results/experiment_metrics/noise003_metrics.json",
                "results/experiment_metrics/sparse751_densephysics_metrics.json",
            ],
            "results/tables/sparse_sampling_summary.csv": [
                "results/tables/final_sigmoid_results.csv",
                "results/experiment_metrics/noise003_metrics.json",
            ],
            "results/tables/method_comparison.csv": [
                "results/tables/table_4_ablation_study.csv"
            ],
            "results/tables/online_repeated_benchmark.csv": [
                "results/experiment_metrics/online_adam_steps_comparison.json"
            ],
        },
    }
    provenance_path = PROVENANCE / "result_provenance.json"
    provenance_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [
        public_time_varying,
        public_sampling,
        public_comparison,
        public_online,
        provenance_path,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    require_sources()
    if arguments.check_only:
        print(f"Validated {len(SOURCE_PATHS)} public source artifacts.")
        return 0
    for output in build():
        print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
