"""Generate public figures from committed validated result artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "experiment_metrics"
TABLES = ROOT / "results" / "tables"
DEFAULT_OUTPUT = ROOT / "results" / "figures"

REQUIRED = [
    METRICS / "main_metrics.json",
    METRICS / "noise003_metrics.json",
    METRICS / "sparse751_densephysics_metrics.json",
    METRICS / "online_adam_steps_comparison.json",
    TABLES / "constant_stiffness_results.csv",
    TABLES / "time_varying_stiffness_summary.csv",
    TABLES / "method_comparison.csv",
    TABLES / "sparse_sampling_summary.csv",
    TABLES / "online_latency_samples_steps345.csv",
    TABLES / "online_stiffness_series_steps345.csv",
]

COLORS = {
    "reference": "#111111",
    "clean": "#0072B2",
    "noise": "#D55E00",
    "sparse": "#009E73",
    "baseline": "#7A5195",
    "threshold": "#CC3311",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=600, pad_inches=0.03)
    figure.savefig(output / f"{name}.pdf", pad_inches=0.03)
    plt.close(figure)


def reference_stiffness(time: np.ndarray, metrics: dict) -> np.ndarray:
    high = metrics["true_k_start_evaluation_only"]
    low = metrics["true_k_final_evaluation_only"]
    center = metrics["true_transition_center_evaluation_only"]
    width = metrics["true_transition_width_evaluation_only"]
    sigmoid = 1 / (1 + np.exp(-(time - center) / width))
    start = 1 / (1 + np.exp(-(time[0] - center) / width))
    finish = 1 / (1 + np.exp(-(time[-1] - center) / width))
    progress = np.clip((sigmoid - start) / (finish - start), 0, 1)
    return high + (low - high) * progress


def estimated_stiffness(time: np.ndarray, metrics: dict) -> np.ndarray:
    high = metrics["estimated_k_high"]
    low = metrics["estimated_k_low"]
    center = metrics["estimated_transition_center"]
    width = metrics["estimated_transition_width"]
    return low + (high - low) / (1 + np.exp((time - center) / width))


def architecture_figure(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    def box(x, y, width, height, text, color, fontsize=8):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#444444", linewidth=0.8,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

    top = [
        (0.2, "Sensor inputs\n$t, M_{em}, \\omega_m, \\omega_l$", "#DCEAF7"),
        (3.1, "Relative quantities\n$v_\\delta, \\delta_{int}$", "#E8F3E8"),
        (6.0, "RelativeStateNet\n$\\hat\\delta, \\hat v_\\delta$", "#F4E8F7"),
        (8.9, "Monotone stiffness\n$\\hat k(t)$", "#FBE7D6"),
    ]
    for x, text, color in top:
        box(x, 4.2, 2.4, 1.0, text, color)
    for start in (2.6, 5.5, 8.4):
        axis.annotate("", xy=(start + 0.45, 4.7), xytext=(start, 4.7),
                      arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#444444"})

    losses = [(0.3, "$L_{data}$"), (2.5, "$L_{kin}$"), (4.7, "$L_{weak}$"), (6.9, "$L_{IC}$")]
    for x, text in losses:
        box(x, 1.15, 1.7, 0.75, text, "#F2F2F2")
    for x in (2.15, 4.35, 6.55):
        axis.text(x, 1.52, "+", ha="center", va="center", fontsize=11, color="#555555")
    axis.annotate("", xy=(9.15, 1.52), xytext=(8.6, 1.52),
                  arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#555555"})
    box(9.15, 1.05, 2.3, 0.95, "Total training loss", "#FFF3BF", fontsize=8)
    axis.text(0.25, 5.55, "State and parameter pathway", weight="bold", fontsize=9)
    axis.text(0.25, 2.45, "Physics-informed objective", weight="bold", fontsize=9)
    save_figure(figure, output, "architecture_weak_first_order")


def stiffness_figure(output: Path) -> None:
    time = np.linspace(0.0, 0.75, 1501)
    main_metrics = read_json(METRICS / "main_metrics.json")
    cases = [
        ("Clean", main_metrics, COLORS["clean"]),
        ("0.3% noise", read_json(METRICS / "noise003_metrics.json"), COLORS["noise"]),
        ("751 labels + dense physics", read_json(METRICS / "sparse751_densephysics_metrics.json"), COLORS["sparse"]),
    ]
    figure, axis = plt.subplots(figsize=(7.1, 3.5))
    axis.plot(
        time,
        reference_stiffness(time, main_metrics),
        "--",
        color=COLORS["reference"],
        lw=2.0,
        label="Reference",
    )
    for label, metrics, color in cases:
        axis.plot(time, estimated_stiffness(time, metrics), color=color, lw=1.6, label=label)
    axis.set(xlabel="Time [s]", ylabel="Torsional stiffness [N m/rad]", xlim=(0, 0.75))
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    save_figure(figure, output, "time_varying_stiffness")


def constant_figure(output: Path) -> None:
    rows = read_csv(TABLES / "constant_stiffness_results.csv")
    true_values = np.asarray([float(row["true_k"]) for row in rows])
    estimates = np.asarray([float(row["estimated_k"]) for row in rows])
    errors = np.asarray([float(row["relative_error_percent"]) for row in rows])
    order = np.argsort(true_values)
    true_values, estimates, errors = true_values[order], estimates[order], errors[order]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    axes[0].plot([230, 365], [230, 365], "--", color="#777777", lw=1.0, label="Ideal")
    axes[0].scatter(true_values, estimates, s=45, color=COLORS["clean"], zorder=3, label="Estimated")
    axes[0].set(xlabel="Reference stiffness [N m/rad]", ylabel="Estimated stiffness [N m/rad]")
    axes[0].legend(frameon=False)
    axes[1].bar([str(int(value)) for value in true_values], errors, color=COLORS["clean"], width=0.6)
    axes[1].set(xlabel="Reference stiffness [N m/rad]", ylabel="Relative error [%]", ylim=(0, 0.4))
    for index, value in enumerate(errors):
        axes[1].text(index, value + 0.012, f"{value:.4f}%", ha="center", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output, "constant_stiffness_validation")


def comparison_figure(output: Path) -> None:
    method_rows = read_csv(TABLES / "method_comparison.csv")
    sampling_rows = read_csv(TABLES / "sparse_sampling_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    methods = ["Second-order\nfree profile", "Weak first-order\nsigmoid"]
    rmse = [float(row["k_relative_RMSE_percent"]) for row in method_rows]
    axes[0].bar(methods, rmse, color=[COLORS["baseline"], COLORS["clean"]], width=0.58)
    axes[0].set(ylabel="Relative stiffness RMSE [%]")
    for index, value in enumerate(rmse):
        axes[0].text(index, value + 0.25, f"{value:.4f}%", ha="center", fontsize=8)

    labels = ["121", "401", "751 + dense\nphysics"]
    samples = [float(row["samples_per_dominant_torsional_period"]) for row in sampling_rows]
    colors = ["#A6CEE3", "#5AA5D1", "#1F78B4"]
    axes[1].bar(labels, samples, color=colors, width=0.58)
    axes[1].axhline(
        2.0,
        color="#666666",
        ls="--",
        lw=1.0,
        label="Nominal Nyquist lower bound",
    )
    axes[1].set(ylabel="Samples per dominant torsional period")
    axes[1].legend(frameon=False, loc="upper left")
    for index, value in enumerate(samples):
        axes[1].text(index, value + 0.08, f"{value:.2f}", ha="center", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output, "method_and_sampling_comparison")


def offline_metrics_figure(output: Path) -> None:
    rows = read_csv(TABLES / "time_varying_stiffness_summary.csv")
    labels = ["Clean", "0.3% noise", "751 labels +\ndense physics"]
    rmse = np.asarray([float(row["k_relative_RMSE_percent"]) for row in rows])
    r2 = np.asarray([float(row["k_R2"]) for row in rows])
    colors = [COLORS["clean"], COLORS["noise"], COLORS["sparse"]]
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].bar(labels, rmse, color=colors, width=0.6)
    axes[0].set(ylabel="Relative stiffness RMSE [%]", ylim=(0, 5))
    axes[1].bar(labels, r2, color=colors, width=0.6)
    axes[1].set(ylabel="$R^2$ for stiffness", ylim=(0.85, 1.0))
    for axis, values, format_string in ((axes[0], rmse, "{:.4f}%"), (axes[1], r2, "{:.4f}")):
        for index, value in enumerate(values):
            axis.text(index, value + (0.08 if axis is axes[0] else 0.003), format_string.format(value), ha="center", fontsize=8)
    figure.tight_layout()
    save_figure(figure, output, "offline_metrics_summary")


def online_figure(output: Path) -> None:
    comparison = read_json(METRICS / "online_adam_steps_comparison.json")
    selected = int(comparison["selected_adam_steps_per_update"])
    series = [
        row for row in read_csv(TABLES / "online_stiffness_series_steps345.csv")
        if int(row["adam_steps_per_update"]) == selected
    ]
    time = np.asarray([float(row["t"]) for row in series])
    truth = np.asarray([float(row["k_true_evaluation_only"]) for row in series])
    estimate = np.asarray([float(row["k_online_causal"]) for row in series])
    latency_rows = read_csv(TABLES / "online_latency_samples_steps345.csv")
    grouped = [
        [float(row["latency_ms"]) for row in latency_rows if int(row["adam_steps_per_update"]) == steps]
        for steps in (3, 4, 5)
    ]
    selected_metrics = comparison["selected_configuration"]

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.0))
    axes[0].plot(time, truth, "--", color=COLORS["reference"], lw=1.8, label="Reference at update times")
    axes[0].step(time, estimate, where="post", color=COLORS["clean"], lw=1.6, label="Causal estimate")
    axes[0].axhline(315.0, color=COLORS["threshold"], ls=":", lw=1.2, label="315 N m/rad threshold")
    axes[0].scatter([selected_metrics["true_threshold_crossing_time_s"]], [315.0], marker="o", color=COLORS["reference"], s=30, zorder=4)
    axes[0].scatter([selected_metrics["estimated_threshold_crossing_time_s"]], [315.0], marker="s", color=COLORS["clean"], s=30, zorder=4)
    axes[0].set(xlabel="Time [s]", ylabel="Torsional stiffness [N m/rad]", xlim=(0, 0.75))
    axes[0].legend(ncol=2, frameon=False)

    box = axes[1].boxplot(grouped, tick_labels=["3", "4", "5"], patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], ["#88CCEE", "#CCBB44", "#EE7733"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[1].axhline(25.0, color=COLORS["threshold"], ls="--", lw=1.2, label="25 ms target")
    axes[1].set(xlabel="Adam steps per update", ylabel="Measured update latency [ms]")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output, "causal_online_monitoring")


def require_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing figure inputs: " + ", ".join(missing))
    for path in REQUIRED:
        if path.suffix == ".json":
            read_json(path)
        elif path.suffix == ".csv":
            rows = read_csv(path)
            if not rows:
                raise ValueError(f"Empty figure input: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    require_inputs()
    if arguments.check_only:
        print(f"Validated {len(REQUIRED)} figure inputs.")
        return 0
    configure_style()
    architecture_figure(arguments.output_dir)
    stiffness_figure(arguments.output_dir)
    constant_figure(arguments.output_dir)
    comparison_figure(arguments.output_dir)
    offline_metrics_figure(arguments.output_dir)
    online_figure(arguments.output_dir)
    print(f"Generated 6 PNG and 6 PDF figures in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
