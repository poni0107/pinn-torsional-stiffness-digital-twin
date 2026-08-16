# Public figure catalogue

All figures are created with `scripts/generate_publication_figures.py` from
committed validated JSON/CSV artifacts. Each figure is exported as 600 dpi PNG
and vector PDF.

| Figure stem | Exact source | Deterministic display transformation |
|---|---|---|
| `architecture_weak_first_order` | `docs/equations.md`; modular model and weak-residual code | schematic only; no numerical data |
| `time_varying_stiffness` | `main_metrics.json`, `noise003_metrics.json`, `sparse751_densephysics_metrics.json` | evaluate saved fitted sigmoid parameters on a 1501-point display grid; reconstruct the evaluation-only reference from the saved true endpoints, center, and width |
| `constant_stiffness_validation` | `constant_stiffness_results.csv` | sort cases by reference stiffness; plot saved estimates and saved relative errors |
| `method_and_sampling_comparison` | `method_comparison.csv`, `sparse_sampling_summary.csv` | bar charts of stored relative RMSE and samples per dominant period |
| `offline_metrics_summary` | `time_varying_stiffness_summary.csv` | bar charts of stored relative RMSE and stiffness R² |
| `causal_online_monitoring` | `online_adam_steps_comparison.json`, `online_stiffness_series_steps345.csv`, `online_latency_samples_steps345.csv` | select the configuration declared in the JSON; step-plot saved causal estimates and box-plot every saved latency sample |

The stiffness curves are reconstructed from the saved fitted sigmoid parameters;
no new optimization is performed. The reference curve is displayed only for
post-training simulation evaluation.
