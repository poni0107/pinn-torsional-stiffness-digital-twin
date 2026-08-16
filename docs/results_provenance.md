# Results provenance

## Policy

Public quantitative values are copied or deterministically normalized from the
validated JSON/CSV artifacts in `results/experiment_metrics` and
`results/tables`. Figure generation never reads checkpoints, private data, or
manuscript files and performs no training.

Run

```bash
python scripts/build_public_results.py
```

to regenerate normalized tables and
`results/provenance/result_provenance.json`. The manifest records SHA-256 hashes
for every source artifact and maps each public claim to its source.

## Claim mapping

| Public result | Canonical source |
|---|---|
| Constant 350/300/245 controls | estimates and errors from `results/tables/constant_stiffness_results.csv`; common initialization from `results/tables/table_2_constant_stiffness_validation.csv` |
| Clean degradation | `results/experiment_metrics/main_metrics.json` |
| 0.3% noise | `results/experiment_metrics/noise003_metrics.json` |
| Sparse751 + dense physics | `results/experiment_metrics/sparse751_densephysics_metrics.json` |
| Second-order baseline | `results/tables/table_4_ablation_study.csv` |
| 121/401 sampling limitations | `results/tables/final_sigmoid_results.csv`; dominant band from `results/experiment_metrics/noise003_metrics.json` |
| Repeated 3/4/5-step online benchmark | `results/experiment_metrics/online_adam_steps_comparison.json` |
| Individual latency samples | `results/tables/online_latency_samples_steps345.csv` |
| Online stiffness trajectory | `results/tables/online_stiffness_series_steps345.csv` |

## Historical timing artifact

`results/experiment_metrics/online_benchmark_summary.json` is retained as an
explicitly labelled historical single-run summary. It is superseded for final
timing claims by the repeated 3/4/5-step benchmark. An additional outlier was
mentioned in development notes, but no original machine-readable artifact for
that observation was found during the final audit; it is not published as
quantitative evidence.

## Scope of reproducibility

The committed result artifacts are sufficient to audit metrics and regenerate
all public figures. Full training reproduction additionally requires the
non-redistributed local MAT input and substantial CPU time. Raw checkpoints are
excluded from Git and are not required to inspect the published numerical
evidence.
