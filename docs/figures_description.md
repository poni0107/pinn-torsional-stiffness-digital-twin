# Final figure descriptions

The previous draft figures have been temporarily removed from the `main`
branch. Final publication figures will be produced only after the remaining
scientific and visual review is complete. The descriptions below are retained
as requirements for the future figure set; they do not indicate that the
corresponding image files are currently published in the repository.

## `final_stiffness_comparison.png`

- **Content:** reference degradation, full-rate clean estimate, 0.3% noise
  estimate, sparse751+dense-physics estimate, and the free-profile second-order
  baseline.
- **Claim tested:** the proposed weak sigmoid model identifies the main
  degradation trend more accurately than the retained free-profile baseline
  and remains robust to the tested noise level.
- **Reference/estimate:** dashed black is `k_true` (evaluation only); colored
  curves are model estimates; gray is the second-order baseline.
- **Use:** main paper and primary GitHub figure.

## `final_metrics_comparison.png`

- **Content:** relative stiffness RMSE, stiffness R², and endpoint errors for
  clean, noise, and sparse751+dense-physics experiments.
- **Claim tested:** quantitative comparison against the prescribed stiffness
  thresholds.
- **Reference/estimate:** horizontal dashed lines are thresholds; bars are
  post-training evaluation metrics.
- **Use:** paper results and GitHub documentation.

## `sparse_limitations_comparison.png`

- **Content:** failed 121- and 401-point stiffness estimates, sampling notes,
  relative-speed reconstruction, and relative-angle reconstruction.
- **Claim tested:** 121-point aliasing and 401-point under-resolution explain
  why uniform sparse-grid experiments fail.
- **Reference/estimate:** black curves are reference simulation values; red and
  green curves are sparse network estimates.
- **Use:** limitations analysis; it must not be presented as a successful main
  result.

## `online_stiffness_tracking.png`

- **Content:** causal online stiffness updates compared with the reference
  stiffness trajectory.
- **Claim tested:** the warm-started sliding-window update tracks degradation
  without future samples.
- **Reference/estimate:** dashed black is evaluation-only `k_true`; the step
  curve is the online estimate.
- **Use:** online benchmark section in the paper and GitHub documentation.

## `latency_distribution.png`

- **Content:** update-latency distributions for the tested stride and Adam-step
  configurations.
- **Claim tested:** measured update times relative to configuration cost and
  deadline feasibility.
- **Reference/estimate:** boxes summarize timed online optimizer updates; they
  exclude offline pretraining and ordinary forward inference.
- **Use:** online performance analysis.

## `latency_vs_accuracy.png`

- **Content:** mean online-update latency against causal stiffness RMSE.
- **Claim tested:** the latency/accuracy trade-off used to select a
  zero-deadline-miss configuration.
- **Reference/estimate:** each point is a tested stride/Adam-step pair; accuracy
  uses `k_true` only after online optimization.
- **Use:** online performance analysis, not hard real-time certification.
