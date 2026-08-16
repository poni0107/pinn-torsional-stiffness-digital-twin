# Validated experiments

No experiment in this document is recomputed during figure generation. Values
come from the artifacts mapped in `results/provenance/result_provenance.json`.

## Constant-stiffness controls

Three separate reference responses use constant stiffness values of 350, 300,
and 245 N m/rad. Each case has its own state pretraining and begins scalar
identification from 288.75 N m/rad. The estimated values are 351.2077,
300.2400, and 245.2341 N m/rad, corresponding to relative errors of 0.3451%,
0.0800%, and 0.0956%.

## Clean time-varying degradation

The main simulation uses 1501 sensor-labelled times and 1501 physics points.
The estimated degradation has relative stiffness RMSE 2.871706%, stiffness
R² 0.952446, initial error 4.912771%, and final error 0.296217%.

## Differential encoder noise

The 0.3% seeded differential-noise case retains 1501 labelled and physics
times. Relative stiffness RMSE is 2.903204%, stiffness R² is 0.951397, and the
initial/final errors are 4.922640% and 1.498450%.

## Sparse labels with dense physics

The final reduced-supervision experiment uses 751 uniformly spaced labelled
times and all 1501 physics-collocation times. Relative stiffness RMSE is
3.748286%, stiffness R² is 0.918983, and initial/final errors are 4.943563% and
0.011552%. Relative-angle R² is 0.924271; relative-speed R² is 0.889694.

This result supports reduced labelled supervision for stiffness identification,
but it does not demonstrate full-rate-quality reconstruction of every state.

## Second-order baseline

The retained free-profile baseline derives `delta_dot` and `delta_ddot` from a
single state output. It records relative stiffness RMSE 10.866806% and
stiffness R² 0.319050. Constant-stiffness loss-landscape diagnostics showed
that state-network second derivatives could displace the network-physics
minimum even when the oracle residual was correctly implemented.

## Sampling limitations

The 121-point uniform case has a 160 Hz effective sample rate, 80 Hz Nyquist
frequency, and only 0.69 samples per dominant torsional period. Its relative
stiffness RMSE is 28.3543%.

The 401-point uniform case has a 533.33 Hz sample rate and 266.67 Hz Nyquist
frequency, but only 2.31 samples per period. Relative stiffness RMSE remains
26.2186%. Nominal Nyquist satisfaction alone is insufficient for accurate
between-sample state reconstruction and trapezoidal relative-angle integration.

## Repeated causal online benchmark

Each of 3, 4, and 5 Adam steps per update was tested in ten repetitions, with
29 timed updates per repetition and 20 warm-up updates. At the 25 ms update
period, the selected five-step configuration records mean 2.540990 ms, median
2.272600 ms, P95 4.030915 ms, P99 6.464917 ms, maximum 6.816200 ms, and zero
deadline exceedances in 290 measurements.

Online relative stiffness RMSE is 6.726746%, stiffness R² is 0.739430, and
final stiffness error is 7.471356%. The 315 N m/rad threshold is triggered
83.896 ms before the corresponding reference crossing. The early trigger and
lower online R² must be considered alongside the timing result.

