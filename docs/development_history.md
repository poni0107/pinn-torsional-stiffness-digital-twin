# Development history

This document records the technically relevant evolution of the final model.
It is not a collection of obsolete source files.

## 1. Forward two-mass model

The first stage established a reference ODE simulation driven by the real `t`
and `Mem` arrays from `jera1.mat`. The generated positions and encoder speeds
were explicitly classified as synthetic simulation sensor data.

## 2. DeltaNet and the second-order baseline

The initial inverse formulation used a Fourier-feature `DeltaNet` for relative
angle. Automatic differentiation produced `delta_dot` and `delta_ddot`, while
a time-dependent stiffness network minimized the second-order physical
residual. This branch remains documented as the free-profile baseline.

## 3. Second-derivative diagnosis

Constant-stiffness loss landscapes separated the ODE residual from network
approximation effects. Oracle landscapes had minima at the reference stiffness,
whereas frozen DeltaNet landscapes could have strongly displaced minima. The
dominant error source was the network-derived second derivative, not the sign
of the physical equation.

## 4. RelativeStateNet

The proposed first-order network was introduced with two outputs:
`delta_hat` and `v_delta_hat`. Fourier time features and channel-specific
normalization preserved oscillatory detail. A kinematic residual enforced
consistency between the two outputs using only first derivatives.

## 5. Weak integral physics

Although the differential first-order form passed state quality gates, its
constant-stiffness landscape was too flat. Integrating the dynamic equation over
overlapping windows removed `d(v_delta)/dt`. Window lengths 51, 101, and 201
were tested diagnostically. The 101-point/all-window configuration placed both
the network landscape and independent closed-form diagnostic near the true
constant stiffness.

## 6. Constant-stiffness validation

With a common initial value of 288.75 Nm/rad and no reference stiffness in the
training loss, the weak first-order formulation produced:

| True k [Nm/rad] | Estimated k [Nm/rad] | Relative error | Gate |
|---:|---:|---:|---|
| 350 | 351.2077 | 0.3451% | PASS |
| 300 | 300.2400 | 0.0800% | PASS |
| 245 | 245.2341 | 0.0956% | PASS |

## 7. Time-varying sigmoid model

The final stiffness model estimates `k_high`, `k_low`, transition center, and
transition width under physical bounds. Three deterministic restarts use the
same neutral physical initialization and are selected only by minimum training
weak loss.

The clean full-rate experiment reached relative stiffness RMSE 2.872% and
stiffness R² 0.9524. The 0.3% differential-noise experiment reached 2.903% and
0.9514, respectively.

## 8. Sparse sampling diagnostics

Uniform 121-point sampling aliases the 228-233 Hz torsional band. Uniform
401-point sampling nominally satisfies Nyquist but has about 2.31 samples per
period, which is insufficient for reliable between-sample reconstruction and
trapezoidal relative-angle integration. These failures are retained in the
final tables as limitations, not hidden or retrained.

The methodological final test uses 751 sensor labels and 1501 dense unlabeled
physics points. Stiffness identification passes (`k` relative RMSE 3.748%, `k`
R² 0.9190), while auxiliary relative-speed reconstruction is below threshold.
Its statuses are therefore stiffness PASS, state reconstruction PARTIAL, and
overall composite FAIL.

## 9. Causal online benchmark

A frozen offline RelativeStateNet supplies causal 101-point window states.
Warm-started sigmoid updates were benchmarked with multiple strides and Adam
step counts. The selected stride-50/five-step case met all 29 tested 25 ms
deadlines, with mean update latency 13.781 ms and online relative stiffness
RMSE 6.727%.

## 10. Final scientific limits

The work remains a simulation validation with a measured torque profile.
Online results are machine-specific and are not hard real-time guarantees.
Sparse sensor supervision improves stiffness identification, but it does not
fully validate every auxiliary state channel under the prescribed composite
gate.
