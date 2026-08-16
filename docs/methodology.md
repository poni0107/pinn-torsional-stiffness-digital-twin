# Methodology

## Information flow

The estimator uses time, electromagnetic torque, motor encoder speed, load
encoder speed, and known `Jm`, `Jl`, and `bv`. Encoder speeds form
`v_delta = omega_m-omega_l`. Relative angle supervision is derived from those
speeds with cumulative trapezoidal integration and the known simulation initial
condition `delta(0)=0`; it is not an additional sensor.

## Reference simulation

Only `t` and `Mem` are loaded from `jera1.mat`. A high-accuracy DOP853
two-inertia simulation produces synthetic `theta_m`, `theta_l`, `omega_m`, and
`omega_l`. The evaluation reference degrades smoothly from 350 to 245 N m/rad.
The simulated encoder channels must not be described as experimental
measurements.

## RelativeStateNet

The state network has two physical outputs, `delta_hat` and `v_delta_hat`.
Separate channel scales prevent the larger velocity amplitude from dominating
the smaller relative angle. Fixed Fourier time features represent the
approximately 228-233 Hz torsional content. Multiplication by normalized time
imposes zero relative angle and relative speed at the initial condition.

## Staged optimization

1. Pretrain `RelativeStateNet` with state data, first-order kinematic physics,
   and initial conditions.
2. For the standard clean/noise experiment, freeze the state network and
   optimize only the four bounded stiffness parameters with weak physics.
3. For sparse751+dense physics, pretrain from only 751 labelled sensor times,
   then jointly fine-tune state and stiffness parameters. The state learning
   rate is `1e-5`; the stiffness learning rate is `5e-3`.
4. Use 101-point weak windows, stride 25, and all valid windows.
5. Select checkpoints and restarts solely by training loss.

All three deterministic stiffness restarts use seeds 2026, 2027, and 2028 and
the same neutral physical initialization. No evaluation label participates in
restart selection.

## Training objective

The public scientific notation follows the compact four-component objective
used in the manuscript:

```math
\mathcal{L}_{\mathrm{total}}
=\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}
+\lambda_{\mathrm{kin}}\mathcal{L}_{\mathrm{kin}}
+\lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}
+\lambda_{\mathrm{IC}}\mathcal{L}_{\mathrm{IC}}.
```

$\mathcal{L}_{\mathrm{data}}$ aggregates supervision from the encoder-derived
relative angle and relative speed. The code evaluates their normalized
contributions separately for scaling and diagnostics, but those contributions
remain an internal decomposition of the single data term. The other components
enforce kinematic consistency, weak integrated dynamics, and the known initial
condition, respectively. Stage-specific weights activate the terms required by
state pretraining, stiffness identification, or sparse joint refinement.

## Noise experiment

The 0.3% case adds seeded differential encoder noise. Its standard deviation is
`0.003*std(omega_m-omega_l)` and the perturbation is split as `+epsilon/2` and
`-epsilon/2` between motor and load speeds. It is not independent noise scaled
by the much larger absolute encoder speeds.

## Sparse supervision versus collocation

Sensor labels and physics points are different resources. The final reduced
supervision experiment uses 751 labelled times and 1501 unlabeled collocation
times. Collocation points contribute only time, known `Mem`, initial
conditions, and physical residuals; they contain no true state or stiffness
labels. No state checkpoint trained on 1501 labels is loaded.

Consequently, 751 instead of 1501 labels means a 49.97% reduction in sensor
supervision, not a corresponding reduction in total optimizer work or physics
collocation.

## Online update

The online benchmark freezes the state network and updates only the four
sigmoid stiffness parameters. At update `i`, the window is `[i-100,i]`; future
samples are unavailable. The preceding sigmoid and Adam states are warm starts.
At least 20 updates are executed before timing. `time.perf_counter_ns()` wraps
only optimizer update work, not offline pretraining or ordinary forward
inference.

The final public timing claim uses the repeated 3/4/5-step benchmark and not the
earlier single-run timing table.
