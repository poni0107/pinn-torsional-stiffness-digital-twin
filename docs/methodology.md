# Methodology

## Physical model

The simulated drivetrain contains a motor inertia `Jm`, load inertia `Jl`,
viscous shaft damping `bv`, and time-varying torsional stiffness `k(t)`. The
known input is the measured motor-torque profile `Mem(t)`. The governing
equations are

```text
Jm * theta_m_ddot + bv * (omega_m - omega_l)
  + k(t) * (theta_m - theta_l) = Mem(t)

Jl * theta_l_ddot - bv * (omega_m - omega_l)
  - k(t) * (theta_m - theta_l) = 0.
```

The ODE model produces synthetic encoder responses for simulation validation.
It does not turn those responses into claims of experimental measurement.

## Relative coordinates

The relative angle and speed are

```text
delta = theta_m - theta_l
v_delta = omega_m - omega_l.
```

Subtracting the two acceleration equations gives the relative dynamic model

```text
dv_delta/dt
  + bv * (1/Jm + 1/Jl) * v_delta
  + k(t) * (1/Jm + 1/Jl) * delta
  - Mem/Jm = 0.
```

## First-order RelativeStateNet

`RelativeStateNet(t)` returns two quantities: `delta_hat(t)` and
`v_delta_hat(t)`. Separate scales normalize both channels, and fixed Fourier
time features represent the oscillatory content. The differential kinematic
constraint is

```text
d(delta_hat)/dt - v_delta_hat = 0.
```

Unlike the retained second-order baseline, the proposed branch does not use
`delta_ddot`. This avoids the observed sensitivity of second-order automatic
differentiation to small state-approximation errors.

## Weak integral residual

For each overlapping window `[t_a, t_b]`, trapezoidal integration produces

```text
r_dynamic_weak = v_delta_hat(t_b) - v_delta_hat(t_a)
  + bv * (1/Jm + 1/Jl) * integral(v_delta_hat dt)
  + (1/Jm + 1/Jl) * integral(k(t) * delta_hat(t) dt)
  - (1/Jm) * integral(Mem(t) dt)

r_kinematic_weak = delta_hat(t_b) - delta_hat(t_a)
  - integral(v_delta_hat dt).
```

The final configuration uses 101 collocation points, stride 25, and all valid
overlapping windows. Integrals involving trainable quantities use
`torch.trapz`, so gradients propagate through the state and stiffness models.

## Sigmoid stiffness parameterization

The monotone degradation model is

```text
k(t) = k_low + (k_high - k_low)
       / (1 + exp((t - t_center) / width)).
```

Sigmoid transforms of unconstrained trainable variables enforce

```text
210 <= k_low <= k_high <= 367.5 Nm/rad
0 <= t_center <= T
0.005*T <= width <= 0.25*T.
```

Neutral initialization is fixed independently of the reference profile:
`k_high=330`, `k_low=270`, `t_center=0.5*T`, and `width=0.10*T`.

## Staged sparse supervision

The final reduced-data experiment separates sensor supervision from physics
collocation:

1. **Phase A:** 751 uniformly spaced encoder-derived labels supervise
   `delta_hat` and `v_delta_hat`; 1501 unlabeled points impose the kinematic
   residual and initial conditions.
2. **Phase B:** `RelativeStateNet` and the four sigmoid parameters are trained
   jointly. Sparse data loss remains restricted to 751 times, while kinematic
   and weak dynamic residuals use all 1501 collocation points.
3. The state network uses a smaller learning rate than the sigmoid parameters.
4. Checkpoints and restarts are selected only by total training loss.

Collocation points use only time, known `Mem(t)`, and physical residuals. They
do not contain true state or stiffness labels.

## Causal online benchmark

The online benchmark freezes the offline-pretrained `RelativeStateNet` and
updates only the four sigmoid stiffness parameters. At update `t_i`, a
101-sample sliding window contains samples no later than `t_i`; the previous
parameter and Adam states provide a warm start. Timing with
`time.perf_counter_ns()` surrounds only the optimizer update. Forward inference
and offline pretraining are reported separately.

The selected reported setting is stride 50 and five Adam steps per update. It
demonstrates causal near-real-time monitoring on the tested CPU, not hard
real-time operation.

## Evaluation policy

`k_true` and the true sigmoid parameters are excluded from all losses,
initialization, early stopping, checkpoint selection, and restart selection.
They enter only after training for RMSE, relative RMSE, R², endpoint-error, and
degradation metrics.
