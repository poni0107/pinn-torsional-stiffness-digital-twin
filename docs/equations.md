# Governing equations and weak formulation

## Two-inertia motor-load model

Let `theta_m`, `theta_l` be motor and load angular positions, with
`omega_m = d(theta_m)/dt` and `omega_l = d(theta_l)/dt`. The model used for the
simulation study is

```text
Jm * d(omega_m)/dt = Mem(t)
  - bv*(omega_m-omega_l) - k(t)*(theta_m-theta_l)

Jl * d(omega_l)/dt =
    bv*(omega_m-omega_l) + k(t)*(theta_m-theta_l).
```

`Jm`, `Jl`, and `bv` are treated as known. The formulation does not include an
independently varying external load torque.

## Relative first-order system

Define

```text
delta   = theta_m - theta_l
v_delta = omega_m - omega_l
alpha   = 1/Jm + 1/Jl.
```

Then

```text
d(delta)/dt = v_delta

d(v_delta)/dt + bv*alpha*v_delta + k(t)*alpha*delta - Mem(t)/Jm = 0.
```

`RelativeStateNet` approximates both `delta` and `v_delta`; the proposed model
does not compute `d²(delta)/dt²`.

## Pointwise kinematic residual

The state pretraining constraint is

```text
r_kin(t) = d(delta_hat)/dt - v_delta_hat.
```

Only this first derivative is obtained by automatic differentiation.

## Weak dynamic residual

Integrating the relative dynamic equation over `[t_a,t_b]` gives

```text
r_weak = v_delta_hat(t_b) - v_delta_hat(t_a)
       + bv*alpha*integral(v_delta_hat dt)
       + alpha*integral(k_hat(t)*delta_hat(t) dt)
       - (1/Jm)*integral(Mem(t) dt).
```

The integral kinematic check is

```text
r_kin,weak = delta_hat(t_b) - delta_hat(t_a)
           - integral(v_delta_hat dt).
```

All integrals are trapezoidal. `torch.trapz` is used when gradients must flow
through network or stiffness outputs. The validated grid uses 101 points per
window, stride 25, and all valid overlapping windows.

## Monotone stiffness parameterization

```text
k_hat(t) = k_low + (k_high-k_low)
           / (1 + exp((t-t_center)/width)).
```

Sigmoid transforms of unconstrained trainable parameters enforce

```text
210 <= k_low <= k_high <= 367.5 N m/rad
0 <= t_center <= T
0.005*T <= width <= 0.25*T.
```

The fixed neutral initialization is `k_high=330`, `k_low=270`,
`t_center=0.5*T`, and `width=0.10*T`. It is independent of the reference
degradation parameters.

## Training objective

State pretraining combines normalized relative-angle data loss, relative-speed
data loss, pointwise kinematic loss, and initial-condition loss. The final weak
identification stage adds the weak dynamic residual. In the sparse+dense
experiment, labelled losses are evaluated only at 751 sensor times while the
kinematic and weak residuals use all 1501 collocation times.

Reference `k_true(t)` is not part of any objective term. It is read after model
selection to compute simulation metrics.

