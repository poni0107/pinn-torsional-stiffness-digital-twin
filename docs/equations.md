# Governing equations and weak formulation

## Two-inertia motor-load model

Let $\theta_m$ and $\theta_l$ be motor and load angular positions, with
$\omega_m=\dot{\theta}_m$ and $\omega_l=\dot{\theta}_l$. The model used for the
simulation study is

```math
\dot{\theta}_m=\omega_m,
\qquad
\dot{\theta}_l=\omega_l,
```

```math
\begin{aligned}
J_m\dot{\omega}_m
  &= M_{\mathrm{em}}(t)
     - b_v\bigl(\omega_m-\omega_l\bigr)
     - k(t)\bigl(\theta_m-\theta_l\bigr), \\
J_l\dot{\omega}_l
  &= b_v\bigl(\omega_m-\omega_l\bigr)
     + k(t)\bigl(\theta_m-\theta_l\bigr).
\end{aligned}
```

$J_m$, $J_l$, and $b_v$ are treated as known. The formulation does not include an
independently varying external load torque.

## Relative first-order system

Define

```math
\delta=\theta_m-\theta_l,
\qquad
v_\delta=\omega_m-\omega_l,
\qquad
\Gamma=\frac{1}{J_m}+\frac{1}{J_l}.
```

Then

```math
\dot{\delta}=v_\delta,
\qquad
\dot{v}_\delta+b_v\Gamma v_\delta+k(t)\Gamma\delta
-\frac{M_{\mathrm{em}}(t)}{J_m}=0.
```

`RelativeStateNet` approximates both $\delta$ and $v_\delta$; the proposed model
does not compute $\ddot{\delta}$.

## Pointwise kinematic residual

The state pretraining constraint is

```math
r_{\mathrm{kin}}(t)
=\frac{\mathrm{d}\hat{\delta}(t)}{\mathrm{d}t}
-\hat{v}_\delta(t).
```

Only this first derivative is obtained by automatic differentiation.

## Weak dynamic residual

Integrating the relative dynamic equation over $[t_a,t_b]$ gives

```math
\begin{aligned}
r_{\mathrm{dyn}}^{[a,b]}
={}&\hat{v}_\delta(t_b)-\hat{v}_\delta(t_a)
+b_v\Gamma\int_{t_a}^{t_b}\hat{v}_\delta(t)\,\mathrm{d}t \\
&+\Gamma\int_{t_a}^{t_b}
   \hat{k}(t)\hat{\delta}(t)\,\mathrm{d}t
-\frac{1}{J_m}\int_{t_a}^{t_b}M_{\mathrm{em}}(t)\,\mathrm{d}t.
\end{aligned}
```

The integral kinematic check is

```math
r_{\mathrm{kin}}^{[a,b]}
=\hat{\delta}(t_b)-\hat{\delta}(t_a)
-\int_{t_a}^{t_b}\hat{v}_\delta(t)\,\mathrm{d}t.
```

All integrals are trapezoidal. `torch.trapz` is used when gradients must flow
through network or stiffness outputs. The validated grid uses 101 points per
window, stride 25, and all valid overlapping windows.

## Monotone stiffness parameterization

```math
\hat{k}(t)
=k_{\mathrm{low}}
+\frac{k_{\mathrm{high}}-k_{\mathrm{low}}}
 {1+\exp\!\left(\dfrac{t-t_c}{w}\right)}.
```

Sigmoid transforms of unconstrained trainable parameters enforce

```math
210\le k_{\mathrm{low}}\le k_{\mathrm{high}}
\le367.5\ \mathrm{N\,m/rad},
\qquad
0\le t_c\le T,
\qquad
0.005T\le w\le0.25T.
```

The fixed neutral initialization is `k_high=330`, `k_low=270`,
`t_center=0.5*T`, and `width=0.10*T`. It is independent of the reference
degradation parameters.

## Training objective

State pretraining combines encoder-derived data loss, pointwise kinematic loss,
and initial-condition loss. The data term aggregates normalized relative-angle
and relative-speed discrepancies. Their separate evaluation in the code is an
implementation detail, not a change to the scientific objective. The final
weak identification stage adds the weak dynamic residual. In the sparse+dense
experiment, labelled losses are evaluated only at 751 sensor times while the
kinematic and weak residuals use all 1501 collocation times.

The staged objective can be written compactly as

```math
\mathcal{L}_{\mathrm{total}}
=\lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}}
+\lambda_{\mathrm{kin}}\mathcal{L}_{\mathrm{kin}}
+\lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}
+\lambda_{\mathrm{IC}}\mathcal{L}_{\mathrm{IC}}.
```

Reference `k_true(t)` is not part of any objective term. It is read after model
selection to compute simulation metrics.
