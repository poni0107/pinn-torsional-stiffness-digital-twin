# Data provenance

## Redistributed data

No private MATLAB dataset, model checkpoint, manuscript, or generated DOCX/PDF
manuscript is distributed in this repository.

## Local input

`jera1.mat` is an authorized local input whose redistribution rights have not
been confirmed. It is excluded by `.gitignore`. The code reads:

| Channel | Origin | Use |
|---|---|---|
| `t` | MAT file | time coordinate |
| `Mem` | MAT file | electromagnetic motor-torque input |
| `THref` | MAT file, if present | deliberately unused by the PINN |

The validated local record has 1501 samples from 0 to 0.75 s.

## Synthetic reference channels

`theta_m`, `theta_l`, `omega_m`, and `omega_l` are generated with the
two-inertia reference ODE driven by `Mem(t)`. They are synthetic simulation
sensor responses, not channels measured in `jera1.mat`.

The reference stiffness profile is also simulated. It is available for
response generation and post-training evaluation because this is a controlled
simulation study. It is not supplied to network losses, initialization, early
stopping, restart selection, checkpoint selection, or online updates.

## Derived channels

`v_delta = omega_m-omega_l` is formed from the two encoder-speed channels.
`delta_integrated` is obtained by cumulative trapezoidal integration using the
actual selected measurement times and `delta(0)=0`. It is a derived training
target, not an additional sensor.

## Sparse labels and collocation

In the sparse751+dense-physics experiment, synthetic encoder-derived
labels are available only at 751 times. The 1501 physics-collocation points use
time, known torque, initial conditions, and governing residuals; they carry no
true state or stiffness labels.
