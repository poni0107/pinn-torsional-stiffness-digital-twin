# Development history and retained diagnostics

This file records why the final public model uses a weak first-order
formulation. It is not the primary method description.

## Second-order baseline

The initial inverse branch represented relative angle with a Fourier-feature
`DeltaNet`; automatic differentiation supplied relative speed and acceleration.
Although reference-equation loss landscapes were correct, network-derived
second derivatives shifted constant-stiffness minima and reduced inverse
accuracy. The direct time-varying baseline records 10.8668% relative stiffness
RMSE and stiffness R² 0.3191.

## First-order state model

`RelativeStateNet` introduced separate relative-angle and relative-speed
outputs and used only first derivatives for kinematic consistency. This passed
state approximation checks, but the pointwise differential stiffness landscape
remained shallow.

## Weak residual

Integrating over overlapping windows removed `d(v_delta)/dt`. Diagnostics with
51, 101, and 201 points identified 101-point windows with all valid overlaps as
the stable configuration. Constant-stiffness controls at 350, 300, and
245 N m/rad then produced relative errors below 0.35%.

## Time-varying bounded model

The final four-parameter monotone sigmoid represents healthy stiffness,
degraded stiffness, transition time, and width. It improved the clean
time-varying result to 2.8717% relative RMSE and stiffness R² 0.9524.

## Reduced sensor-supervision interpretation

Uniformly reducing both labels and physics grids to 121 or 401 points exposed
aliasing and under-resolution. The final sparse-supervision experiment instead
retains 1501 unlabeled physics points while reducing labels to 751. This
preserves stiffness accuracy but leaves relative-speed reconstruction weaker
than the full-rate case.

## Online timing history

An early single-run benchmark reported mean 13.781 ms and maximum 19.701 ms for
five Adam steps. The final controlled 3/4/5-step comparison uses ten
repetitions per configuration; the selected five-step case recorded maximum
6.8162 ms and zero exceedances in 290 updates.

The repeated comparison is the only source for final timing claims. A further
historical outlier was discussed during development, but no original
machine-readable artifact for it is present in the reviewed repository or
validated output archive. It is therefore not published as a repository
result.
