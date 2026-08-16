# Input data

The training script expects a local MATLAB file at `data/jera1.mat` with the
following column vectors:

| Variable | Expected shape | Use |
|---|---:|---|
| `t` | 1501 x 1 | recorded time coordinate |
| `Mem` | 1501 x 1 | recorded electromagnetic motor-torque input |
| `THref` | 1501 x 1 | present in the source file but not used in the PINN loss |

Public redistribution permission for `jera1.mat` has not been confirmed. The
file is therefore excluded by `.gitignore`. Place an authorized local copy at
`data/jera1.mat` before running the scripts.

Only `t` and `Mem` are treated as data originating from the MAT file. The
encoder responses used in this simulation study (`theta_m`, `theta_l`,
`omega_m`, and `omega_l`) are synthetic sensor signals produced with the
reference two-mass ODE model. They must not be described as measured variables
from `jera1.mat`.
