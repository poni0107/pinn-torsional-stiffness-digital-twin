# PINN Digital Twin for Torsional Stiffness Estimation

## Project overview

This repository was developed for **Digital Twin Design for Torsional
Stiffness Estimation in Motor-Load Oscillatory Systems Using Physics-Informed
Neural Networks**.

It provides a reproducible simulation study of a motor-load digital twin that
identifies a degrading torsional stiffness profile from encoder-speed
information and a measured motor-torque profile. The final proposed model uses
relative coordinates, a first-order state network, and weak integral physics.

## Scientific objective

The objective is to estimate the unknown time-varying shaft stiffness `k(t)`
without including the reference stiffness or its true sigmoid parameters in
the training loss, initialization, checkpoint selection, early stopping, or
restart selection. Reference stiffness is used only after training for
simulation evaluation.

## Main contribution

The proposed method combines:

- a Fourier-feature `RelativeStateNet` that predicts relative angle and speed;
- a first-order formulation that avoids unreliable second derivatives;
- weak, windowed integral residuals evaluated with the trapezoidal rule;
- a physically constrained monotone sigmoid representation of stiffness;
- staged sparse sensor supervision with dense unlabeled physics collocation;
- a causal sliding-window near-real-time benchmark.

## Data origin

The real time vector `t` and motor torque `Mem` are read from `jera1.mat`.
`THref` is intentionally excluded from the PINN loss. The angular positions and
encoder speeds are synthetic sensor responses produced with the reference ODE
model. Therefore, this is a simulation validation driven by a measured input
profile, not an experiment with measured encoder outputs.

Dataset redistribution permission has not been confirmed. See
[`data/README.md`](data/README.md) and place an authorized local copy at
`data/jera1.mat`.

## Physical two-mass model

The motor and load equations are

```text
Jm * theta_m_ddot + bv * (omega_m - omega_l)
    + k(t) * (theta_m - theta_l) = Mem(t)

Jl * theta_l_ddot - bv * (omega_m - omega_l)
    - k(t) * (theta_m - theta_l) = 0
```

The reference simulation uses a smooth degradation from approximately 350 to
245 Nm/rad.

## Relative first-order formulation

With `delta = theta_m - theta_l` and `v_delta = omega_m - omega_l`, the proposed
model uses

```text
r_kinematic = d(delta_hat)/dt - v_delta_hat

r_dynamic = d(v_delta_hat)/dt
    + bv * (1/Jm + 1/Jl) * v_delta_hat
    + k(t) * (1/Jm + 1/Jl) * delta_hat
    - Mem/Jm
```

`RelativeStateNet` outputs both `delta_hat` and `v_delta_hat`; no
`delta_ddot` is computed in the proposed branch.

## Weak physics-informed residual

For each causal or offline time window `[t_a, t_b]`, the dynamic residual is

```text
r_weak = v_delta_hat(t_b) - v_delta_hat(t_a)
    + bv * (1/Jm + 1/Jl) * integral(v_delta_hat dt)
    + (1/Jm + 1/Jl) * integral(k(t) * delta_hat(t) dt)
    - (1/Jm) * integral(Mem(t) dt)
```

The integral kinematic residual is

```text
r_kinematic_weak = delta_hat(t_b) - delta_hat(t_a)
    - integral(v_delta_hat dt)
```

The final offline configuration uses 101 collocation points per window, stride
25, all valid overlapping windows, and `torch.trapz` integration.

## Repository structure

```text
data/       local MAT-file interface and data notes
src/        final implementation and diagnostic utilities
scripts/    reproducible Windows CMD entry points
results/    committed final tables, metrics, and figures
docs/       methodology, history, and figure descriptions
paper/      working-paper notice and PDF draft when available
outputs/    ignored training checkpoints and intermediate artifacts
```

## Installation

Python 3.12 was used for the reported benchmark.

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\check_data.py
```

The final environment used PyTorch 2.5.1 (CPU), NumPy 2.0.2, SciPy 1.16.0,
and Matplotlib 3.10.0.

## Reproduction commands

Run from the repository root:

```cmd
scripts\run_main.cmd
scripts\run_noise.cmd
scripts\run_sparse_densephysics.cmd
scripts\run_online_benchmark.cmd
```

These commands create ignored intermediate checkpoints under `outputs/`. The
committed files in `results/` are the preserved final results; long training is
not required merely to inspect them.

## Main results

| Scenario | Sensor labels | Physics points | Relative k-RMSE | k R² | Interpretation |
|---|---:|---:|---:|---:|---|
| Full-rate clean | 1501 | 1501 | 2.872% | 0.9524 | PASS |
| Noise 0.3% | 1501 | 1501 | 2.903% | 0.9514 | PASS |
| Sparse supervision + dense physics | 751 | 1501 | 3.748% | 0.9190 | Stiffness PASS; state reconstruction partial |
| Uniform sparse 401 | 401 | 401 | 26.219% | -2.9640 | FAIL |
| Uniform sparse 121 | 121 | 121 | 28.354% | -3.6361 | FAIL due to aliasing |

## Figures

Publication-quality figures are currently under final scientific and visual
review. Numerical results and reproducibility files are available in the
`results/tables` and `results/experiment_metrics` directories.

## Sparse sensor supervision

The final sparse-supervision experiment uses labels only at 751 uniformly
spaced sensor times and 1501 dense unlabeled physics collocation points. The
collocation grid uses known time and `Mem(t)` but no true state or stiffness
labels. No RelativeStateNet checkpoint trained on all 1501 sensor labels is
loaded.

The result must be reported transparently:

- `stiffness_identification_gate`: **PASS**;
- `state_reconstruction_gate`: **PARTIAL** (`v_delta R² = 0.88969`);
- `overall_composite_gate`: **FAIL**.

Thus the stiffness curve satisfies every prescribed identification threshold,
but the experiment is not presented as a full composite PASS.

## Online near-real-time benchmark

Selected causal configuration on the tested desktop CPU:

| Quantity | Value |
|---|---:|
| Stride | 50 samples |
| Adam steps/update | 5 |
| Update period | 25.000 ms |
| Mean latency | 13.781 ms |
| Median latency | 14.272 ms |
| p95 latency | 17.166 ms |
| Maximum latency | 19.701 ms |
| Missed deadlines | 0/29 |
| Online relative k-RMSE | 6.727% |
| Online k R² | 0.7394 |
| Final stiffness error | 7.471% |

**Causal near-real-time stiffness monitoring was demonstrated on the tested
desktop CPU.** This is not a hard real-time proof. `RelativeStateNet` was
pretrained offline on the full simulation trajectory, and online accuracy is
lower than the full offline estimate.

## Sampling limitations

Uniform 121-point sampling has a 160 Hz sample rate and an 80 Hz Nyquist
frequency, below the approximately 228-233 Hz dominant torsional band; it is an
aliasing failure. Uniform 401-point sampling satisfies the nominal Nyquist
condition but provides only about 2.31 samples per dominant period, leaving
relative-speed reconstruction and trapezoidal angle integration unreliable.

## Scientific limitations

- Only `t` and `Mem` originate from `jera1.mat`; encoder outputs are simulated.
- The reference stiffness is known only because this is a simulation study.
- Identifiability depends on the torque excitation and known inertial/damping parameters.
- The main and noise estimates place the initial stiffness near the imposed upper bound.
- Near-real-time measurements were obtained on one desktop CPU and do not establish worst-case hard real-time behavior.
- Sparse751+dense-physics stiffness identification passes, but the auxiliary relative-speed reconstruction threshold does not.

## Related work and reference implementation

The project was methodologically developed from the flexible motor-load example
and PINN code at [`imilos/pinn-motor`](https://github.com/imilos/pinn-motor)
and the related work *Inverse Modeling of Flexible Rotational Systems via
Hybrid Physics-Informed and Data-Driven Learning*.

The original work addresses inverse mapping toward the drive torque and uses an
RFF-LSTM surrogate. This repository instead estimates a time-varying torsional
stiffness and introduces relative coordinates, a first-order
`RelativeStateNet`, and a weak/integral physics residual. It also evaluates
degradation, noise, sparse supervision, and causal online updating. The
original implementation is credited as related work and is not presented as
an original contribution of this repository.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until the
conference record is finalized, cite the project as the accompanying MVM 2026
working paper.

## Authors

- Marijana Jeremić — University of Kragujevac, Serbia
- Lazar Krstić — University of Kragujevac, Serbia
- Miloš Ivanović — Faculty of Information Studies, Novo Mesto, Slovenia
- Mihailo Lazarević — University of Belgrade, Serbia
- Milan Matijević — University of Kragujevac, Serbia

## License

No permissive open-source license has been assigned automatically because the
reference `imilos/pinn-motor` repository does not expose a clear license and
third-party provenance must be confirmed. See [`LICENSE`](LICENSE). Author
confirmation is required before a formal licensed release.
