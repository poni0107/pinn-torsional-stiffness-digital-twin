# Reproducibility

## Environment

The validated environment used Python 3.12, PyTorch 2.5.1 CPU, NumPy 2.0.2,
SciPy 1.16.0, and Matplotlib 3.10.0. Create and activate a virtual environment,
then install the package:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows activation is `.venv\Scripts\activate`; POSIX activation is
`source .venv/bin/activate`.

## Lightweight verification

The following commands require no private data and perform no training:

```bash
python -m unittest discover -s tests -v
python scripts/build_public_results.py --check-only
python scripts/generate_publication_figures.py --check-only
```

Regenerate the portable tables and figures with

```bash
python scripts/build_public_results.py
python scripts/generate_publication_figures.py
```

The public builder maps validated artifacts without changing stored metrics.

## Dataset check

Place an authorized copy at `data/jera1.mat`, then run

```bash
pinn-torsional-twin check-data --mat data/jera1.mat
```

The expected record contains 1501 valid samples over 0.75 s. Only `t` and
`Mem` are consumed from the MAT file.

## Long experiment commands

Each TOML command delegates to the validated compatibility entry point and
writes ignored outputs under `outputs/`:

```bash
pinn-torsional-twin run --config configs/main_clean.toml --mat data/jera1.mat
pinn-torsional-twin run --config configs/noise_0p3pct.toml --mat data/jera1.mat
pinn-torsional-twin run --config configs/sparse751_dense_physics.toml --mat data/jera1.mat
```

Use `--dry-run` to inspect the translated legacy command without executing it:

```bash
pinn-torsional-twin run --config configs/main_clean.toml --mat data/jera1.mat --dry-run
```

The historical `.cmd` wrappers remain available for Windows, but the package
CLI is the cross-platform public interface.

## Online repeated benchmark

First reproduce the clean state checkpoint under
`outputs/first_order_weak_sigmoid_main`. The default paths then work directly:

```bash
python scripts/run_online_adam_steps_comparison.py
```

Alternative locations can be supplied with `MVM2026_MAT_FILE`,
`MVM2026_EXPERIMENT_ROOT`, and `MVM2026_RELATIVE_STATE_CHECKPOINT`. For example,
PowerShell uses

```powershell
$env:MVM2026_MAT_FILE = "data/jera1.mat"
$env:MVM2026_RELATIVE_STATE_CHECKPOINT = "outputs/first_order_weak_sigmoid_main/relative_state_pretrained_sigmoid.pt"
python scripts/run_online_adam_steps_comparison.py
```

The script keeps `RelativeStateNet` frozen, runs ten repetitions for 3, 4, and
5 Adam steps, and writes every `perf_counter_ns` latency sample.

The committed benchmark tables allow inspection and figure regeneration
without re-running the machine-specific timing experiment.

## Determinism and model selection

Python, NumPy, and PyTorch seeds are set by the experiment configuration.
Stiffness restarts use seeds 2026, 2027, and 2028. Checkpoints and restarts are
selected only by training loss. Evaluation stiffness is accessed after
selection.

Machine scheduling can still affect latency measurements. The repeated timing
dataset, rather than a single run, is therefore the final timing source.
