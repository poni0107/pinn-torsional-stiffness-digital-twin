# Publication figures

Regenerate every PNG/PDF pair from validated public artifacts with

```bash
python scripts/build_public_results.py
python scripts/generate_publication_figures.py
```

The generator performs no training and does not read `jera1.mat`, checkpoints,
or manuscript files. Figure inputs and their scientific interpretation are
listed in `docs/figures_description.md`.
