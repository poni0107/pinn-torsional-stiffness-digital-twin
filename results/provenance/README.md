# Result provenance

`result_provenance.json` is created by
`python scripts/build_public_results.py`. It maps each public quantitative
claim to its validated JSON/CSV source and records SHA-256 checksums of those
source artifacts.

The public tables are compact, portable extracts. Training checkpoints and the
non-redistributable `jera1.mat` file are intentionally not included. Their
absence means that inspecting and regenerating figures requires no private
data, while full model retraining requires an authorized local dataset copy.

No training, model selection, or metric adjustment occurs in the provenance
builder.
