# Train for narrow decisions. Run on a CPU.

A personal learning experiment: adapt a compact text encoder for a defined
step in a business process, then run the saved model locally. This example
checks whether a purchase request contains enough identifying information
for buyer review across six purchasing categories.

**Research/demo prototype.** Artificial policies, AI-authored labels, no independent
human label review. Ready means buyer review, not order approval or engineering suitability.

## Run the trained demo

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are needed. No GPU or model API
is required for inference. Download/setup needs internet; inference can be offline.

```sh
uv sync --frozen --group benchmark
uv run --frozen --group benchmark python scripts/install_artifact.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --group benchmark python scripts/serve_demo.py
```

Open http://127.0.0.1:8769. If occupied, add `--port 8770`.
The pinned model archive is available in the [v0.1.0-demo release](https://github.com/skye0402/narrow-decision-model/releases/tag/v0.1.0-demo).
For offline setup, use `python scripts/install_artifact.py --archive /path/to/prism-cpu-demo.tar.gz`.
Training archives listed in artifacts.json are prepared locally and are not
yet published; the quick-start model installer does not require them.
Archive sizes and SHA-256 values are in [artifacts.json](artifacts.json).
The installer checks the archive; the runtime checks the bundle and inference source.

## Understand and test it

Read [ARCHITECTURE.md](ARCHITECTURE.md), [MODEL_CARD.md](MODEL_CARD.md),
and [REPRODUCING.md](REPRODUCING.md).

```sh
uv run --frozen --group benchmark python -m unittest discover -s tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --frozen --group benchmark python scripts/smoke_model.py
```

The software tests cover comparison/provenance and demo boundaries. The smoke
script runs ten public examples through actual model inference. Neither is a
blind model-quality evaluation. The reserved final cohort remains private and unopened.

## Licenses

New code: [MIT](LICENSE). Adapted Ettin encoder: upstream MIT declaration.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component attribution.
Training data and weights are separate release assets, not Git history.
Personal narration, API credentials and cloud-account configuration are excluded.
