# Reproduction boundaries

## Frozen inference: self-contained with the model release asset

Install the pinned model archive using scripts/install_artifact.py, install
uv.lock dependencies, and run scripts/smoke_model.py or scripts/serve_demo.py.
All runtime model files, tokenizer, calibration, mock policy and source-hash
checks are supplied. No cloud account is needed. The archive retains its
original bundle identity rather than silently rebinding it to changed code.

## Training: prepared locally, not yet published

The following training archives are not part of the first public release.
The default model installer and CPU demo work without them.

### Source and exact final-stage optimizer inputs

The `training-inputs.tar.gz` release asset contains the final round's train and
development optimizer inputs, pilot configuration and original manifest.
Install with `python scripts/install_artifact.py --training` (or `--archive PATH`).
Inputs are AI-authored, synthetic and unreviewed. The original manifest refers
to the wider research procedure; only train/development optimizer inputs are
included. Reserved evaluation, calibration and threshold-selection rows are
not redistributed here. No final cohort is opened by these instructions.

The neural training implementation is in prism_model/teacher_transfer_training.py,
with scripts/train_teacher_transfer.py as the entrypoint. It takes explicit
--data-dir, --warm-dir, --teacher-dir, --output-dir, --source-root and
--source-manifest arguments; --seed 83 and --view-version 3 selected the
final continuation, consisting of two extraction-transfer and two readout epochs.
GPU dependency pins are in requirements/training.txt.

**This first public snapshot does not reproduce the entire multi-stage training
history from scratch.** Exact retraining additionally requires the earlier
warm-start and teacher artifacts and their source/audit manifests. Those
ancestors are not in this initial package. Do not substitute the final demo
bundle as a teacher and call it reproduction of the published training run.
The exact exported training source is supplied as `training-source.tar.gz`;
current source includes later inference fixes. Its checksum is in artifacts.json.

A future training release should add ancestor checkpoints, the complete fit-role
provenance and a documented fresh holdout procedure. The current package
supports running and inspecting the frozen candidate without those dependencies.
