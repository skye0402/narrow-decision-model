# Reused-component licenses

Checked 2 November 2026. The user accepts reuse of components under MIT, Apache 2.0, or comparable open licenses. Prefer permissive licenses and record the exact terms of each model artifact and library when adopting it. Noncommercial-only or bespoke restricted model weights are outside the selected approach.

The agreed architecture reuses a pretrained text encoder and implements/trains our decision head. Initial encoder candidate: Ettin-150M. Freezing versus fine-tuning the encoder remains an experiment. Reusing pretrained language representations is now agreed; training every weight from scratch is not required for this project.

| Component | License evidence | Project status |
|---|---|---|
| Ettin-encoder-150M checkpoint/tokenizer repository, revision `45d08642849e5c5701b162671ac811b7654bfd9f` | Upstream model card and revision metadata declare MIT. [Model card](https://huggingface.co/jhu-clsp/ettin-encoder-150m); [pinned source](https://huggingface.co/jhu-clsp/ettin-encoder-150m/blob/45d08642849e5c5701b162671ac811b7654bfd9f/README.md). | Downloaded, digest-verified, and CPU-profiled in the first encoder experiment. |
| Transformers 4.57.3 | [Apache 2.0 at the exact release tag](https://github.com/huggingface/transformers/blob/v4.57.3/LICENSE). | Used by the pinned tokenizer and CPU encoder experiments. |
| Tokenizers 0.22.2 | [Apache 2.0 at the exact release tag](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE). | Used by the pinned tokenizer and CPU encoder experiments. |
| PyTorch 2.9.1 | [BSD-style license at the exact release tag](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE), with redistribution, attribution, and non-endorsement conditions. | Selected for the first local CPU experiment; installed through the pinned uv benchmark group. GPU training remains a separate environment to validate. |
| Our decision head and application | New project work; outgoing project license remains to be selected before public release. | Application and shared head implemented; small development head-training check passed. Task training/calibration pending. |

[Machine-readable Ettin evidence](ettin-license-evidence.json) records the exact revision, declared license, source URLs, and model-card hash. The inspected snapshot declares MIT in metadata/model-card frontmatter and lists no separate LICENSE/NOTICE file. This records the upstream declaration rather than asserting a separate license document was present.

This is a record of the named components, not a completed audit of all transitive dependencies or future deployment bundles. Keep upstream license/attribution notices with redistributed artifacts and record any additional terms when pinning the actual runtime and packaging it. Training datasets and alternative checkpoints need their own license records; a paper citation or an open code repository does not establish the license of its released weights or data.
