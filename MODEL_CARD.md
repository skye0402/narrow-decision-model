# Model card: demo candidate

Adapted Ettin-150M encoder and trained decision/evidence heads, composed with
explicit comparison code. Candidate artifact:
`82bd496705c18bd8355baf4555dc61d08e4010f94519bb95c1dbadcaa368f98a`.
Hybrid calibration identity:
`4df4339744a865859f4cf2d71741d3713866aced0537378ee174d82d5b851de0`.

## Intended use

Learning, inspection and local demos of narrow purchase-request routing.
Not production procurement, autonomous order placement, SKU selection or engineering advice.
Fictional policies and AI-authored, unreviewed labels. English request text.

## Results and limits

Exposed-100 regression: 25/34 genuinely Ready identified, zero incorrect Ready,
identification F1 0.706. Added wording/category challenge F1 0.519; combined
verification F1 0.84963 fails the 0.85 target. Threshold selection contains one
incorrect Ready among 693 predicted Ready. No blanket zero-error claim applies.
See research/2026-11-15-category-gate-results.json for the recorded evidence.
The category correction is demo-only. No model is promoted. The fresh reserved
cohort was not opened and is excluded from this release.

Warm native Apple Silicon Mac CPU medians were approximately 202–220 ms,
p95 269–319 ms for the model and policy path with two threads. This excludes
cold start, browser and network latency. Other hardware needs measurement;
this candidate has not repeated constrained-device qualification.

## Training provenance

Multi-stage synthetic and natural-wording transfer with adapted pretrained
representations, learned extraction and task heads; the current category gates
were fitted separately without changing the trained neural weights. See
REPRODUCING.md for the boundary between inference reproduction and training.
