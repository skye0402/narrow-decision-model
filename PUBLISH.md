# First publication

The public destination exists at https://github.com/skye0402/narrow-decision-model.
This repository has a fresh history, separate from the private research archive.
The initial public release contains the frozen demo model. Training archives
remain local pending a separate training-data release review.
To update source or prepare subsequent releases, use this directory:

```sh
git push -u origin main
gh release create v0.1.0-demo release-artifacts/*.tar.gz \
  --repo skye0402/narrow-decision-model \
  --title "CPU decision model: first demo snapshot" \
  --notes-file RELEASE_NOTES.md
```

The README downloads the pinned demo-model release asset. Upload training
archives only as part of a separately reviewed training release.
The archives contain only the pinned demo model, exposed training/development
optimizer inputs and exact training source. The private final evaluation
cohort and personal voice/media credentials are excluded. A credential-pattern
scan passed, but review the outgoing files before public upload. The exact
training-source archive also contains historical synthetic-data generators;
review their effect on the reserved-cohort protocol before publishing that archive.

The MIT license covers new project work; THIRD_PARTY_NOTICES.md records upstream
components. The original research repository is unaffected by this new history.
