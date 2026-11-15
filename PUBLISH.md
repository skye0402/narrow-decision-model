# First publication

The public destination exists at https://github.com/skye0402/narrow-decision-model.
This local repository has a fresh history. Source and release assets have not
yet been uploaded. Review README.md, MODEL_CARD.md, LICENSE and the selected
files, then run from this directory:

```sh
git push -u origin main
gh release create v0.1.0-demo release-artifacts/*.tar.gz \
  --repo skye0402/narrow-decision-model \
  --title "CPU decision model: first demo snapshot" \
  --notes-file RELEASE_NOTES.md
```

The model download in the README works after the release assets are uploaded.
The archives contain only the pinned demo model, exposed training/development
optimizer inputs and exact training source. The private final evaluation
cohort and personal voice/media credentials are excluded. A credential-pattern
scan passed, but review the outgoing files before public upload. The exact
training-source archive also contains historical synthetic-data generators;
review their effect on the reserved-cohort protocol before publishing that archive.

The MIT license covers new project work; THIRD_PARTY_NOTICES.md records upstream
components. The original research repository is unaffected by this new history.
