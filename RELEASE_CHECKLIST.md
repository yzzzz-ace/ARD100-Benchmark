# Release checklist

Do not make the repository public until every unchecked item is resolved.

- [x] Replace the author entry and repository URL in `CITATION.cff`.
- [x] Replace `YOUR_GITHUB_USERNAME` in `docs/UPLOAD_TO_GITHUB.md`.
- [x] Confirm the preferred public repository name.
- [ ] Confirm permission/terms for any checkpoint release assets.
- [ ] Run `python scripts/verify_release.py` and require `PASS`.
- [ ] Create tag `v1.0.0` after the final verification commit.
- [ ] Attach only the six selected `best.pt` checkpoints, not `last.pt` files
  or optimizer state.
- [ ] Optionally archive tag `v1.0.0` in Zenodo and insert its DOI in the paper.
- [ ] Replace the manuscript's public archival URL before submission.
