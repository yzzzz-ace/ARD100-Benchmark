# Upload to GitHub

The folder is prepared as a small source repository. It intentionally contains
no Git history or remote configuration.

## Command-line upload

Create an empty GitHub repository named `ARD100-Benchmark`, then
run from this folder:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/yzzzz-ace/ARD100-Benchmark.git
git push -u origin main
```

Run `python scripts/verify_release.py` before `git add` and require a `PASS`
result.

## Checkpoint release

Do not commit `.pt` files. After pushing the source repository, create a GitHub
Release tagged `v1.0.0` and attach only the six selected best checkpoints after
confirming their distribution terms. The checksum table is in
`weights/README.md`.
