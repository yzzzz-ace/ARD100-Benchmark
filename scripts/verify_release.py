#!/usr/bin/env python3
"""Fail if the public package contains private paths, secrets, or large data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".cff"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache"}
FORBIDDEN = (
    re.compile(r"[A-Za-z]:[\\/]Yzzzz", re.IGNORECASE),
    re.compile(r"C:[\\/]Users[\\/]", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
)
EXECUTABLE_SUFFIXES = {".py", ".ps1", ".yaml", ".yml", ".json"}
LEGACY_PROJECT = re.compile(r"LCA" + r"-DR", re.IGNORECASE)


def lines(relative: str) -> list[str]:
    return [line.strip() for line in (ROOT / relative).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-metadata-placeholders",
        action="store_true",
        help="permit author/repository placeholders while checking all technical content",
    )
    args = parser.parse_args()
    errors: list[str] = []
    required = (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "configs/test_lock.json",
        "manifests/locked_test_manifest.json",
        "results/primary/main_results.csv",
        "results/resolution/main_results.csv",
        "weights/README.md",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"required file missing: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_PARTS.intersection(path.parts):
            continue
        relative = path.relative_to(ROOT)
        raw = path.read_bytes()
        for label, needle in (
            ("private workspace path", b"F:\\Yzzzz"),
            ("private user path", b"C:\\Users"),
            ("private workspace path (UTF-16)", "F:\\Yzzzz".encode("utf-16-le")),
            ("private user path (UTF-16)", "C:\\Users".encode("utf-16-le")),
        ):
            if needle.lower() in raw.lower():
                errors.append(f"{label} embedded in file: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB: {relative}")
        if path.suffix.lower() in {".pt", ".pth", ".mp4", ".avi"}:
            errors.append(f"excluded binary type present: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON {relative}: {exc}")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                errors.append(f"forbidden pattern {pattern.pattern!r}: {relative}")
        if path.suffix.lower() in EXECUTABLE_SUFFIXES and LEGACY_PROJECT.search(text):
            errors.append(f"legacy project dependency in executable/config: {relative}")

    official_train = set(lines("splits/official_train_videos.txt"))
    official_test = set(lines("splits/official_test_videos.txt"))
    learned_train = set(lines("splits/learned_train_videos.txt"))
    learned_val = set(lines("splits/learned_val_videos.txt"))
    if tuple(map(len, (official_train, official_test, learned_train, learned_val))) != (65, 35, 52, 13):
        errors.append("video split counts are not 65/35 and 52/13")
    if official_train & official_test or learned_train & learned_val:
        errors.append("video split overlap detected")
    if learned_train | learned_val != official_train:
        errors.append("learned 52/13 split does not reconstruct the 65-video source partition")

    random_train_path = ROOT / "splits/random_frame_train_sample_ids.txt"
    random_val_path = ROOT / "splits/random_frame_val_sample_ids.txt"
    random_train = set(lines(str(random_train_path.relative_to(ROOT))))
    random_val = set(lines(str(random_val_path.relative_to(ROOT))))
    if (len(random_train), len(random_val)) != (35708, 8938):
        errors.append("random-frame split counts are not 35,708/8,938")
    if random_train & random_val:
        errors.append("random-frame sample IDs overlap")
    random_manifest = json.loads((ROOT / "manifests/random_frame_split_manifest.json").read_text(encoding="utf-8"))
    for key, path in (("train", random_train_path), ("val", random_val_path)):
        if random_manifest["files"][key]["sha256"] != sha256(path):
            errors.append(f"random-frame {key} checksum mismatch")

    for relative in ("results/primary/main_results.csv", "results/resolution/main_results.csv"):
        with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 6 or any(not row.get("model") for row in rows):
            errors.append(f"expected six model rows in {relative}")
    runtime_count = len(list((ROOT / "results/runtime").glob("*.json")))
    if runtime_count != 12:
        errors.append(f"expected 12 runtime records, found {runtime_count}")

    lock_path = ROOT / "configs/test_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked_manifest = json.loads((ROOT / "manifests/locked_test_manifest.json").read_text(encoding="utf-8"))
    if len(lock.get("entries", {})) != 12 or len(locked_manifest.get("entries", [])) != 12:
        errors.append("test lock or locked manifest does not contain 12 model-resolution entries")
    if locked_manifest.get("portable_test_lock_sha256") != sha256(lock_path):
        errors.append("portable test-lock checksum mismatch")
    locked_records = {record["output_id"]: record for record in locked_manifest.get("entries", [])}
    for identifier, entry in lock.get("entries", {}).items():
        record = locked_records.get(identifier)
        if record is None:
            errors.append(f"locked test record missing: {identifier}")
            continue
        if entry.get("weights_sha256") != record.get("weights_sha256"):
            errors.append(f"locked checkpoint digest mismatch: {identifier}")
        if float(entry.get("threshold", -1)) != float(record.get("threshold", -2)):
            errors.append(f"locked threshold mismatch: {identifier}")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "REPLACE_BEFORE_RELEASE" in citation and not args.allow_metadata_placeholders:
        errors.append("CITATION.cff still contains release metadata placeholders")
    if errors:
        print("RELEASE CHECK: FAIL")
        for error in sorted(set(errors)):
            print(f"- {error}")
        sys.exit(1)
    included = [path for path in ROOT.rglob("*") if path.is_file() and not SKIP_PARTS.intersection(path.parts)]
    count = len(included)
    size = sum(path.stat().st_size for path in included)
    print(f"RELEASE CHECK: PASS ({count} files, {size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
