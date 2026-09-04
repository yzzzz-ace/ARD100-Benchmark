#!/usr/bin/env python3
"""Build the frozen ARD100 full-frame detector benchmark.

Train/validation RGB and motion samples are reused byte-for-byte from the
audited official-split workspace. Official-test frames are decoded here. Only
frames with an XML file are materialized; an XML with no target is retained as
a genuine negative frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from public_paths import OFFICIAL_ROOT, REPO_ROOT


BENCHMARK_ROOT = REPO_ROOT
SOURCE_DATASET = OFFICIAL_ROOT / "datasets" / "YOLOMG_OfficialTrain_v1"
DATASET_ROOT = BENCHMARK_ROOT / "datasets" / "ARD100_FullFrame_v1"
LINEAGE_CSV = OFFICIAL_ROOT / "manifests" / "dataset_lineage.csv"
ANNOTATION_ROOT = OFFICIAL_ROOT / "annotations" / "test"
CLASS_NAMES = {"drone", "uav"}
STEM_RE = re.compile(r"^([A-Za-z]+\d+)_([0-9]+)$")


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_xml(path: Path) -> tuple[list[Box], tuple[int, int]]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes: list[Box] = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() not in CLASS_NAMES:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            box = Box(
                float(bb.findtext("xmin", "0")),
                float(bb.findtext("ymin", "0")),
                float(bb.findtext("xmax", "0")),
                float(bb.findtext("ymax", "0")),
            )
        except ValueError:
            continue
        if box.width >= 2 and box.height >= 2:
            boxes.append(box)
    return boxes, (width, height)


def xml_index(path: Path) -> int:
    match = STEM_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unrecognized XML name: {path}")
    return int(match.group(2))


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image, 1.0
    scale = max_side / float(max(height, width))
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


_HANNING: dict[tuple[int, int], np.ndarray] = {}


def phase_frame(gray: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = max(gray.shape)
    if maximum <= 640:
        return np.asarray(gray, dtype=np.float32), 1.0
    scale = 640.0 / maximum
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.asarray(small, dtype=np.float32), scale


def camera_shift(prev: np.ndarray | None, curr: np.ndarray, scale: float) -> tuple[float, float, float]:
    if prev is None or prev.shape != curr.shape:
        return 0.0, 0.0, 0.0
    key = (curr.shape[1], curr.shape[0])
    window = _HANNING.setdefault(key, cv2.createHanningWindow(key, cv2.CV_32F))
    try:
        (dx_small, dy_small), response = cv2.phaseCorrelate(prev, curr, window)
        dx = float(np.clip(dx_small / scale, -80.0, 80.0))
        dy = float(np.clip(dy_small / scale, -80.0, 80.0))
        return dx, dy, float(response)
    except cv2.error:
        return 0.0, 0.0, 0.0


def aligned_diff(prev: np.ndarray | None, curr: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    if prev is None:
        return np.zeros_like(curr)
    matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    warped = cv2.warpAffine(
        prev,
        matrix,
        (curr.shape[1], curr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.absdiff(warped, curr)


def motion_image(
    prev2: np.ndarray | None,
    prev1: np.ndarray | None,
    curr: np.ndarray,
    shift: tuple[float, float],
) -> np.ndarray:
    recent = aligned_diff(prev1, curr, shift)
    older = aligned_diff(prev2, prev1, (0.0, 0.0)) if prev1 is not None else np.zeros_like(curr)
    diff = cv2.GaussianBlur(cv2.max(recent, older), (3, 3), 0)
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)


def yolo_lines(boxes: list[Box], width: int, height: int) -> list[str]:
    return [
        f"0 {box.cx / width:.8f} {box.cy / height:.8f} "
        f"{box.width / width:.8f} {box.height / height:.8f}"
        for box in boxes
    ]


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise OSError(f"Failed to write {path}")


def local_attributes(gray: np.ndarray, box: Box) -> tuple[float, float, float]:
    height, width = gray.shape
    x1 = max(0, min(width - 1, int(math.floor(box.xmin))))
    y1 = max(0, min(height - 1, int(math.floor(box.ymin))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box.xmax))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box.ymax))))
    target = gray[y1:y2, x1:x2]

    margin_x = max(6, int(round(box.width * 2.0)))
    margin_y = max(6, int(round(box.height * 2.0)))
    ox1, oy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    ox2, oy2 = min(width, x2 + margin_x), min(height, y2 + margin_y)
    context = gray[oy1:oy2, ox1:ox2]
    ring_mask = np.ones(context.shape, dtype=bool)
    ring_mask[y1 - oy1 : y2 - oy1, x1 - ox1 : x2 - ox1] = False
    ring = context[ring_mask]

    target_mean = float(target.mean()) if target.size else 0.0
    ring_mean = float(ring.mean()) if ring.size else target_mean
    contrast = abs(target_mean - ring_mean) / 255.0
    blur = float(cv2.Laplacian(context, cv2.CV_64F).var()) if context.size else 0.0
    edges = cv2.Canny(context, 60, 180)
    edge_density = float((edges[ring_mask] > 0).mean()) if ring.size else 0.0
    return contrast, blur, edge_density


def test_rows() -> list[dict[str, str]]:
    with LINEAGE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        return sorted(
            (row for row in csv.DictReader(handle) if row["official_split"] == "test"),
            key=lambda row: row["official_id"].lower(),
        )


def build_test(max_side: int, quality: int) -> list[dict[str, object]]:
    image_dir = DATASET_ROOT / "images" / "test"
    motion_dir = DATASET_ROOT / "images2" / "test"
    label_dir = DATASET_ROOT / "labels" / "test"
    for directory in (image_dir, motion_dir, label_dir, DATASET_ROOT / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, object]] = []
    for row in test_rows():
        video_metadata_start = len(metadata)
        video_id = row["official_id"]
        video_path = Path(row["official_video"])
        annotation_dir = ANNOTATION_ROOT / video_id
        xml_by_index = {xml_index(path): path for path in annotation_dir.glob("*.xml")}
        parsed = {index: parse_xml(path) for index, path in xml_by_index.items()}

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        prev2_gray: np.ndarray | None = None
        prev1_gray: np.ndarray | None = None
        prev_phase: np.ndarray | None = None
        prev_box: Box | None = None
        prev_annotated_index: int | None = None
        decoded = 0
        started = time.time()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_phase, phase_scale = phase_frame(gray)
                dx, dy, phase_response = camera_shift(prev_phase, current_phase, phase_scale)
                item = parsed.get(decoded)
                if item is not None:
                    boxes, (xml_width, xml_height) = item
                    height, width = frame.shape[:2]
                    if (xml_width, xml_height) != (width, height):
                        raise ValueError(
                            f"Size mismatch {video_id} frame {decoded}: "
                            f"video={width}x{height}, XML={xml_width}x{xml_height}"
                        )
                    stem = f"{video_id}_{decoded:06d}"
                    rgb_resized, scale = resize_max_side(frame, max_side)
                    motion = motion_image(prev2_gray, prev1_gray, gray, (dx, dy))
                    motion_resized, _ = resize_max_side(motion, max_side)
                    write_jpg(image_dir / f"{stem}.jpg", rgb_resized, quality)
                    write_jpg(motion_dir / f"{stem}.jpg", motion_resized, quality)
                    (label_dir / f"{stem}.txt").write_text(
                        "\n".join(yolo_lines(boxes, width, height)) + ("\n" if boxes else ""),
                        encoding="utf-8",
                    )

                    primary = boxes[0] if boxes else None
                    if primary is not None:
                        contrast, blur, clutter = local_attributes(gray, primary)
                        if prev_box is not None and prev_annotated_index is not None:
                            delta_frames = decoded - prev_annotated_index
                            target_dx = primary.cx - prev_box.cx
                            target_dy = primary.cy - prev_box.cy
                            target_displacement = math.hypot(target_dx, target_dy) / max(delta_frames, 1)
                        else:
                            target_dx = target_dy = target_displacement = float("nan")
                    else:
                        contrast = blur = clutter = float("nan")
                        target_dx = target_dy = target_displacement = float("nan")

                    metadata.append(
                        {
                            "sample_id": stem,
                            "video_id": video_id,
                            "frame_index": decoded,
                            "fps": float(row["fps"]),
                            "container_reported_frames": int(row["frames"]),
                            "sequentially_decoded_frames": "",
                            "original_width": width,
                            "original_height": height,
                            "stored_width": rgb_resized.shape[1],
                            "stored_height": rgb_resized.shape[0],
                            "stored_scale": scale,
                            "box_count": len(boxes),
                            "target_present": int(bool(boxes)),
                            "target_xmin": primary.xmin if primary else "",
                            "target_ymin": primary.ymin if primary else "",
                            "target_xmax": primary.xmax if primary else "",
                            "target_ymax": primary.ymax if primary else "",
                            "target_width_px": primary.width if primary else "",
                            "target_height_px": primary.height if primary else "",
                            "target_sqrt_area_px": math.sqrt(primary.width * primary.height) if primary else "",
                            "target_contrast": contrast,
                            "target_context_laplacian_var": blur,
                            "context_edge_density": clutter,
                            "target_dx_per_frame": target_dx,
                            "target_dy_per_frame": target_dy,
                            "target_displacement_per_frame": target_displacement,
                            "camera_dx_px": dx,
                            "camera_dy_px": dy,
                            "camera_displacement_px": math.hypot(dx, dy),
                            "phase_response": phase_response,
                        }
                    )
                    prev_box = primary
                    prev_annotated_index = decoded

                prev2_gray, prev1_gray = prev1_gray, gray
                prev_phase = current_phase
        finally:
            capture.release()
        last_xml_index = max(parsed) if parsed else 0
        if decoded < last_xml_index:
            raise RuntimeError(
                f"Annotated frame cannot be decoded for {video_id}: "
                f"decoded={decoded}, last_xml_index={last_xml_index}"
            )
        for item in metadata[video_metadata_start:]:
            item["sequentially_decoded_frames"] = decoded
        print(
            f"[DONE] {video_id}: decoded={decoded} container_reported={row['frames']} "
            f"annotated={len(parsed)} "
            f"seconds={time.time() - started:.1f}"
        )

    metadata_path = DATASET_ROOT / "metadata" / "test_frames.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    return metadata


def image_files(branch: str, split: str) -> list[Path]:
    root = SOURCE_DATASET if split in {"train", "val"} else DATASET_ROOT
    return sorted((root / branch / split).glob("*.jpg"))


def label_path(image_path: Path, split: str) -> Path:
    root = SOURCE_DATASET if split in {"train", "val"} else DATASET_ROOT
    return root / "labels" / split / f"{image_path.stem}.txt"


def parse_yolo(path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        _, cx, cy, bw, bh = map(float, fields)
        box_width, box_height = bw * width, bh * height
        boxes.append([cx * width - box_width / 2, cy * height - box_height / 2, box_width, box_height])
    return boxes


def build_coco(split: str) -> dict[str, int]:
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(image_files("images", split), start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        match = STEM_RE.match(image_path.stem)
        if not match:
            raise ValueError(f"Unexpected image stem {image_path.stem}")
        video_id, frame_index = match.group(1), int(match.group(2))
        images.append(
            {
                "id": image_id,
                "file_name": image_path.as_posix(),
                "width": width,
                "height": height,
                "video_id": video_id,
                "frame_index": frame_index,
            }
        )
        for x, y, box_width, box_height in parse_yolo(label_path(image_path, split), width, height):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": "ARD100 leakage-free full-frame detector benchmark",
            "version": "ard100-det-v1",
            "split": split,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "Drone", "supercategory": "object"}],
        "images": images,
        "annotations": annotations,
    }
    annotation_dir = DATASET_ROOT / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    output = annotation_dir / f"instances_{split}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations), "sha256": sha256(output)}


def write_lists_and_yaml() -> None:
    split_dir = DATASET_ROOT / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for branch, suffix in (("images", "rgb"), ("images2", "motion")):
            paths = image_files(branch, split)
            (split_dir / f"{split}_{suffix}.txt").write_text(
                "\n".join(path.as_posix() for path in paths) + "\n",
                encoding="utf-8",
            )

    rgb_yaml = (
        f"path: {DATASET_ROOT.as_posix()}\n"
        f"train: {(split_dir / 'train_rgb.txt').as_posix()}\n"
        f"val: {(split_dir / 'val_rgb.txt').as_posix()}\n"
        f"test: {(split_dir / 'test_rgb.txt').as_posix()}\n"
        "nc: 1\nnames:\n  0: Drone\n"
    )
    (DATASET_ROOT / "ard100_rgb.yaml").write_text(rgb_yaml, encoding="utf-8")
    dual_yaml = (
        f"path: {DATASET_ROOT.as_posix()}\n"
        f"train: {(split_dir / 'train_rgb.txt').as_posix()}\n"
        f"train02: {(split_dir / 'train_motion.txt').as_posix()}\n"
        f"val: {(split_dir / 'val_rgb.txt').as_posix()}\n"
        f"val2: {(split_dir / 'val_motion.txt').as_posix()}\n"
        f"test: {(split_dir / 'test_rgb.txt').as_posix()}\n"
        f"test2: {(split_dir / 'test_motion.txt').as_posix()}\n"
        "nc: 1\nnames: ['Drone']\n"
    )
    (DATASET_ROOT / "ard100_yolomg.yaml").write_text(dual_yaml, encoding="utf-8")


def audit(coco: dict[str, dict[str, int]], metadata: list[dict[str, object]]) -> None:
    expected = {"train": 35708, "val": 8938, "test": 72631}
    errors: list[str] = []
    for split, count in expected.items():
        if coco[split]["images"] != count:
            errors.append(f"{split}_image_count={coco[split]['images']} expected={count}")
        rgb_stems = {path.stem for path in image_files("images", split)}
        motion_stems = {path.stem for path in image_files("images2", split)}
        if rgb_stems != motion_stems:
            errors.append(f"{split}_rgb_motion_pair_mismatch")
    test_videos = {str(row["video_id"]) for row in metadata}
    if len(test_videos) != 35:
        errors.append(f"test_video_count={len(test_videos)} expected=35")
    if sum(int(row["target_present"]) for row in metadata) != 71633:
        errors.append("test_positive_count_mismatch")
    if len(metadata) - sum(int(row["target_present"]) for row in metadata) != 998:
        errors.append("test_negative_count_mismatch")

    manifest = {
        "protocol_version": "ard100-det-v1",
        "source_lineage": str(LINEAGE_CSV),
        "source_lineage_sha256": sha256(LINEAGE_CSV),
        "source_dataset": str(SOURCE_DATASET),
        "coco": coco,
        "test_frames": len(metadata),
        "test_videos": len(test_videos),
        "test_positive_frames": sum(int(row["target_present"]) for row in metadata),
        "test_negative_frames": len(metadata) - sum(int(row["target_present"]) for row in metadata),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    output = DATASET_ROOT / "dataset_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError("Dataset audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--skip-test-decode", action="store_true")
    args = parser.parse_args()

    if args.skip_test_decode:
        metadata_path = DATASET_ROOT / "metadata" / "test_frames.csv"
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            metadata = list(csv.DictReader(handle))
    else:
        metadata = build_test(args.max_side, args.jpeg_quality)
    write_lists_and_yaml()
    coco = {split: build_coco(split) for split in ("train", "val", "test")}
    audit(coco, metadata)


if __name__ == "__main__":
    main()
