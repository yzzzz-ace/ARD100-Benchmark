#!/usr/bin/env python3
"""Batch-1 in-memory latency benchmark with a common wall-time boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from public_paths import DATA_ROOT, REPO_ROOT, YOLOMG_CODE_ROOT


ROOT = REPO_ROOT
SOURCE = DATA_ROOT
DEFAULT_CODE = YOLOMG_CODE_ROOT
ULTRALYTICS_MODELS = {"yolov8s", "rtdetr_l"}
TORCHVISION_MODELS = {"fasterrcnn_r50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"}
YOLOMG_MODELS = {"yolomg_paper"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool(size: int, model_id: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    base = ROOT / "datasets" / "ARD100_YOLOMG_Paper_v1" if model_id == "yolomg_paper" else SOURCE
    paths = sorted((base / "images" / "val").glob("*.jpg"))[:size]
    if len(paths) != size:
        raise ValueError(f"Requested pool={size}, found {len(paths)} validation images")
    rgb_pool: list[np.ndarray] = []
    motion_pool: list[np.ndarray] = []
    for path in paths:
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise OSError(path)
        rgb_pool.append(rgb)
        if model_id in YOLOMG_MODELS:
            motion_path = base / "images2" / "val" / path.name
            motion = cv2.imread(str(motion_path), cv2.IMREAD_COLOR)
            if motion is None:
                raise OSError(motion_path)
            motion_pool.append(motion)
    return rgb_pool, motion_pool


def parameter_count(model: object) -> int | None:
    candidates = [model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "parameters"):
            return int(sum(parameter.numel() for parameter in candidate.parameters()))
    return None


def ultralytics_runner(model_id: str, weights: Path, half: bool, input_size: int) -> tuple[Callable, int | None]:
    from ultralytics import RTDETR, YOLO

    model = RTDETR(str(weights)) if model_id == "rtdetr_l" else YOLO(str(weights))

    def run(rgb: np.ndarray, _motion: np.ndarray | None) -> dict[str, float]:
        result = model.predict(
            source=rgb,
            imgsz=input_size,
            conf=0.001,
            iou=0.60,
            max_det=300,
            device=0,
            half=half,
            verbose=False,
            save=False,
        )[0]
        return {key: float(result.speed.get(key, float("nan"))) for key in ("preprocess", "inference", "postprocess")}

    return run, parameter_count(model.model)


def torchvision_runner(model_id: str, weights: Path, half: bool, input_size: int) -> tuple[Callable, int | None]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_torchvision_baseline import build_model

    model = build_model(model_id, pretrained=False, input_size=input_size)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to("cuda:0").eval()
    if half:
        model.half()

    def run(rgb: np.ndarray, _motion: np.ndarray | None) -> dict[str, float]:
        contiguous_rgb = np.ascontiguousarray(rgb[:, :, ::-1])
        tensor = torch.from_numpy(contiguous_rgb).permute(2, 0, 1).float().div_(255.0).to("cuda:0")
        if half:
            tensor = tensor.half()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=half, dtype=torch.float16):
            model([tensor])
        return {}

    return run, parameter_count(model)


def yolomg_runner(
    weights: Path,
    half: bool,
    code_root: Path,
    input_size: int,
) -> tuple[Callable, int | None]:
    if half:
        raise ValueError("The audited YOLOMG wrapper is FP32-only; do not report an unvalidated FP16 number")
    sys.path.insert(0, str(ROOT / "scripts"))
    from yolomg_adapter import YOLOMGDetector

    detector = YOLOMGDetector(code_root, weights, input_size, "0", 0.001, 0.60)

    def run(rgb: np.ndarray, motion: np.ndarray | None) -> dict[str, float]:
        if motion is None:
            raise ValueError("YOLOMG runtime requires its paired motion input")
        detector.detect(rgb, motion)
        return {}

    return run, parameter_count(detector.model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        choices=sorted(ULTRALYTICS_MODELS | TORCHVISION_MODELS | YOLOMG_MODELS),
        required=True,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--imgsz", type=int, choices=(640, 1280), default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    half = args.precision == "fp16"
    rgb_pool, motion_pool = load_pool(args.pool_size, args.model_id)
    if args.model_id in ULTRALYTICS_MODELS:
        runner, parameters = ultralytics_runner(args.model_id, args.weights, half, args.imgsz)
    elif args.model_id in TORCHVISION_MODELS:
        runner, parameters = torchvision_runner(args.model_id, args.weights, half, args.imgsz)
    else:
        runner, parameters = yolomg_runner(args.weights, half, args.code_root, args.imgsz)

    cuda_device = torch.device("cuda:0")
    # Some lazy-loading wrappers (notably Ultralytics) keep the model on CPU
    # until the first prediction, so initialize CUDA before resetting counters.
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_device)
    for index in range(args.warmup):
        runner(rgb_pool[index % len(rgb_pool)], motion_pool[index % len(motion_pool)] if motion_pool else None)
    torch.cuda.synchronize()

    latency_ms: list[float] = []
    components: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": []}
    for index in range(args.iterations):
        rgb = rgb_pool[index % len(rgb_pool)]
        motion = motion_pool[index % len(motion_pool)] if motion_pool else None
        torch.cuda.synchronize()
        started = time.perf_counter()
        component = runner(rgb, motion)
        torch.cuda.synchronize()
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        for key in components:
            value = component.get(key, float("nan"))
            if np.isfinite(value):
                components[key].append(value)

    latency = np.asarray(latency_ms, dtype=np.float64)
    payload: dict[str, object] = {
        "protocol_version": "ard100-det-v1",
        "model": args.model_id,
        "precision": args.precision,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "hardware": torch.cuda.get_device_name(0),
        "batch": 1,
        "input_long_side": args.imgsz,
        "decode_included": False,
        "common_boundary": "decoded BGR ndarray through final post-NMS detections",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "pool_size": args.pool_size,
        "parameters": parameters,
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_median": float(np.median(latency)),
        "latency_ms_p95": float(np.quantile(latency, 0.95)),
        "throughput_fps_from_mean": float(1000.0 / latency.mean()),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(cuda_device) / (1024.0**2)),
        "framework_components_ms": {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for key, values in components.items()
            if values
        },
        "component_warning": "Framework components are descriptive only; the common wall-time boundary is used for cross-model comparison.",
    }
    output_dir = ROOT / "outputs" / "runtime" / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"i{args.imgsz}_{args.precision}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
