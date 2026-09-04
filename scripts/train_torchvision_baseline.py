#!/usr/bin/env python3
"""Train Faster R-CNN, RetinaNet, or FCOS on the frozen ARD100 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    fcos_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import pil_to_tensor

from public_paths import DATA_ROOT, REPO_ROOT


ROOT = REPO_ROOT
SOURCE = DATA_ROOT
PROJECT = ROOT / "outputs" / "training" / "torchvision"
SEED = 20260821


class YoloFrameDataset(Dataset):
    def __init__(self, split: str, horizontal_flip: bool, target_label: int) -> None:
        self.split = split
        self.horizontal_flip = horizontal_flip
        self.target_label = target_label
        self.images = sorted((SOURCE / "images" / split).glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(SOURCE / "images" / split)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        label_path = SOURCE / "labels" / self.split / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            _, cx, cy, box_width, box_height = map(float, fields)
            bw, bh = box_width * width, box_height * height
            boxes.append([cx * width - bw / 2, cy * height - bh / 2, cx * width + bw / 2, cy * height + bh / 2])
        image_tensor = pil_to_tensor(image).float().div_(255.0)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.horizontal_flip and torch.rand(()) < 0.5:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            if box_tensor.numel():
                old_x1 = box_tensor[:, 0].clone()
                old_x2 = box_tensor[:, 2].clone()
                box_tensor[:, 0] = width - old_x2
                box_tensor[:, 2] = width - old_x1
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
            if box_tensor.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": box_tensor,
            "labels": torch.full((len(box_tensor),), self.target_label, dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
        }
        return image_tensor, target


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(worker_id: int) -> None:
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(model_id: str, pretrained: bool = True, input_size: int = 640) -> nn.Module:
    common = dict(min_size=input_size, max_size=input_size)
    if model_id == "fasterrcnn_r50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            **common,
        )
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
        model.roi_heads.score_thresh = 0.001
        model.roi_heads.detections_per_img = 300
        return model
    if model_id == "retinanet_resnet50_fpn_v2":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            **common,
        )
        old = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            old.conv[0][0].in_channels,
            old.num_anchors,
            1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        model.score_thresh = 0.001
        model.detections_per_img = 300
        return model
    if model_id == "fcos_resnet50_fpn":
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            **common,
        )
        old = model.head.classification_head
        model.head.classification_head = FCOSClassificationHead(
            old.conv[0].in_channels,
            old.num_anchors,
            1,
            num_convs=4,
            norm_layer=partial(nn.GroupNorm, 32),
        )
        model.score_thresh = 0.001
        model.detections_per_img = 300
        return model
    raise ValueError(model_id)


def target_label_for_model(model_id: str) -> int:
    # Faster R-CNN reserves label 0 for explicit background; RetinaNet and FCOS
    # use sigmoid heads whose single foreground class is label 0.
    return 1 if model_id == "fasterrcnn_r50_fpn_v2" else 0


def box_iou(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    if many.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(one[:2], many[:, :2])
    bottom_right = torch.minimum(one[2:], many[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area_one = (one[2] - one[0]).clamp(min=0) * (one[3] - one[1]).clamp(min=0)
    area_many = (many[:, 2] - many[:, 0]).clamp(min=0) * (many[:, 3] - many[:, 1]).clamp(min=0)
    return intersection / (area_one + area_many - intersection).clamp(min=1e-9)


def interpolated_ap(scores: list[float], matches: list[int], positives: int) -> float:
    if positives <= 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    truth = np.asarray(matches, dtype=np.float64)[order]
    tp = np.cumsum(truth)
    fp = np.cumsum(1.0 - truth)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1e-12)
    samples = np.linspace(0.0, 1.0, 101)
    values = [precision[recall >= threshold].max() if np.any(recall >= threshold) else 0.0 for threshold in samples]
    return float(np.mean(values))


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, target_label: int) -> dict[str, float]:
    model.eval()
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    scores = {threshold: [] for threshold in thresholds}
    matches = {threshold: [] for threshold in thresholds}
    positives = 0
    started = time.time()
    for images, targets in loader:
        outputs = model([image.to(device, non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            gt = target["boxes"].cpu()
            positives += len(gt)
            predicted_boxes = output["boxes"].detach().cpu()
            predicted_scores = output["scores"].detach().cpu()
            predicted_labels = output["labels"].detach().cpu()
            keep = predicted_labels == target_label
            predicted_boxes = predicted_boxes[keep]
            predicted_scores = predicted_scores[keep]
            order = torch.argsort(predicted_scores, descending=True)
            for threshold in thresholds:
                used: set[int] = set()
                for prediction_index in order.tolist():
                    score = float(predicted_scores[prediction_index])
                    scores[threshold].append(score)
                    ious = box_iou(predicted_boxes[prediction_index], gt)
                    if ious.numel() == 0:
                        matches[threshold].append(0)
                        continue
                    best_iou, best_index = torch.max(ious, dim=0)
                    gt_index = int(best_index)
                    is_match = float(best_iou) >= threshold and gt_index not in used
                    matches[threshold].append(int(is_match))
                    if is_match:
                        used.add(gt_index)
    ap = {f"ap{int(threshold * 100)}": interpolated_ap(scores[threshold], matches[threshold], positives) for threshold in thresholds}
    ap["map50_95"] = float(np.mean(list(ap.values())))
    ap["positive_boxes"] = positives
    ap["seconds"] = time.time() - started
    return ap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("fasterrcnn_r50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.val_every < 1:
        raise ValueError("--val-every must be positive")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must lie in [1, epochs]")
    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience must be positive")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    run_name = f"{args.model}_ard100_det_v1_e{args.epochs}_b{args.batch}"
    run_dir = PROJECT / run_name
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    target_label = target_label_for_model(args.model)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        YoloFrameDataset("train", horizontal_flip=True, target_label=target_label),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        YoloFrameDataset("val", horizontal_flip=False, target_label=target_label),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
        worker_init_fn=seed_worker,
    )

    model = build_model(args.model).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = 0.005 * args.batch / 16.0
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=learning_rate * 0.01)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    best_map = -1.0
    best_epoch = 0
    validation_events_without_improvement = 0
    history: list[dict[str, float]] = []

    last_checkpoint = run_dir / "last.pt"
    if args.resume and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint["best_map"])
        best_epoch = int(checkpoint.get("best_epoch", 0))
        validation_events_without_improvement = int(
            checkpoint.get("validation_events_without_improvement", 0)
        )
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        started = time.time()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images_gpu = [image.to(device, non_blocking=True) for image in images]
            targets_gpu = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                losses = model(images_gpu, targets_gpu)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={step}: {losses}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(images)
            epoch_loss += float(loss.detach()) * batch_size
            seen += batch_size
            if step % 200 == 0:
                print(
                    f"[TRAIN] model={args.model} epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(train_loader)} loss={epoch_loss / seen:.5f}"
                )
        scheduler.step()
        record: dict[str, float] = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / seen,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": time.time() - started,
        }
        should_validate = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        stop_after_epoch = False
        if should_validate:
            metrics = validate(model, val_loader, device, target_label)
            record.update({f"val_{key}": value for key, value in metrics.items()})
            print(f"[VAL] {json.dumps(record, ensure_ascii=False)}")
            if metrics["map50_95"] > best_map:
                best_map = metrics["map50_95"]
                best_epoch = epoch + 1
                validation_events_without_improvement = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, run_dir / "best.pt")
            else:
                validation_events_without_improvement += 1
            stop_after_epoch = (
                epoch + 1 >= args.min_epochs
                and validation_events_without_improvement >= args.early_stop_patience
            )
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_map": best_map,
            "best_epoch": best_epoch,
            "validation_events_without_improvement": validation_events_without_improvement,
            "history": history,
            "model_id": args.model,
        }
        torch.save(checkpoint, last_checkpoint)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if stop_after_epoch:
            print(
                f"[EARLY-STOP] model={args.model} epoch={epoch + 1} best_epoch={best_epoch} "
                f"validation_events_without_improvement={validation_events_without_improvement}"
            )
            break

    best = run_dir / "best.pt"
    if not best.exists():
        raise RuntimeError("No best checkpoint was produced")
    frozen_dir = ROOT / "weights"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_best = frozen_dir / f"{args.model}_ard100_det_v1_best.pt"
    frozen_last = frozen_dir / f"{args.model}_ard100_det_v1_last.pt"
    shutil.copy2(best, frozen_best)
    shutil.copy2(last_checkpoint, frozen_last)
    metadata = {
        "protocol_version": "ard100-det-v1",
        "model": args.model,
        "epochs": args.epochs,
        "epochs_completed": int(history[-1]["epoch"]),
        "batch": args.batch,
        "validation_interval_epochs": args.val_every,
        "minimum_epochs_before_early_stop": args.min_epochs,
        "early_stop_patience_validation_events": args.early_stop_patience,
        "seed": SEED,
        "target_label": target_label,
        "best_map50_95": best_map,
        "best_epoch": best_epoch,
        "best": str(frozen_best),
        "best_sha256": sha256(frozen_best),
        "last": str(frozen_last),
        "last_sha256": sha256(frozen_last),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (frozen_dir / f"{args.model}_ard100_det_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
