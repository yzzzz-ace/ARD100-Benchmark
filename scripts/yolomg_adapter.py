"""Standalone adapter for the externally installed GPL-3.0 YOLOMG tree."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float


def clip_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


class YOLOMGDetector:
    def __init__(
        self,
        code_root: Path,
        weights: Path,
        imgsz: int = 640,
        device: str = "0",
        conf_thres: float = 0.001,
        iou_thres: float = 0.60,
    ) -> None:
        if not code_root.exists():
            raise FileNotFoundError(code_root)
        if not weights.exists():
            raise FileNotFoundError(weights)
        sys.path.insert(0, str(code_root))
        from models.common import DetectMultiBackend  # type: ignore
        from utils.general import non_max_suppression  # type: ignore
        try:
            from utils.general import scale_boxes  # type: ignore
        except ImportError:
            from utils.general import scale_coords as scale_boxes  # type: ignore
        from utils.torch_utils import select_device  # type: ignore
        try:
            from utils.augmentations import letterbox  # type: ignore
        except ImportError:
            from utils.datasets import letterbox  # type: ignore

        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.model = DetectMultiBackend(str(weights), device=self.device, dnn=False, data=None, fp16=False)
        self.stride = int(getattr(self.model, "stride", 32))
        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.letterbox = letterbox

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        image = self.letterbox(image_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])
        tensor = torch.from_numpy(image).to(self.device).float().div_(255.0)
        return tensor[None] if tensor.ndimension() == 3 else tensor

    def _forward(self, rgb: torch.Tensor, motion: torch.Tensor) -> Any:
        for candidate in (
            lambda: self.model(rgb, motion),
            lambda: self.model.model(rgb, motion),
            lambda: self.model(rgb),
            lambda: self.model.model(rgb),
        ):
            try:
                return candidate()
            except Exception:
                continue
        raise RuntimeError("YOLOMG forward failed for all supported call signatures")

    def detect(self, rgb_bgr: np.ndarray, motion_bgr: np.ndarray) -> list[Detection]:
        rgb = self._preprocess(rgb_bgr)
        motion = self._preprocess(motion_bgr)
        with torch.no_grad():
            prediction = self._forward(rgb, motion)
        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        nms = self.non_max_suppression(
            prediction, self.conf_thres, self.iou_thres, classes=None, agnostic=False
        )
        if not nms or nms[0] is None or len(nms[0]) == 0:
            return []
        det = nms[0]
        try:
            det[:, :4] = self.scale_boxes(rgb.shape[2:], det[:, :4], rgb_bgr.shape).round()
        except Exception:
            det[:, :4] = self.scale_boxes(rgb.shape[2:], det[:, :4], rgb_bgr.shape[:2]).round()
        height, width = rgb_bgr.shape[:2]
        result = [
            Detection(clip_box(list(xyxy), width, height), float(conf))
            for *xyxy, conf, _class_id in det.detach().cpu().numpy().tolist()
        ]
        return sorted(result, key=lambda item: item.conf, reverse=True)

