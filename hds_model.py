"""HDS regression, ROI-local caching, and S-Only virtual fruit removal."""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA不可用：本版本代码已设置为只使用 GPU/CUDA，不会自动回退到 CPU。"
            "请检查 NVIDIA 驱动、CUDA 版 PyTorch、显卡状态和当前 conda 环境。"
        )
    from config import CUDA_DEVICE_INDEX
    torch.cuda.set_device(CUDA_DEVICE_INDEX)
    device = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
except ImportError as exc:
    raise RuntimeError(
        "当前程序需要 PyTorch 和 torchvision 才能运行。请先安装 torch 和 torchvision 后再运行。"
    ) from exc

from config import (
    DIFFICULTY_MODEL_PATH, YOLO_USE_HALF,
    ROI_SIDE_FACTOR, ROI_TOTAL_EXPANSION_FACTOR,
    ENABLE_ROI_DIFFICULTY_CACHE, ROI_DIFFICULTY_CACHE_MAX_ITEMS,
    ENABLE_FULL_STATE_DIFFICULTY_CACHE, FULL_STATE_DIFFICULTY_CACHE_MAX_ITEMS,
)
from perception import _analyze_binary_mask, unpack_detection_box

CACHED_DIFFICULTY_MODEL = None
GLOBAL_DIFFICULTY_VALUE_CACHE = OrderedDict()
GLOBAL_DIFFICULTY_CACHE_STATS = {"hits": 0, "misses": 0, "stores": 0}
GLOBAL_FULL_STATE_DIFFICULTY_CACHE = OrderedDict()
GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS = {"hits": 0, "misses": 0, "stores": 0}

def _hash_roi_image_for_cache(roi_320: np.ndarray) -> str:
    """对模型实际输入的 320×320 ROI 图像做哈希，用于判断局部场景是否发生变化。"""
    if roi_320 is None:
        return ""
    roi_contiguous = np.ascontiguousarray(roi_320)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(roi_contiguous.shape).encode("utf-8"))
    h.update(str(roi_contiguous.dtype).encode("utf-8"))
    h.update(roi_contiguous.tobytes())
    return h.hexdigest()

def _make_roi_difficulty_cache_key(roi_320: np.ndarray) -> Optional[Tuple]:
    """构建难度缓存 key：只依赖模型、ROI扩展参数和模型实际输入ROI像素。"""
    if roi_320 is None:
        return None
    roi_hash = _hash_roi_image_for_cache(roi_320)
    if not roi_hash:
        return None
    model_id = os.path.normcase(os.path.abspath(DIFFICULTY_MODEL_PATH))
    return (
        "difficulty_roi_v2",
        model_id,
        float(ROI_SIDE_FACTOR),
        int(ROI_TOTAL_EXPANSION_FACTOR * 1000),
        "resize_pad_320_with_bbox_annotation",
        roi_hash,
    )

def _get_cached_difficulty(cache_key: Optional[Tuple]) -> Optional[float]:
    """读取全局 ROI 难度缓存，并维护 LRU 顺序。"""
    if not ENABLE_ROI_DIFFICULTY_CACHE or cache_key is None:
        return None
    if cache_key in GLOBAL_DIFFICULTY_VALUE_CACHE:
        GLOBAL_DIFFICULTY_CACHE_STATS["hits"] += 1
        GLOBAL_DIFFICULTY_VALUE_CACHE.move_to_end(cache_key)
        return float(GLOBAL_DIFFICULTY_VALUE_CACHE[cache_key])
    GLOBAL_DIFFICULTY_CACHE_STATS["misses"] += 1
    return None

def _store_cached_difficulty(cache_key: Optional[Tuple], difficulty_value: float) -> None:
    """写入全局 ROI 难度缓存；超出容量时按 LRU 删除最旧项。"""
    if not ENABLE_ROI_DIFFICULTY_CACHE or cache_key is None:
        return
    try:
        GLOBAL_DIFFICULTY_VALUE_CACHE[cache_key] = float(difficulty_value)
        GLOBAL_DIFFICULTY_VALUE_CACHE.move_to_end(cache_key)
        GLOBAL_DIFFICULTY_CACHE_STATS["stores"] += 1
        while len(GLOBAL_DIFFICULTY_VALUE_CACHE) > int(ROI_DIFFICULTY_CACHE_MAX_ITEMS):
            GLOBAL_DIFFICULTY_VALUE_CACHE.popitem(last=False)
    except Exception:
        pass

def reset_difficulty_cache_stats(clear_cache: bool = False) -> Dict[str, int]:
    """重置命中率统计；clear_cache=True 时同时清空缓存内容。"""
    old_stats = dict(GLOBAL_DIFFICULTY_CACHE_STATS)
    old_stats["full_state_hits"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("hits", 0))
    old_stats["full_state_misses"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("misses", 0))
    old_stats["full_state_stores"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("stores", 0))

    GLOBAL_DIFFICULTY_CACHE_STATS["hits"] = 0
    GLOBAL_DIFFICULTY_CACHE_STATS["misses"] = 0
    GLOBAL_DIFFICULTY_CACHE_STATS["stores"] = 0
    GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["hits"] = 0
    GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["misses"] = 0
    GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["stores"] = 0
    if clear_cache:
        GLOBAL_DIFFICULTY_VALUE_CACHE.clear()
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE.clear()
    return old_stats

def get_difficulty_cache_stats() -> Dict[str, int]:
    """获取当前缓存统计。"""
    stats = dict(GLOBAL_DIFFICULTY_CACHE_STATS)
    stats["cache_size"] = len(GLOBAL_DIFFICULTY_VALUE_CACHE)
    stats["full_state_hits"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("hits", 0))
    stats["full_state_misses"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("misses", 0))
    stats["full_state_stores"] = int(GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS.get("stores", 0))
    stats["full_state_cache_size"] = len(GLOBAL_FULL_STATE_DIFFICULTY_CACHE)
    return stats

def _make_full_state_difficulty_cache_key(image_id, target_idx, picked_indices) -> Optional[Tuple]:
    """构建完整状态难度缓存 key：图片ID + 目标ID + 已采集合。"""
    if not ENABLE_FULL_STATE_DIFFICULTY_CACHE:
        return None
    if image_id is None or str(image_id) == "":
        return None
    try:
        picked_state = tuple(sorted(int(v) for v in set(picked_indices or [])))
        model_id = os.path.normcase(os.path.abspath(DIFFICULTY_MODEL_PATH))
        return (
            "difficulty_full_state_v1",
            model_id,
            float(ROI_SIDE_FACTOR),
            str(image_id),
            int(target_idx),
            picked_state,
        )
    except Exception:
        return None

def _get_cached_full_state_difficulty(cache_key: Optional[Tuple]) -> Optional[float]:
    """读取完整状态难度缓存。"""
    if not ENABLE_FULL_STATE_DIFFICULTY_CACHE or cache_key is None:
        return None
    if cache_key in GLOBAL_FULL_STATE_DIFFICULTY_CACHE:
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["hits"] += 1
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE.move_to_end(cache_key)
        return float(GLOBAL_FULL_STATE_DIFFICULTY_CACHE[cache_key])
    GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["misses"] += 1
    return None

def _store_cached_full_state_difficulty(cache_key: Optional[Tuple], difficulty_value: float) -> None:
    """写入完整状态难度缓存，并按 LRU 控制容量。"""
    if not ENABLE_FULL_STATE_DIFFICULTY_CACHE or cache_key is None:
        return
    try:
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE[cache_key] = float(difficulty_value)
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE.move_to_end(cache_key)
        GLOBAL_FULL_STATE_DIFFICULTY_CACHE_STATS["stores"] += 1
        while len(GLOBAL_FULL_STATE_DIFFICULTY_CACHE) > int(FULL_STATE_DIFFICULTY_CACHE_MAX_ITEMS):
            GLOBAL_FULL_STATE_DIFFICULTY_CACHE.popitem(last=False)
    except Exception:
        pass

def expand_bbox_per_side(bbox, img_size, side_factor=3.0):
    img_w, img_h = img_size
    cx, cy, w, h = bbox
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    new_w = w * (1.0 + 2.0 * side_factor)
    new_h = h * (1.0 + 2.0 * side_factor)

    x_min = int(cx - new_w / 2)
    y_min = int(cy - new_h / 2)
    x_max = int(cx + new_w / 2)
    y_max = int(cy + new_h / 2)

    pad_left = max(0, -x_min)
    pad_top = max(0, -y_min)
    pad_right = max(0, x_max - img_w)
    pad_bottom = max(0, y_max - img_h)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return (x_min, y_min, x_max, y_max), (pad_left, pad_top, pad_right, pad_bottom)

def resizeAndPadToSquare(img, target_size=320):
    h, w = img.shape[:2]
    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img
        new_h, new_w = h, w

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return img_padded

def _compute_roi_geometry_from_box(base_img_shape, box_xyxy, bbox_norm=None) -> Dict:
    x1, y1, x2, y2 = box_xyxy
    h, w = base_img_shape[:2]
    if bbox_norm is not None and len(bbox_norm) == 4:
        cx, cy, box_w, box_h = bbox_norm
    else:
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h

    expanded_bbox, padding = expand_bbox_per_side(
        [cx, cy, box_w, box_h],
        (w, h),
        side_factor=ROI_SIDE_FACTOR,
    )
    roi_x1, roi_y1, roi_x2, roi_y2 = expanded_bbox
    pad_left, pad_top, pad_right, pad_bottom = padding
    rel_box = (
        int(x1 - roi_x1 + pad_left),
        int(y1 - roi_y1 + pad_top),
        int(x2 - roi_x1 + pad_left),
        int(y2 - roi_y1 + pad_top),
    )
    return {
        "original_bbox": (x1, y1, x2, y2),
        "roi_coords": (roi_x1, roi_y1, roi_x2, roi_y2),
        "roi_size": (roi_x2 - roi_x1, roi_y2 - roi_y1),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "relative_box": rel_box,
        "final_size": (320, 320),
        "expansion_side_factor": ROI_SIDE_FACTOR,
        "total_expansion_factor": ROI_TOTAL_EXPANSION_FACTOR,
    }

def preprocess_roi_image_for_model(base_img_any, box_xyxy, bbox_norm=None, roi_geometry=None):
    if roi_geometry is None:
        roi_geometry = _compute_roi_geometry_from_box(base_img_any.shape, box_xyxy, bbox_norm=bbox_norm)

    x1, y1, x2, y2 = roi_geometry["original_bbox"]
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_geometry["roi_coords"]
    pad_left, pad_top, pad_right, pad_bottom = roi_geometry["padding"]

    roi = base_img_any[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    if roi.size == 0:
        return None, (roi_x1, roi_y1, roi_x2, roi_y2), {}

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        roi = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    roi_with_annotation = roi.copy()
    if len(roi_with_annotation.shape) == 3:
        rel_x1, rel_y1, rel_x2, rel_y2 = roi_geometry["relative_box"]
        cv2.rectangle(roi_with_annotation, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)

    roi_320 = resizeAndPadToSquare(roi_with_annotation, target_size=320)

    roi_metadata = {
        "original_bbox": (x1, y1, x2, y2),
        "roi_coords": (roi_x1, roi_y1, roi_x2, roi_y2),
        "roi_size": roi_geometry.get("roi_size", (roi_x2 - roi_x1, roi_y2 - roi_y1)),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "final_size": (320, 320),
        "expansion_side_factor": ROI_SIDE_FACTOR,
        "total_expansion_factor": ROI_TOTAL_EXPANSION_FACTOR,
    }
    return roi_320, (roi_x1, roi_y1, roi_x2, roi_y2), roi_metadata

def roi_image_to_model_tensor(roi_320):
    if roi_320 is None:
        return None
    roi_rgb = cv2.cvtColor(roi_320, cv2.COLOR_BGR2RGB)
    roi_tensor = transforms.ToTensor()(roi_rgb)
    roi_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(roi_tensor)
    return roi_tensor.unsqueeze(0)

class ImageDeductionProcessor:
    def __init__(self, original_image_or_path, detection_boxes):
        if isinstance(original_image_or_path, str):
            self.original_image_path = original_image_or_path
            self.original_image = cv2.imread(original_image_or_path)
            if self.original_image is None:
                raise ValueError(f"无法加载图像: {original_image_or_path}")
        else:
            self.original_image = np.copy(original_image_or_path)
            self.original_image_path = None

        self.detection_boxes = detection_boxes
        self.current_image = self.original_image.copy()
        self.prediction_image = self.original_image.copy()
        self.picked_boxes = []
        self.index_to_mask = None
        self.mask_bool_cache = {}
        self.mask_bbox_cache = {}
        self.mask_centroid_cache = {}
        self.mask_area_cache = {}
        self.current_image_id = None
        self.difficulty_model = None
        self.roi_geometry_cache: Dict[int, Dict] = {}
        self._precompute_roi_geometry_cache()

        global CACHED_DIFFICULTY_MODEL
        if CACHED_DIFFICULTY_MODEL is not None:
            self.difficulty_model = CACHED_DIFFICULTY_MODEL
            if TORCH_AVAILABLE and device is not None:
                self.difficulty_model.to(device)
        else:
            self._load_difficulty_model()

    def set_index_to_mask(self, index_to_mask, mask_metadata: Optional[Dict[int, Dict]] = None):
        self.index_to_mask = index_to_mask
        self._precompute_mask_cache(mask_metadata=mask_metadata)

    def _precompute_mask_cache(self, mask_metadata: Optional[Dict[int, Dict]] = None):
        self.mask_bool_cache = {}
        self.mask_bbox_cache = {}
        self.mask_centroid_cache = {}
        self.mask_area_cache = {}

        if not isinstance(self.index_to_mask, dict):
            return

        metadata_dict = mask_metadata if isinstance(mask_metadata, dict) else {}

        for idx, mask in self.index_to_mask.items():
            try:
                if mask is None:
                    continue

                idx_int = int(idx)
                meta = metadata_dict.get(idx_int, {})
                mask_bool = meta.get("mask_bool") if isinstance(meta, dict) else None
                mask_bbox = meta.get("mask_bbox") if isinstance(meta, dict) else None
                centroid = meta.get("mask_centroid") if isinstance(meta, dict) else None
                mask_area = meta.get("mask_area") if isinstance(meta, dict) else None

                metadata_valid = (
                    isinstance(mask_bool, np.ndarray)
                    and mask_bool.shape[:2] == mask.shape[:2]
                    and mask_bbox is not None
                    and len(mask_bbox) == 4
                    and centroid is not None
                    and len(centroid) == 2
                    and mask_area is not None
                    and int(mask_area) > 0
                )

                if not metadata_valid:
                    analyzed = _analyze_binary_mask(mask)
                    if analyzed is None:
                        continue
                    mask_bool = analyzed["mask_bool"]
                    mask_bbox = analyzed["mask_bbox"]
                    centroid = analyzed["mask_centroid"]
                    mask_area = analyzed["mask_area"]

                self.mask_bool_cache[idx_int] = mask_bool
                self.mask_bbox_cache[idx_int] = tuple(int(v) for v in mask_bbox)
                self.mask_centroid_cache[idx_int] = (
                    float(centroid[0]),
                    float(centroid[1]),
                )
                self.mask_area_cache[idx_int] = int(mask_area)
            except Exception:
                continue

    def get_cached_mask_centroid(self, strawberry_idx: int):
        return getattr(self, "mask_centroid_cache", {}).get(int(strawberry_idx))

    def get_cached_mask_area(self, strawberry_idx: int):
        return getattr(self, "mask_area_cache", {}).get(int(strawberry_idx))

    def _get_roi_affecting_picked_state(self, target_idx: int, picked_indices) -> Tuple[int, ...]:
        try:
            picked_unique = sorted(int(v) for v in set(picked_indices or []))
        except Exception:
            return tuple()
        if not picked_unique:
            return tuple()
        geometry = self._get_roi_geometry(int(target_idx))
        if geometry is None:
            return tuple(picked_unique)
        rx1, ry1, rx2, ry2 = [int(v) for v in geometry.get("roi_coords", (0, 0, 0, 0))]
        affecting = []
        for pid in picked_unique:
            if pid == int(target_idx):
                affecting.append(pid)
                continue
            mb = getattr(self, "mask_bbox_cache", {}).get(pid)
            if mb is None:
                box = self._get_detection_box(pid)
                mb = box if box is not None else None
            if mb is None:
                affecting.append(pid)
                continue
            bx1, by1, bx2, by2 = [int(v) for v in mb[:4]]
            ix1, iy1 = max(rx1, bx1), max(ry1, by1)
            ix2, iy2 = min(rx2, bx2), min(ry2, by2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            mask_bool = getattr(self, "mask_bool_cache", {}).get(pid)
            if mask_bool is not None and mask_bool.shape[:2] == self.prediction_image.shape[:2]:
                try:
                    if not np.any(mask_bool[iy1:iy2, ix1:ix2]):
                        continue
                except Exception:
                    affecting.append(pid)
                    continue
            affecting.append(pid)
        return tuple(affecting)

    def _precompute_roi_geometry_cache(self):
        self.roi_geometry_cache = {}
        if self.original_image is None:
            return
        for det_idx, detection_box in enumerate(self.detection_boxes, start=1):
            if detection_box is None or len(detection_box) < 4:
                continue
            try:
                x1, y1, x2, y2 = detection_box[:4]
                self.roi_geometry_cache[int(det_idx)] = _compute_roi_geometry_from_box(
                    self.original_image.shape,
                    (x1, y1, x2, y2),
                )
            except Exception:
                continue

    def _get_roi_geometry(self, strawberry_idx: int):
        geometry = self.roi_geometry_cache.get(int(strawberry_idx))
        if geometry is not None:
            return geometry
        box = self._get_detection_box(strawberry_idx)
        if box is None or self.original_image is None:
            return None
        geometry = _compute_roi_geometry_from_box(self.original_image.shape, box)
        self.roi_geometry_cache[int(strawberry_idx)] = geometry
        return geometry

    def _get_detection_box(self, strawberry_idx: int):
        if not (1 <= strawberry_idx <= len(self.detection_boxes)):
            return None
        return unpack_detection_box(self.detection_boxes[strawberry_idx - 1])

    def _apply_white_fill(self, idx: int, x1: int, y1: int, x2: int, y2: int, update_current: bool = False, update_prediction: bool = True):
        target_shape = self.prediction_image.shape[:2]
        idx_int = int(idx) if isinstance(idx, (int, np.integer)) else idx
        mask_bool = getattr(self, "mask_bool_cache", {}).get(idx_int)
        mask_bbox = getattr(self, "mask_bbox_cache", {}).get(idx_int)

        if mask_bool is not None and mask_bbox is not None and mask_bool.shape[:2] == target_shape:
            bx1, by1, bx2, by2 = mask_bbox
            # 局部 mask 操作与原来的整图 self.prediction_image[mask > 0] 等价，
            # 但避免每次扫描整张图，目标多、路径评估多时 CPU 开销明显更小。
            local_mask = mask_bool[by1:by2, bx1:bx2]
            if update_current:
                current_region = self.current_image[by1:by2, bx1:bx2]
                current_region[local_mask] = (255, 255, 255)
            if update_prediction:
                prediction_region = self.prediction_image[by1:by2, bx1:bx2]
                prediction_region[local_mask] = (255, 255, 255)
            return True

        use_mask = self.index_to_mask is not None and isinstance(idx, int) and idx in self.index_to_mask
        if use_mask:
            mask = self.index_to_mask[idx]
            if mask is not None and mask.shape[:2] == target_shape:
                mask_bool = mask > 0
                if update_current:
                    self.current_image[mask_bool] = (255, 255, 255)
                if update_prediction:
                    self.prediction_image[mask_bool] = (255, 255, 255)
                return True
        if update_current:
            self.current_image[y1:y2, x1:x2] = (255, 255, 255)
        if update_prediction:
            self.prediction_image[y1:y2, x1:x2] = (255, 255, 255)
        return False

    def _load_difficulty_model(self):
        global CACHED_DIFFICULTY_MODEL
        if CACHED_DIFFICULTY_MODEL is not None:
            self.difficulty_model = CACHED_DIFFICULTY_MODEL
            if TORCH_AVAILABLE and device is not None:
                self.difficulty_model.to(device)
            return
        try:
            model_path = DIFFICULTY_MODEL_PATH
            if not os.path.exists(model_path):
                print(f"❌ 难度预测模型文件不存在: {model_path}")
                raise RuntimeError(f"必须的回归模型文件不存在: {model_path}")

            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, nn.Module):
                self.difficulty_model = checkpoint
                self.difficulty_model.to(device)
                self.difficulty_model.eval()
            elif isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.difficulty_model = checkpoint["model"]
                    self.difficulty_model.to(device)
                    if hasattr(self.difficulty_model, "eval"):
                        self.difficulty_model.eval()
                elif "model_state_dict" in checkpoint:
                    try:
                        from ultralytics.nn.tasks import DetectionModel
                        from ultralytics.utils import YAML
                        from ultralytics.nn.modules.head import Regress

                        cfg = YAML().load("ultralytics/cfg/models/11/yolo11-regression.yaml")
                        model = DetectionModel(cfg)
                        with torch.no_grad():
                            test_input = torch.randn(1, 3, 320, 320)
                            x = test_input
                            for layer in model.model[:-1]:
                                x = layer(x)
                            c1 = x.shape[1]
                        regress_head = Regress(
                            c1=c1,
                            c2=1,
                            activation="sigmoid",
                            num_residual=1,
                            se_reduction=16,
                            cbam_reduction=16,
                            dropout_p=0.5,
                            c3_n=1,
                        )
                        regress_head.i = len(model.model) - 1
                        regress_head.f = -1
                        regress_head.type = "Regress"
                        regress_head.np = sum(p.numel() for p in regress_head.parameters())
                        model.model[-1] = regress_head
                        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                        if missing:
                            print(f"   ⚠️ 丢失权重键: {list(missing)[:5]}... 共{len(missing)} 个")
                        if unexpected:
                            print(f"   ⚠️ 多余权重键: {list(unexpected)[:5]}... 共{len(unexpected)} 个")
                        model.eval()
                        self.difficulty_model = model
                        self.difficulty_model.to(device)
                    except Exception as e:
                        print(f"❌  YOLO11 回归结构重建失败: {e}")
                        raise RuntimeError(f"YOLO11 回归结构重建失败: {e}")
                elif "state_dict" in checkpoint:
                    raise RuntimeError("只找到state_dict，需要模型架构信息才能重建模型")
                else:
                    raise RuntimeError(f"未知的checkpoint格式: {type(checkpoint)}")
            else:
                raise RuntimeError(f"模型类型未知：{type(checkpoint)}")

            if self.difficulty_model is None:
                raise RuntimeError("模型加载后为None")

            CACHED_DIFFICULTY_MODEL = self.difficulty_model
            if TORCH_AVAILABLE and device is not None:
                CACHED_DIFFICULTY_MODEL.to(device)
        except Exception as e:
            print(f"❌ 加载难度预测模型失败: {e}")
            print("❌ 难度预测模型最终加载状态: 失败")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"加载回归模型失败: {e}")

    def deduct_single_strawberry(self, strawberry_idx):
        if not (1 <= strawberry_idx <= len(self.detection_boxes)):
            raise IndexError(f"草莓索引 {strawberry_idx} 超出范围")
        box = self._get_detection_box(strawberry_idx)
        if box is None:
            raise ValueError(f"检测框 {strawberry_idx} 格式错误或不存在")
        x1, y1, x2, y2 = box

        used_mask = self._apply_white_fill(
            strawberry_idx,
            x1,
            y1,
            x2,
            y2,
            update_current=False,
            update_prediction=True,
        )
        if self.index_to_mask is not None and strawberry_idx in self.index_to_mask and not used_mask:
            print("      ⚠️  mask missing/size mismatch, fallback to white bbox")

        if strawberry_idx not in self.picked_boxes:
            self.picked_boxes.append(strawberry_idx)

    def predict_difficulty_for_point(self, target_idx, picked_indices=None, input_image=None):
        if picked_indices is None:
            picked_indices = []
        if len(picked_indices) > 0 and input_image is None:
            if len(picked_indices) != len(self.picked_boxes):
                print(f"⚠️  状态不一致警告：picked_indices={picked_indices}, 实际已采摘={self.picked_boxes}")
                print("    建议：使用明确的input_image参数或传入空列表[]")
        if not (1 <= target_idx <= len(self.detection_boxes)):
            raise IndexError(f"目标索引 {target_idx} 超出范围")
        target_box = self.detection_boxes[target_idx - 1]
        if len(target_box) >= 4:
            x1, y1, x2, y2 = target_box[:4]
        else:
            raise ValueError(f"检测框 {target_idx} 格式错误: {target_box}")

        # ROI影响状态缓存：只要目标ROI内的实际像素状态相同，就复用难度值。
        # 该优化不跳过重规划、不改变模型输入；只是避免远处已采目标造成缓存key无效。
        picked_state_for_cache = self._get_roi_affecting_picked_state(
            target_idx,
            picked_indices if picked_indices is not None else self.picked_boxes,
        )
        full_state_cache_key = _make_full_state_difficulty_cache_key(
            getattr(self, "current_image_id", None),
            target_idx,
            picked_state_for_cache,
        )
        cached_full_state_difficulty = _get_cached_full_state_difficulty(full_state_cache_key)
        if cached_full_state_difficulty is not None:
            return cached_full_state_difficulty

        input_img = input_image if input_image is not None else self.prediction_image
        roi_geometry = self._get_roi_geometry(target_idx)
        roi_320, _, _ = preprocess_roi_image_for_model(
            input_img,
            (x1, y1, x2, y2),
            roi_geometry=roi_geometry,
        )
        if roi_320 is None:
            raise RuntimeError(f"目标 {target_idx} 的ROI区域无效")

        cache_key = _make_roi_difficulty_cache_key(roi_320)
        cached_difficulty = _get_cached_difficulty(cache_key)
        if cached_difficulty is not None:
            final_difficulty = cached_difficulty
            _store_cached_full_state_difficulty(full_state_cache_key, final_difficulty)
            return final_difficulty

        roi_tensor = roi_image_to_model_tensor(roi_320)
        if roi_tensor is None:
            raise RuntimeError(f"目标 {target_idx} 的ROI tensor生成失败")

        if self.difficulty_model is not None and TORCH_AVAILABLE:
            if device is None or getattr(device, "type", "cpu") != "cuda":
                raise RuntimeError("❌ 当前代码要求单颗难度预测必须使用GPU/CUDA，不允许CPU推理。")
            try:
                self.difficulty_model.eval()
                with torch.inference_mode():
                    roi_tensor = roi_tensor.to(device, non_blocking=True)
                    output = self.difficulty_model(roi_tensor)
                    if isinstance(output, (list, tuple)):
                        base_difficulty = output[0].item()
                    else:
                        base_difficulty = output.item()
                    base_difficulty = max(0.0, min(1.0, base_difficulty))
            except Exception as e:
                print(f"❌ 回归模型预测失败: {e}")
                raise RuntimeError(f"必须使用回归模型进行难度预测，但模型预测失败: {e}")
        else:
            if self.difficulty_model is None:
                raise RuntimeError("回归模型未加载，无法进行难度预测")
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch不可用，无法运行回归模型")
            raise RuntimeError("未知错误，无法进行难度预测")

        final_difficulty = base_difficulty
        _store_cached_difficulty(cache_key, final_difficulty)
        _store_cached_full_state_difficulty(full_state_cache_key, final_difficulty)
        return final_difficulty

    def predict_difficulty_for_points_batched(self, indices_list, input_image=None):
        results = {}
        if not indices_list:
            return results
        if self.difficulty_model is None or not TORCH_AVAILABLE:
            raise RuntimeError("回归模型未加载或PyTorch不可用，无法进行批量预测")
        if device is None or getattr(device, "type", "cpu") != "cuda":
            raise RuntimeError("❌ 当前代码要求批量难度预测必须使用GPU/CUDA，不允许CPU推理。")

        base_img = self.prediction_image if input_image is None else input_image
        roi_tensors = []
        roi_meta = []
        roi_cache_keys = []
        for idx in indices_list:
            if not (1 <= idx <= len(self.detection_boxes)):
                raise IndexError(f"目标索引 {idx} 超出范围")
            target_box = self.detection_boxes[idx - 1]
            if len(target_box) < 4:
                raise ValueError(f"检测框 {idx} 格式错误: {target_box}")
            x1, y1, x2, y2 = target_box[:4]
            roi_geometry = self._get_roi_geometry(idx)
            roi_320, _, _ = preprocess_roi_image_for_model(
                base_img,
                (x1, y1, x2, y2),
                roi_geometry=roi_geometry,
            )
            if roi_320 is None:
                raise RuntimeError(f"目标 {idx} 的ROI区域无效")

            cache_key = _make_roi_difficulty_cache_key(roi_320)
            cached_difficulty = _get_cached_difficulty(cache_key)
            if cached_difficulty is not None:
                results[idx] = float(cached_difficulty)
                continue

            roi_tensor = roi_image_to_model_tensor(roi_320)
            if roi_tensor is None:
                raise RuntimeError(f"目标 {idx} 的ROI tensor生成失败")
            roi_tensors.append(roi_tensor)
            roi_meta.append(idx)
            roi_cache_keys.append(cache_key)
        if not roi_tensors:
            return results

        batch = torch.cat(roi_tensors, dim=0).to(device, non_blocking=True)
        self.difficulty_model.eval()
        self.difficulty_model.to(device)

        with torch.inference_mode():
            if YOLO_USE_HALF:
                from torch import autocast
                with autocast(device_type="cuda", dtype=torch.float16):
                    output = self.difficulty_model(batch)
            else:
                output = self.difficulty_model(batch)

        if isinstance(output, (list, tuple)):
            output_vals = output[0].detach().float().cpu().numpy().reshape(-1)
        else:
            out = output.detach().float().cpu()
            output_vals = out.view(-1).numpy()
        output_vals = np.clip(output_vals, 0.0, 1.0)
        for idx, val, cache_key in zip(roi_meta, output_vals, roi_cache_keys):
            results[idx] = float(val)
            _store_cached_difficulty(cache_key, float(val))
        return results
