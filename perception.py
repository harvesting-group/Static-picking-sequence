"""YOLO detection, instance segmentation, mask analysis, and model warm-up."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from config import (
    CUDA_DEVICE_INDEX, YOLO_USE_HALF, YOLO_AUGMENT,
    DETECTION_CONF_THRESHOLD, DETECTION_IOU_THRESHOLD,
    SEGMENTATION_CONF_THRESHOLD, SEGMENTATION_IOU_THRESHOLD,
    SEGMENT_MATCH_SCORE_THRESHOLD, SEGMENT_MATCH_MIN_BBOX_IOU,
)

TORCH_AVAILABLE = True

def unpack_detection_box(detection_box):
    """安全解包检测框，返回 (x1, y1, x2, y2) 或 None。"""
    if detection_box is None or len(detection_box) < 4:
        return None
    x1, y1, x2, y2 = detection_box[:4]
    return x1, y1, x2, y2

def build_mask_metadata_from_detection_info(detection_info) -> Dict[int, Dict]:
    """
    从检测/分割阶段已计算的 mask 元数据构建 1-based 缓存映射。
    面积、边界框、bool 图和质心直接复用，不在处理器初始化时重复扫描 mask。
    """
    metadata: Dict[int, Dict] = {}
    for det_idx, info in enumerate(detection_info or [], start=1):
        if not isinstance(info, dict):
            continue
        metadata[det_idx] = {
            "mask_bool": info.get("mask_bool"),
            "mask_bbox": info.get("mask_bbox"),
            "mask_centroid": info.get("mask_centroid"),
            "mask_area": info.get("mask_area"),
        }
    return metadata

def create_polygon_mask(polygon_points, img_size):
    h, w = img_size
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon_points) < 3:
        return mask
    cleaned_points = []
    for i, point in enumerate(polygon_points):
        if i == 0 or point != polygon_points[i - 1]:
            cleaned_points.append(point)
    if len(cleaned_points) < 3:
        return mask
    try:
        points_array = np.array(cleaned_points, dtype=np.float32)
        perimeter = cv2.arcLength(points_array, closed=True)
        epsilon = max(0.005 * perimeter, 0.5)
        simplified_points = cv2.approxPolyDP(points_array, epsilon, closed=True)
        if len(simplified_points) < 3:
            simplified_points = points_array
    except Exception:
        simplified_points = np.array(cleaned_points, dtype=np.int32)
    points = np.array(simplified_points, dtype=np.int32)
    if points.ndim == 1:
        if len(points) % 2 == 0:
            points = points.reshape(-1, 2)
        else:
            points = np.array(cleaned_points, dtype=np.int32)
    elif points.shape[1] != 2:
        points = np.array(cleaned_points, dtype=np.int32)
    if points.shape[1] == 2:
        points[:, 0] = np.clip(points[:, 0], 0, w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    else:
        return mask
    cv2.fillPoly(mask, [points], 255)
    kernel_close = np.ones((3, 3), np.uint8)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_smooth = np.ones((2, 2), np.uint8)
    mask_smoothed = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel_smooth)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_smoothed, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_final = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask_final = mask_smoothed
    if np.sum(mask_final > 0) < 5:
        return mask
    return mask_final

def load_ultralytics_model(model_path: str, model_name: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name}权重文件不存在: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("当前流程需要 ultralytics 才能使用检测模型和分割模型。请先安装 ultralytics 后再运行。") from e
    model = YOLO(model_path)
    return model

def _get_yolo_predict_device():
    """强制 YOLO 使用指定 GPU。CUDA 不可用时直接报错，不回退 CPU。"""
    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        raise RuntimeError("❌ CUDA不可用：YOLO推理已设置为只使用GPU/CUDA，不允许CPU回退。")
    return CUDA_DEVICE_INDEX

def _run_yolo_predict(model, image, conf: float, iou: float):
    predict_kwargs = {
        "source": image,
        "conf": conf,
        "iou": iou,
        "verbose": False,
        "device": _get_yolo_predict_device(),
        "half": YOLO_USE_HALF,
        "augment": YOLO_AUGMENT,
    }
    try:
        return model.predict(**predict_kwargs)
    except TypeError:
        predict_kwargs.pop("half", None)
        predict_kwargs.pop("augment", None)
        return model.predict(**predict_kwargs)

def _synchronize_cuda_for_warmup():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()

def warmup_models_before_formal_timing(
    detection_model,
    segmentation_model,
    image_files,
    test_image_dir,
):
    
    from hds_model import ImageDeductionProcessor, reset_difficulty_cache_stats

    if not image_files:
        return

    warmup_image = None
    for image_file in image_files:
        candidate_path = os.path.join(test_image_dir, image_file)
        candidate_image = cv2.imread(candidate_path)
        if candidate_image is not None:
            warmup_image = candidate_image
            break

    if warmup_image is None:
        raise RuntimeError("模型预热失败：测试文件夹中没有可读取的图片。")

    _synchronize_cuda_for_warmup()
    _run_yolo_predict(
        detection_model,
        warmup_image,
        conf=DETECTION_CONF_THRESHOLD,
        iou=DETECTION_IOU_THRESHOLD,
    )

    if segmentation_model is not detection_model:
        _run_yolo_predict(
            segmentation_model,
            warmup_image,
            conf=SEGMENTATION_CONF_THRESHOLD,
            iou=SEGMENTATION_IOU_THRESHOLD,
        )
    _synchronize_cuda_for_warmup()

    image_h, image_w = warmup_image.shape[:2]
    x1 = max(0, image_w // 3)
    y1 = max(0, image_h // 3)
    x2 = min(image_w, max(x1 + 2, (2 * image_w) // 3))
    y2 = min(image_h, max(y1 + 2, (2 * image_h) // 3))
    dummy_detection_boxes = [(x1, y1, x2, y2, 1.0)]

    warmup_processor = ImageDeductionProcessor(
        warmup_image,
        dummy_detection_boxes,
    )
    warmup_processor.current_image_id = "__model_warmup__"
    warmup_processor.predict_difficulty_for_point(
        1,
        picked_indices=[],
        input_image=warmup_image,
    )
    _synchronize_cuda_for_warmup()

    reset_difficulty_cache_stats(clear_cache=True)

def _tensor_or_array_to_numpy(value):
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None

def _bbox_iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _create_bbox_mask(bbox, img_h: int, img_w: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask

def _analyze_binary_mask(mask: np.ndarray) -> Optional[Dict]:
    if mask is None:
        return None
    mask_bool = mask > 0
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max_inclusive = int(xs.max())
    y_max_inclusive = int(ys.max())

    return {
        "mask_bool": mask_bool,
        "mask_area": int(xs.size),
        "mask_bbox": (
            x_min,
            y_min,
            x_max_inclusive + 1,
            y_max_inclusive + 1,
        ),

        "bbox": (
            x_min,
            y_min,
            x_max_inclusive,
            y_max_inclusive,
        ),
        "mask_centroid": (float(xs.mean()), float(ys.mean())),
    }

def _extract_segmentation_candidates(seg_results, img_h: int, img_w: int) -> List[Dict]:
    candidates: List[Dict] = []
    if not seg_results:
        return candidates

    result = seg_results[0]
    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)
    if masks_obj is None:
        return candidates

    confs = []
    classes = []
    if boxes_obj is not None:
        confs_np = _tensor_or_array_to_numpy(getattr(boxes_obj, "conf", None))
        cls_np = _tensor_or_array_to_numpy(getattr(boxes_obj, "cls", None))
        confs = confs_np.reshape(-1).tolist() if confs_np is not None and confs_np.size else []
        classes = cls_np.reshape(-1).tolist() if cls_np is not None and cls_np.size else []

    polygons = getattr(masks_obj, "xy", None)
    if polygons is not None and len(polygons) > 0:
        for k, poly in enumerate(polygons):
            try:
                poly_np = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            except Exception:
                continue
            if poly_np.shape[0] < 3:
                continue

            polygon_points = []
            for x, y in poly_np:
                px = max(0, min(img_w - 1, int(round(float(x)))))
                py = max(0, min(img_h - 1, int(round(float(y)))))
                polygon_points.append((px, py))

            mask = create_polygon_mask(polygon_points, (img_h, img_w))
            mask_meta = _analyze_binary_mask(mask)
            if mask_meta is None:
                continue

            candidates.append(
                {
                    "seg_idx": k + 1,
                    "class": int(classes[k]) if k < len(classes) else 0,
                    "confidence": float(confs[k]) if k < len(confs) else 1.0,
                    "bbox": mask_meta["bbox"],
                    "polygon": polygon_points,
                    "mask": mask,
                    "mask_bool": mask_meta["mask_bool"],
                    "mask_area": mask_meta["mask_area"],
                    "mask_bbox": mask_meta["mask_bbox"],
                    "mask_centroid": mask_meta["mask_centroid"],
                }
            )

    if not candidates:
        masks_data = _tensor_or_array_to_numpy(getattr(masks_obj, "data", None))
        if masks_data is not None:
            masks_data = np.asarray(masks_data)
            if masks_data.ndim == 2:
                masks_data = masks_data[None, ...]

            for k, m in enumerate(masks_data):
                mask = (m > 0.5).astype(np.uint8) * 255
                if mask.shape[:2] != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                mask_meta = _analyze_binary_mask(mask)
                if mask_meta is None:
                    continue

                candidates.append(
                    {
                        "seg_idx": k + 1,
                        "class": int(classes[k]) if k < len(classes) else 0,
                        "confidence": float(confs[k]) if k < len(confs) else 1.0,
                        "bbox": mask_meta["bbox"],
                        "polygon": [],
                        "mask": mask,
                        "mask_bool": mask_meta["mask_bool"],
                        "mask_area": mask_meta["mask_area"],
                        "mask_bbox": mask_meta["mask_bbox"],
                        "mask_centroid": mask_meta["mask_centroid"],
                    }
                )

    return candidates

def detect_targets_with_models(original_image, detection_model, segmentation_model):
    img_h, img_w = original_image.shape[:2]
    same_model = detection_model is segmentation_model

    detect_start = time.time()
    det_results = _run_yolo_predict(
        detection_model,
        original_image,
        conf=DETECTION_CONF_THRESHOLD,
        iou=DETECTION_IOU_THRESHOLD,
    )
    detect_time = time.time() - detect_start

    if same_model:
        seg_results = det_results
        seg_time = 0.0
    else:
        seg_start = time.time()
        seg_results = _run_yolo_predict(
            segmentation_model,
            original_image,
            conf=SEGMENTATION_CONF_THRESHOLD,
            iou=SEGMENTATION_IOU_THRESHOLD,
        )
        seg_time = time.time() - seg_start

    detection_info: List[Dict] = []
    if not det_results:
        return [], [], {}, detect_time, seg_time

    det_result = det_results[0]
    boxes_obj = getattr(det_result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return [], [], {}, detect_time, seg_time

    xyxy = _tensor_or_array_to_numpy(boxes_obj.xyxy)
    confs = _tensor_or_array_to_numpy(getattr(boxes_obj, "conf", None))
    classes = _tensor_or_array_to_numpy(getattr(boxes_obj, "cls", None))

    if xyxy is None or len(xyxy) == 0:
        return [], [], {}, detect_time, seg_time

    xyxy = np.asarray(xyxy, dtype=np.float32)
    confs = (
        np.asarray(confs, dtype=np.float32).reshape(-1)
        if confs is not None
        else np.ones((xyxy.shape[0],), dtype=np.float32)
    )
    classes = (
        np.asarray(classes, dtype=np.float32).reshape(-1)
        if classes is not None
        else np.zeros((xyxy.shape[0],), dtype=np.float32)
    )

    for raw_idx, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = box[:4]
        x1 = max(0, min(img_w - 1, int(round(float(x1)))))
        y1 = max(0, min(img_h - 1, int(round(float(y1)))))
        x2 = max(0, min(img_w - 1, int(round(float(x2)))))
        y2 = max(0, min(img_h - 1, int(round(float(y2)))))
        if x2 <= x1 or y2 <= y1:
            print(f"  ⚠️ 检测模型第{raw_idx}个 bbox 无效，跳过: {(x1, y1, x2, y2)}")
            continue

        conf = float(confs[raw_idx - 1]) if raw_idx - 1 < len(confs) else 1.0
        cls = int(classes[raw_idx - 1]) if raw_idx - 1 < len(classes) else 0
        detection_info.append(
            {
                "raw_index": raw_idx,
                "source_line": raw_idx,
                "source": "detection_model",
                "class": cls,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "bottom_y": int(y2),
            }
        )

    seg_candidates = _extract_segmentation_candidates(seg_results, img_h, img_w)
    detection_info.sort(key=lambda item: item["bottom_y"], reverse=True)

    detection_boxes = []
    index_to_mask = {}
    used_seg_indices = set()

    for new_idx, item in enumerate(detection_info, start=1):
        x1, y1, x2, y2 = item["bbox"]
        det_area = max(1, (x2 - x1) * (y2 - y1))
        best_candidate = None
        best_score = -1.0
        best_overlap = 0
        best_iou = 0.0

        for cand in seg_candidates:
            seg_idx = cand.get("seg_idx")
            if seg_idx in used_seg_indices:
                continue

            mask = cand.get("mask")
            mask_bool = cand.get("mask_bool")
            if mask is None or mask.shape[:2] != (img_h, img_w):
                continue
            if mask_bool is None or mask_bool.shape[:2] != (img_h, img_w):
                mask_meta = _analyze_binary_mask(mask)
                if mask_meta is None:
                    continue
                cand.update(mask_meta)
                mask_bool = mask_meta["mask_bool"]

            overlap = int(np.count_nonzero(mask_bool[y1:y2, x1:x2]))
            if overlap <= 0:
                continue

            mask_area_value = cand.get("mask_area")
            if mask_area_value is None:
                mask_area_value = int(np.count_nonzero(mask_bool))
                cand["mask_area"] = mask_area_value
            mask_area = max(1, int(mask_area_value))

            coverage = overlap / float(det_area)
            mask_precision = overlap / float(mask_area)
            bbox_iou = _bbox_iou_xyxy(
                (x1, y1, x2, y2),
                cand.get("bbox", (0, 0, 0, 0)),
            )
            score = 0.50 * coverage + 0.35 * mask_precision + 0.15 * bbox_iou
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_overlap = overlap
                best_iou = bbox_iou

        if (
            best_candidate is not None
            and best_score >= SEGMENT_MATCH_SCORE_THRESHOLD
            and best_iou >= SEGMENT_MATCH_MIN_BBOX_IOU
        ):
            mask = best_candidate["mask"].copy()
            selected_meta = {
                "mask_bool": best_candidate.get("mask_bool"),
                "mask_area": best_candidate.get("mask_area"),
                "mask_bbox": best_candidate.get("mask_bbox"),
                "mask_centroid": best_candidate.get("mask_centroid"),
            }
            used_seg_indices.add(best_candidate.get("seg_idx"))
            item["mask_source"] = "segmentation_model"
            item["segmentation_orig_idx"] = best_candidate.get("seg_idx")
            item["segmentation_match_score"] = float(best_score)
            item["segmentation_overlap_pixels"] = int(best_overlap)
            item["segmentation_bbox_iou"] = float(best_iou)
            item["polygon"] = best_candidate.get("polygon", [])
        else:
            mask = _create_bbox_mask((x1, y1, x2, y2), img_h, img_w)
            selected_meta = _analyze_binary_mask(mask)
            item["mask_source"] = "bbox_fallback_from_detection"
            item["segmentation_orig_idx"] = None
            item["segmentation_match_score"] = float(best_score if best_score >= 0 else 0.0)
            item["segmentation_overlap_pixels"] = int(best_overlap)
            item["segmentation_bbox_iou"] = float(best_iou)
            item["polygon"] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if selected_meta is None or selected_meta.get("mask_bool") is None:
            selected_meta = _analyze_binary_mask(mask)
        if selected_meta is None:
            continue

        item["strawberry_idx"] = new_idx
        item["mask"] = mask
        item["mask_bool"] = selected_meta["mask_bool"]
        item["mask_area"] = int(selected_meta["mask_area"])
        item["mask_bbox"] = tuple(selected_meta["mask_bbox"])
        item["mask_centroid"] = tuple(selected_meta["mask_centroid"])

        detection_boxes.append(
            (x1, y1, x2, y2, float(item.get("confidence", 1.0)))
        )
        index_to_mask[new_idx] = mask

    return detection_boxes, detection_info, index_to_mask, detect_time, seg_time
