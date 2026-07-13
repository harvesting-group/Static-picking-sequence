"""Algorithm-agnostic rolling global replanning."""
from __future__ import annotations

import time
from typing import Dict, List, Type

from config import FIXED_SEED, SAVE_OPTIMIZATION_TRACE, USE_FIXED_SEED
from objective import make_stable_seed_from_image_round, seed_everything, truncate_float


def rolling_global_replanning(
    processor,
    path_config,
    all_indices: List[int],
    image_id: str,
    optimizer_cls: Type,
    algorithm_key: str,
):
    """Re-optimize all remaining targets after each virtually executed target."""
    algorithm_key = algorithm_key.lower()
    if algorithm_key not in {"ga", "sa"}:
        raise ValueError(f"Unsupported algorithm_key: {algorithm_key}")

    remaining_indices = list(all_indices)
    final_order: List[int] = []
    rolling_records: List[Dict] = []
    total_replanning_time = 0.0

    while remaining_indices:
        round_id = len(final_order) + 1

        if USE_FIXED_SEED:
            round_seed = make_stable_seed_from_image_round(
                image_id,
                round_id,
                base_seed=FIXED_SEED,
            )
            seed_everything(round_seed)

        round_start = time.time()
        planner = optimizer_cls(
            processor,
            path_config,
            candidate_indices=remaining_indices,
            base_prediction_image=processor.prediction_image,
            base_picked_boxes=processor.picked_boxes,
        )
        best_suffix, best_fitness, round_info = planner.optimize_path()
        round_time = time.time() - round_start
        total_replanning_time += round_time

        if not best_suffix:
            break

        chosen_idx = int(best_suffix[0])
        try:
            chosen_difficulty_raw = processor.predict_difficulty_for_point(
                chosen_idx,
                picked_indices=planner.base_picked_boxes.copy(),
                input_image=planner.base_prediction_image,
            )
            chosen_difficulty = truncate_float(float(chosen_difficulty_raw), 2)
        except Exception as exc:
            raise RuntimeError(
                f"滚动重规划第 {round_id} 轮目标 {chosen_idx} 的HDS确认失败。"
            ) from exc

        processor.current_image = planner.base_prediction_image
        processor.prediction_image = planner.base_prediction_image.copy()
        processor.picked_boxes = planner.base_picked_boxes.copy()
        deduct_start = time.time()
        processor.deduct_single_strawberry(chosen_idx)
        deduct_time = time.time() - deduct_start

        final_order.append(chosen_idx)
        if chosen_idx in remaining_indices:
            remaining_indices.remove(chosen_idx)

        record_item = {
            "step": round_id,
            "picked_strawberry": chosen_idx,
            "chosen_strawberry_difficulty": chosen_difficulty,
            "remaining_before_replan": planner.candidate_indices.copy(),
            "best_suffix": best_suffix.copy(),
            "best_fitness": float(best_fitness),
            "round_time": float(round_time),
            "deduct_time": float(deduct_time),
            "picked_before_step": planner.base_picked_boxes.copy(),
            "picked_after_step": processor.picked_boxes.copy(),
        }
        trace_key = f"{algorithm_key}_trace"
        if SAVE_OPTIMIZATION_TRACE:
            record_item[trace_key] = [dict(item) for item in round_info.get(trace_key, [])]
        rolling_records.append(record_item)

    last_round_best_fitness = (
        float(rolling_records[-1].get("best_fitness", float("inf")))
        if rolling_records else float("inf")
    )
    label = "GA" if algorithm_key == "ga" else "SA"
    optimization_info = {
        "algorithm": f"Rolling Global Replanning ({label} each step)",
        "replanning_rounds": len(rolling_records),
        "rolling_records": rolling_records,
        "total_replanning_time": total_replanning_time,
        "final_order": final_order.copy(),
        "last_round_best_fitness": last_round_best_fitness,
        "best_objective_value": last_round_best_fitness,
    }
    return final_order, total_replanning_time, rolling_records, optimization_info
