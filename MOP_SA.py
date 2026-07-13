"""MOP-SA entry point and simulated-annealing path optimizer."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    SAPathPlanningConfig as PathPlanningConfig,
    SAVE_OPTIMIZATION_TRACE, USE_FIXED_SEED, FIXED_SEED,
)
from objective import (
    calculate_path_length_pixels, calculate_weighted_objective,
    get_strawberry_position_from_processor, seed_everything, truncate_float,
)
from perception import unpack_detection_box

class PathOptimizer:
    def __init__(
        self,
        processor: "ImageDeductionProcessor",
        config: PathPlanningConfig,
        candidate_indices: Optional[List[int]] = None,
        base_prediction_image=None,
        base_picked_boxes: Optional[List[int]] = None,
    ):
        self.processor = processor
        self.config = config
        self.n_strawberries = len(processor.detection_boxes)
        self.candidate_indices = list(candidate_indices) if candidate_indices is not None else list(range(1, self.n_strawberries + 1))
        self.base_prediction_image = np.copy(base_prediction_image) if base_prediction_image is not None else processor.prediction_image.copy()
        # 适应度评价复用同一张预分配工作图，避免每次评价都重新申请整幅图像内存。
        self._working_prediction_image = np.empty_like(self.base_prediction_image)
        self.base_picked_boxes = list(base_picked_boxes) if base_picked_boxes is not None else processor.picked_boxes.copy()

        self.prefilter_details = []
        self.eligible_strawberries = self._filter_eligible_strawberries()
        self.n_eligible = len(self.eligible_strawberries)


        self.strawberry_positions = self._extract_strawberry_positions()
        self.all_strawberry_positions = self._extract_all_strawberry_positions()
        self.distance_matrix = self._precompute_distance_matrix()
        self.eligible_index_map = {idx: i for i, idx in enumerate(self.eligible_strawberries)}
        self.difficulty_cache: Dict[Tuple[frozenset, int], float] = {}
        self.path_fitness_cache: Dict[Tuple, float] = {}
        self.optimization_stats = {
            "sa_iterations": 0,
            "temperature_updates": 0,
            "accepted_moves": 0,
            "uphill_accepted_moves": 0,
            "total_evaluations": 0,
            "path_fitness_cache_hits": 0,
            "path_fitness_cache_stores": 0,
            "best_fitness_history": [],
            "sa_trace": [],
        }

    def _reset_processor_state(self):
        np.copyto(self._working_prediction_image, self.base_prediction_image)
        self.processor.current_image = self.base_prediction_image
        self.processor.prediction_image = self._working_prediction_image
        self.processor.picked_boxes = self.base_picked_boxes.copy()

    def _get_detection_box(self, strawberry_idx: int):
        i = strawberry_idx - 1
        if not (0 <= i < len(self.processor.detection_boxes)):
            return None
        return unpack_detection_box(self.processor.detection_boxes[i])

    def _get_strawberry_center(self, strawberry_idx: int) -> Tuple[float, float]:
        if hasattr(self.processor, "get_cached_mask_centroid"):
            centroid = self.processor.get_cached_mask_centroid(strawberry_idx)
            if centroid is not None:
                return centroid
        return get_strawberry_position_from_processor(self.processor, strawberry_idx)

    def _filter_eligible_strawberries(self) -> List[int]:
        eligible_strawberries = []
        self.prefilter_details = []

        for strawberry_idx in self.candidate_indices:
            try:
                difficulty = self.processor.predict_difficulty_for_point(
                    strawberry_idx,
                    picked_indices=self.base_picked_boxes.copy(),
                    input_image=self.base_prediction_image,
                )
                difficulty_raw = float(difficulty)
                difficulty_for_filter = truncate_float(difficulty_raw, 2)
                is_kept = difficulty_for_filter <= self.config.difficulty_threshold
                status = "kept" if is_kept else "filtered"

                self.prefilter_details.append(
                    {
                        "strawberry_idx": strawberry_idx,
                        "initial_difficulty_raw": difficulty_raw,
                        "initial_difficulty_trunc2": difficulty_for_filter,
                        "difficulty_threshold": float(self.config.difficulty_threshold),
                        "prefilter_status": status,
                    }
                )

                if is_kept:
                    eligible_strawberries.append(strawberry_idx)
            except Exception as exc:
                raise RuntimeError(
                    f"目标 {strawberry_idx} 的HDS预测失败，当前图片不应继续参与实验统计。"
                ) from exc

        return eligible_strawberries

    def _extract_strawberry_positions(self) -> List[Tuple[float, float]]:
        return [self._get_strawberry_center(strawberry_idx) for strawberry_idx in self.eligible_strawberries]

    def _extract_all_strawberry_positions(self) -> List[Tuple[float, float]]:
        return [self._get_strawberry_center(strawberry_idx) for strawberry_idx in range(1, self.n_strawberries + 1)]

    def _precompute_distance_matrix(self) -> List[List[float]]:
        n = len(self.strawberry_positions)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            x1, y1 = self.strawberry_positions[i]
            for j in range(i + 1, n):
                x2, y2 = self.strawberry_positions[j]
                d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    def evaluate_path_fitness(self, path: List[int], return_details: bool = True) -> Tuple[float, Dict]:
        if len(path) != self.n_eligible:
            details = {"error": f"路径长度不匹配: 期望{self.n_eligible}，实际{len(path)}"}
            return float("inf"), details if return_details else {}

        path_cache_key = (tuple(self.base_picked_boxes), tuple(int(v) for v in path))
        if (not return_details) and (path_cache_key in self.path_fitness_cache):
            self.optimization_stats["total_evaluations"] += 1
            self.optimization_stats["path_fitness_cache_hits"] += 1
            return float(self.path_fitness_cache[path_cache_key]), {}

        self._reset_processor_state()

        total_difficulty = 0.0
        total_distance = 0.0
        difficulty_sequence = [] if return_details else None
        distance_sequence = [] if return_details else None
        skipped_strawberries = [] if return_details else None
        picked_strawberries = []
        distance_segments = 0

        last_picked_strawberry = (
            int(self.base_picked_boxes[-1])
            if getattr(self, "base_picked_boxes", None)
            else None
        )
        base_start_strawberry = last_picked_strawberry

        path_skipped_count = 0
        for strawberry_idx in path:
            if not (1 <= strawberry_idx <= self.n_strawberries):
                details = {"error": f"无效的草莓索引: {strawberry_idx}"}
                return float("inf"), details if return_details else {}

            current_picked_set = frozenset(self.processor.picked_boxes)
            cache_key = (current_picked_set, int(strawberry_idx))

            if cache_key in self.difficulty_cache:
                current_difficulty = self.difficulty_cache[cache_key]
            else:
                current_difficulty = self.processor.predict_difficulty_for_point(
                    strawberry_idx,
                    picked_indices=self.processor.picked_boxes.copy(),
                    input_image=self.processor.prediction_image,
                )
                self.difficulty_cache[cache_key] = current_difficulty

            current_difficulty_used = truncate_float(float(current_difficulty), 2)

            if current_difficulty_used > self.config.difficulty_threshold:
                path_skipped_count += 1
                if return_details:
                    skipped_strawberries.append((strawberry_idx, current_difficulty_used))
                    difficulty_sequence.append(current_difficulty_used)
                    distance_sequence.append(0.0)
                continue

            if last_picked_strawberry is None:
                distance = 0.0
            else:
                distance = self._calculate_distance(last_picked_strawberry, strawberry_idx)
                if not math.isfinite(distance):
                    distance = 0.0
                distance_segments += 1

            if return_details:
                difficulty_sequence.append(current_difficulty_used)
                distance_sequence.append(distance)
            total_difficulty += current_difficulty_used
            picked_strawberries.append(strawberry_idx)

            self.processor.deduct_single_strawberry(strawberry_idx)
            last_picked_strawberry = strawberry_idx

        picked_count = len(picked_strawberries)
        all_positions = self.all_strawberry_positions
        distance_path = picked_strawberries
        if base_start_strawberry is not None and picked_strawberries:
            distance_path = [base_start_strawberry] + picked_strawberries
        total_distance = calculate_path_length_pixels(distance_path, all_positions)
        avg_difficulty = total_difficulty / picked_count if picked_count > 0 else 1.0

        if hasattr(self.processor, "original_image") and self.processor.original_image is not None:
            img_height, img_width = self.processor.original_image.shape[:2]
            max_possible_distance = math.sqrt(img_width**2 + img_height**2)
        else:
            max_possible_distance = 1000.0

        if max_possible_distance > 0 and distance_segments > 0:
            normalized_distance = total_distance / (max_possible_distance * distance_segments)
        else:
            normalized_distance = 0.0

        candidate_count = len(self.candidate_indices)
        prefiltered_skipped = candidate_count - self.n_eligible
        total_skipped = prefiltered_skipped + path_skipped_count
        skip_rate = total_skipped / candidate_count if candidate_count > 0 else 0.0

        fitness_value = calculate_weighted_objective(
            avg_difficulty, normalized_distance, skip_rate, self.config
        )

        self.optimization_stats["total_evaluations"] += 1
        if not return_details:
            self.path_fitness_cache[path_cache_key] = float(fitness_value)
            self.optimization_stats["path_fitness_cache_stores"] += 1
            return fitness_value, {}

        evaluation_details = {
            "total_difficulty": total_difficulty,
            "avg_difficulty": avg_difficulty,
            "total_distance": total_distance,
            "distance_segments": distance_segments,
            "normalized_distance": normalized_distance,
            "max_possible_distance": max_possible_distance,
            "base_start_strawberry": base_start_strawberry,
            "skip_rate": skip_rate,
            "candidate_count": candidate_count,
            "skipped_count": total_skipped,
            "prefiltered_skipped": prefiltered_skipped,
            "path_skipped": path_skipped_count,
            "picked_count": picked_count,
            "picked_strawberries": picked_strawberries,
            "skipped_strawberries": skipped_strawberries,
            "difficulty_sequence": difficulty_sequence,
            "distance_sequence": distance_sequence,
            "fitness_breakdown": {
                "difficulty_component": self.config.w1_difficulty * avg_difficulty,
                "distance_component": self.config.w2_distance * normalized_distance,
                "skip_rate_component": self.config.w3_skip_rate * skip_rate,
            },
        }
        return fitness_value, evaluation_details

    def _normalize_path(self, path: List[int]) -> List[int]:
        eligible_set = set(self.eligible_strawberries)
        normalized = []
        seen = set()
        for idx in path:
            if idx in eligible_set and idx not in seen:
                normalized.append(idx)
                seen.add(idx)
        for idx in self._build_bottom_to_top_path():
            if idx not in seen:
                normalized.append(idx)
                seen.add(idx)
        return normalized

    def _get_initial_difficulty_for_sort(self, strawberry_idx: int) -> float:
        for item in getattr(self, "prefilter_details", []):
            if item.get("strawberry_idx") == strawberry_idx:
                val = item.get("initial_difficulty_trunc2", item.get("initial_difficulty_raw", None))
                if val is not None:
                    return float(val)
        return float("inf")

    def _get_bottom_y_for_sort(self, strawberry_idx: int) -> float:
        box = self._get_detection_box(strawberry_idx)
        if box is None:
            return -1.0
        _, _, _, y2 = box
        return float(y2)

    def _build_bottom_to_top_path(self) -> List[int]:
        return sorted(self.eligible_strawberries, key=lambda idx: (-self._get_bottom_y_for_sort(idx), idx))

    def _build_difficulty_ascending_path(self) -> List[int]:
        return sorted(
            self.eligible_strawberries,
            key=lambda idx: (self._get_initial_difficulty_for_sort(idx), -self._get_bottom_y_for_sort(idx), idx),
        )

    def _build_nearest_neighbor_path(self) -> List[int]:
        if not self.eligible_strawberries:
            return []
        unvisited = set(self.eligible_strawberries)
        current = max(unvisited, key=lambda idx: (self._get_bottom_y_for_sort(idx), -idx))
        path = [current]
        unvisited.remove(current)
        while unvisited:
            next_idx = min(
                unvisited,
                key=lambda idx: (self._calculate_distance(current, idx), -self._get_bottom_y_for_sort(idx), idx),
            )
            path.append(next_idx)
            unvisited.remove(next_idx)
            current = next_idx
        return path

    def _calculate_distance(self, idx1: int, idx2: int) -> float:
        if idx1 == idx2:
            return 0.0
        pos1_idx = getattr(self, "eligible_index_map", {}).get(idx1)
        pos2_idx = getattr(self, "eligible_index_map", {}).get(idx2)
        if pos1_idx is not None and pos2_idx is not None:
            return self.distance_matrix[pos1_idx][pos2_idx]

        x1, y1 = self._get_strawberry_center(idx1)
        x2, y2 = self._get_strawberry_center(idx2)
        if not all(math.isfinite(v) for v in [x1, y1, x2, y2]):
            return float("inf")
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def generate_initial_solution(self) -> List[int]:
        if self.n_eligible == 0:
            self._initial_solution_cached_fitness = float("inf")
            return []

        heuristic_candidates = [
            ("bottom_to_top", self._build_bottom_to_top_path()),
            ("difficulty_ascending", self._build_difficulty_ascending_path()),
            ("nearest_neighbor", self._build_nearest_neighbor_path()),
        ]

        unique_candidates = []
        seen_paths = set()
        for name, path in heuristic_candidates:
            normalized = self._normalize_path(path)
            key = tuple(normalized)
            if key not in seen_paths:
                unique_candidates.append((name, normalized))
                seen_paths.add(key)


        best_name = None
        best_path = None
        best_fitness = float("inf")
        heuristic_initial_solutions = {}

        for name, path in unique_candidates:
            fitness, _ = self.evaluate_path_fitness(path, return_details=False)
            heuristic_initial_solutions[name] = {
                "path": path.copy(),
                "fitness": float(fitness),
            }
            if fitness < best_fitness:
                best_name = name
                best_path = path.copy()
                best_fitness = fitness

        if best_path is None:
            best_path = self._build_bottom_to_top_path()
            best_name = "bottom_to_top_fallback"
            best_fitness, _ = self.evaluate_path_fitness(best_path, return_details=False)

        self.optimization_stats["initial_solution_method"] = (
            "heuristic_best_of_three: bottom_to_top + difficulty_ascending + nearest_neighbor"
        )
        self.optimization_stats["initial_heuristic_solutions"] = heuristic_initial_solutions
        self.optimization_stats["selected_initial_solution_name"] = best_name
        self.optimization_stats["selected_initial_solution_fitness"] = float(best_fitness)
        self._initial_solution_cached_fitness = float(best_fitness)

        return best_path

    def _generate_neighbor_solution(self, current_path: List[int]) -> List[int]:
        """生成模拟退火邻域解：随机采用交换、区间逆序或插入移动。"""
        if len(current_path) < 2:
            return current_path.copy()

        neighbor = current_path.copy()
        move_selector = random.random()
        swap_boundary = self.config.sa_swap_probability
        reverse_boundary = self.config.sa_swap_probability + self.config.sa_reverse_probability

        if move_selector < swap_boundary:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif move_selector < reverse_boundary:
            i, j = sorted(random.sample(range(len(neighbor)), 2))
            neighbor[i : j + 1] = reversed(neighbor[i : j + 1])
        else:
            i, j = random.sample(range(len(neighbor)), 2)
            item = neighbor.pop(i)
            neighbor.insert(j, item)

        return neighbor

    def simulated_annealing_optimization(self) -> Tuple[List[int], float]:
        """纯模拟退火优化，并记录每次迭代的当前解、候选解、最优解与目标函数值。"""
        self.optimization_stats["sa_trace"] = []
        self.optimization_stats["best_fitness_history"] = []

        current_path = self.generate_initial_solution()
        if hasattr(self, "_initial_solution_cached_fitness"):
            current_fitness = float(self._initial_solution_cached_fitness)
        else:
            current_fitness, _ = self.evaluate_path_fitness(current_path, return_details=False)
        best_path = current_path.copy()
        best_fitness = current_fitness

        temperature = float(self.config.sa_initial_temperature)
        iteration = 0
        no_improve_temperature_levels = 0

        if SAVE_OPTIMIZATION_TRACE:
            self.optimization_stats["sa_trace"].append(
                {
                    "iteration": 0,
                    "label": "initial_solution",
                    "temperature": float(temperature),
                    "current_path": current_path.copy(),
                    "current_fitness": float(current_fitness),
                    "candidate_path": current_path.copy(),
                    "candidate_fitness": float(current_fitness),
                    "accepted": True,
                    "acceptance_probability": 1.0,
                    "delta": 0.0,
                    "global_best_path": best_path.copy(),
                    "global_best_fitness": float(best_fitness),
                }
            )
        self.optimization_stats["best_fitness_history"].append(float(best_fitness))

        while temperature > self.config.sa_final_temperature and iteration < self.config.sa_max_iterations:
            improved_in_this_temperature = False

            for _ in range(self.config.sa_iterations_per_temperature):
                if iteration >= self.config.sa_max_iterations:
                    break

                iteration += 1
                candidate_path = self._generate_neighbor_solution(current_path)
                candidate_fitness, _ = self.evaluate_path_fitness(candidate_path, return_details=False)
                delta = candidate_fitness - current_fitness

                if delta <= 0:
                    acceptance_probability = 1.0
                    accepted = True
                else:
                    acceptance_probability = math.exp(-delta / temperature) if temperature > 0 else 0.0
                    accepted = random.random() < acceptance_probability

                if accepted:
                    current_path = candidate_path.copy()
                    current_fitness = candidate_fitness
                    self.optimization_stats["accepted_moves"] += 1
                    if delta > 0:
                        self.optimization_stats["uphill_accepted_moves"] += 1

                if current_fitness < best_fitness:
                    best_path = current_path.copy()
                    best_fitness = current_fitness
                    improved_in_this_temperature = True

                self.optimization_stats["sa_iterations"] += 1
                self.optimization_stats["best_fitness_history"].append(float(best_fitness))
                if SAVE_OPTIMIZATION_TRACE:
                    self.optimization_stats["sa_trace"].append(
                        {
                            "iteration": iteration,
                            "label": f"sa_iteration_{iteration}",
                            "temperature": float(temperature),
                            "current_path": current_path.copy(),
                            "current_fitness": float(current_fitness),
                            "candidate_path": candidate_path.copy(),
                            "candidate_fitness": float(candidate_fitness),
                            "accepted": bool(accepted),
                            "acceptance_probability": float(acceptance_probability),
                            "delta": float(delta),
                            "global_best_path": best_path.copy(),
                            "global_best_fitness": float(best_fitness),
                        }
                    )


            self.optimization_stats["temperature_updates"] += 1
            if improved_in_this_temperature:
                no_improve_temperature_levels = 0
            else:
                no_improve_temperature_levels += 1

            if no_improve_temperature_levels >= self.config.early_stop_patience:
                break

            temperature *= self.config.sa_cooling_rate

        return best_path, best_fitness

    def optimize_path(self) -> Tuple[List[int], float, Dict]:
        if self.n_eligible == 0:
            return [], float("inf"), {
                "algorithm": "Heuristic-seeded Simulated Annealing (三目标优化)",
                "best_objective_value": float("inf"),
                "optimization_stats": self.optimization_stats.copy(),
                "best_fitness_history": [],
                "sa_trace": [],
            }

        best_path, best_fitness = self.simulated_annealing_optimization()

        final_objective_value, final_details = self.evaluate_path_fitness(best_path, return_details=True)
        final_details["final_re_evaluated"] = True


        optimization_info = {
            "algorithm": "Heuristic-seeded Simulated Annealing (三目标优化)",
            "best_objective_value": final_objective_value,
            "sa_best_objective_value": best_fitness,
            "current_round_objective_value": final_objective_value,
            "current_round_details": final_details,
            "optimization_stats": self.optimization_stats.copy(),
            "best_fitness_history": self.optimization_stats["best_fitness_history"].copy(),
            "sa_trace": [dict(item) for item in self.optimization_stats.get("sa_trace", [])] if SAVE_OPTIMIZATION_TRACE else [],
        }
        return best_path, final_objective_value, optimization_info

def main():
    from evaluation import process_test_folder_images_sa

    if USE_FIXED_SEED:
        seed_everything(FIXED_SEED)
    result = process_test_folder_images_sa(PathOptimizer, PathPlanningConfig)
    if not result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
