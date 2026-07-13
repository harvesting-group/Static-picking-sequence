"""MOP-GA entry point and genetic path optimizer."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    GAPathPlanningConfig as PathPlanningConfig,
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
        self.path_fitness_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
        self.optimization_stats = {
            "ga_iterations": 0,
            "total_evaluations": 0,
            "path_fitness_cache_hits": 0,
            "path_fitness_cache_stores": 0,
            "best_fitness_history": [],
            "ga_trace": [],
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
                # 难度值截断为两位小数，不进行四舍五入；只过滤截断后 > 当前设定阈值的草莓。
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

        # 三目标适应度仅包含：难度项 + 路径长度项 + 漏采项。
        # 保留旧变量名 skip_rate / w3_skip_rate，避免影响现有结果文件和下游统计代码。
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

    def _add_unique_path(self, population: List[List[int]], seen_paths: set, path: List[int]) -> bool:
        if len(population) >= self.config.population_size:
            return False
        normalized = self._normalize_path(path)
        key = tuple(normalized)
        if key in seen_paths:
            return False
        population.append(normalized)
        seen_paths.add(key)
        return True

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

    def _interleave_paths(self, primary: List[int], secondary: List[int]) -> List[int]:
        result = []
        seen = set()
        max_len = max(len(primary), len(secondary))
        for i in range(max_len):
            for source in (primary, secondary):
                if i < len(source) and source[i] not in seen:
                    result.append(source[i])
                    seen.add(source[i])
        return self._normalize_path(result)

    def generate_initial_population(self) -> List[List[int]]:
        population: List[List[int]] = []
        seen_paths = set()
        if self.n_eligible == 0:
            return population

        bottom_to_top_path = self._build_bottom_to_top_path()
        difficulty_ascending_path = self._build_difficulty_ascending_path()
        nearest_neighbor_path = self._build_nearest_neighbor_path()
        heuristic_paths = [
            ("bottom_to_top", bottom_to_top_path),
            ("difficulty_ascending", difficulty_ascending_path),
            ("nearest_neighbor", nearest_neighbor_path),
        ]

        self.optimization_stats["initial_heuristic_paths"] = {name: path.copy() for name, path in heuristic_paths}
        self.optimization_stats["initial_population_method"] = (
            "heuristic_seeded_no_random_shuffle: bottom_to_top + difficulty_ascending + nearest_neighbor + deterministic_variations"
        )


        for _, path in heuristic_paths:
            self._add_unique_path(population, seen_paths, path)

        n = self.n_eligible
        if n <= 1:
            while len(population) < self.config.population_size:
                population.append(bottom_to_top_path.copy())
            self.optimization_stats["initial_population_unique_count"] = len(seen_paths)
            return population

        # 1) 确定性交换扰动
        for _, base_path in heuristic_paths:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = base_path.copy()
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    self._add_unique_path(population, seen_paths, candidate)
                    if len(population) >= self.config.population_size:
                        break
                if len(population) >= self.config.population_size:
                    break
            if len(population) >= self.config.population_size:
                break

        # 2) 确定性区间逆序扰动
        if len(population) < self.config.population_size:
            for _, base_path in heuristic_paths:
                for start in range(n - 1):
                    for end in range(start + 2, n + 1):
                        candidate = base_path.copy()
                        candidate[start:end] = reversed(candidate[start:end])
                        self._add_unique_path(population, seen_paths, candidate)
                        if len(population) >= self.config.population_size:
                            break
                    if len(population) >= self.config.population_size:
                        break
                if len(population) >= self.config.population_size:
                    break

        # 3) 确定性插入扰动
        if len(population) < self.config.population_size:
            for _, base_path in heuristic_paths:
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        candidate = base_path.copy()
                        value = candidate.pop(i)
                        candidate.insert(j, value)
                        self._add_unique_path(population, seen_paths, candidate)
                        if len(population) >= self.config.population_size:
                            break
                    if len(population) >= self.config.population_size:
                        break
                if len(population) >= self.config.population_size:
                    break

        # 4) 启发式路径交错组合
        if len(population) < self.config.population_size:
            for _, path_a in heuristic_paths:
                for _, path_b in heuristic_paths:
                    if path_a == path_b:
                        continue
                    self._add_unique_path(population, seen_paths, self._interleave_paths(path_a, path_b))
                    if len(population) >= self.config.population_size:
                        break
                if len(population) >= self.config.population_size:
                    break

        # 5) 仍不足时复制已有个体补齐
        fill_idx = 0
        while len(population) < self.config.population_size:
            source = population[fill_idx % len(population)] if population else bottom_to_top_path
            population.append(source.copy())
            fill_idx += 1

        self.optimization_stats["initial_population_unique_count"] = len(seen_paths)
        return population

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

    def tournament_selection(self, population: List[List[int]], fitness_values: List[float], tournament_size: int = 3) -> List[int]:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        if len(parent1) < 2:
            return parent1.copy(), parent2.copy()
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [-1] * size
        child2 = [-1] * size
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        fill1 = [x for x in parent2 if x not in child1]
        fill2 = [x for x in parent1 if x not in child2]
        ptr1 = 0
        ptr2 = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = fill1[ptr1]
                ptr1 += 1
            if child2[i] == -1:
                child2[i] = fill2[ptr2]
                ptr2 += 1
        return child1, child2

    def swap_mutation(self, individual: List[int]) -> List[int]:
        if len(individual) < 2:
            return individual.copy()
        mutated = individual.copy()
        if random.random() < self.config.mutation_rate:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def genetic_algorithm_optimization(self, initial_population: List[List[int]]) -> Tuple[List[int], float]:
        population = initial_population.copy()
        self.optimization_stats["ga_trace"] = []
        fitness_values = self._evaluate_population_fitness(population)

        best_idx = fitness_values.index(min(fitness_values))
        best_path = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        initial_avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else float("inf")
        if SAVE_OPTIMIZATION_TRACE:
            self.optimization_stats["ga_trace"].append(
                {
                    "generation": 0,
                    "label": "initial_population",
                    "generation_best_path": best_path.copy(),
                    "generation_best_fitness": float(best_fitness),
                    "global_best_path": best_path.copy(),
                    "global_best_fitness": float(best_fitness),
                    "avg_fitness": float(initial_avg_fitness),
                }
            )

        no_improve_rounds = 0
        for generation in range(self.config.generations):
            elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[: self.config.elite_size]
            new_population = [population[i].copy() for i in elite_indices]

            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                child1 = self.swap_mutation(child1)
                child2 = self.swap_mutation(child2)
                new_population.extend([child1, child2])

            population = new_population[: self.config.population_size]
            fitness_values = self._evaluate_population_fitness(population)

            current_best_idx = fitness_values.index(min(fitness_values))
            current_best_fitness = fitness_values[current_best_idx]
            current_best_path = population[current_best_idx].copy()
            avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else float("inf")

            if current_best_fitness < best_fitness:
                best_path = current_best_path.copy()
                best_fitness = current_best_fitness
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            self.optimization_stats["ga_iterations"] += 1
            self.optimization_stats["best_fitness_history"].append(best_fitness)
            if SAVE_OPTIMIZATION_TRACE:
                self.optimization_stats["ga_trace"].append(
                    {
                        "generation": generation + 1,
                        "label": f"generation_{generation + 1}",
                        "generation_best_path": current_best_path.copy(),
                        "generation_best_fitness": float(current_best_fitness),
                        "global_best_path": best_path.copy(),
                        "global_best_fitness": float(best_fitness),
                        "avg_fitness": float(avg_fitness),
                    }
                )

            if no_improve_rounds >= self.config.early_stop_patience:
                break

        return best_path, best_fitness

    def optimize_path(self) -> Tuple[List[int], float, Dict]:
        if self.n_eligible == 0:
            return [], float("inf"), {
                "algorithm": "Heuristic-seeded Genetic Algorithm (三目标优化)",
                "best_objective_value": float("inf"),
                "optimization_stats": self.optimization_stats.copy(),
                "best_fitness_history": [],
                "ga_trace": [],
            }

        population = self.generate_initial_population()
        best_path, best_fitness = self.genetic_algorithm_optimization(population)

        # 当前轮最优后缀路径复评：只用于当前轮GA结果，不再生成全局最终完整复评顺序。
        final_objective_value, final_details = self.evaluate_path_fitness(best_path, return_details=True)
        final_details["final_re_evaluated"] = True


        optimization_info = {
            "algorithm": "Heuristic-seeded Genetic Algorithm (三目标优化)",
            "best_objective_value": final_objective_value,
            "ga_best_objective_value": best_fitness,
            "current_round_objective_value": final_objective_value,
            "current_round_details": final_details,
            "optimization_stats": self.optimization_stats.copy(),
            "best_fitness_history": self.optimization_stats["best_fitness_history"].copy(),
            "ga_trace": [dict(item) for item in self.optimization_stats.get("ga_trace", [])] if SAVE_OPTIMIZATION_TRACE else [],
        }
        return best_path, final_objective_value, optimization_info

    def _evaluate_population_fitness(self, population: List[List[int]]) -> List[float]:
        fitness_values = []
        for individual in population:
            fitness, _ = self.evaluate_path_fitness(individual, return_details=False)
            fitness_values.append(fitness)
        return fitness_values

def main():
    from evaluation import process_test_folder_images_ga

    if USE_FIXED_SEED:
        seed_everything(FIXED_SEED)
    result = process_test_folder_images_ga(PathOptimizer, PathPlanningConfig)
    if not result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
