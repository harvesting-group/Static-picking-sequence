from __future__ import annotations

import os
from dataclasses import dataclass

# Prevent OpenCV/Qt from requiring a desktop display when figures are saved on servers.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CUDA_DEVICE_INDEX = 0
YOLO_USE_HALF = False
YOLO_AUGMENT = False

# By default, all paths are resolved relative to this source-code directory.
# Every path can still be overridden through the corresponding environment variable.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.environ.get("MOP_BASE_DIR", PROJECT_DIR))
TEST_IMAGE_DIR = os.path.abspath(
    os.environ.get("MOP_TEST_IMAGE_DIR", os.path.join(BASE_DIR, "fit_5.8"))
)
OUTPUT_ROOT_DIR = os.path.abspath(
    os.environ.get("MOP_OUTPUT_ROOT_DIR", os.path.join(BASE_DIR, "outputs"))
)
GA_OUTPUT_DIR = os.path.abspath(
    os.environ.get("MOP_GA_OUTPUT_DIR", os.path.join(OUTPUT_ROOT_DIR, "mop_ga"))
)
SA_OUTPUT_DIR = os.path.abspath(
    os.environ.get("MOP_SA_OUTPUT_DIR", os.path.join(OUTPUT_ROOT_DIR, "mop_sa"))
)
DIFFICULTY_MODEL_PATH = os.path.abspath(
    os.environ.get(
        "MOP_DIFFICULTY_MODEL_PATH",
        os.path.join(BASE_DIR, "weights", "Difficulty", "best_model.pt"),
    )
)
STRAWBERRY_SEGMENT_MODEL_PATH = os.path.abspath(
    os.environ.get(
        "MOP_SEGMENT_MODEL_PATH",
        os.path.join(BASE_DIR, "weights", "Detection_segmentation", "best.pt"),
    )
)
DETECTION_MODEL_PATH = STRAWBERRY_SEGMENT_MODEL_PATH
SEGMENTATION_MODEL_PATH = STRAWBERRY_SEGMENT_MODEL_PATH

DETECTION_CONF_THRESHOLD = 0.75
DETECTION_IOU_THRESHOLD = 0.65
SEGMENTATION_CONF_THRESHOLD = 0.75
SEGMENTATION_IOU_THRESHOLD = 0.65
SEGMENT_MATCH_SCORE_THRESHOLD = 0.30
SEGMENT_MATCH_MIN_BBOX_IOU = 0.10
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# The value is a per-side expansion factor. A value of 3.0 gives a theoretical
# total width/height of 1 + 2*3 = 7 times the original bounding-box size.
ROI_SIDE_FACTOR = 3.0
ROI_TOTAL_EXPANSION_FACTOR = 1.0 + 2.0 * ROI_SIDE_FACTOR
ENABLE_ROI_DIFFICULTY_CACHE = True
ROI_DIFFICULTY_CACHE_MAX_ITEMS = 200000
ENABLE_FULL_STATE_DIFFICULTY_CACHE = True
FULL_STATE_DIFFICULTY_CACHE_MAX_ITEMS = 300000

SAVE_OPTIMIZATION_TRACE = False
SAVE_OPTIMIZATION_PROCESS_TXT = False
USE_FIXED_SEED = True
FIXED_SEED = 42
RESET_ROI_CACHE_EACH_IMAGE = True


@dataclass
class GAPathPlanningConfig:
    population_size: int = 40
    generations: int = 70
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 4
    w1_difficulty: float = 0.60
    w2_distance: float = 0.35
    w3_skip_rate: float = 0.05
    difficulty_threshold: float = 0.75
    early_stop_patience: int = 10

    def __post_init__(self):
        assert self.population_size > 0, "种群规模必须大于0"
        assert self.generations > 0, "最大代数必须大于0"
        assert 0 < self.elite_size <= self.population_size, "精英数量必须位于(0, population_size]"
        assert 0.0 <= self.mutation_rate <= 1.0, "变异率必须在[0,1]范围内"
        assert 0.0 <= self.crossover_rate <= 1.0, "交叉率必须在[0,1]范围内"
        assert 0.0 <= self.w1_difficulty <= 1.0, "难度权重必须在[0,1]范围内"
        assert 0.0 <= self.w2_distance <= 1.0, "距离权重必须在[0,1]范围内"
        assert 0.0 <= self.w3_skip_rate <= 1.0, "漏采率权重必须在[0,1]范围内"
        assert abs(self.w1_difficulty + self.w2_distance + self.w3_skip_rate - 1.0) < 1e-6, "三权重之和必须为1"
        assert 0.0 < self.difficulty_threshold < 1.0, "难度阈值必须在(0,1)范围内"
        assert self.early_stop_patience > 0, "早停耐心值必须大于0"


@dataclass
class SAPathPlanningConfig:
    sa_initial_temperature: float = 1200.0
    sa_final_temperature: float = 0.1
    sa_cooling_rate: float = 0.85
    sa_iterations_per_temperature: int = 10
    sa_max_iterations: int = 500
    sa_swap_probability: float = 0.50
    sa_reverse_probability: float = 0.30
    sa_insert_probability: float = 0.20
    w1_difficulty: float = 0.60
    w2_distance: float = 0.35
    w3_skip_rate: float = 0.05
    difficulty_threshold: float = 0.75
    early_stop_patience: int = 10

    def __post_init__(self):
        assert self.sa_initial_temperature > 0.0, "SA初始温度必须大于0"
        assert self.sa_final_temperature > 0.0, "SA终止温度必须大于0"
        assert self.sa_initial_temperature > self.sa_final_temperature, "SA初始温度必须大于终止温度"
        assert 0.0 < self.sa_cooling_rate < 1.0, "SA降温系数必须在(0,1)范围内"
        assert self.sa_iterations_per_temperature > 0, "每个温度层迭代次数必须大于0"
        assert self.sa_max_iterations > 0, "SA最大迭代次数必须大于0"
        assert 0.0 <= self.sa_swap_probability <= 1.0, "交换邻域概率必须在[0,1]范围内"
        assert 0.0 <= self.sa_reverse_probability <= 1.0, "逆序邻域概率必须在[0,1]范围内"
        assert 0.0 <= self.sa_insert_probability <= 1.0, "插入邻域概率必须在[0,1]范围内"
        assert abs(self.sa_swap_probability + self.sa_reverse_probability + self.sa_insert_probability - 1.0) < 1e-6, "三种邻域概率之和必须为1"
        assert 0.0 <= self.w1_difficulty <= 1.0, "难度权重必须在[0,1]范围内"
        assert 0.0 <= self.w2_distance <= 1.0, "距离权重必须在[0,1]范围内"
        assert 0.0 <= self.w3_skip_rate <= 1.0, "漏采率权重必须在[0,1]范围内"
        assert abs(self.w1_difficulty + self.w2_distance + self.w3_skip_rate - 1.0) < 1e-6, "三权重之和必须为1"
        assert 0.0 < self.difficulty_threshold < 1.0, "难度阈值必须在(0,1)范围内"
        assert self.early_stop_patience > 0, "早停耐心值必须大于0"
