# HDS-Aware Harvesting Sequence Optimization for Clustered Strawberries

This repository contains the offline inference, sequence-planning, and evaluation code associated with the paper:

**Harvesting Sequence Optimization for Clustered Strawberries Using a Fruit-Level Harvesting Difficulty Score**

The code implements two proposed harvesting-sequence planners:

- **MOP-GA**: multi-objective genetic algorithm;
- **MOP-SA**: multi-objective simulated annealing.

Both planners combine fruit-level harvesting difficulty score (HDS), image-space movement distance, and missed picking rate in a rolling global re-planning framework. After each virtually selected fruit is removed from the image, the HDS values of the remaining fruits are updated and the remaining sequence is re-optimized.

## 1. Method overview

For a candidate harvesting sequence \(\Pi\), the weighted objective is

\[
f(\Pi)=0.60D(\Pi)+0.35L(\Pi)+0.05R(\Pi),
\]

where:

- \(D(\Pi)\) is the average dynamically updated HDS of the retained fruits;
- \(L(\Pi)\) is the mean normalized movement distance;
- \(R(\Pi)\) is the missed picking rate.

The default HDS threshold is **0.75**. A target whose truncated HDS is greater than this threshold is excluded from the current candidate-sequence evaluation.

The implemented pipeline includes:

1. YOLOv11 strawberry detection and instance segmentation;
2. per-side expanded ROI extraction for HDS prediction;
3. S-Only virtual fruit removal;
4. dynamic HDS updating;
5. mask–ROI geometric intersection checking;
6. BLAKE2b hashing of the actual 320 × 320 HDS-model input ROI;
7. ROI-local and full-state HDS caching;
8. rolling global re-planning after every virtually executed target;

## 2. Repository structure

```text
project_root/
├── config.py                  # Paths, thresholds, cache options, and GA/SA parameters
├── perception.py              # YOLO detection, segmentation, mask analysis, and model warm-up
├── hds_model.py               # HDS regression, ROI processing, caching, and S-Only removal
├── objective.py               # Objective function, distance calculation, and reproducibility helpers
├── rolling_replanning.py      # Algorithm-independent rolling global re-planning
├── MOP_GA.py                  # MOP-GA optimizer and executable entry point
├── MOP_SA.py                  # MOP-SA optimizer and executable entry point
├── README.md
└── weights/
    ├── Difficulty/
    │   └── best_model.pt      # HDS regression checkpoint
    └── Detection_segmentation/
        └── best.pt            # YOLOv11 detection/segmentation checkpoint
                   

```

Supported test-image extensions are `.jpg`, `.jpeg`, `.png`, and `.bmp`.


## 3. Required data and model files

Before running the code, place the required files at the default locations:

```text
fit_5.8/                                  # Test images
weights/Difficulty/best_model.pt          # HDS model
weights/Detection_segmentation/best.pt    # Detection/segmentation model
```


## 6. Running the experiments

Run all images in the configured test directory with MOP-GA:

```bash
python MOP_GA.py
```

Run all images with MOP-SA:

```bash
python MOP_SA.py
```
