# Assignment 6: MLP vs CNN on Indian Traffic Signs (TECH 405)

Project: classify 58 Indian road-sign categories. Two different models are compared on the **same leakage-aware split**:

| Model | Type | Parameters |
|---|---|---|
| MLP (fully connected) | neural network | 6,438,202 |
| CNN (3 conv blocks + 2 FC layers) | deep learning | 1,087,434 |

Metrics: confusion matrix, precision, recall, F1 and **AUC-ROC** (one-vs-rest).

Dataset (ground truth = class label of each image folder / `labels.csv`):
https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification

## Files
- `data_prep.py` / `common.py` - MD5 de-duplication (4,170 -> 1,672 unique), block-grouped 60/20/20 split, 64x64 RGB arrays (same pipeline as the Assignment 7 repo).
- `models.py` - MLP, CNN, training loop, and `evaluate()` (precision / recall / F1 / AUC-ROC).
- `sweep_mlp.py` - 12-setting hyperparameter sweep for the MLP, scored on validation macro-F1. (The CNN settings come from the Assignment 7 sweep.)
- `compare.py` - trains each model with 3 seeds, evaluates once on the test blocks, writes figures and tables to `results/`.

## Run it
```bash
pip install -r requirements.txt
# Kaggle layout in the current folder: DATA/<ClassId>/*.png and labels.csv
python sweep_mlp.py     # ~5 min on 1 CPU core  -> mlp_sweep.json, mlp_sweep.log
python compare.py       # ~8 min                -> final.log, results/*
```
Other locations: `DATA_DIRS=dir1,dir2 LABELS_CSV=path/labels.csv python compare.py`.

## Results (313 test images, 18 classes, mean +- SD over 3 seeds)
| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro AUC-ROC |
|---|---|---|---|---|---|
| MLP | 0.590 +- 0.008 | 0.698 +- 0.015 | 0.627 +- 0.007 | 0.597 +- 0.011 | 0.950 +- 0.001 |
| CNN | 0.738 +- 0.016 | 0.789 +- 0.021 | 0.752 +- 0.013 | 0.733 +- 0.024 | 0.978 +- 0.006 |

The CNN beats the MLP on every metric and in every seed, with about one sixth of the parameters.

## Caveats
- Small validation/test sets and only 3 seeds; only 18 of 58 classes can be evaluated.
- ClassId 43 has no training images after the buffer purge (F1 = 0); it stays in the macro averages.
- AUC-ROC is high for both models even where F1 is low, because one-vs-rest ROC curves are dominated by the many negative images.
