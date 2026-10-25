# DTRL Inference

Minimal inference implementation for hyperspectral change detection with the
released DTRL checkpoints.

## Contents

- `infer.py`: preprocessing, model inference, and Otsu thresholding
- `model.py`: encoder definition
- `checkpoints/`: selected checkpoints for the three supported datasets
- `preprocessing/`: fitted PCA transforms required by the checkpoints

Training code, evaluation labels, experiment outputs, and ablation utilities
are not included.

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input

The input is a MATLAB file containing two arrays named `T1` and `T2`. Both
arrays must have the same `H x W x B` shape. Ground-truth labels are not read.

## Usage

```bash
python infer.py \
  --dataset Farmland \
  --input data/Farmland.mat \
  --output output/farmland
```

Available presets are `Farmland`, `Hermiston`, and `River`. Use
`--device cpu` when CUDA is unavailable.

The output directory contains:

- `change_score.npy`: continuous change score
- `change_map.npy`: binary change map
- `change_map.png`: binary change map preview
