#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from skimage.filters import threshold_otsu

from model import SimpleConvEncoder


ROOT = Path(__file__).resolve().parent
PRESETS = {
    "Farmland": (128, ROOT / "preprocessing/pca_farmland.npy", ROOT / "checkpoints/farmland.pth"),
    "Hermiston": (64, ROOT / "preprocessing/pca_hermiston.npy", ROOT / "checkpoints/hermiston.pth"),
    "River": (128, ROOT / "preprocessing/pca_river.joblib", ROOT / "checkpoints/river.pth"),
}


def load_pair(path: Path):
    data = sio.loadmat(path, variable_names=["T1", "T2"])
    if "T1" not in data or "T2" not in data:
        raise KeyError("The MAT file must contain T1 and T2 arrays.")
    t1 = data["T1"].astype(np.float32)
    t2 = data["T2"].astype(np.float32)
    if t1.shape != t2.shape or t1.ndim != 3:
        raise ValueError("T1 and T2 must have the same H x W x B shape.")
    low = min(float(t1.min()), float(t2.min()))
    high = max(float(t1.max()), float(t2.max()))
    scale = high - low + 1e-6
    return (t1 - low) / scale, (t2 - low) / scale


def transform(image: np.ndarray, pca, channels: int):
    height, width, bands = image.shape
    projected = pca.transform(image.reshape(-1, bands))
    return projected.reshape(height, width, channels).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run DTRL change detection inference.")
    parser.add_argument("--dataset", choices=PRESETS, required=True)
    parser.add_argument("--input", type=Path, required=True, help="MAT file containing T1 and T2")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    channels, pca_path, checkpoint_path = PRESETS[args.dataset]
    device = torch.device(args.device)
    t1, t2 = load_pair(args.input)
    pca = joblib.load(pca_path)
    t1 = transform(t1, pca, channels)
    t2 = transform(t2, pca, channels)
    x1 = torch.from_numpy(t1).permute(2, 0, 1).unsqueeze(0).to(device)
    x2 = torch.from_numpy(t2).permute(2, 0, 1).unsqueeze(0).to(device)

    model = SimpleConvEncoder(channels).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        score = torch.linalg.vector_norm(model(x1) - model(x2), dim=1)[0].cpu().numpy()
    score = score.astype(np.float16).astype(np.float32)

    threshold = float(threshold_otsu(score))
    prediction = (score > threshold).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "change_score.npy", score.astype(np.float32))
    np.save(args.output / "change_map.npy", prediction)
    Image.fromarray(prediction * 255).save(args.output / "change_map.png")
    print(f"threshold={threshold:.8f}")
    print(f"outputs={args.output.resolve()}")


if __name__ == "__main__":
    main()
