"""
Extract VideoMAEv2 embeddings for the 50 EEG-aligned 8-second video windows.

Identical frame-sampling logic as extract_videomae.py:
  - Two consecutive 4-second clips concatenated → 16 frames sampled uniformly
  - Mean pool over spatiotemporal tokens of last_hidden_state → (D,) per window

Key differences from VideoMAE v1:
  - Models from OpenGVLab require trust_remote_code=True (custom ViT code)
  - VideoMAEImageProcessor returns (B, T, C, H, W); these models expect (B, C, T, H, W)
    so pixel_values are permuted before the forward pass.

Output shape: (W, D)
  W : 50 windows
  D : hidden_size (768 / 1024 / 1280 / 1408 for base / large / huge / giant)

Output files:
    embeddings/VideoMAEv2/videomaev2_base.npz
    embeddings/VideoMAEv2/videomaev2_large.npz
    embeddings/VideoMAEv2/videomaev2_huge.npz
    embeddings/VideoMAEv2/videomaev2_giant.npz

Usage:
    python scripts/extract_videomaev2.py --size base
    python scripts/extract_videomaev2.py --size giant
    python scripts/extract_videomaev2.py          # runs all four
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

CB_ROOT   = Path(__file__).parents[1]
CLIPS_DIR = CB_ROOT / "clips"
EMBED_DIR = CB_ROOT / "embeddings" / "VideoMAEv2"

HF_MODELS = {
    "base":  "OpenGVLab/VideoMAEv2-Base",
    "large": "OpenGVLab/VideoMAEv2-Large",
    "huge":  "OpenGVLab/VideoMAEv2-Huge",
    "giant": "OpenGVLab/VideoMAEv2-giant",
}

N_FRAMES   = 16
FRAME_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_frames(frames: list[np.ndarray], n: int = N_FRAMES) -> list[np.ndarray]:
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def load_window_frames(start: int) -> list[np.ndarray]:
    clip_idx = start // 5
    clip_a = CLIPS_DIR / f"{clip_idx:06d}.mp4"
    clip_b = CLIPS_DIR / f"{clip_idx + 1:06d}.mp4"
    return sample_frames(load_clip_frames(clip_a) + load_clip_frames(clip_b), N_FRAMES)


def embed_window(processor, model, frames: list[np.ndarray]) -> np.ndarray:
    inputs = processor(frames, return_tensors="pt").to(DEVICE)
    # VideoMAEImageProcessor → (B, T, C, H, W); VideoMAEv2 custom forward → (B, C, T, H, W)
    pv = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        out = model(pixel_values=pv)
    # VideoMAEv2 returns a plain (1, D) tensor — already mean-pooled (use_mean_pooling=True)
    return out.squeeze(0).cpu().float().numpy()


def run(size_key: str):
    print(f"\n{'='*60}")
    print(f"=== VideoMAEv2-{size_key} ===")
    print(f"{'='*60}")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBED_DIR / f"videomaev2_{size_key}.npz"

    ref = np.load(CB_ROOT / "embeddings" / "VideoMAE" / "videomae_base.npz", allow_pickle=True)
    window_starts = ref["window_starts"]
    W = len(window_starts)
    print(f"  Windows: {W}  |  Device: {DEVICE}")

    hf_id = HF_MODELS[size_key]
    print(f"  Loading {hf_id}...")
    t0 = time.time()
    config    = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(hf_id)
    model     = AutoModel.from_pretrained(hf_id, config=config,
                                          trust_remote_code=True).to(DEVICE).eval()
    # VideoMAEv2 config nests model params under model_config dict
    D = model.config.model_config['embed_dim']
    print(f"  Loaded in {time.time()-t0:.1f}s  |  embed_dim={D}")

    # Some sizes (giant) use use_mean_pooling=False, which takes only the CLS token.
    # For MAE-pretrained models the CLS token is uninformative; force mean-pooling
    # over all patch tokens by swapping norm → fc_norm (reuses the trained weights).
    if not config.model_config.get('use_mean_pooling', True):
        import copy
        inner = model.model
        inner.fc_norm = copy.deepcopy(inner.norm)
        inner.norm = torch.nn.Identity()
        print(f"  [patched] use_mean_pooling=False → forced mean-pool via fc_norm")

    embeddings = np.zeros((W, D), dtype=np.float32)

    for w_idx, start in enumerate(window_starts):
        t1 = time.time()
        frames = load_window_frames(int(start))
        embeddings[w_idx] = embed_window(processor, model, frames)
        print(f"  window {w_idx+1:>2}/{W}  (seg {start:>5d} "
              f"-> clips {start//5:06d}+{start//5+1:06d}): "
              f"{time.time()-t1:.2f}s")

    np.savez(out_path, embeddings=embeddings, window_starts=window_starts,
             size=size_key, model_id=hf_id)
    print(f"\n  Saved {embeddings.shape} to {out_path}")

    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(HF_MODELS.keys()),
                    help="Model size to run. Omit to run all four.")
    args = ap.parse_args()

    sizes = [args.size] if args.size else list(HF_MODELS.keys())
    for size in sizes:
        run(size)


if __name__ == "__main__":
    main()
