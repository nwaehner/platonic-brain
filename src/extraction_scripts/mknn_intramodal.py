"""
MKNN intramodal alignment across NeuroLM model sizes.

Loads embeddings for B, L, XL (same windows, same subjects), then computes
Mutual k-Nearest Neighbor alignment for every model-size pair:
  B <-> L,  L <-> XL,  B <-> XL

Two representations per model:
  1. Stimulus space   : mean over subjects → (W, D), captures shared stimulus geometry
  2. Full space       : (W*S, D), subject-mean centered + L2-normed

Random MKNN baseline ≈ k / (n - 1).

Usage:
    python scripts/mknn_intramodal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "code" / "src"))
from pb.metrics.neighbors import mknn, mknn_pairwise

EMBED_DIR = Path(__file__).parents[1] / "embeddings"
NPZ = {
    "B": EMBED_DIR / "neurolm_b.npz",
    "L": EMBED_DIR / "neurolm_l.npz",
    "XL": EMBED_DIR / "neurolm_xl.npz",
}


def load_emb(path: Path) -> np.ndarray:
    """Returns (W, S, D) float32."""
    d = np.load(path, allow_pickle=True)
    return d["embeddings"].astype(np.float32)


def subject_mean_center(emb: np.ndarray) -> np.ndarray:
    """emb: (W, S, D) → subject-mean centered."""
    return emb - emb.mean(axis=0, keepdims=True)


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def stimulus_repr(emb: np.ndarray) -> np.ndarray:
    """Mean over subjects → (W, D), L2-normed."""
    z = subject_mean_center(emb)
    mu = z.mean(axis=1)        # (W, D)
    return l2_norm(mu)


def full_repr(emb: np.ndarray) -> np.ndarray:
    """Subject-mean centered, L2-normed, flattened → (W*S, D)."""
    z = subject_mean_center(emb)
    W, S, D = z.shape
    flat = z.reshape(W * S, D)
    return l2_norm(flat)


def per_subject_repr(emb: np.ndarray) -> list[np.ndarray]:
    """
    For each subject, return their (W, D) embedding matrix, L2-normed.
    No subject-mean centering here — we want each subject's own geometry
    across their W windows.
    """
    W, S, D = emb.shape
    out = []
    for s in range(S):
        x = emb[:, s, :]          # (W, D)
        out.append(l2_norm(x))
    return out


def run(k: int = 5):
    # Load available models
    embs = {}
    for size, path in NPZ.items():
        if path.exists():
            embs[size] = load_emb(path)
            W, S, D = embs[size].shape
            print(f"Loaded NeuroLM-{size}: ({W}, {S}, {D})")
        else:
            print(f"Missing: {path.name} — skipping {size}")

    if len(embs) < 2:
        print("Need at least 2 model sizes. Exiting.")
        return

    sizes = sorted(embs)
    W, S = list(embs.values())[0].shape[:2]
    random_baseline = k / (W - 1)

    print(f"\n=== MKNN Intramodal Alignment — Per Subject (k={k}) ===")
    print(f"N windows per subject : {W}")
    print(f"N subjects            : {S}")
    print(f"Random baseline       : {random_baseline:.4f}")

    # Per-subject MKNN: for each subject, compare their W-window geometry across model sizes
    subj_reprs = {size: per_subject_repr(embs[size]) for size in sizes}
    pairs = [(a, b) for i, a in enumerate(sizes) for b in sizes[i + 1:]]

    all_scores: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}

    print("\n--- Per-subject MKNN (W windows per subject, L2-normed) ---")
    header = "subject  " + "  ".join(f"{a}<->{b}" for a, b in pairs)
    print(header)
    print("-" * len(header))

    for s in range(S):
        row = f"sub-{s+1:04d} "
        for a, b in pairs:
            score = mknn(subj_reprs[a][s], subj_reprs[b][s], k=k)
            all_scores[(a, b)].append(score)
            row += f"  {score:.4f}      "
        print(row)

    print("\n--- Summary (mean ± std across subjects) ---")
    for (a, b), scores in all_scores.items():
        arr = np.array(scores)
        print(f"  {a} <-> {b}: {arr.mean():.4f} ± {arr.std():.4f}  ({arr.mean()/random_baseline:.1f}× random)")

    # Convergence trend
    if all(s in sizes for s in ["B", "L", "XL"]):
        print("\n--- Convergence trend per subject ---")
        bl = np.array(all_scores[("B", "L")])
        lx = np.array(all_scores[("L", "XL")])
        bx = np.array(all_scores[("B", "XL")])
        print(f"  B<->L  mean: {bl.mean():.4f}")
        print(f"  L<->XL mean: {lx.mean():.4f}")
        print(f"  B<->XL mean: {bx.mean():.4f}")
        if lx.mean() > bl.mean():
            print("  -> L<->XL > B<->L: scale convergence holds at the individual subject level")
        else:
            print("  -> No clear scale-dependent convergence at individual subject level")

    return all_scores


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    run(k=args.k)
