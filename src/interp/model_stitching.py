"""
model_stitching.py — Study (3): functional alignment by fitting maps between spaces.

"Stitching" here = fit a map from one frozen latent space to another on a TRAIN split of
windows and measure how well it predicts the held-out windows. Two map families, both in a
shared PCA subspace (so different-width models are comparable and the rotation is defined):

  • orthogonal Procrustes  (rotation/reflection only) → isometry-up-to-rotation R²
  • ridge affine A→B                                   → linear-predictivity R²
  • temporal SHIFT-NULL   (refit on time-shifted pairings) → significance, controlling the
    repo's temporal-autocorrelation confound.

All models are placed on ONE shared window grid by choosing a reference EEG model (`--eeg-ref`):
its family fixes the vision window scheme and the LLM caption directory, and the EEG members
are the models sharing that family. Layer pair is chosen by max mKNN (or last layer). Run once
per `--eeg-ref` to cover all families.

Figures (→ outputs/):
  stitching_matrix      — models×models linear & Procrustes R² heatmaps
  stitching_proc_vs_lin — Procrustes vs ridge R² scatter
  stitching_vs_mknn     — stitching R² vs max mKNN (do the two alignment notions agree?)
  stitching_layer_heatmap — layer×layer ridge R² for a flagship cross-modal pair

Usage:
  python src/interp/model_stitching.py                       # eeg-ref neurolm, largest sizes
  python src/interp/model_stitching.py --eeg-ref femba --all-sizes
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import _interp_common as IC
import _common as C

MOD_COLOR = {"eeg": "#1a9850", "vision": "#d95f02", "llm": "#7570b3"}


# ── model specs on one shared grid ──────────────────────────────────────────────
def build_specs(eeg_ref, all_sizes):
    vis_family = C.EEG[eeg_ref]["family"]
    specs = []
    for m, cfg in C.EEG.items():
        if cfg["family"] != vis_family:
            continue
        for s in (cfg["sizes"] if all_sizes else cfg["sizes"][-1:]):
            specs.append(dict(kind="eeg", name=m, size=s, mod="eeg",
                              label=f"{C.DISPLAY[m]}-{s}"))
    for arch, cfg in C.VISION.items():
        for s in (cfg["sizes"] if all_sizes else cfg["sizes"][-1:]):
            specs.append(dict(kind="vision", name=arch, size=s, mod="vision",
                              vis_family=vis_family, label=C.vision_label(arch, s)))
    for fam, stems in C.LLM.items():
        for st in (stems if all_sizes else stems[-1:]):
            specs.append(dict(kind="llm", name=fam, size=st, mod="llm",
                              llm_dir=eeg_ref, label=C.LLM_LABEL.get(st, st)))
    return specs, vis_family


def load_spec(sp):
    if sp["kind"] == "eeg":
        e = IC.try_load_emb(C.EEG[sp["name"]]["fname"](sp["size"]))
        return None if e is None else e.mean(axis=2)          # (L,W,D)
    if sp["kind"] == "vision":
        e = IC.try_load_emb(C.VISION[sp["name"]]["path"](sp["size"], sp["vis_family"]))
    else:
        e = IC.load_llm_grid(sp["llm_dir"], sp["size"])    # triniborrell (nitrox639 lacks most)
    if e is None:
        return None
    return e[None] if e.ndim == 2 else e


# ── stitching primitives ─────────────────────────────────────────────────────────
def _r2(Y, Yh):
    ss = ((Y - Yh) ** 2).sum()
    st = ((Y - Y.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss / (st + 1e-12))


def _reduce(A, tr, n_pc):
    p = PCA(n_components=min(n_pc, A.shape[1], len(tr) - 1)).fit(A[tr])
    return p.transform(A)


def stitch(A, B, n_pc=50, train_frac=0.7, seed=0):
    """Fit Procrustes + ridge A→B in a shared PCA subspace. Returns (proc_r2, lin_r2,
    Az, Bz, tr, te)."""
    W = A.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(W)
    ntr = int(train_frac * W)
    tr, te = idx[:ntr], idx[ntr:]
    Az = _reduce(A, tr, n_pc)
    Bz = _reduce(B, tr, n_pc)
    d = min(Az.shape[1], Bz.shape[1])
    Az, Bz = Az[:, :d], Bz[:, :d]
    M = Az[tr].T @ Bz[tr]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt                                       # orthogonal map
    proc_r2 = _r2(Bz[te], Az[te] @ R)
    rg = Ridge(alpha=1.0).fit(Az[tr], Bz[tr])
    lin_r2 = _r2(Bz[te], rg.predict(Az[te]))
    return proc_r2, lin_r2, Az, Bz, tr, te


def shift_null_r2(Az, Bz, tr, te, shifts):
    """Ridge test-R² for the true pairing vs time-shifted pairings. Returns
    (r2_by_shift dict, p) where p = frac(shift R² ≥ unshifted R²)."""
    out = {}
    for d in shifts:
        Bs = np.roll(Bz, d, axis=0)
        rg = Ridge(alpha=1.0).fit(Az[tr], Bs[tr])
        out[d] = _r2(Bs[te], rg.predict(Az[te]))
    obs = out.get(0, max(out.values()))
    dist = np.array(list(out.values()))
    p = float((1 + (dist >= obs).sum()) / (len(dist) + 1))
    return out, p


# ── orchestration ────────────────────────────────────────────────────────────────
def run(args):
    IC.set_token(args.hf_token)
    specs, vis_family = build_specs(args.eeg_ref, args.all_sizes)
    print(f"eeg-ref={args.eeg_ref}  grid family={vis_family}  candidate models={len(specs)}")

    embs, nns, labels, mods = {}, {}, {}, {}
    W_ref = None
    for sp in specs:
        e = load_spec(sp)
        if e is None:
            continue
        if W_ref is None:
            W_ref = e.shape[1]
        if e.shape[1] != W_ref:
            print(f"  skip {sp['label']}: W={e.shape[1]} ≠ {W_ref}")
            continue
        key = sp["label"]
        embs[key] = e
        nns[key] = C.precompute_knn_layers(e, args.k)
        mods[key] = sp["mod"]
        print(f"  loaded {key}: {e.shape}")

    keys = list(embs)
    if len(keys) < 2:
        print("Need ≥2 models on a shared grid.")
        return

    rows = []
    shift_set = [0, 10, 20, 50, 100, 200, -10, -20, -50, -100, -200]
    for a, b in combinations(keys, 2):
        la, lb, mk = IC.best_layer_pair(nns[a], nns[b]) if args.best_layer \
            else (embs[a].shape[0] - 1, embs[b].shape[0] - 1,
                  C.mknn_1d(nns[a][-1], nns[b][-1]))
        A, B = embs[a][la], embs[b][lb]
        proc, lin, Az, Bz, tr, te = stitch(A, B, args.n_pc, seed=args.seed)
        _, p = shift_null_r2(Az, Bz, tr, te, shift_set)
        rows.append(dict(a=a, b=b, la=la, lb=lb, mknn=mk, proc_r2=proc,
                         lin_r2=lin, shift_p=p, mod_a=mods[a], mod_b=mods[b]))
        print(f"  {a:>18} ↔ {b:<18}  mKNN={mk:.3f}  proc={proc:.3f}  "
              f"lin={lin:.3f}  shift-p={p:.3f}")

    plot_matrix(keys, rows, mods)
    plot_proc_vs_lin(rows)
    plot_vs_mknn(rows)
    if args.flagship and not args.no_layer_heatmap:
        plot_layer_heatmap(embs, nns, args)
    save(keys, rows, args.eeg_ref)
    print("Done.")


# ── plots ────────────────────────────────────────────────────────────────────────
def _matrix(keys, rows, field):
    M = np.full((len(keys), len(keys)), np.nan)
    idx = {k: i for i, k in enumerate(keys)}
    for r in rows:
        i, j = idx[r["a"]], idx[r["b"]]
        M[i, j] = M[j, i] = r[field]
    np.fill_diagonal(M, 1.0)
    return M


def plot_matrix(keys, rows, mods):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
    for ax, field, title in zip(axes[0], ["lin_r2", "proc_r2"],
                                ["ridge (linear predictivity) R²",
                                 "orthogonal Procrustes R²"]):
        M = _matrix(keys, rows, field)
        im = ax.imshow(M, vmin=0, vmax=max(0.05, np.nanmax(M[~np.eye(len(keys), dtype=bool)])),
                       cmap="viridis")
        ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=7)
        ax.set_yticklabels(keys, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("test R²")
        ax.set_title(title)
    fig.suptitle("Model stitching: predict one latent space from another", y=1.0)
    fig.tight_layout()
    IC.savefig(fig, "model_stitching/stitching_matrix.png")


def plot_proc_vs_lin(rows):
    fig, ax = plt.subplots(figsize=(11, 11))
    for r in rows:
        cross = r["mod_a"] != r["mod_b"]
        ax.scatter(r["proc_r2"], r["lin_r2"], s=55,
                   color=MOD_COLOR[r["mod_a"]] if not cross else "black",
                   marker="o" if not cross else "^",
                   edgecolor="black", lw=0.4, alpha=0.8)
    lim = [min(0, min(r["proc_r2"] for r in rows)),
           max(r["lin_r2"] for r in rows) * 1.05]
    ax.plot(lim, lim, color="grey", ls="--", lw=1)
    ax.set_xlabel("Procrustes R² (rotation only)")
    ax.set_ylabel("ridge R² (free linear map)")
    ax.set_title("Rotation-only vs free-linear stitching (▲ = cross-modal pair)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "model_stitching/stitching_proc_vs_lin.png")


def plot_vs_mknn(rows):
    fig, ax = plt.subplots(figsize=(11, 9))
    for r in rows:
        cross = r["mod_a"] != r["mod_b"]
        ax.scatter(r["lin_r2"], r["mknn"], s=55,
                   color="black" if cross else MOD_COLOR[r["mod_a"]],
                   marker="^" if cross else "o", edgecolor="black", lw=0.4, alpha=0.8)
    ax.set_xlabel("ridge stitching R²")
    ax.set_ylabel("max mKNN (neighbour-overlap alignment)")
    ax.set_title("Do stitching and mKNN agree on which models align? (▲ = cross-modal)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    IC.savefig(fig, "model_stitching/stitching_vs_mknn.png")


def plot_layer_heatmap(embs, nns, args, max_per_axis=8):
    names = [s for s in args.flagship if s in embs]
    if len(names) < 2:
        print(f"  flagship pair {args.flagship} not both present — skipping heatmap")
        return
    a, b = names[0], names[1]
    A_all, B_all = embs[a], embs[b]
    la_idx = np.unique(np.linspace(0, A_all.shape[0] - 1, max_per_axis).astype(int))
    lb_idx = np.unique(np.linspace(0, B_all.shape[0] - 1, max_per_axis).astype(int))
    M = np.zeros((len(la_idx), len(lb_idx)))
    for i, la in enumerate(la_idx):
        for j, lb in enumerate(lb_idx):
            _, lin, *_ = stitch(A_all[la], B_all[lb], args.n_pc, seed=args.seed)
            M[i, j] = lin
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(lb_idx))); ax.set_xticklabels(lb_idx)
    ax.set_yticks(range(len(la_idx))); ax.set_yticklabels(la_idx)
    ax.set_xlabel(f"{b} layer"); ax.set_ylabel(f"{a} layer")
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("ridge R²")
    ax.set_title(f"Layer×layer stitching: {a} → {b}")
    fig.tight_layout()
    IC.savefig(fig, "model_stitching/stitching_layer_heatmap.png")


def save(keys, rows, eeg_ref):
    IC.save_npz(f"model_stitching/model_stitching__{eeg_ref}.npz",
                keys=np.array(keys),
                pairs=np.array([(r["a"], r["b"]) for r in rows]),
                la=np.array([r["la"] for r in rows]),
                lb=np.array([r["lb"] for r in rows]),
                mknn=np.array([r["mknn"] for r in rows]),
                proc_r2=np.array([r["proc_r2"] for r in rows]),
                lin_r2=np.array([r["lin_r2"] for r in rows]),
                shift_p=np.array([r["shift_p"] for r in rows]))


def main():
    ap = argparse.ArgumentParser(description="Study 3: model stitching.")
    ap.add_argument("--eeg-ref", default="neurolm", choices=list(C.EEG),
                    help="reference EEG model fixing the shared window grid.")
    ap.add_argument("--all-sizes", action="store_true",
                    help="include every size (default: largest of each family).")
    ap.add_argument("--n-pc", type=int, default=50)
    ap.add_argument("--k", type=int, default=C.K_MKNN_DEFAULT)
    ap.add_argument("--best-layer", action="store_true",
                    help="select layer pair by max mKNN (default: last layers).")
    ap.add_argument("--flagship", nargs=2, default=None,
                    help="two model labels for the layer×layer heatmap.")
    ap.add_argument("--no-layer-heatmap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
