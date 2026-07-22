"""
extract_fmri.py

Build the "raw fMRI" representation of CineBrain Season 7, per subject, aligned to
the exact same window scheme every EEG / video / LLM extractor uses.

Data source : HuggingFace  Fudan-fMRI/CineBrain  →  sub-00XX/fMRI_preprocessed_data.tar
Content     : the preprocessed tar holds ONE .npy per fMRI TR (0.8 s frame), named
              `visual_audio/<tr_index>.npy`, each a flat (18946,) float64 masked-voxel
              vector (already z-scored). There are ~27,000 TRs per subject (20 episodes
              of 18 min at TR=0.8 s). This is NOT atlas-parcellated — it is the raw
              preprocessed grayordinate/voxel vector, which is exactly the representation
              we want as "the fMRI itself".

Alignment   : the fMRI TR index is the same 0.8 s frame index as the EEG segment index.
              Season 7 = the first 10 episodes = TR indices 0..13499 (13500 TRs = 10800 s),
              mirroring MAX_SEGMENT=13500 in extract_neurolm.py and the canonical
              `window_starts_s = arange(0, 10800, win_sec)` tiling. A 4 s video clip
              (clip id c, Season 7 = c in 0..2699) corresponds to TRs [5c, 5c+5).

Output      : one memory-mappable .npy per subject (so downstream analysis can mmap it
              and never hold all 6 subjects — 6 GB — resident at once):
                fmri_raw_<subject>.npy   (13500, D=18946)  float32   rows = Season-7 TRs
              (row index == TR index == EEG segment index; TR = 0.8 s; all constants known,
              so no sidecar metadata is needed.)
              This raw per-TR matrix is the flexible base: window pooling for any EEG
              family and any HRF shift is derived from it downstream (see fmri_windows.py).

Usage:
    # Verify the TR-id mapping on one subject WITHOUT building the matrix:
    python extract_fmri.py --subjects sub-0001 --verify-only

    # Build the raw matrix for all 6 subjects (downloads ~4 GB tar each, cached):
    python extract_fmri.py

    # Single subject, custom output dir:
    python extract_fmri.py --subjects sub-0001 --out-dir /data/fmri

Requirements:  pip install huggingface_hub numpy   (already in venv)

HuggingFace token: set HF_TOKEN, or it is read from .env / tokens/hf_token.txt.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

# ── Dataset constants ─────────────────────────────────────────────────────────
HF_DATASET_REPO = "Fudan-fMRI/CineBrain"
ALL_SUBJECTS    = [f"sub-{i:04d}" for i in range(1, 7)]

TR_SECONDS      = 0.8
SEASON7_N_TR    = 13500          # first 10 episodes × 18 min ÷ 0.8 s  (== EEG MAX_SEGMENT)
CLIP_SECONDS    = 4.0
TR_PER_CLIP     = 5              # 4 s clip ↔ 5 fMRI TRs
EXPECTED_D      = 18946          # flat masked-voxel dim per TR (asserted, not hard-coded use)

_MEMBER_RE = re.compile(r"(?:^|/)(\d+)\.npy$")


# ── Token resolution (mirror _common.resolve_hf_token, self-contained) ─────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_hf_token(cli_token=None):
    if cli_token:
        return cli_token.strip()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    env = _PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    for p in [_PROJECT_ROOT / "tokens" / "hf_token.txt", Path("tokens/hf_token.txt"),
              Path.home() / ".cache" / "huggingface" / "token"]:
        if p.exists():
            return p.read_text().strip()
    return None


# ── Tar indexing ───────────────────────────────────────────────────────────────
def index_tar(tar_path: Path):
    """Return {tr_id: TarInfo} for every visual_audio/<id>.npy member, using random
    access over the on-disk tar (reads headers only, not the 4 GB of data)."""
    idx = {}
    with tarfile.open(tar_path, mode="r:") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            g = _MEMBER_RE.search(m.name)
            if g:
                idx[int(g.group(1))] = m
    return idx


def report_inventory(idx: dict, subject: str):
    ids = sorted(idx)
    n = len(ids)
    lo, hi = ids[0], ids[-1]
    contiguous_full = ids == list(range(lo, hi + 1))
    s7_present = [i for i in range(SEASON7_N_TR) if i in idx]
    s7_missing = SEASON7_N_TR - len(s7_present)
    beyond = [i for i in ids if i >= 27000]
    print(f"  [{subject}] {n} TR files | id range {lo}..{hi} | "
          f"contiguous={contiguous_full}")
    print(f"  [{subject}] Season-7 ids 0..{SEASON7_N_TR-1}: "
          f"{len(s7_present)}/{SEASON7_N_TR} present ({s7_missing} missing)")
    if beyond:
        print(f"  [{subject}] WARNING: {len(beyond)} ids >= 27000 (first few {beyond[:5]})")
    return s7_missing == 0


def build_matrix(tar_path: Path, idx: dict, subject: str):
    """Stack visual_audio/{0..13499}.npy into (13500, D) float32."""
    mat = None
    t0 = time.time()
    with tarfile.open(tar_path, mode="r:") as tf:
        for i in range(SEASON7_N_TR):
            m = idx.get(i)
            if m is None:
                raise ValueError(f"{subject}: missing Season-7 TR file id {i}")
            v = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=True)
            v = np.asarray(v).ravel()
            if mat is None:
                D = v.shape[0]
                if D != EXPECTED_D:
                    print(f"  [{subject}] NOTE: D={D} (expected {EXPECTED_D})")
                mat = np.empty((SEASON7_N_TR, D), dtype=np.float32)
            elif v.shape[0] != mat.shape[1]:
                raise ValueError(f"{subject}: TR {i} dim {v.shape[0]} != {mat.shape[1]}")
            mat[i] = v.astype(np.float32)
            if (i + 1) % 2000 == 0:
                print(f"  [{subject}] {i+1}/{SEASON7_N_TR} TRs "
                      f"({time.time()-t0:.0f}s)")
    return mat


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS,
                    help="Subset of sub-000X to process (default: all 6).")
    ap.add_argument("--out-dir", default=str(_PROJECT_ROOT / "embeddings" / "fmri"))
    ap.add_argument("--data-repo", default=HF_DATASET_REPO)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="Index the tar and report the TR-id inventory; do not build/save.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    token = resolve_hf_token(args.hf_token)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        out = out_dir / f"fmri_raw_{subject}.npy"
        if out.exists() and not args.overwrite and not args.verify_only:
            print(f"[{subject}] exists, skipping ({out}). Use --overwrite to redo.")
            continue

        print(f"[{subject}] downloading fMRI_preprocessed_data.tar (~4 GB, cached)...")
        tar_path = Path(hf_hub_download(
            args.data_repo, f"{subject}/fMRI_preprocessed_data.tar",
            repo_type="dataset", token=token))

        print(f"[{subject}] indexing tar members...")
        idx = index_tar(tar_path)
        ok = report_inventory(idx, subject)
        if not ok:
            print(f"[{subject}] Season-7 TR ids are NOT all present — inspect before "
                  f"trusting the alignment. Skipping build for this subject.")
            continue
        if args.verify_only:
            continue

        print(f"[{subject}] building (13500, D) matrix...")
        mat = build_matrix(tar_path, idx, subject)
        np.save(out, mat)                                    # plain .npy → mmappable
        print(f"[{subject}] saved {out}  shape={mat.shape}  "
              f"({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
