# CineBrain — Cross-modal embedding extraction

Scripts for extracting layerwise embeddings from EEG foundation models, video models,
and image models on the [CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)
Season 7 data, for cross-modal alignment analysis (Platonic Representation Hypothesis).

## EEG extractors

Each writes `embeddings/<model>_<size>_layerwise.npz` of shape
`(n_layers, W, S, D)` — n_layers per model, W windows, S=6 subjects, D embedding dim.

| Script                  | Model            | Window | # Windows | Sizes                  | Repo arg |
|-------------------------|------------------|--------|-----------|------------------------|----------|
| `extract_femba.py`      | FEMBA            | 5 s    | 2160      | tiny / base / large    | `--femba-repo` |
| `extract_luna.py`       | LUNA             | 5 s    | 2160      | base / large / huge    | `--luna-repo` |
| `extract_steegformer.py`| ST-EEGFormer     | 6 s    | 1800      | small / base / large   | `--steegformer-repo` |
| `extract_neurolm.py`    | NeuroLM          | 8 s    | 1350      | b / l / xl             | `--neurolm-repo` |
| `extract_reve.py`       | REVE (gated)     | 10 s   | 1080      | base / large           | none (loads via HF) |

## Video / image extractors

Aligned 1-to-1 with the EEG windows of the corresponding family. Both are
**layerwise** too: each writes `<model>_<size>__<eeg_family>.npz` of shape
`(n_layers, W, D)` — for every transformer block, DINOv2 takes the CLS token
(matching the `v[:, 0, :]` pooling in Huh et al. 2024 / platonic-rep) averaged
over the window's frames, and VideoMAE (which has no CLS token) mean-pools its
spatiotemporal tokens. This mirrors the EEG `(n_layers, W, S, D)` arrays so
cross-modal alignment can be measured between every pair of layers.

| Script               | Model          | Sizes                          | Per window |
|----------------------|----------------|--------------------------------|------------|
| `extract_videomae.py`| VideoMAE (pretrained) | base / large            | 16 frames (uniform downsample) |
| `extract_videomae_ft_kinetics.py` | VideoMAE (Kinetics-400 fine-tuned) | base / large / huge | 16 frames (uniform downsample) |
| `extract_dinov2.py`  | DINOv2         | small / base / large / giant   | 1 frame per second |
| `extract_vjepa2.py`  | V-JEPA 2       | large / huge / giant           | all native 8 fps frames |

All scripts download `videos.tar` (2.59 GB) from HuggingFace once, extract clips locally,
then produce one NPZ per `(size, eeg_family)`. EEG families:
`femba_luna` (5 s), `steegformer` (6 s), `neurolm` (8 s), `reve` (10 s).

`extract_videomae_ft_kinetics.py` uses `MCG-NJU/videomae-{base,large,huge}-finetuned-kinetics`.
The classification head is stripped and only the encoder backbone is extracted, identical
in shape and semantics to the pretrained-only NPZs. Output files are named
`videomae_ft_kinetics_{size}__{eeg_family}.npz`. This is the checkpoint set used by
Gröger et al. (2026) for the VideoMAE huge model (no pretrained-only huge exists).

### V-JEPA 2 details

`extract_vjepa2.py` uses Meta's V-JEPA 2 video encoder
(`facebook/vjepa2-vit{l,h,g}-fpc64-256`) loaded via `AutoModel` from HuggingFace.
Unlike VideoMAE, it feeds **all native 8 fps frames** covering the window without
uniform downsampling, so the frame count varies with window length:

| Window | Frames | EEG family |
|--------|--------|------------|
| 5 s    | ~42    | femba_luna |
| 6 s    | ~50    | steegformer |
| 8 s    | ~66    | neurolm |
| 10 s   | ~82    | reve |

V-JEPA 2 has no CLS token — embeddings are mean-pooled over all spatiotemporal
patch tokens (`tubelet_size=2`, `patch_size=16`, input resolution 256×256).
Output shape: `(n_layers, W, D)` — large: 24 layers / D=1024, huge: 32 / 1280,
giant: 40 / 1408.

## Setup

```bash
# Clone the model repos that the EEG extractors need (kept under models/)
git clone https://github.com/pulp-bio/BioFoundation.git   models/BioFoundation
git clone https://github.com/935963004/NeuroLM.git        models/NeuroLM
git clone https://github.com/LiuyinYang1101/STEEGFormer.git models/STEEGFormer

# Python deps — creates ./venv/, installs pinned torch + requirements.txt
# (+ torcheeg, and mamba-ssm/causal-conv1d if nvcc is available, for FEMBA)
bash setup.sh
source venv/bin/activate

# HuggingFace token (REVE is gated; CineBrain dataset may also require login)
echo "hf_xxxx..." > hf_token.txt   # or set HF_TOKEN env var
```

## Running everything

```bash
# EEG (all sizes per family)
python extract_femba.py       --femba-repo       /path/to/BioFoundation
python extract_luna.py        --luna-repo        /path/to/BioFoundation
python extract_neurolm.py     --neurolm-repo     /path/to/NeuroLM
python extract_steegformer.py --steegformer-repo /path/to/STEEGFormer
python extract_reve.py

# Video / image (one command per EEG family)
python extract_videomae.py --eeg-family femba_luna  --window-seconds 5
python extract_videomae.py --eeg-family steegformer --window-seconds 6
python extract_videomae.py --eeg-family neurolm     --window-seconds 8
python extract_videomae.py --eeg-family reve        --window-seconds 10

python extract_dinov2.py   --eeg-family femba_luna  --window-seconds 5
python extract_dinov2.py   --eeg-family steegformer --window-seconds 6
python extract_dinov2.py   --eeg-family neurolm     --window-seconds 8
python extract_dinov2.py   --eeg-family reve        --window-seconds 10

python extract_vjepa2.py   --eeg-family femba_luna  --window-seconds 5
python extract_vjepa2.py   --eeg-family steegformer --window-seconds 6
python extract_vjepa2.py   --eeg-family neurolm     --window-seconds 8
python extract_vjepa2.py   --eeg-family reve        --window-seconds 10

# VideoMAE Kinetics-finetuned (base / large / huge — all three sizes)
python extract_videomae_ft_kinetics.py --eeg-family femba_luna  --window-seconds 5
python extract_videomae_ft_kinetics.py --eeg-family steegformer --window-seconds 6
python extract_videomae_ft_kinetics.py --eeg-family neurolm     --window-seconds 8
python extract_videomae_ft_kinetics.py --eeg-family reve        --window-seconds 10

# Single size or half-precision (recommended for huge / giant on tight VRAM):
python extract_vjepa2.py              --eeg-family neurolm     --window-seconds 8  --size large
python extract_vjepa2.py              --eeg-family reve        --window-seconds 10 --dtype fp16
python extract_videomae_ft_kinetics.py --eeg-family neurolm    --window-seconds 8  --size huge
```

## Notes

- All EEG extractors tile non-overlapping windows over Season 7 (0–10800 s).
- Video / image scripts cap clip access at index 2699 (Season 7 boundary).
- FEMBA and LUNA share the same 5 s tiling, so they share one video / image NPZ
  tagged `femba_luna`.
- `mamba-ssm` requires CUDA at install time (FEMBA only).
- Extraction scripts live in `src/extraction_scripts/`, analysis notebooks in
  `notebooks/`, and the cloned model repos in `models/` (git-ignored).

## Analysis notebooks

| Notebook | Purpose |
|---|---|
| `analyze_mknn_hf.ipynb` | Layerwise cross-modal mKNN between EEG models (all sizes) and VideoMAE / DINOv2 / V-JEPA2. Includes Aristotelian calibration and k-ablation. Per-subject scores are computed then averaged. |
| `analyze_mknn_avgsubj_hf.ipynb` | Same analyses but EEG embeddings are **averaged across subjects first** (per stimulus) before computing kNN. Produces a single calibrated and uncalibrated mKNN score per (EEG size, vision size) pair, measuring alignment of the shared/consensus neural representation rather than individual-level alignment. |
| `visualize_mknn_stimuli.ipynb` | Stimulus visualizer: for each EEG model (max size) × best vision model, shows the top-5 windows with the highest mutual k-NN overlap as animated GIFs. Each anchor window is shown alongside three rows — **shared neighbors** (EEG ∩ vision), **EEG-only**, and **vision-only** — to let you inspect what semantic features drive cross-modal agreement. Runs for both VideoMAE and DINOv2, and repeats with temporal exclusion (±10 windows removed from the neighbor pool) to control for shot-continuity confounds. |
| `temporal_distance_analysis.ipynb` | For each max-size EEG model × max-size vision model (VideoMAE-large, DINOv2-giant), plots the full-range distribution of k-NN neighbor temporal distances (windows apart), overlaying standard and temporal-exclusion ±20 k-NN. Separate figures for VideoMAE and DINOv2. Includes a deep-dive into the **LUNA × VideoMAE** case, which shows periodic peaks in the EEG distance histogram on top of the expected exponential decay: detects peak positions via smooth-baseline residuals, checks whether the same peaks appear in the VideoMAE distribution (visual recurrence vs. EEG-specific artifact), and displays video clips for a concrete anchor whose nearest EEG neighbours fall at those peak distances. |
