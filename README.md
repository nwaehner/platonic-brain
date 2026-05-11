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
| `extract_videomae.py`| VideoMAE       | base / large                   | 16 frames |
| `extract_dinov2.py`  | DINOv2         | small / base / large / giant   | 1 frame per second |

Both download `videos.tar` (2.59 GB) from HuggingFace once, extract clips locally,
then produce one NPZ per `(size, eeg_family)`. EEG families:
`femba_luna` (5 s), `steegformer` (6 s), `neurolm` (8 s), `reve` (10 s).

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
```

## Notes

- All EEG extractors tile non-overlapping windows over Season 7 (0–10800 s).
- Video / image scripts cap clip access at index 2699 (Season 7 boundary).
- FEMBA and LUNA share the same 5 s tiling, so they share one video / image NPZ
  tagged `femba_luna`.
- `mamba-ssm` requires CUDA at install time (FEMBA only).
- Extraction scripts live in `src/extraction_scripts/`, analysis notebooks in
  `notebooks/`, and the cloned model repos in `models/` (git-ignored).
