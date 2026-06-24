# Running the figures (alignment stage 4) + interp studies on Scaleway

This is the **render / analysis** runbook: it produces every figure and every interp
study **from embeddings already on HuggingFace** — no GPU extraction, no re-upload.
For the heavy GPU *extraction* pipeline (computing the embeddings in the first place)
see [`alignment_plots/RUN_SCALEWAY.md`](alignment_plots/RUN_SCALEWAY.md) instead.

It covers, in order:

1. **Alignment stage 4** — `run_all.sh` stage `[4]` (intramodal [EEG+vision+LLM],
   video/language × EEG, video×language). CPU-only.
2. **Interp components 1–5** — `src/interp/run-components.sh` (full).
3. **Interp extra studies** — `src/interp/run-extra-studies.sh` (full).

Everything downloads from HF and writes locally; nothing is uploaded.

---

## 0. What you need (TL;DR)

| Thing | Why | Notes |
|---|---|---|
| **Any Linux instance** | the work is mostly CPU (sklearn/numpy/matplotlib) | a **GPU only speeds interp Component 1** (CLIP over movie frames); a plain CPU box also works, just slower there |
| **Block volume, ≥ 100 GB free** | the HF embedding cache is large (~40 GB) + `videos.tar` (2.6 GB) + CLIP weights | point `HF_HOME` at it; **reuse the same `HF_HOME`** for stage 4 *and* interp so embeddings download once |
| **Your HF token** | gated-dataset reads (nitrox639 + triniborrell) | one token (the `triniborrell` user) reads both repos |

A fresh clone already ships `tokens/llm_bpb.json` (the real 1-BPB x-axis), so stage 4
needs **no** GPU BPB run. The only files a clone is missing are your **token** and the
**`.env`** — both created in §2.

---

## 1. Instance + volume

```bash
# on the instance: mount the block volume and point HF_HOME at it
sudo mkfs.ext4 /dev/nvme1n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
export HF_HOME=/scratch/hf          # ← used by BOTH stage 4 and interp (download once)
export TMPDIR=/scratch/tmp && mkdir -p "$TMPDIR"
```

(Device name may differ — check `lsblk`. If you use a CPU-only instance, skip nothing;
the commands below are identical.)

## 2. Clone + set up (once)

```bash
git clone <repo-url> && cd platonic-brain
python -m venv venv && source venv/bin/activate
pip install -U pip

# Focused deps for rendering + interp (NOT the full EEG/video extraction stack):
#   torch+transformers → CLIP tier of interp Component 1; statsmodels → ANCOVA;
#   scikit-image/empath → interp visual+semantic feature tiers;
#   opencv-python-headless + pillow → Component 1 decodes the movie mp4 frames (cv2/PIL).
#   (headless OpenCV avoids the libGL.so.1 dependency missing on bare servers.)
pip install torch transformers huggingface_hub \
            numpy scipy pandas scikit-learn matplotlib \
            scikit-image empath statsmodels \
            opencv-python-headless pillow
# (If the instance already ships a CUDA torch, drop `torch` above to keep that build.)

# Your HF token (you already have this file's contents):
printf '%s' "<PASTE_HF_TOKEN>" > tokens/hf_token.txt

# .env: routes the embeddings WE generated (llms/* + 4 s clip4s vision) to triniborrell,
# while eeg/ and EEG-grid vision keep loading from nitrox639. run_all.sh auto-sources it.
printf 'PLATONIC_LLM_REPO=triniborrell/platonic-embeddings\n' > .env
```

> The interp scripts read the token from `tokens/hf_token.txt` automatically and hardcode
> the triniborrell 4 s repo, so they don't need `.env`. The `.env` is only for stage 4.

---

## 3. Alignment **stage 4** — render all figures (CPU)

Skip the three GPU stages (BPB / LLM extraction / video extraction); the BPB JSON and all
embeddings are already present/on HF, so stage 4 just downloads + renders.

```bash
HF_HOME=/scratch/hf \
  SKIP_BPB=1 SKIP_EXTRACT=1 SKIP_VIDEO=1 \
  bash src/alignment_plots/run_all.sh
```

- Output → `src/alignment_plots/outputs/` (PNGs + NPZs), log → `run_all_<ts>.log`.
- Includes the new figures: `intramodal__{dinov2,videomae_ft,vjepa2,bloom,openllama}.png`
  (+ `…_vision_combined` / `…_llm_combined`) and `language_vs_eeg_nonperf__<family>.png`.
- mKNN runs at **k = 5** (the default; nothing to set).
- **Faster stats:** add `PLOT_KPERM=50` (≈ 4× faster permutation null, p-resolution ~0.02).
- **Cheap wiring check first (optional):** `SMOKE=1 SKIP_BPB=1 SKIP_EXTRACT=1 SKIP_VIDEO=1 bash src/alignment_plots/run_all.sh`
  (renders quickly on a log-param x-axis — not the final figures).

First run downloads ~30–40 GB of embeddings into `HF_HOME`; reruns are fast (cached).

---

## 4. Interp **components 1–5** (full)

Component 1 builds the feature caches that 2–5 read, so the script runs them in order. It
downloads the movie clips (`videos.tar`, 2.6 GB) and a CLIP model the first time; a GPU
speeds the CLIP pass but it falls back to CPU (and, if torch is truly absent, just skips
the CLIP tier and continues).

```bash
source venv/bin/activate           # ensure `python` = the venv
HF_HOME=/scratch/hf bash src/interp/run-components.sh
# wiring check first (optional):  HF_HOME=/scratch/hf bash src/interp/run-components.sh --smoke
```

Output → `src/interp/outputs/`.

## 5. Interp **extra studies** (full)

Four independent studies; `model_stitching` + `attribute_probing` run once per distinct
EEG grid (`femba neurolm reve steegformer`; femba covers luna). Each `attribute_probing`
ref first (re)builds its Component-1 cache, so run §4 first (or just run this — it rebuilds
what it needs).

```bash
HF_HOME=/scratch/hf bash src/interp/run-extra-studies.sh
# wiring check first (optional):  HF_HOME=/scratch/hf bash src/interp/run-extra-studies.sh --smoke
```

Output → `src/interp/outputs/`.

> If `python` isn't the venv interpreter, pass it explicitly:
> `PYTHON=/scratch/.../venv/bin/python bash src/interp/run-extra-studies.sh`.

---

## 6. Retrieve results

```bash
tar czf results.tgz src/alignment_plots/outputs src/interp/outputs
# then scp/download results.tgz off the instance
```

`src/interp/outputs/` is gitignored (regenerable); `src/alignment_plots/outputs/` is
tracked, so you can also just `git add`/commit those PNGs if you want them in the repo.

---

## Why one `HF_HOME` for everything

Stage 4 and interp read the **same** EEG / vision / LLM embeddings from HF. Pointing both
at `/scratch/hf` means the ~40 GB of `*_layerwise.npz` is fetched **once** and reused —
the interp run then only adds `videos.tar` + the CLIP weights on top.

## Cost summary

| Step | GPU? | Main cost |
|---|---|---|
| Stage 4 figures | no | downloading embeddings (~30–40 GB), then CPU permutation stats |
| Interp comp 1 | optional | `videos.tar` (2.6 GB) + CLIP encode of movie frames (GPU ≫ CPU here) |
| Interp comp 2–5, extra studies | no | CPU (kNN, linear probes, ANCOVA) over cached embeddings |
