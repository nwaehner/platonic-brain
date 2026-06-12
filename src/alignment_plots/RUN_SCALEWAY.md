# Running the full alignment pipeline on a Scaleway GPU instance

The plotting is cheap (CPU); the cost is **(a)** computing LLM 1-BPB scores and
**(b)** extracting LLM caption embeddings for every EEG family × LLM family — both
need a GPU and a lot of disk (LLaMA-65B weights alone are ~130 GB in bf16). EEG and
the EEG-windowed vision embeddings are already on HuggingFace, so they are *not*
recomputed; only the **LLM captions** and the **4 s `clip4s` video** embeddings are new.

## TL;DR — one command does the whole thing

```bash
# from the repo root on the instance (deps installed / volume mounted — see §2)
DO_SETUP=1 HF_TOKEN=hf_xxx bash src/alignment_plots/run_all.sh
```

`run_all.sh` runs, in order: **[1]** `measure_llm_bpb --all` → `tokens/llm_bpb.json` (the real
1−BPB x-axis), **[2]** LLM caption embeddings for every window scheme incl. `clip4s` × every LLM
family, **[3]** the four 4 s `clip4s` video extractions (placed at the `vision/<arch>/…` layout the
loader expects), **[4]** every figure **with significance** → `src/alignment_plots/outputs/`.
EEG + EEG-windowed vision come from HF; only the LLM and `clip4s` video embeddings are new and are
read locally via `PLATONIC_LOCAL_DIR`.

It **refuses to fall back to the log-param proxy** on a full run: if `tokens/llm_bpb.json` is missing
it errors before plotting (override with `REQUIRE_BPB=0`). The figure titles also print
`[x = measured 1−BPB …]` vs `[x = log-param proxy …]` so you can see which axis is live.

### Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `HF_TOKEN` | — | written to `tokens/hf_token.txt`; required (or pre-create that file) |
| `DO_SETUP` | `0` | `1` → `pip install` python deps (**not** torch — keep the CUDA build) |
| `EEG_MODELS` | all 5 | EEG models to extract LLM captions for |
| `LLM_FAMILIES` | `bloom openllama llama` | LLM families (drop `llama` to skip the heavy 65B) |
| `VL_SCHEMES` | `clip4s` | window scheme(s) for `video_vs_language` (4 s clips) |
| `HF_HOME` / `TMPDIR` / `PLATONIC_LOCAL_DIR` | `./.hf_cache` / `./.tmp` / `./embeddings` | caches + local embeddings (put on the scratch volume) |
| `BPB_TOKENS` | `4000000` | OpenWebText tokens for the 1−BPB estimate (lower = faster/noisier) |
| `REQUIRE_BPB` | `1` | `0` allows the proxy on a full run |
| `SKIP_BPB`/`SKIP_EXTRACT`/`SKIP_VIDEO`/`SKIP_PLOTS` | `0` | skip a stage (e.g. re-render figures only) |
| `SMOKE` | `0` | wiring check: `--test` extraction/bpb, `--smoke` plots, **skips the video stage** (stays on the proxy) |

Examples:
```bash
# put caches on a scratch volume, drop LLaMA-65B
HF_HOME=/scratch/hf TMPDIR=/scratch/tmp PLATONIC_LOCAL_DIR=/scratch/embeddings \
  LLM_FAMILIES="bloom openllama" HF_TOKEN=hf_xxx bash src/alignment_plots/run_all.sh

# re-render figures only (after extraction already ran)
SKIP_BPB=1 SKIP_EXTRACT=1 SKIP_VIDEO=1 bash src/alignment_plots/run_all.sh
```

The four steps below are exactly what the one command automates — listed for reference.

```bash
# [1] python src/extraction_scripts/measure_llm_bpb.py --all
# [2] for eeg in neurolm femba luna steegformer reve clip4s; do
#       for fam in bloom openllama llama; do
#         python src/extraction_scripts/extract_llm_captions.py --eeg-model $eeg --all-$fam --out-dir embeddings/llms
#       done; done
# [3] for arch dir in videomae|videomae dinov2|dinov2 vjepa2|vjepa2 videomae_ft_kinetics|videomae_finetuned:
#       python src/extraction_scripts/extract_<arch>.py --eeg-family clip4s --window-seconds 4 --out-dir embeddings/vision/<dir>
# [4] intramodal.py; video_vs_eeg_nonperf.py; language_vs_eeg.py; video_vs_eeg.py;
#     video_vs_language.py --window-scheme clip4s
```

## 1. Pick an instance

| Need | Recommendation |
|---|---|
| Up to LLaMA-13B / OpenLLaMA-13B | 1× **L40S (48 GB)** or **H100 (80 GB)** (`H100-1-80G`) |
| Including **LLaMA-30B** | 1× **H100 80 GB** (bf16 fits) |
| Including **LLaMA-65B** | **2×+ H100** (bf16 ≈130 GB) — needs model sharding (`device_map="auto"`), or run 65B in 8-bit, or skip it |

Attach a **Block Storage volume ≥ 1 TB** for model weights + caches + embeddings.

> LLaMA-65B on a single GPU will OOM. Options: provision a multi-GPU instance and add
> `device_map="auto"` to `load_text_model`, load 65B in 8-bit (`bitsandbytes`), or set
> `LLM_FAMILIES="bloom openllama"` and add LLaMA later.

## 2. Provision

```bash
# on the instance
sudo mkfs.ext4 /dev/nvme1n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
export HF_HOME=/scratch/hf            # model + dataset caches
export TMPDIR=/scratch/tmp && mkdir -p $TMPDIR

git clone <repo-url> && cd platonic-brain
python -m venv venv && source venv/bin/activate
pip install torch transformers datasets huggingface_hub scikit-learn scipy matplotlib requests
# (mamba-ssm / setup.sh are only needed to re-extract FEMBA EEG — not needed here)

printf '%s' "<HF_TOKEN>" > tokens/hf_token.txt      # gated dataset access
```

## 3. Run

```bash
# full run (all EEG families, all LLM families, real BPB, real figures)
HF_HOME=/scratch/hf PLATONIC_LOCAL_DIR=/scratch/embeddings \
  bash src/alignment_plots/run_all.sh

# scope it down (e.g. skip 65B / one family)
EEG_MODELS="neurolm femba luna steegformer reve" LLM_FAMILIES="bloom openllama" \
  bash src/alignment_plots/run_all.sh

# resume: stages are idempotent (existing *_layerwise.npz and llm_bpb.json are skipped)
SKIP_BPB=1 SKIP_EXTRACT=1 bash src/alignment_plots/run_all.sh   # just re-render figures
```

`run_all.sh` runs three stages: `measure_llm_bpb.py --all` → `tokens/llm_bpb.json`;
`extract_llm_captions.py` for each EEG×LLM family → `$PLATONIC_LOCAL_DIR/llms/<eeg>/`;
then every figure. `PLATONIC_LOCAL_DIR` lets the plots read the new embeddings locally
while pulling EEG/vision from HF (see `_common.fetch`).

## 4. Validate first (cheap)

```bash
SMOKE=1 EEG_MODELS=neurolm LLM_FAMILIES=bloom bash src/alignment_plots/run_all.sh
```

Smoke uses `--test` extraction (50 windows, writes `*_test.npz`), a 50k-token BPB run,
and `--smoke` plots — it checks wiring end-to-end, not the final data.

## 4b. 4 s windows for `video_vs_language` (`clip4s`)

`video_vs_language` defaults to `--window-scheme clip4s` (native 4 s clips: 1 caption ↔ 1
clip, 2700 windows, no aggregation on either side). Those embeddings are **not** on HF — both
sides must be extracted at 4 s:

```bash
# LLM captions at 4 s (1 caption/window) → embeddings/llms/clip4s/
python src/extraction_scripts/extract_llm_captions.py --eeg-model clip4s --all-bloom      --out-dir embeddings/llms
python src/extraction_scripts/extract_llm_captions.py --eeg-model clip4s --all-openllama  --out-dir embeddings/llms
python src/extraction_scripts/extract_llm_captions.py --eeg-model clip4s --all-llama      --out-dir embeddings/llms

# Video at 4 s for every arch (tag = clip4s, window = 4 s). Each call does all sizes.
for arch in videomae dinov2 vjepa2 videomae_ft_kinetics; do
  python src/extraction_scripts/extract_${arch}.py --eeg-family clip4s --window-seconds 4
done
# Place outputs so the loader finds them (HF layout): vision/<arch>/<arch>_<size>__clip4s.npz
#   e.g. embeddings/vision/videomae/videomae_base__clip4s.npz
```

Then render with the local override:

```bash
PLATONIC_LOCAL_DIR=$PWD/embeddings \
  python src/alignment_plots/video_vs_language.py --window-scheme clip4s
```

`run_all.sh` already automates all of the above — stage 2 extracts the `clip4s` captions
(`LLM_SCHEMES = EEG_MODELS ∪ VL_SCHEMES`) and stage 3 runs the four 4 s video extractions — so you
normally don't run these by hand; they're here for reference / re-running a single piece.

## 5. Retrieve / persist

```bash
tar czf outputs.tgz src/alignment_plots/outputs            # figures + NPZ
# optional: publish the new embeddings so others skip extraction
python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="/scratch/embeddings/llms", path_in_repo="llms",
                      repo_id="nitrox639/platonic-embeddings", repo_type="dataset")
PY
```

After upload, anyone can render the figures from HF alone (no `PLATONIC_LOCAL_DIR`).
