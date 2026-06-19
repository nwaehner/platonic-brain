#!/usr/bin/env bash
# run_all.sh — ONE command: full alignment pipeline on a (Scaleway) GPU instance.
#
# Stages (each idempotent; skip individually with SKIP_*):
#   [1] measure_llm_bpb  → tokens/llm_bpb.json   (the real 1-BPB x-axis; not the proxy)
#   [2] extract LLM caption embeddings for every window scheme × LLM family
#       (EEG models + clip4s — the 4 s scheme used by video_vs_language)
#   [3] extract the 4 s clip4s VIDEO embeddings (only when clip4s ∈ VL_SCHEMES)
#   [4] render every figure (intramodal, video_vs_eeg_nonperf, language_vs_eeg,
#       video_vs_eeg, video_vs_language) WITH significance → src/alignment_plots/outputs/
#
# EEG + EEG-windowed vision embeddings come from HuggingFace; only the LLM and the
# clip4s video embeddings are new and are read locally via PLATONIC_LOCAL_DIR
# (the _common.fetch override). So no re-upload is needed to render the figures.
#
# One-line full run:
#   DO_SETUP=1 HF_TOKEN=hf_xxx bash src/alignment_plots/run_all.sh
#
# Config (env, all optional):
#   PY (python)  EEG_MODELS(5)  LLM_FAMILIES(bloom openllama llama)  VL_SCHEMES(clip4s)
#   HF_HOME(.hf_cache)  TMPDIR(.tmp)  PLATONIC_LOCAL_DIR(./embeddings)  BPB_TOKENS(4000000)
#   DO_SETUP=1   pip-install python deps (NOT torch — keep the instance's CUDA build)
#   SMOKE=1      wiring check: --test extraction/bpb, --smoke plots (stays on the proxy)
#   REQUIRE_BPB=0  allow a full run to fall back to the log-param proxy (default: forbid)
#   SKIP_BPB / SKIP_EXTRACT / SKIP_VIDEO / SKIP_PLOTS = 1   skip a stage
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Load local config (HF_TOKEN, PLATONIC_LLM_REPO, …) from .env if present (gitignored,
# so personal repo ids / tokens stay out of the committed code).
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

PY="${PY:-python}"
EEG_MODELS="${EEG_MODELS:-neurolm femba luna steegformer reve}"
LLM_FAMILIES="${LLM_FAMILIES:-bloom openllama llama}"
VL_SCHEMES="${VL_SCHEMES:-clip4s}"
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}"; mkdir -p "$TMPDIR"
export PLATONIC_LOCAL_DIR="${PLATONIC_LOCAL_DIR:-$ROOT/embeddings}"
BPB_TOKENS="${BPB_TOKENS:-4000000}"
SMOKE="${SMOKE:-0}"
EMB_LLM="$PLATONIC_LOCAL_DIR/llms"
EMB_VIS="$PLATONIC_LOCAL_DIR/vision"

LOG="${LOG:-$ROOT/run_all_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1                       # mirror all output to a run log

# ── HF token: prefer $HF_TOKEN, persist to tokens/hf_token.txt, then read back ──
if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p tokens; printf '%s' "$HF_TOKEN" > tokens/hf_token.txt
fi
[ -f tokens/hf_token.txt ] && export HF_TOKEN="$(cat tokens/hf_token.txt)"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: no HF token. Set HF_TOKEN=... or create tokens/hf_token.txt"; exit 1
fi

# ── window schemes that need LLM captions = EEG models ∪ VL_SCHEMES (deduped) ───
LLM_SCHEMES="$EEG_MODELS"
for s in $VL_SCHEMES; do
  case " $LLM_SCHEMES " in *" $s "*) ;; *) LLM_SCHEMES="$LLM_SCHEMES $s" ;; esac
done
case " $VL_SCHEMES " in *" clip4s "*) HAS_CLIP4S=1 ;; *) HAS_CLIP4S=0 ;; esac

PLOT_FLAG=""; BPB_FLAG="--n-tokens $BPB_TOKENS"; EXTRACT_FLAG=""
if [ "$SMOKE" = "1" ]; then PLOT_FLAG="--smoke"; BPB_FLAG="--test"; EXTRACT_FLAG="--test"; fi

banner() { echo; echo "==================== $* ===================="; date; }

echo "ROOT=$ROOT  HF_HOME=$HF_HOME  LOCAL=$PLATONIC_LOCAL_DIR  LOG=$LOG"
echo "EEG_MODELS=[$EEG_MODELS]  LLM_FAMILIES=[$LLM_FAMILIES]  VL_SCHEMES=[$VL_SCHEMES]"
echo "LLM_SCHEMES=[$LLM_SCHEMES]  HAS_CLIP4S=$HAS_CLIP4S  SMOKE=$SMOKE"

# ── [0] python deps (opt-in; never reinstalls torch) ───────────────────────────
if [ "${DO_SETUP:-0}" = "1" ]; then
  banner "[0] install python deps (transformers/datasets/… — NOT torch)"
  $PY -m pip install -U transformers datasets huggingface_hub scikit-learn scipy matplotlib requests
fi

# ── [1] LLM performance: 1-BPB over OpenWebText → tokens/llm_bpb.json ───────────
if [ "${SKIP_BPB:-0}" != "1" ]; then
  banner "[1/4] measure_llm_bpb (--all $BPB_FLAG)"
  $PY src/extraction_scripts/measure_llm_bpb.py --all $BPB_FLAG
fi

# ── [2] LLM caption embeddings → HuggingFace (upload each file, delete local after) ─
# extract_llm_captions.py (with PLATONIC_HF_UPLOAD set) writes each .npz to the
# PERSISTENT --out-dir, uploads it to HF, VERIFIES it, then deletes the local copy — so a
# wiped scratch disk can never lose work. FEMBA↔LUNA share one 5 s/2160-window tiling:
# computed once and copied to the twin automatically (no second forward pass).
# IMPORTANT: keep PLATONIC_LOCAL_DIR on the persistent disk (NOT ephemeral /scratch).
if [ "${SKIP_EXTRACT:-0}" != "1" ]; then
  banner "[2/4] extract_llm_captions → HF"
  # Dataset to upload llms/ to — configured via env/.env, not hardcoded.
  : "${PLATONIC_LLM_REPO:=${HF_REPO_ID:-}}"
  if [ -z "${PLATONIC_LLM_REPO:-}" ]; then
    echo "ERROR: PLATONIC_LLM_REPO not set — the HF dataset to upload llms/ to."
    echo "       Add 'PLATONIC_LLM_REPO=<user>/<dataset>' to .env (or export it)."
    exit 1
  fi
  export PLATONIC_LLM_REPO                       # so the plot scripts read from the same place
  export PLATONIC_HF_UPLOAD="$PLATONIC_LLM_REPO"
  echo "upload target = $PLATONIC_HF_UPLOAD   local(persistent) = $EMB_LLM"

  # pre-flight inventory: exactly what is already on HF vs still missing
  $PY - "$PLATONIC_HF_UPLOAD" "$LLM_SCHEMES" "$LLM_FAMILIES" <<'PY' || true
import sys
from huggingface_hub import HfApi
repo, schemes, fams = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
LLM = {"bloom":["bloomz-560m","bloomz-1b1","bloomz-1b7","bloomz-3b","bloomz-7b1"],
       "openllama":["open_llama_3b","open_llama_7b","open_llama_13b"],
       "llama":["llama-13b"]}
stems = [s for f in fams for s in LLM.get(f, [])]
have = set(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("llms/"))
tot = miss = 0
print(f"[2] INVENTORY on {repo}  (FEMBA=LUNA, luna copied from femba):")
for sch in schemes:
    cells = []
    for st in stems:
        ok = f"llms/{sch}/{st}_layerwise.npz" in have
        tot += 1; miss += 0 if ok else 1
        cells.append(f"{st}{'·OK' if ok else '·MISSING'}")
    print(f"   {sch:12s} " + "  ".join(cells))
print(f"[2] {tot-miss}/{tot} already on HF · up to {miss} to produce (luna twins copied, not recomputed)")
PY

  for eeg in $LLM_SCHEMES; do
    for fam in $LLM_FAMILIES; do
      echo "-- llm  $eeg × $fam --"
      $PY src/extraction_scripts/extract_llm_captions.py \
          --eeg-model "$eeg" --all-"$fam" --out-dir "$EMB_LLM" $EXTRACT_FLAG
    done
  done
fi

# ── [3] 4 s clip4s VIDEO embeddings (needed by video_vs_language clip4s) ────────
# (the video extractors have no --test mode, so SMOKE skips this heavy stage)
if [ "${SKIP_VIDEO:-0}" != "1" ] && [ "$HAS_CLIP4S" = "1" ] && [ "$SMOKE" != "1" ]; then
  banner "[3/4] extract clip4s video → HF (--eeg-family clip4s --window-seconds 4)"
  # Same dataset as llms/ — the video extractors write to the PERSISTENT --out-dir; we
  # then upload each .npz, VERIFY it, and delete the local copy (never before).
  : "${PLATONIC_LLM_REPO:=${HF_REPO_ID:-}}"
  if [ -z "${PLATONIC_LLM_REPO:-}" ]; then
    echo "ERROR: PLATONIC_LLM_REPO not set — the HF dataset to upload clip4s vision/ to."
    echo "       Add 'PLATONIC_LLM_REPO=<user>/<dataset>' to .env (or export it)."
    exit 1
  fi
  export PLATONIC_LLM_REPO
  echo "upload target = $PLATONIC_LLM_REPO   local(persistent) = $EMB_VIS"
  #   script:vision-subdir:#expected-clip4s-files (for auto-resume skip)
  for spec in \
    "extract_videomae.py:videomae:2" \
    "extract_dinov2.py:dinov2:4" \
    "extract_vjepa2.py:vjepa2:3" \
    "extract_videomae_ft_kinetics.py:videomae_finetuned:3"; do
    script="${spec%%:*}"; rest="${spec#*:}"; sub="${rest%%:*}"; exp="${rest##*:}"
    # auto-resume: skip this extractor entirely if all its clip4s files are already on HF
    have_n=$($PY - "$PLATONIC_LLM_REPO" "$sub" <<'PY'
import sys, os
from huggingface_hub import HfApi
repo, sub = sys.argv[1], sys.argv[2]
fs = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="dataset")
      if f.startswith(f"vision/{sub}/") and f.endswith("__clip4s.npz")]
print(len(fs))
PY
)
    if [ "${have_n:-0}" -ge "$exp" ]; then
      echo "-- video  $sub: $have_n/$exp clip4s on HF — skipping (done) --"; continue
    fi
    echo "-- video  $script → $EMB_VIS/$sub  ($have_n/$exp on HF) --"
    $PY src/extraction_scripts/"$script" \
        --eeg-family clip4s --window-seconds 4 --out-dir "$EMB_VIS/$sub"
    # upload this arch's clip4s npz(s) → HF, verify, then delete local
    $PY - "$PLATONIC_LLM_REPO" "$PLATONIC_LOCAL_DIR" "$EMB_VIS/$sub" <<'PY'
import os, sys, glob
from huggingface_hub import HfApi
repo, local_root, subdir = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ.get("HF_TOKEN"))
have = set(api.list_repo_files(repo, repo_type="dataset"))
for f in sorted(glob.glob(os.path.join(subdir, "*__clip4s.npz"))):
    dest = os.path.relpath(f, local_root)          # vision/<sub>/<file>
    if dest in have:
        print(f"  [hf] already on HF: {dest}"); os.remove(f); continue
    print(f"  [hf] UPLOAD → {repo}:{dest} ({os.path.getsize(f)/1e6:.0f} MB)")
    api.upload_file(path_or_fileobj=f, path_in_repo=dest, repo_id=repo,
                    repo_type="dataset", commit_message=f"step3: {dest}")
    if dest not in set(api.list_repo_files(repo, repo_type="dataset")):
        raise SystemExit(f"  [hf] verify FAILED for {dest} — keeping local copy")
    print(f"  [hf] verified on HF: {dest}")
    os.remove(f); print(f"  [hf] deleted local {f} (safe: verified on HF)")
PY
  done
fi

# ── BPB guarantee: a full run must use measured 1-BPB, never the proxy ──────────
if [ "${SKIP_PLOTS:-0}" != "1" ] && [ "$SMOKE" != "1" ] && [ "${REQUIRE_BPB:-1}" = "1" ] \
   && [ ! -f tokens/llm_bpb.json ]; then
  echo "ERROR: tokens/llm_bpb.json missing — figures would use the log-param proxy."
  echo "       Run stage [1] (measure_llm_bpb) first, or set REQUIRE_BPB=0 to allow the proxy."
  exit 1
fi

# ── [4] figures (with significance) ────────────────────────────────────────────
# CPU-ONLY stage — run this LOCALLY, not on the paid GPU instance (it's no faster there).
# PLOT_KPERM sets the permutation-null count (--k-perm): default 200; 50 is ~4x faster
# with p-resolution ~0.02, fine for these alignment figures.
if [ "${SKIP_PLOTS:-0}" != "1" ]; then
  banner "[4/4] figures"
  KPERM_FLAG=""; [ -n "${PLOT_KPERM:-}" ] && KPERM_FLAG="--k-perm $PLOT_KPERM"
  echo "k-perm = ${PLOT_KPERM:-200 (default)}"
  $PY src/alignment_plots/intramodal.py             $PLOT_FLAG $KPERM_FLAG
  $PY src/alignment_plots/video_vs_eeg_nonperf.py   $PLOT_FLAG $KPERM_FLAG
  $PY src/alignment_plots/language_vs_eeg.py         $PLOT_FLAG $KPERM_FLAG
  $PY src/alignment_plots/video_vs_eeg.py            $PLOT_FLAG $KPERM_FLAG
  for sch in $VL_SCHEMES; do
    $PY src/alignment_plots/video_vs_language.py --window-scheme "$sch" $PLOT_FLAG $KPERM_FLAG
  done
fi

banner "DONE — figures in src/alignment_plots/outputs/  (log: $LOG)"
