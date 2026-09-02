# Request: regenerate the 16 cross-modal decodability-vs-alignment plots

## What I want

All 16 variants of the EEG-decodability (x) vs cross-modal-alignment (y) scatter, written to
`src/interp/outputs/eeg_alignment/`:

    {llm,vision}_crossmodal_{mknn,cka}_{aggregated,disaggregated}[_no-arch-confounders].png

i.e. 2 modalities x 2 metrics x 2 aggregation levels x 2 confound treatments.

- `aggregated`    = one point per EEG variant, y = mean calibrated score over all partners
- `disaggregated` = one subplot per partner (9 LLM stems / 12 video models)
- `no-arch-confounders` = the ANCOVA partial-regression version: both axes residualised on
  EEG family, so each point is a delta from its own family's mean. This removes the
  window-length confound (femba W=2160 vs reve W=1080).

## The script

`src/interp/eeg_features/eeg_crossmodal_decodability.py`

Prerequisite: `src/interp/outputs/eeg_alignment/summary.npz` must exist — it carries the
decodability axis (`perf`, mean R^2 over the 17 NICE features). Run `eeg_feature_alignment.py`
first if it's missing.

## Heads-up: 4 of the 16 cannot be produced as the code stands

I traced the three plotting paths. None emits the full set:

| path | metrics | ANCOVA variants? | emits |
|---|---|---|---|
| `main()` (full compute) | mk + ck | NO — never calls the `_ancova` plotters | 4 files (no `_no-arch-confounders`) |
| `--plot-only` (`_replot_from_npz`) | mk + ck | NO — same omission | 4 files |
| `--from-hf` (`_plots_from_hf_gridsearch`) | **mk only**, hardcoded `for measure in ("mk",)` | yes, all 4 plotters | 4 mKNN files, zero CKA |

That comment on the `--from-hf` loop says *"only mKNN (CKA not available in gridsearch rows)"*
and the function sets `ck_obs/ck_cal/ck_tau/ck_p = np.nan`.

This matches what's on disk: the 8 mKNN files are from a `--from-hf` run (all 4 variants
present) and the CKA files are older, with only `aggregated` + `disaggregated` and no
`_no-arch-confounders` twin.

So:
- **12 of 16 are reproducible today** — 8 mKNN via `--from-hf`, 4 CKA via a real compute run.
- **4 are not**: `{llm,vision}_crossmodal_cka_{aggregated,disaggregated}_no-arch-confounders.png`.
  They need a small code change first (below).

## The code change needed for the last 4

`_plot_aggregated_ancova(agg, modality, path)` and `_plot_disaggregated_ancova(rows, modality, path)`
have no `measure` parameter — they hardcode mKNN:

    _plot_aggregated_ancova    ->  xs = [agg[k]["mean_mk"] for k in keys]
    _plot_disaggregated_ancova ->  xs = [r["mk_cal"] for r in sub]
    (and `_plot_perfeature_partner_ancova` likewise)

plus hardcoded axis labels `"Delta mKNN"` and the suptitle `"mKNN vs EEG decodability"`.

Change: thread `measure` through those three functions the way `_plot_aggregated` /
`_plot_disaggregated` already do it — pick `mean_mk`/`mk_cal` vs `mean_ck`/`ck_cal`, set
`tag`/label from `measure` — then call all four plotters from inside the existing
`for measure in ("mk", "ck")` loop in `main()` (and in `_replot_from_npz` for the
`--plot-only` path).

## Commands

Full recompute, both modalities, both metrics (this is the one that gives real CKA):

    python src/interp/eeg_features/eeg_crossmodal_decodability.py --modality both

Defaults: `--k` = `C.K_MKNN_DEFAULT`, `--K 100` permutation null, `--alpha 0.05`.
Please keep the defaults so the numbers stay comparable with what we have.

Cheap replot from the cached `summary_crossmodal.npz` (no recompute, but only works for
metrics already in the cache — CKA is currently NaN there if the cache came from `--from-hf`):

    python src/interp/eeg_features/eeg_crossmodal_decodability.py --plot-only

Smoke test first if you want a fast sanity check: add `--smoke`.

## One thing to check before you start

The script is **untracked** — it is not on GitHub, and neither is `summary_crossmodal.npz`
or the `src/interp/outputs/eeg_alignment/` outputs. Current `origin/main` is at
`51543cb`. So you will not get this file from a pull; I need to send it to you, or I commit
and push it first. Tell me which you prefer.

## What to send back

The 16 PNGs, plus the regenerated `summary_crossmodal.npz`. If you make the `measure`
change, please commit it on a branch rather than pushing straight to `main`.
