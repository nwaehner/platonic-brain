"""
block_b.py — intra-arch / same-modality (EEG): decodability vs intramodal alignment.

Each point is a consecutive size pair within a family (femba tiny↔base, base↔large, …;
9 pairs incl reve base↔large). Marker + colour identify the EEG family.
  x = mean decodability R² of the pair's two sizes (per feature tier).
  y = intramodal calibrated alignment between the two sizes (mKNN or CKA, subject-mean).
Per panel we report Spearman ρ, p over the 9 pairs.

One figure per feature tier {eeg, low, semantic} + one "mean" figure (x = mean R² across all
three tiers), each with two panels (mKNN | CKA). Reads only the two cached artifacts.

x error bars are the standard error of the tier-mean R² across the 6 nested-CV folds. R² is a
held-out-subject × held-out-time-block estimate and is negative for weakly-encoded tiers; the
chance floor is itself negative (see _cache/NEGATIVE_R2_DIAGNOSIS.md), so read the ordering
across variants, not the sign.

Outputs → data/paper-plots/intra-arch_same-modality_EEG__semantic-nestedcv/
  intra__<tier>.png   (3) · intra__mean.png (1) · stats_block_b.csv · report.md

Its own folder: both the tiers (mid+high → semantic) and the probe (nested subject × time CV)
changed, so these are NOT comparable with the originals in `intra-arch_same-modality_EEG/`,
which are kept intact. `data/paper-plots` is not under version control.

Usage:  python src/paper_plots/block_b.py
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

import _paper_common as P

METRICS = [("mk_cal", "mKNN (%)"), ("ck_cal", "CKA (%)")]


def _panel(ax, xs, ys, models, ylabel, xlabel, xerr=None):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xerr is not None:
        ax.errorbar(xs, ys, xerr=np.asarray(xerr, float), fmt="none",
                    ecolor="#999", elinewidth=1.0, capsize=2.5, zorder=2)
    for x, y, m in zip(xs, ys, models):
        ax.scatter(x, y, s=90, color=P.FAM_COLORS.get(m, "#333"),
                   marker=P.FAM_MARKERS.get(m, "o"), edgecolor="k", lw=0.5, zorder=3)
    msk = np.isfinite(xs) & np.isfinite(ys)
    if msk.sum() >= 2 and np.ptp(xs[msk]) > 0:
        b, a = np.polyfit(xs[msk], ys[msk], 1)
        xf = np.linspace(xs[msk].min(), xs[msk].max(), 20)
        ax.plot(xf, a + b * xf, "k--", lw=1.2, zorder=2)
    rho, p = P.spearman(xs, ys)
    ax.set_title(f"ρ={rho:+.2f}  p={p:.3g}{P.stars(p)}", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.3)
    return rho, p, int(msk.sum())


def _pair_x(dec_tier_map, model, a, b):
    """Mean tier-R² of the two sizes (nan-safe)."""
    va, vb = dec_tier_map.get(f"{model}-{a}"), dec_tier_map.get(f"{model}-{b}")
    vals = [v for v in (va, vb) if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _family_legend(fig, models_present):
    handles = [plt.Line2D([], [], marker=P.FAM_MARKERS[m], ls="", color=P.FAM_COLORS[m],
               label=m, markeredgecolor="k") for m in P.FAM_COLORS if m in models_present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))


def _write_report(out_dir, stats_out, dec):
    """Factual report: setup + the measured ρ/p table + the decodability floors. Any narrative
    interpretation belongs in the paper, not here — it would go stale on the next recompute."""
    by = {(r["tier"], r["metric"]): r for r in stats_out}
    tiers = [t for t in P.TIERS] + ["mean"]
    lines = [
        "# Block (b) — Intra-architecture, same modality: scaling within an EEG family\n",
        "**Question.** As an EEG foundation-model family scales, do consecutive sizes become "
        "both more **decodable** (their embeddings predict interpretable features) and more "
        "**internally aligned** (consecutive sizes agree on representational geometry)?\n",
        "## Setup",
        "- **point** — one consecutive size pair within a family (femba tiny↔base, base↔large, "
        "…); **9 pairs** across the five families (reve contributes its single base↔large pair).",
        "- **x** — mean decodability R² of the pair's two sizes, per feature tier. Error bars are "
        "the SE of that tier mean across the 6 nested-CV folds.",
        "- **y** — intramodal Aristotelian-calibrated alignment between the two sizes (mKNN% or "
        "CKA%, subject-mean), on their shared native grid.",
        "- **stat** — Spearman ρ, p over the 9 pairs, per tier × metric.",
        "- **tiers** — eeg (16 NICE EEG-derived), low (12 low-level visual), semantic "
        "(768-d ModernBERT [CLS] of the window's captions). Plus a \"mean\" panel.\n",
        "## Decodability probe\n",
        "R² comes from the **nested subject × time CV** (`compute_decodability.py`): α and layer "
        "are chosen on a validation split, R² is reported on a held-out subject × held-out "
        "contiguous time block, 6 Latin-square folds. The old probe selected both α and layer on "
        "the test folds and never split by subject.\n",
        "**The chance floor is negative.** Under block CV the intercept comes from train while "
        "R²'s denominator is the test block's variance about its own mean, so a model that has "
        "learned nothing scores below 0. Compare tiers and variants with each other, not with "
        "zero. See `_cache/NEGATIVE_R2_DIAGNOSIS.md`.\n",
        "| tier | mean R² | chance floor (`perf_null`) | skill | carries signal? |",
        "|---|---|---|---|---|"]
    for t in P.TIERS:
        r2 = np.nanmean(list(P.tier_mean_r2(dec, t).values()))
        nul = (np.nanmean(list(P.tier_mean_r2(dec, t, key="perf_null").values()))
               if "perf_null" in dec else np.nan)
        at_chance, skill = P.tier_at_chance(dec, t)
        lines.append(f"| {P.TIER_LABEL[t]} | {r2:+.3f} | {nul:+.3f} | {skill:+.3f} | "
                     f"{'**no — at chance**' if at_chance else 'yes'} |")
    lines.append("\nA tier at its chance floor has no x-spread, so the ρ and p in its panels are "
                 "fitted to noise and should not be interpreted.")
    lines += ["\n## Results (Spearman over the 9 size pairs)\n",
              "| tier | mKNN ρ (p) | CKA ρ (p) |", "|---|---|---|"]
    for t in tiers:
        mk, ck = by.get((t, "mknn")), by.get((t, "cka"))
        f = lambda d: (f"{d['rho']:+.2f} ({d['p']:.3g}){P.stars(d['p'])}"
                       if d and np.isfinite(d["rho"]) else "—")
        lines.append(f"| {P.TIER_LABEL.get(t, t)} | {f(mk)} | {f(ck)} |")
    lines.append("\nPer-feature breakdown: `intra__disent__<tier>__{mknn,cka}.png` + "
                 "`stats_block_b_disentangled.csv` (BH-FDR corrected within each panel set). "
                 "The semantic tier disaggregates over its top-24 PCA components.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Block (b): decodability vs intramodal alignment.")
    ap.add_argument("--x", choices=["r2", "skill"], default="r2",
                    help="x-axis: raw nested-CV R² (default), or the skill score "
                         "(R²−floor)/(1−floor), which is the fairer thing to average across "
                         "features whose chance floors differ by an order of magnitude.")
    args = ap.parse_args()
    xk = P.x_key(args.x)

    P.ensure_dirs()
    dec = P.load_decodability()
    intra = P.load_intramodal()                 # key '<m>__<a>__<b>' → mk_cal/ck_cal (S,)
    tier_maps = {t: P.tier_mean_r2(dec, t, key=xk) for t in P.TIERS}
    sem_maps = {t: P.tier_sem_r2(dec, t) for t in P.TIERS}
    # x for the "mean" figure = mean over the three tier means
    mean_map = {k: float(np.nanmean([tier_maps[t].get(k, np.nan) for t in P.TIERS]))
                for k in dec["keys"]}
    mean_sem = {k: float(np.nanmean([sem_maps[t].get(k, np.nan) for t in P.TIERS]))
                for k in dec["keys"]} if all(sem_maps.values()) else {}

    pairs = [intra[k] for k in intra]           # only pairs that were computed
    models_present = {d["model"] for d in pairs}

    def _xy(metric_key, xmap, emap):
        xs, ys, ms, es = [], [], [], []
        for d in pairs:
            m, a, b = d["model"], d["a"], d["b"]
            xs.append(_pair_x(xmap, m, a, b))
            ys.append(100.0 * float(np.nanmean(d[metric_key])))   # calibrated score → %
            ms.append(m)
            es.append(_pair_x(emap, m, a, b) if emap else np.nan)
        return xs, ys, ms, (es if emap else None)

    stats_out = []
    suffix = "" if args.x == "r2" else "__skill"
    panels = [(t, tier_maps[t], sem_maps[t],
               f"{P.TIER_LABEL[t]} — {P.X_LABEL[xk]}", f"intra__{t}{suffix}.png")
              for t in P.TIERS]
    panels.append(("mean", mean_map, mean_sem,
                   f"All tiers — {P.X_LABEL[xk]}", f"intra__mean{suffix}.png"))

    for tier, xmap, emap, xlabel, fname in panels:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (mkey, mlabel) in zip(axes, METRICS):
            xs, ys, ms, es = _xy(mkey, xmap, emap)
            rho, p, n = _panel(ax, xs, ys, ms, mlabel, xlabel, xerr=es)
            stats_out.append(dict(tier=tier, metric=("mknn" if mkey == "mk_cal" else "cka"),
                                  rho=rho, p=p, n=n))
        _family_legend(fig, models_present)
        at_chance, skill = P.tier_at_chance(dec, tier) if tier in P.TIERS else (False, np.nan)
        warn = (f"\n[!] this tier sits on its chance floor for every variant "
                f"(mean skill {skill:+.3f}) — the x-axis has no real spread and the fit is noise"
                if at_chance else "")
        fig.suptitle(f"Intra-architecture (EEG): {P.TIER_LABEL.get(tier, tier)} "
                     f"decodability vs intramodal alignment{warn}",
                     fontsize=13, color=("#b30000" if at_chance else "black"))
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        path = P.DIR_B_NESTED / fname
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {path.name}")

    with open(P.DIR_B_NESTED / f"stats_block_b{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tier", "metric", "rho", "p", "n"])
        w.writeheader(); w.writerows(stats_out)
    print(f"  saved stats_block_b.csv  ({len(stats_out)} rows)")
    _write_report(P.DIR_B_NESTED, stats_out, dec)
    print("  saved report.md")


if __name__ == "__main__":
    main()
