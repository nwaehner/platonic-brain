"""
block_c.py — cross-modal, per-variant: does EEG decodability track cross-modal alignment?

Each point is one EEG variant (marker+colour = family). For each partner (LLM or video model)
and feature group we relate the variant's decodability R² to its calibrated alignment (mKNN/CKA).

Two controls (``--control``):
  family : ANCOVA `R² ~ align + C(family)`; axes are within-family residuals (ΔR² vs Δalign).
           This removes the architecture-family confound.  (the confound-removed study)
  none   : plain OLS `R² ~ align` with NO family term; axes are the RAW R² vs alignment.
           This keeps the confound (the "with-confound" study).

Two partner sets (``--partners``): llm (9 LLM stems) | video (12 video-model variants).

Figures (per folder): gross per-partner (rows=metric × cols=3 feature tiers {eeg, low,
semantic}), disaggregated per (tier×metric) (rows=features capped by --disagg-cap,
cols=partners), mean-over-partners per group (rows=metric). Plus stats_block_c.csv + report.md.

Decodability R² comes from the nested subject × time CV (compute_decodability.py) and is
negative for weakly-encoded tiers — its chance floor is negative too, so compare variants to
each other, not to zero.

Usage:
  python src/paper_plots/block_c.py                                        # LLM, family (default)
  python src/paper_plots/block_c.py --control none --out-dir <…_with-confound>
  python src/paper_plots/block_c.py --partners video --crossmodal-npz <…video…> --out-dir <…EEG-to-Video>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _paper_common as P
from _paper_common import C

METRICS = [("mk_cal", "mKNN"), ("ck_cal", "CKA")]


def _partner_order(kind, rows):
    present = set(r["partner"] for r in rows)
    if kind == "llm":
        return [s for s in P.LLM_STEMS if s in present]
    order = [f"{a}-{s}" for a in C.VISION for s in C.VISION[a]["sizes"]]
    return [p for p in order if p in present]


# ── data assembly ──────────────────────────────────────────────────────────────────
def build(rows, dec, partners):
    vkeys = [k for k in dec["keys"] if any(r["eeg"] == k for r in rows)]
    model = {r["eeg"]: r["eeg_model"] for r in rows}
    family = {r["eeg"]: r["family"] for r in rows}
    align = {m: {(r["eeg"], r["partner"]): r[m] for r in rows} for m, _ in METRICS}
    feat_names = dec["feat_names"]
    feat_col = {n: dec["perf"][:, i] for i, n in enumerate(feat_names)}
    key_row = {k: i for i, k in enumerate(dec["keys"])}
    return dict(vkeys=vkeys, model=model, family=family, stems=partners, align=align,
                feat_names=feat_names, feat_tier=dec["feat_tier"], feat_col=feat_col,
                key_row=key_row, tier_maps={t: P.tier_mean_r2(dec, t) for t in P.TIERS})


def _fit(B, y_r2, align_map, keys, control):
    """Return (stat, models). stat.resid_y = x-axis series (ΔR² or raw R²),
    stat.resid_x = y-axis series (Δalign or raw align); beta/p from the (family-ANCOVA or
    plain-OLS) fit; the annotated β is d(R²)/d(align)."""
    ys = [y_r2.get(k, np.nan) for k in keys]
    xs = [align_map.get(k, np.nan) for k in keys]
    ms = [B["model"].get(k, "?") for k in keys]
    if control == "family":
        st = P.ancova(ys, xs, [B["family"].get(k, "?") for k in keys])
    else:                                                   # no family control → raw OLS
        st = P.ancova(ys, xs, ["_"] * len(keys))            # single group ⇒ plain OLS β/p/R²
        st = dict(st, resid_y=np.asarray(ys, float), resid_x=np.asarray(xs, float))
    return st, ms


def _scatter(ax, stat, models, title, dprefix, at_chance=False):
    dx = np.asarray(stat["resid_y"], float)                 # x-axis
    dy = np.asarray(stat["resid_x"], float)                 # y-axis
    n = min(len(dx), len(dy), len(models))
    for x, y, m in zip(dx[:n], dy[:n], models[:n]):
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, s=55, color=P.FAM_COLORS.get(m, "#333"),
                       marker=P.FAM_MARKERS.get(m, "o"), edgecolor="k", lw=0.4, zorder=3)
    fin = np.isfinite(dx[:n]) & np.isfinite(dy[:n])
    if fin.sum() >= 2 and np.ptp(dx[:n][fin]) > 0:
        b, a = np.polyfit(dx[:n][fin], dy[:n][fin], 1)
        xf = np.linspace(dx[:n][fin].min(), dx[:n][fin].max(), 20)
        ax.plot(xf, a + b * xf, "k--", lw=1, zorder=2)
    if dprefix:                                             # residual plot → reference axes at 0
        ax.axhline(0, color="gray", lw=0.4, ls=":"); ax.axvline(0, color="gray", lw=0.4, ls=":")
    ax.set_title(f"{title}\nβ={stat['beta']:+.3f}  p={stat['p']:.3g}{P.stars(stat['p'])}"
                 + ("\n[tier at chance — fit is noise]" if at_chance else ""),
                 fontsize=8, color=("#b30000" if at_chance else "black"))
    ax.tick_params(labelsize=6)


def _famlegend(fig, models_present, y=-0.01):
    h = [plt.Line2D([], [], marker=P.FAM_MARKERS[m], ls="", color=P.FAM_COLORS[m],
         label=m, markeredgecolor="k") for m in P.FAM_COLORS if m in models_present]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8, bbox_to_anchor=(0.5, y))


def _write_report(out_dir, kind, control, plabel, n_partners, stats, dec=None):
    import collections
    sig = collections.Counter((r["scope"], r["metric"]) for r in stats
                              if np.isfinite(r["p"]) and r["p"] < 0.05)
    tot = collections.Counter((r["scope"], r["metric"]) for r in stats)
    ctl = ("within-family residuals (family confound removed)" if control == "family"
           else "raw values (architecture-family confound NOT removed)")
    lines = [f"# Block (c) — EEG ↔ {plabel} · {ctl}\n",
             f"Each point = one EEG variant (14). x = {'Δ' if control=='family' else ''}R² (EEG-feature "
             f"decodability), y = {'Δ' if control=='family' else ''}alignment (calibrated mKNN/CKA). "
             f"β = d(R²)/d(align); "
             + ("`R² ~ align + C(family)` (ANCOVA)." if control == "family"
                else "`R² ~ align` (plain OLS, no family term).") + "\n",
             f"Partners: {n_partners} {plabel} models. Figures: gross per-partner, disaggregated "
             f"per (tier×metric), mean-over-partners.\n",
             "## Significant panels (p<0.05)\n", "| scope | metric | # significant |",
             "|---|---|---|"]
    for k in sorted(tot):
        lines.append(f"| {k[0]} | {k[1]} | {sig.get(k,0)} / {tot[k]} |")
    if control == "none":
        lines.append("\nRaw (confounded) relationship — compare with the family-controlled folder; "
                     "any apparent effect here may be driven by between-family differences.")
    if dec is not None:
        lines += ["\n## Which tiers actually carry signal\n",
                  "Decodability comes from the nested time-block CV with honest hyperparameter "
                  "selection (`compute_decodability.py`). Skill = (R²−floor)/(1−floor): 0 means "
                  "no better than predicting the mean.\n",
                  "| tier | mean R² | chance floor | skill | carries signal? |",
                  "|---|---|---|---|---|"]
        for t in P.TIERS:
            r2 = np.nanmean(list(P.tier_mean_r2(dec, t).values()))
            nul = (np.nanmean(list(P.tier_mean_r2(dec, t, key="perf_null").values()))
                   if "perf_null" in dec else np.nan)
            at_chance, skill = P.tier_at_chance(dec, t)
            lines.append(f"| {P.TIER_LABEL[t]} | {r2:+.3f} | {nul:+.3f} | {skill:+.3f} | "
                         f"{'**no — at chance**' if at_chance else 'yes'} |")
        flat = [P.TIER_LABEL[t] for t in P.TIERS if P.tier_at_chance(dec, t)[0]]
        if flat:
            lines.append(f"\n**{', '.join(flat)}** sit on the chance floor for every EEG variant, "
                         "so the x-axis of those columns has no real spread and their β/p are "
                         "fitted to noise. Do not interpret them. Panels are marked in the figures.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partners", choices=["llm", "video"], default="llm")
    ap.add_argument("--control", choices=["family", "none"], default="family")
    ap.add_argument("--crossmodal-npz", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--disagg-cap", type=int, default=24)
    args = ap.parse_args()

    npz = Path(args.crossmodal_npz) if args.crossmodal_npz else (
        P.CROSSMODAL_VIDEO_NPZ if args.partners == "video" else P.CROSSMODAL_NPZ)
    out_dir = Path(args.out_dir) if args.out_dir else (
        P.DIR_C_VIDEO_NESTED if args.partners == "video"
        else P.DIR_C_CONFOUND_NESTED if args.control == "none" else P.DIR_C_NESTED)
    out_dir.mkdir(parents=True, exist_ok=True)
    plabel = "video" if args.partners == "video" else "LLM"
    dprefix = "Δ " if args.control == "family" else ""

    rows = P.load_crossmodal(npz)
    dec = P.load_decodability()
    partners = _partner_order(args.partners, rows)
    B = build(rows, dec, partners)
    keys, stems = B["vkeys"], partners
    models_present = {B["model"][k] for k in keys}
    stats = []

    def align_stem(metric, stem):
        return {k: B["align"][metric].get((k, stem), np.nan) for k in keys}

    def align_mean(metric):
        out = {}
        for k in keys:
            vals = [v for s in stems if np.isfinite(v := B["align"][metric].get((k, s), np.nan))]
            out[k] = float(np.mean(vals)) if vals else np.nan
        return out

    # ── (A) per-partner gross ──────────────────────────────────────────────────────
    for stem in stems:
        fig, axes = plt.subplots(2, len(P.TIERS), figsize=(4.0 * len(P.TIERS), 8), squeeze=False)
        for ri, (mkey, mlabel) in enumerate(METRICS):
            amap = align_stem(mkey, stem)
            for ci, tier in enumerate(P.TIERS):
                st, ms = _fit(B, B["tier_maps"][tier], amap, keys, args.control)
                _scatter(axes[ri][ci], st, ms, P.TIER_LABEL[tier], dprefix,
                         at_chance=P.tier_at_chance(dec, tier)[0])
                if ci == 0:
                    axes[ri][ci].set_ylabel(f"{dprefix}{mlabel}", fontsize=9)
                if ri == 1:
                    axes[ri][ci].set_xlabel(f"{dprefix}R²", fontsize=9)
                stats.append(dict(scope="gross", partner=stem, feature=tier, metric=mlabel,
                                  beta=st["beta"], p=st["p"], r2=st["r2_full"], n=st["n"]))
        _famlegend(fig, models_present)
        fig.suptitle(f"EEG ↔ {stem}: decodability vs alignment", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.savefig(out_dir / f"gross__{stem}.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── (A) disaggregated per (tier × metric) ──────────────────────────────────────
    for tier in P.TIERS:
        idx = np.where(B["feat_tier"] == P.disagg_tier(tier))[0]
        feats = [B["feat_names"][i] for i in idx]
        capped = len(feats) > args.disagg_cap
        if capped:
            order = np.argsort([np.nanmean(B["feat_col"][f]) for f in feats])[::-1]
            feats = [feats[i] for i in order[:args.disagg_cap]]
        for mkey, mlabel in METRICS:
            nr, nc = len(feats), len(stems)
            fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 1.7 * nr), squeeze=False)
            for fi, feat in enumerate(feats):
                y_r2 = {k: B["feat_col"][feat][B["key_row"][k]] for k in keys}
                for si, stem in enumerate(stems):
                    st, ms = _fit(B, y_r2, align_stem(mkey, stem), keys, args.control)
                    _scatter(axes[fi][si], st, ms, f"{feat}\n{stem}" if fi == 0 else feat, dprefix)
                    if si == 0:
                        axes[fi][si].set_ylabel(f"{dprefix}{mlabel}", fontsize=6)
                    stats.append(dict(scope="disagg", partner=stem, feature=feat, metric=mlabel,
                                      beta=st["beta"], p=st["p"], r2=st["r2_full"], n=st["n"]))
            _famlegend(fig, models_present, y=-0.005)
            cap = f"  [top-{args.disagg_cap} of {len(idx)} by mean R²]" if capped else ""
            fig.suptitle(f"Disaggregated · {P.TIER_LABEL[tier]} · {mlabel}  "
                         f"(rows: feature, cols: {plabel}){cap}", fontsize=12)
            fig.tight_layout(rect=[0, 0.02, 1, 0.99])
            fig.savefig(out_dir / f"disagg__{tier}__{'mknn' if mkey=='mk_cal' else 'cka'}.png",
                        dpi=110, bbox_inches="tight"); plt.close(fig)

    # ── (B) mean over partners ──────────────────────────────────────────────────────
    for tier in P.TIERS:
        fig, axes = plt.subplots(2, 1, figsize=(5.2, 9), squeeze=False)
        for ri, (mkey, mlabel) in enumerate(METRICS):
            st, ms = _fit(B, B["tier_maps"][tier], align_mean(mkey), keys, args.control)
            _scatter(axes[ri][0], st, ms, P.TIER_LABEL[tier], dprefix,
                     at_chance=P.tier_at_chance(dec, tier)[0])
            axes[ri][0].set_ylabel(f"{dprefix}{mlabel}", fontsize=9)
            axes[ri][0].set_xlabel(f"{dprefix}R²", fontsize=9)
            stats.append(dict(scope="mean", partner="mean", feature=tier, metric=mlabel,
                              beta=st["beta"], p=st["p"], r2=st["r2_full"], n=st["n"]))
        _famlegend(fig, models_present)
        fig.suptitle(f"EEG ↔ {plabel} (mean): {P.TIER_LABEL[tier]} decodability vs alignment", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.savefig(out_dir / f"mean__{tier}.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    with open(out_dir / "stats_block_c.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "partner", "feature", "metric", "beta", "p", "r2", "n"])
        w.writeheader(); w.writerows(stats)
    _write_report(out_dir, args.partners, args.control, plabel, len(stems), stats, dec)
    print(f"  block_c [{args.partners}/{args.control}] → {out_dir.name}: "
          f"{len(stems)} gross + {len(P.TIERS)*2} disagg + {len(P.TIERS)} mean + stats + report")


if __name__ == "__main__":
    main()
