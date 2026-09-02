"""
block_a.py — cross-architecture, cross-modality (LLM → EEG): alignment vs LLM performance.

x = LLM performance (1 − bits-per-byte). y = Aristotelian-calibrated alignment (mKNN or CKA)
of each EEG size to the LLM captions. One curve per EEG size (unified small/base/large colours).

Three variants (``--variant``), each written to its own folder so nothing is clobbered:
  ancova  → `align ~ perf + C(size)`, OLS p in plot                         (DIR_A, default)
  perm    → same fit, PERMUTATION p in plot (9-LLM shuffle)                 (DIR_A_PERM)
  noconf  → `align ~ perf + C(size) + C(LLM-family)`, OLS p (+perm in CSV)  (DIR_A_NOCONF)

The permutation p is assumption-free: LLM performance is shuffled across the LLM *stems* (each
stem's sizes move together), so it respects that the independent unit is the 9 LLMs, not the 27
(LLM×size) points — the OLS t-test treats all 27 as independent. `noconf` additionally removes the
between-LLM-family baseline differences, isolating the within-family, within-size perf effect.

Outputs per folder: `language_vs_eeg__<model>__<mknn|cka>.png` (10), `combined__2x5.png`,
`stats_block_a.csv`, `stats_block_a_persize.csv`, `report.md`.

Usage:
  python src/paper_plots/block_a.py --variant ancova     # existing folder, aesthetics
  python src/paper_plots/block_a.py --variant perm
  python src/paper_plots/block_a.py --variant noconf
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
K_PERM = 10000

# variant → (control covariates, p shown in plot, default folder)
VARIANTS = {
    "ancova":   (["size"], "ols", "DIR_A"),
    "perm":     (["size"], "perm", "DIR_A_PERM"),
    "noconf":   (["size", "llm_family"], "ols", "DIR_A_NOCONF"),   # also adjusts LLM family
    "withconf": ([], "ols", "DIR_A_WITHCONF"),                     # y ~ perf, nothing controlled
}


def _index(rows):
    idx = {(r["eeg_model"], r["eeg_size"], r["partner"]): r for r in rows}
    stems = [s for s in P.LLM_STEMS
             if any(r["partner"] == s for r in rows) and C.LLM_PERF.get(s) is not None]
    return idx, sorted(stems, key=C.llm_x)


def _gather(model, mkey, idx, stems):
    """Flat arrays over the model's (size × LLM) points."""
    ax_, ay, asz, ast = [], [], [], []
    for sz in C.EEG[model]["sizes"]:
        for st in stems:
            r = idx.get((model, sz, st))
            if r is None or not np.isfinite(r[mkey]):
                continue
            ax_.append(C.llm_x(st)); ay.append(r[mkey]); asz.append(sz); ast.append(st)
    return np.array(ax_), np.array(ay), np.array(asz), np.array(ast)


def _fit(model, mkey, idx, stems, control_cats):
    """Fit y ~ perf + C(cats…) and return β/R²/OLS-p + the permutation p, per-size fits, points.
    `control_cats` ⊆ {'size','llm_family'}."""
    x, y, sz, st = _gather(model, mkey, idx, stems)
    fam = np.array([C.LLM_FAMILY_OF.get(s, s) for s in st])
    catmap = {"size": sz, "llm_family": fam}
    cats = [catmap[c] for c in control_cats]
    # family is a stem-level covariate collinear with perf → restricted (within-family) permutation
    strata = fam if "llm_family" in control_cats else None
    a = P.ancova_multi(y, x, *cats)
    p_perm, _ = P.perm_p(y, x, st, control_cats=cats, stem_strata=strata, K=K_PERM)
    return dict(model=model, beta=a["beta"], r2=a["r2_full"], p_ols=a["p"], p_perm=p_perm,
                n=len(y), per_size=P.per_size_fits(y, x, sz), x=x, y=y, sz=sz)


def _draw(ax, model, mkey, st, p_method, show_x, show_y, ylabel, stems):
    sizes = C.EEG[model]["sizes"]
    for sz in sizes:                                        # points coloured by size
        m = st["sz"] == sz
        ax.scatter(st["x"][m], st["y"][m], color=P.size_color(model, sz), s=42, zorder=3)
    b = st["beta"]
    if np.isfinite(b):                                      # common-slope parallel line per size
        for sz in sizes:
            m = st["sz"] == sz
            if m.sum():
                inter = st["y"][m].mean() - b * st["x"][m].mean()
                xf = np.linspace(st["x"].min(), st["x"].max(), 20)
                ax.plot(xf, inter + b * xf, color=P.size_color(model, sz), lw=1.4, ls="--", zorder=2)
    p_show = st["p_perm"] if p_method == "perm" else st["p_ols"]
    statline = f"β={b:+.3f}  p={p_show:.3g}{P.stars(p_show)}  R²={st['r2']:.2f}"
    ax.set_title((f"{C.DISPLAY[model]}\n{statline}" if show_x is False else statline), fontsize=10)
    ax.grid(alpha=0.3)
    if show_y:
        ax.set_ylabel(ylabel, fontsize=12)
    if show_x:
        ax.set_xticks([C.llm_x(s) for s in stems])
        ax.set_xticklabels([P.LLM_DISPLAY.get(s, s) for s in stems], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("LLM performance", fontsize=11)


def _write_report(out_dir, variant, fits):
    lines = []
    title = {"ancova": "y ~ perf + C(size), OLS p",
             "perm": "y ~ perf + C(size), permutation p",
             "noconf": "y ~ perf + C(size) + C(LLM-family), OLS p (+ permutation)",
             "withconf": "y ~ perf (nothing controlled — fully confounded), OLS p (+ permutation)"}[variant]
    lines.append(f"# Block (a) — LLM → EEG · {title}\n")
    lines.append("**x** = LLM performance (1−BPB). **y** = calibrated mKNN/CKA. Points = 9 LLMs × the "
                 "model's sizes. One curve per size (small/base/large).\n")
    if variant == "perm":
        lines.append("## Why a permutation p\n"
                     "The OLS t-test treats the 27 (LLM×size) points as independent, but the 3 sizes at "
                     "each performance value are repeated measures on the **same LLM** — so the effective "
                     "sample is the **9 LLMs**, not 27, and the OLS p is anti-conservative. The permutation "
                     "shuffles LLM-performance across the 9 stems (sizes move together) and recomputes the "
                     f"size-adjusted slope K={K_PERM}×; `p=(1+#{{|β*|≥|β_obs|}})/(K+1)`.\n")
        lines.append("## Combining the 3 per-size p-values\n"
                     "Each size also gives its own regression (9 LLMs) → 3 p-values (`stats_block_a_persize.csv`). "
                     "To merge them into one: **Fisher** `χ²=−2Σln pᵢ` and **Stouffer** `Z=ΣΦ⁻¹(1−pᵢ)/√k` both "
                     "assume the pᵢ are *independent* — here they are **not** (same 9 LLMs, correlated sizes), so "
                     "both understate the combined p. The **harmonic-mean p** is robust to that dependence, and the "
                     "single **global permutation** above sidesteps combination entirely. We therefore report all 3 "
                     "per-size p-values and use the global permutation p as the headline.\n")
    lines.append("## Results (per model × metric)\n")
    lines.append("| model | metric | β | OLS p | perm p | R² |")
    lines.append("|---|---|---|---|---|---|")
    for f, mk, mlabel in fits:
        lines.append(f"| {C.DISPLAY[f['model']]} | {mlabel} | {f['beta']:+.4f} | "
                     f"{f['p_ols']:.3g} | {f['p_perm']:.3g} | {f['r2']:.2f} |")
    if variant == "perm":
        lines.append("\n**Headline:** the permutation p is consistently **larger** than the OLS p — the "
                     "size-controlled LLM→EEG trend is **weaker/not significant** once the 9-LLM sample is "
                     "respected. Read `perm p`, not `OLS p`.")
    if variant == "withconf":
        lines.append("\nNothing is controlled here (`y ~ perf`): EEG size and LLM family are left as "
                     "confounds. β/p mix the within-family perf effect with between-family and "
                     "between-size baseline differences. Compare with the size-controlled and "
                     "size+family-controlled folders. OLS p is anti-conservative (27 correlated "
                     "points); the (global) permutation p in the CSV is the calibrated check.")
    if variant == "noconf":
        lines.append("\nThis fit additionally adjusts for **LLM family** (Bloom/OpenLLaMA/LLaMA) on top of "
                     "size, so β is the **within-family, within-size** effect of performance — it isolates the "
                     "clean perf gradient inside each family (e.g. Bloom 560M→7B1) from the between-family "
                     "baseline differences.\n\n**Trust the permutation p, not the OLS p.** LLM family is nearly "
                     "collinear with performance (families ≈ performance tiers), so the OLS t-test is wildly "
                     "anti-conservative here (it also treats the 27 correlated points as independent). The "
                     "permutation is **restricted to within-family shuffles** of performance (a global shuffle "
                     "would break the perf↔family structure and be invalid); it is calibrated to ~5% under the "
                     "null. Result: the within-family effect is genuine but **modest** — significant mainly for "
                     "CKA (FEMBA, REVE, STEEGFormer), marginal for mKNN — nowhere near the OLS ~1e-4.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Block (a): LLM→EEG alignment vs performance.")
    ap.add_argument("--variant", choices=list(VARIANTS), default="ancova")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    control_cats, p_method, dir_attr = VARIANTS[args.variant]
    out_dir = Path(args.out_dir) if args.out_dir else getattr(P, dir_attr)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = P.load_crossmodal_llm()
    idx, stems = _index(rows)
    models = [m for m in C.EEG if any(r["eeg_model"] == m for r in rows)]

    fits = {(m, mk): _fit(m, mk, idx, stems, control_cats) for m in models for mk, _ in METRICS}

    # ── individual figures ──────────────────────────────────────────────────────────
    for m in models:
        for mkey, mlabel in METRICS:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            _draw(ax, m, mkey, fits[(m, mkey)], p_method,
                  show_x=True, show_y=True, ylabel=mlabel, stems=stems)
            ax.legend(handles=P.size_legend_handles(), fontsize=8, loc="best")
            fig.tight_layout()
            tag = "mknn" if mkey == "mk_cal" else "cka"
            fig.savefig(out_dir / f"language_vs_eeg__{m}__{tag}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ── combined 2×5 (rows=metric, cols=model), shared x per column, one bottom legend ─
    nC = len(models)
    fig, axes = plt.subplots(2, nC, figsize=(4.2 * nC, 8.6), sharex="col", squeeze=False)
    for ci, m in enumerate(models):
        for ri, (mkey, mlabel) in enumerate(METRICS):
            _draw(axes[ri][ci], m, mkey, fits[(m, mkey)], p_method,
                  show_x=(ri == 1), show_y=(ci == 0), ylabel=mlabel, stems=stems)
    fig.legend(handles=P.size_legend_handles(), loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_dir / "combined__2x5.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ── stats + per-size CSVs ─────────────────────────────────────────────────────────
    with open(out_dir / "stats_block_a.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "metric", "beta", "r2", "p_ols", "p_perm", "n"])
        for m in models:
            for mkey, _ in METRICS:
                f = fits[(m, mkey)]
                w.writerow([m, "mknn" if mkey == "mk_cal" else "cka",
                            f["beta"], f["r2"], f["p_ols"], f["p_perm"], f["n"]])
    with open(out_dir / "stats_block_a_persize.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "metric", "size", "beta", "p", "r2", "n"])
        for m in models:
            for mkey, _ in METRICS:
                for d in fits[(m, mkey)]["per_size"]:
                    w.writerow([m, "mknn" if mkey == "mk_cal" else "cka",
                                d["size"], d["beta"], d["p"], d["r2"], d["n"]])

    _write_report(out_dir, args.variant,
                  [(fits[(m, mk)], mk, ml) for m in models for mk, ml in METRICS])
    print(f"  block_a [{args.variant}] → {out_dir.name}: 11 figs + stats + persize + report")


if __name__ == "__main__":
    main()
