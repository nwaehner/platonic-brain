"""
block_size.py — alignment vs MODEL SIZE (parameter count), both directions.

Block (a) plots EEG↔LLM alignment against LLM *competence* (1−BPB). This asks the other
scaling question: does alignment grow with sheer parameter count, on either side?

  (a) llm-size__2x5.png   x = log10 LLM params      y = calibrated CKA (row 1) / mKNN (row 2)
                          one column per EEG family, one curve per EEG size.
                          Fit: align ~ log10(params) + C(size).
  (b) eeg-size__2x9.png   x = log10 EEG-FM params   y = calibrated CKA (row 1) / mKNN (row 2)
                          one column per LLM stem, points coloured by EEG family.
                          Fit: align ~ log10(params) + C(family).

Same aesthetics as block_a's combined__2x5.png (shared size palette, parallel per-group fit
lines, β/p/R² in the panel title). Reads only _cache/crossmodal_llm_calibrated.npz — no new
alignment computation.

Both OLS and permutation p are reported. The OLS t-test treats every (partner × variant) point
as independent when they are repeated measures on the same partner, so it is anti-conservative;
the permutation shuffles the size→partner mapping (all points of a partner move together).

Outputs → data/paper-plots/size-vs-alignment/
  llm-size__2x5.png · eeg-size__2x9.png · stats_block_size.csv · report.md

Usage:  python src/paper_plots/block_size.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _paper_common as P
from _paper_common import C

# row 1 = CKA, row 2 = mKNN (as requested)
METRICS = [("ck_cal", "CKA"), ("mk_cal", "mKNN")]
K_PERM = 10000


def _log_params(v):
    return float(np.log10(v)) if v and v > 0 else np.nan


def _fmt_p(m):
    return f"{m/1000:.0f}B" if m >= 1000 else f"{m:.0f}M"


# ── (a) x = LLM parameter count, one column per EEG family ─────────────────────────
def gather_llm_size(rows, model, mkey, stems):
    x, y, sz, st = [], [], [], []
    idx = {(r["eeg_model"], r["eeg_size"], r["partner"]): r for r in rows}
    for s in C.EEG[model]["sizes"]:
        for stem in stems:
            r = idx.get((model, s, stem))
            if r is None or not np.isfinite(r[mkey]):
                continue
            x.append(_log_params(C.LLM_PARAMS.get(stem)))
            y.append(r[mkey]); sz.append(s); st.append(stem)
    return np.array(x), np.array(y), np.array(sz), np.array(st)


def draw_llm_size(ax, model, st, stems, show_x, show_y, ylabel):
    for s in C.EEG[model]["sizes"]:
        m = st["g"] == s
        ax.scatter(st["x"][m], st["y"][m], color=P.size_color(model, s), s=42, zorder=3)
    b = st["beta"]
    if np.isfinite(b):                                   # common-slope parallel line per size
        for s in C.EEG[model]["sizes"]:
            m = st["g"] == s
            if m.sum():
                inter = st["y"][m].mean() - b * st["x"][m].mean()
                xf = np.linspace(st["x"].min(), st["x"].max(), 20)
                ax.plot(xf, inter + b * xf, color=P.size_color(model, s), lw=1.4, ls="--", zorder=2)
    ax.set_title((f"{C.DISPLAY[model]}\n" if not show_x else "") +
                 f"β={b:+.3f}  p={st['p_perm']:.3g}{P.stars(st['p_perm'])}  R²={st['r2']:.2f}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    if show_y:
        ax.set_ylabel(ylabel, fontsize=12)
    if show_x:
        ticks = sorted({_log_params(C.LLM_PARAMS[s]) for s in stems})
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_p(10 ** t) for t in ticks], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("LLM parameters", fontsize=11)


# ── (b) x = EEG-FM parameter count, one column per LLM stem ────────────────────────
def gather_eeg_size(rows, stem, mkey):
    x, y, fam, key = [], [], [], []
    for r in rows:
        if r["partner"] != stem or not np.isfinite(r[mkey]):
            continue
        p = C.PARAMS.get(r["eeg_model"], {}).get(r["eeg_size"])
        x.append(_log_params(p)); y.append(r[mkey])
        fam.append(r["eeg_model"]); key.append(r["eeg"])
    return np.array(x), np.array(y), np.array(fam), np.array(key)


def draw_eeg_size(ax, st, stem, show_x, show_y, ylabel):
    for x, y, f in zip(st["x"], st["y"], st["g"]):
        ax.scatter(x, y, s=55, color=P.FAM_COLORS.get(f, "#333"),
                   marker=P.FAM_MARKERS.get(f, "o"), edgecolor="k", lw=0.4, zorder=3)
    b = st["beta"]
    if np.isfinite(b):
        for f in dict.fromkeys(st["g"].tolist()):
            m = st["g"] == f
            inter = st["y"][m].mean() - b * st["x"][m].mean()
            xf = np.linspace(st["x"].min(), st["x"].max(), 20)
            ax.plot(xf, inter + b * xf, color=P.FAM_COLORS.get(f, "#333"), lw=1.1, ls="--", zorder=2)
    ax.set_title((f"{P.LLM_DISPLAY.get(stem, stem)}\n" if not show_x else "") +
                 f"β={b:+.3f}  p={st['p_perm']:.3g}{P.stars(st['p_perm'])}  R²={st['r2']:.2f}",
                 fontsize=9)
    ax.grid(alpha=0.3)
    if show_y:
        ax.set_ylabel(ylabel, fontsize=12)
    if show_x:
        ax.set_xlabel("EEG-FM parameters", fontsize=10)
        # ticks at the actual variant sizes, labelled in M/B — not raw log10
        ticks = sorted({round(v, 3) for v in st["x"] if np.isfinite(v)})
        keep = [ticks[0]]                                  # thin out near-coincident sizes
        for t in ticks[1:]:
            if t - keep[-1] > 0.22:
                keep.append(t)
        ax.set_xticks(keep)
        ax.set_xticklabels([_fmt_p(10 ** t) for t in keep], rotation=45, ha="right", fontsize=7)
    ax.tick_params(labelsize=7)


def _fit(x, y, groups, unit_ids, strata=None):
    """ANCOVA y ~ x + C(groups) plus the unit-level permutation p."""
    a = P.ancova_multi(y, x, groups)
    p_perm, _ = P.perm_p(y, x, unit_ids, control_cats=[groups], stem_strata=strata, K=K_PERM)
    return dict(x=x, y=y, g=groups, beta=a["beta"], r2=a["r2_full"],
                p_ols=a["p"], p_perm=p_perm, n=a["n"])


def _write_report(out_dir, stems, models, stats):
    miss = [s for s in C.LLM_PARAMS if s not in stems]
    lines = [
        "# Size vs alignment — EEG foundation models ↔ LLMs\n",
        "Alignment here is the Aristotelian-calibrated CKA / mKNN already computed in "
        "`_cache/crossmodal_llm_calibrated.npz`; only the x-axis is new. Block (a) asks whether "
        "alignment grows with LLM *competence* (1−BPB); this asks whether it grows with raw "
        "**parameter count**, on each side in turn.\n",
        "| figure | x | y | columns | fit |",
        "|---|---|---|---|---|",
        "| `llm-size__2x5.png` | log10 LLM params | CKA (row 1), mKNN (row 2) | 5 EEG families | "
        "`align ~ log10(params) + C(EEG size)` |",
        "| `eeg-size__2x9.png` | log10 EEG-FM params | CKA (row 1), mKNN (row 2) | 9 LLM stems | "
        "`align ~ log10(params) + C(EEG family)` |\n",
        "## Which p to read\n",
        "The **permutation p** is in the panel titles. The OLS p (in the CSV) treats every point "
        "as independent, but the points in a column are repeated measures on the same partner "
        "models, so it is anti-conservative. The permutation shuffles the size→partner mapping "
        f"(all points of a partner move together), K={K_PERM}.\n",
        "## Coverage\n",
        f"LLM stems: {len(stems)} — {', '.join(stems)}.\n",
    ]
    if miss:
        lines.append(f"**Absent:** {', '.join(miss)}. The alignment cache was built over the LLM "
                     "stems with a measured 1−BPB (`P.LLM_STEMS`), so these are missing from the "
                     "cache even though their parameter counts are known. Extending "
                     "`compute_crossmodal_llm.py` to them would fill in the size axis, since the "
                     "size figure does not itself need 1−BPB.\n")
    lines += [f"EEG variants: {sum(len(C.EEG[m]['sizes']) for m in models)} across "
              f"{len(models)} families ({_fmt_p(min(C.PARAMS[m][s] for m in models for s in C.EEG[m]['sizes']))}"
              f"–{_fmt_p(max(C.PARAMS[m][s] for m in models for s in C.EEG[m]['sizes']))}).\n",
              "## Results\n", "| figure | column | metric | β | OLS p | perm p | R² | n |",
              "|---|---|---|---|---|---|---|---|"]
    for r in stats:
        lines.append(f"| {r['figure']} | {r['column']} | {r['metric']} | {r['beta']:+.4f} | "
                     f"{r['p_ols']:.3g} | {r['p_perm']:.3g} | {r['r2']:.2f} | {r['n']} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Alignment vs parameter count, both directions.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else P.PAPER_DIR / "size-vs-alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = P.load_crossmodal_llm()
    present = {r["partner"] for r in rows}
    stems = [s for s in P.LLM_STEMS if s in present and C.LLM_PARAMS.get(s)]
    models = [m for m in C.EEG if any(r["eeg_model"] == m for r in rows)]
    stats = []

    # ── (a) x = LLM params, columns = EEG families ─────────────────────────────────
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 8.6),
                             sharex="col", squeeze=False)
    for ci, m in enumerate(models):
        for ri, (mkey, mlabel) in enumerate(METRICS):
            x, y, sz, st_ = gather_llm_size(rows, m, mkey, stems)
            f = _fit(x, y, sz, st_)
            draw_llm_size(axes[ri][ci], m, f, stems, show_x=(ri == 1), show_y=(ci == 0),
                          ylabel=mlabel)
            stats.append(dict(figure="llm-size", column=m, metric=mlabel, beta=f["beta"],
                              p_ols=f["p_ols"], p_perm=f["p_perm"], r2=f["r2"], n=f["n"]))
    fig.legend(handles=P.size_legend_handles(), loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Alignment vs LLM size (EEG ↔ LLM, calibrated)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "llm-size__2x5.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved llm-size__2x5.png  (2 × {len(models)})")

    # ── (b) x = EEG-FM params, columns = LLM stems ─────────────────────────────────
    fig, axes = plt.subplots(2, len(stems), figsize=(3.0 * len(stems), 8.0),
                             sharex="col", sharey="row", squeeze=False)
    for ci, stem in enumerate(stems):
        for ri, (mkey, mlabel) in enumerate(METRICS):
            x, y, fam, key = gather_eeg_size(rows, stem, mkey)
            f = _fit(x, y, fam, key, strata=fam)
            draw_eeg_size(axes[ri][ci], f, stem, show_x=(ri == 1), show_y=(ci == 0),
                          ylabel=mlabel)
            stats.append(dict(figure="eeg-size", column=stem, metric=mlabel, beta=f["beta"],
                              p_ols=f["p_ols"], p_perm=f["p_perm"], r2=f["r2"], n=f["n"]))
    h = [plt.Line2D([], [], marker=P.FAM_MARKERS[m], ls="", color=P.FAM_COLORS[m], label=m,
                    markeredgecolor="k") for m in P.FAM_COLORS if m in models]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("Alignment vs EEG foundation-model size (EEG ↔ LLM, calibrated)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "eeg-size__2x9.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved eeg-size__2x9.png  (2 × {len(stems)})")

    with open(out_dir / "stats_block_size.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["figure", "column", "metric", "beta",
                                           "p_ols", "p_perm", "r2", "n"])
        w.writeheader(); w.writerows(stats)
    _write_report(out_dir, stems, models, stats)
    print(f"  block_size → {out_dir.name}: 2 figs + stats + report")


if __name__ == "__main__":
    main()
