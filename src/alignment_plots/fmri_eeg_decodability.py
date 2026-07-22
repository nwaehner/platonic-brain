"""
fmri_eeg_decodability.py — does a "better" EEG model align more with fMRI, AFTER
removing the between-family confound (window length differs per EEG family)?

This is the family-confound ANCOVA / added-variable plot used elsewhere in the repo
(mirrors src/interp/_interp_common.ancova_family): we regress
    alignment  ~  decodability + C(family)
and show the PARTIAL-REGRESSION plot — each axis residualised on family, i.e. every
point is expressed as a DELTA from its own family's mean:
    X = Δ decodability | family   (resid of EEG feature R² after removing family means)
    Y = Δ alignment    | family   (resid of fMRI–EEG CKA / mKNN after removing family means)
The family-adjusted slope β_adj is the within-family effect of decodability on alignment;
we report whether β_adj > 0 and its one-sided significance.

Inputs (both already computed, local):
  decodability R² : src/interp/outputs/eeg_alignment/summary.npz  → perf_mean (mean R²
                    of decoding the 16 NICE EEG features from the EEG embedding)
  fMRI–EEG align. : src/alignment_plots/outputs/fmri_vs_all__*.npz (from fmri_vs_all.py)

Usage:  python fmri_eeg_decodability.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
SUMMARY = _ROOT / "src" / "interp" / "outputs" / "eeg_alignment" / "summary.npz"

# which family npz holds each EEG model's fMRI alignment
FAM_OF = {"femba": "femba_luna", "luna": "femba_luna", "steegformer": "steegformer",
          "neurolm": "neurolm", "reve": "reve"}
COLOR = {"femba": "#1b9e77", "luna": "#d95f02", "neurolm": "#7570b3",
         "reve": "#e7298a", "steegformer": "#66a61e"}


def load_decodability():
    d = np.load(SUMMARY, allow_pickle=True)
    return {str(k): float(v) for k, v in zip(d["keys"], d["perf_mean"])}


def load_fmri_eeg_alignment():
    """key 'model-size' -> (cka, mknn_cal) for every EEG model, from the family npzs."""
    out = {}
    for fam in set(FAM_OF.values()):
        p = OUT / f"fmri_vs_all__{fam}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        for lbl, kind, ck, mk in zip(d["labels"], d["kinds"], d["cka"], d["mknn_cal"]):
            if str(kind) == "eeg":
                out[str(lbl)] = (float(ck), float(mk))
    return out


def _residualize_on_family(v, fam):
    """Remove each family's mean from v → the within-family delta (added-variable resid)."""
    v = np.asarray(v, float); out = np.empty_like(v)
    for f in set(fam):
        idx = [i for i in range(len(fam)) if fam[i] == f]
        out[idx] = v[idx] - v[idx].mean()
    return out


def family_ancova(x, y, fam):
    """Family-adjusted slope of y on x (removing family means), with a proper t-test.
    Mirrors _interp_common.ancova_family: partial regression on family dummies.
    df = n − n_family − 1 (family intercepts + one slope). Returns dict."""
    rx = _residualize_on_family(x, fam)
    ry = _residualize_on_family(y, fam)
    n, nfam = len(x), len(set(fam))
    df = n - nfam - 1
    sxx = float((rx * rx).sum())
    if df <= 0 or sxx == 0:
        return dict(beta=np.nan, se=np.nan, t=np.nan, p_one=np.nan, r2=np.nan,
                    rx=rx, ry=ry, df=df)
    beta = float((rx * ry).sum() / sxx)              # slope through residualised origin
    resid = ry - beta * rx
    sigma2 = float((resid * resid).sum() / df)
    se = float(np.sqrt(sigma2 / sxx))
    tval = beta / se if se > 0 else np.nan
    p_two = float(2 * tdist.sf(abs(tval), df))
    p_one = p_two / 2 if beta > 0 else 1 - p_two / 2  # H1: beta > 0
    ss_tot = float((ry * ry).sum())
    r2 = 1 - float((resid * resid).sum()) / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, se=se, t=tval, p_one=p_one, r2=r2, rx=rx, ry=ry, df=df)


def main():
    decod = load_decodability()
    align = load_fmri_eeg_alignment()
    keys = [k for k in decod if k in align]
    if not keys:
        raise SystemExit("no overlapping EEG keys between summary.npz and fmri npzs")

    x = np.array([decod[k] for k in keys])
    cka = np.array([align[k][0] for k in keys])
    mk = np.array([align[k][1] for k in keys])
    fam = [k.rsplit("-", 1)[0] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    stats = {}
    for ax, y, name in [(axes[0], cka, "fMRI–EEG linear CKA"),
                        (axes[1], mk, "fMRI–EEG mKNN (calibrated)")]:
        r = family_ancova(x, y, fam)
        # added-variable scatter: Δdecodability|family vs Δalignment|family
        for f in sorted(set(fam)):
            idx = [i for i in range(len(keys)) if fam[i] == f]
            ax.scatter(r["rx"][idx], r["ry"][idx], color=COLOR.get(f, "gray"),
                       s=75, label=f, zorder=3)
        xs = np.linspace(r["rx"].min(), r["rx"].max(), 50)
        ax.plot(xs, r["beta"] * xs, color="black", lw=2, zorder=2, label="family-adj fit")
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        sig = "YES" if (r["beta"] > 0 and r["p_one"] < 0.05) else "no"
        ci = 1.96 * r["se"]
        ax.set_title(f"Δ{name}  vs  Δ decodability   (within-family)\n"
                     f"β_adj={r['beta']:+.3f} ± {r['se']:.3f}  "
                     f"(95% CI [{r['beta']-ci:+.3f}, {r['beta']+ci:+.3f}])\n"
                     f"t={r['t']:+.2f} (df={r['df']}), one-sided p(β>0)={r['p_one']:.3f} "
                     f"→ sig>0: {sig}", fontsize=10)
        ax.set_xlabel("Δ EEG decodability | family   (resid R²)")
        ax.set_ylabel(f"Δ {name} | family")
        ax.grid(alpha=0.3)
        stats[name] = r
    axes[0].legend(title="EEG family", fontsize=8)
    fig.tight_layout()
    p = OUT / "fmri_eeg_vs_decodability.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("saved", p)
    for name, s in stats.items():
        verdict = ("POSITIVE & significant (p<0.05)" if s["beta"] > 0 and s["p_one"] < 0.05
                   else "positive but NOT significant" if s["beta"] > 0 else "negative")
        print(f"  {name:28} β_adj={s['beta']:+.3f}±{s['se']:.3f}  t={s['t']:+.2f} "
              f"(df={s['df']})  one-sided p={s['p_one']:.3f}  → {verdict}")


if __name__ == "__main__":
    main()
