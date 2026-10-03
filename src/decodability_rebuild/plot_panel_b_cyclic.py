#!/usr/bin/env python
"""
plot_panel_b_cyclic.py — panel (b) with the y axis calibrated by the CYCLIC null.

Same figure as plot_panel_b.py, same visual grammar as prh_axes.py. The only change is
the y axis: instead of the raw max-over-layer-pairs mKNN that `summary.npz` ships, each
pair is calibrated against the full cyclic permutation group C_n, following
aristotelian.pdf Eqs. (15)-(17) with Pi_n = C_n (see CYCLIC_NULL.md).

Per model pair, from the shift curve T(d) computed by compute_mknn_cyclic.py:

    T_obs = T(0)                                   the true pairing
    T_null = { T(d) : d = 1 .. W-1 }               K = W-1 = 1079 rotations, ALL of them
    tau_a  = order statistic ceil((1-a)(K+1)) of the COMBINED multiset {T_obs} u T_null
    p      = (1 + #{d != 0 : T(d) >= T_obs}) / (K+1)
    T_cal  = max( (T_obs - tau_a) / (1 - tau_a), 0 )        s_max = 1 for mKNN

NO FAR-TAIL RESTRICTION, by request. The null therefore contains d = 1, 2, 3 ..., which
for smooth signals are near-identical to d = 0. This is the exact cyclic test and it is
maximally conservative: tau_a sits close to T_obs and many T_cal will floor at 0.
`curves` is saved in full, so a tail-restricted variant needs no recomputation.

x is the from-scratch NICE decodability of compute_decodability.py (subject-mean view,
the published design), so the only thing that changed relative to panel_b_rebuilt.png is
the calibration of y.

Usage:
    python plot_panel_b_cyclic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "alignment_plots"))

from plot_panel_b import CACHE, OUT, THEME, load_x, draw  # noqa: E402


def calibrate_cyclic(curve, alpha=0.05):
    """aristotelian.pdf Eqs. (15)-(17) with the permutation group set to C_n."""
    T_obs = float(curve[0])
    T_null = np.asarray(curve[1:], float)          # every non-identity rotation
    K = len(T_null)
    combined = np.sort(np.concatenate([[T_obs], T_null]))
    idx = int(np.ceil((1 - alpha) * (K + 1))) - 1              # 1-indexed -> 0-indexed
    tau = float(combined[min(idx, len(combined) - 1)])
    p = float((1 + (T_null >= T_obs).sum()) / (K + 1))
    T_cal = max((T_obs - tau) / (1.0 - tau), 0.0) if tau < 1 else 0.0
    return T_obs, tau, T_cal, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--metric", default="mknn", choices=["mknn", "cka"],
                    help="mknn = the published panel (b) quantity (local neighbourhood); "
                         "cka = the same axis measured globally, so metric and modality "
                         "pair stop being confounded across panels (b) and (c).")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(THEME)

    src = {"mknn": "cyclic_mknn.npz", "cka": "cyclic_cka_eeg.npz"}[args.metric]
    MLAB = {"mknn": "mKNN", "cka": "CKA"}[args.metric]
    d = np.load(CACHE / src, allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    fam_all = np.array([str(f) for f in d["family"]])
    pairs, curves = d["pairs"], d["curves"]
    n = len(keys)

    MK_cal = np.full((n, n), np.nan)
    MK_obs = np.full((n, n), np.nan)
    TAU = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    for (i, j), c in zip(pairs, curves):
        o, t, cal, p = calibrate_cyclic(c, args.alpha)
        for a, b in ((i, j), (j, i)):
            MK_obs[a, b] = o; TAU[a, b] = t; MK_cal[a, b] = cal; P[a, b] = p

    # panel (b) uses CROSS-FAMILY pairs only (siblings align ~5x higher and would leak
    # Axis 1 into Axis 2); all 91 were computed, 78 are used here.
    same = fam_all[:, None] == fam_all[None, :]
    cal_x = np.where(same, np.nan, MK_cal)
    obs_x = np.where(same, np.nan, MK_obs)
    y_cal = np.nanmean(cal_x, axis=1)
    y_obs = np.nanmean(obs_x, axis=1)

    keys_x, fam_x, x_mv, _ = load_x()
    order = [keys_x.index(k) for k in keys]
    x = x_mv[order]
    fam = np.array([fam_x[i] for i in order])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    rho, p = draw(ax, x, y_cal, fam,
                  title="(b) Cross-model, within EEG\nbetter models agree more with "
                        "other architectures",
                  xlabel="NICE decodability",
                  ylabel=f"cyclic-null calibrated {MLAB}\nto other EEG families")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    f1 = OUT / f"panel_b_cyclic_{args.metric}.png"
    fig.savefig(f1, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {f1.name}   rho_adj={rho:+.3f} p={p:.4f}")

    print(f"\n{'model':20}{MLAB+' obs':>10}{'tau_.05':>9}{'cal':>9}{'worst p':>9}")
    for i, k in enumerate(keys):
        m = ~same[i]
        print(f"  {k:18}{y_obs[i]:>10.4f}{np.nanmean(np.where(m, TAU[i], np.nan)):>9.4f}"
              f"{y_cal[i]:>9.4f}{np.nanmax(np.where(m, P[i], np.nan)):>9.4f}")
    nz = int((y_cal > 0).sum())
    print(f"\n  {nz}/{n} models have a non-zero calibrated score")
    print(f"  cross-family pairs with p<=0.05: "
          f"{int((np.where(same, np.nan, P) <= 0.05).sum() // 2)}/78")


if __name__ == "__main__":
    main()
