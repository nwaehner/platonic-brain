"""
test_nested_cv.py — the leakage assertions for `nested_ridge_r2`.

The whole point of the nested subject × time CV is that a reported R² cannot be produced by
memorising a stimulus window or a subject. These tests check that mechanically (fold
construction) and statistically (a shuffled-target control must land on the chance floor).

Run:  .venv-interp/bin/python src/paper_plots/test_nested_cv.py
      (pytest also works where it is installed; the venv has no pip, hence the bare runner
       at the bottom — every test is a plain no-arg function with asserts.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
for _d in [_SRC / "interp", _SRC / "interp" / "feature_search", _SRC / "alignment_plots"]:
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from feature_gridsearch import nested_cv_folds, nested_ridge_r2   # noqa: E402

W, S, D, F, NB = 300, 6, 24, 5, 5


# ── fold construction ──────────────────────────────────────────────────────────────
def test_test_cell_never_appears_in_any_training_or_val_split():
    """The one that matters: the test subject AND the test windows must be absent from every
    split the model ever sees — the final refit, and every inner train/val used to pick α."""
    for f in nested_cv_folds(W, S, NB):
        te_s, te_w = set(f["test"][0]), set(f["test"][1])
        for name, (s, w) in [("remaining", f["remaining"])] + \
                [(f"inner{i}-{k}", sp) for i, pair in enumerate(f["inner"])
                 for k, sp in zip(("train", "val"), pair)]:
            assert set(s) & te_s == set(), f"test subject leaks into {name}"
            assert set(w) & te_w == set(), f"test windows leak into {name}"


def test_inner_val_never_appears_in_its_own_inner_train():
    for f in nested_cv_folds(W, S, NB):
        for (tr_s, tr_w), (va_s, va_w) in f["inner"]:
            assert set(tr_s) & set(va_s) == set(), "val subject appears in inner train"
            assert set(tr_w) & set(va_w) == set(), "val windows appear in inner train"


def test_every_subject_tested_exactly_once():
    tested = [int(f["test"][0][0]) for f in nested_cv_folds(W, S, NB)]
    assert sorted(tested) == list(range(S))


def test_final_refit_uses_all_remaining_cells():
    """S-1 subjects × NB-1 contiguous blocks — the refit should not waste the val cell."""
    for f in nested_cv_folds(W, S, NB):
        assert len(f["remaining"][0]) == S - 1
        assert abs(len(f["remaining"][1]) - W * (NB - 1) / NB) <= NB


def test_time_only_mode_still_isolates_the_test_windows():
    """hold_out_subject=False keeps every subject in every split (X is the subject-MEAN, so
    there is no subject axis to generalise across), but the TEST WINDOWS must still never be
    seen — that is what the whole block-CV design is protecting."""
    folds = nested_cv_folds(W, S, NB, hold_out_subject=False)
    assert len(folds) == NB, "one outer fold per time block"
    for f in folds:
        te_w = set(f["test"][1])
        assert set(f["remaining"][1]) & te_w == set(), "test windows leak into the refit"
        for (tr_s, tr_w), (va_s, va_w) in f["inner"]:
            assert set(tr_w) & te_w == set(), "test windows leak into inner train"
            assert set(va_w) & te_w == set(), "test windows leak into inner val"
            assert set(tr_w) & set(va_w) == set(), "inner val leaks into inner train"
    assert sorted(w for f in folds for w in f["test"][1]) == list(range(W)), \
        "every window tested exactly once"


def test_time_only_mode_runs_end_to_end():
    emb, Y = _fake_data(seed=21, signal=3.0, drift=False)
    out = nested_ridge_r2(emb, Y, n_blocks=NB, hold_out_subject=False)
    assert out["r2_folds"].shape[0] == NB
    assert np.nanmean(out["r2"]) > 0.3, "planted signal not recovered in time-only mode"


def test_selection_uses_several_inner_folds():
    """A single val cell makes the per-feature argmax over ~40 (layer, α) options overfit the
    val split; averaging over inner folds is what keeps test R² at or above the floor."""
    for f in nested_cv_folds(W, S, NB, n_inner=4):
        assert len(f["inner"]) >= 3


def test_test_windows_are_contiguous():
    """Blocks must stay contiguous in time, otherwise temporal autocorrelation leaks."""
    for f in nested_cv_folds(W, S, NB):
        w = f["test"][1]
        assert np.all(np.diff(w) == 1)


def test_selected_model_is_never_worse_than_the_floor_on_pure_noise():
    """With no signal at all, a correctly-tuned probe backs off to the intercept, so test R²
    must not fall meaningfully BELOW `r2_null`. This is the regression test for the selection
    overfitting that a single val cell caused."""
    emb, Y = _fake_data(seed=11, signal=0.0)
    out = nested_ridge_r2(emb, Y, n_blocks=NB)
    r2, null = np.nanmean(out["r2"]), np.nanmean(out["r2_null"])
    assert r2 >= null - 0.05, f"R²={r2:+.3f} sits below the floor {null:+.3f} → selection noise"


# ── statistical control ────────────────────────────────────────────────────────────
def _fake_data(seed=0, signal=0.0, drift=True):
    """(L,W,S,D) embedding and a (W,F) stimulus-locked target sharing `signal` variance.
    `drift=True` makes the target a random walk, like the real season-long features."""
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(W, F))
    if drift:
        Y = np.cumsum(Y, axis=0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    emb = rng.normal(size=(2, W, S, D))
    if signal:
        emb[:, :, :, :F] += signal * Y[None, :, None, :]    # plant a decodable component
    return emb, Y


def test_shuffled_target_lands_on_the_chance_floor():
    """With y shuffled along time, the embedding carries NO information about it. The mean
    test R² must sit at the intercept-only floor (which is negative under block CV), not
    above zero. A subject-only split would fail this: the same window's y would be in train."""
    emb, Y = _fake_data(seed=1, signal=1.5)
    Y_shuf = Y[np.random.default_rng(7).permutation(W)]
    out = nested_ridge_r2(emb, Y_shuf, n_blocks=NB)
    r2, null = np.nanmean(out["r2"]), np.nanmean(out["r2_null"])
    se = np.nanmean(out["r2_sd"]) / np.sqrt(out["r2_folds"].shape[0])
    assert r2 < 0.05, f"shuffled target still decodable (R²={r2:+.3f}) → leakage"
    assert r2 <= null + 2 * se + 0.05, f"R²={r2:+.3f} sits above the floor {null:+.3f}"


def test_planted_signal_is_recovered_stationary():
    """Sanity in the other direction: a genuinely decodable component must be recovered, so
    the shuffled-target test above is not passing merely because the probe is broken."""
    emb, Y = _fake_data(seed=2, signal=3.0, drift=False)
    out = nested_ridge_r2(emb, Y, n_blocks=NB)
    assert np.nanmean(out["r2"]) > 0.3, f"planted signal not recovered: {np.nanmean(out['r2']):+.3f}"


def test_planted_signal_beats_the_floor_when_the_target_drifts():
    """The finding that matters for reading the paper's x-axis: with a season-long DRIFTING
    target, even a perfectly encoded feature scores NEGATIVE in absolute R² — the test block
    lies outside the train range, so the train-fitted intercept is wrong there. The signal is
    still unmistakable relative to the intercept-only floor. R² must be read against
    `r2_null`, not against zero."""
    emb, Y = _fake_data(seed=2, signal=3.0, drift=True)
    out = nested_ridge_r2(emb, Y, n_blocks=NB)
    r2, null = np.nanmean(out["r2"]), np.nanmean(out["r2_null"])
    assert r2 > null + 1.0, f"planted signal not above its floor (R²={r2:+.3f}, null={null:+.3f})"
    assert r2 < 0.0, "expected the absolute R² of a drifting target to stay negative"


def test_null_floor_is_negative_under_block_cv():
    """Documents the diagnosis: the chance floor is not zero. The intercept comes from the
    train mean while R²'s denominator is the test block's variance about its OWN mean, so
    block-level drift is charged as error."""
    emb, Y = _fake_data(seed=3, signal=0.0)
    out = nested_ridge_r2(emb, Y, n_blocks=NB)
    assert np.nanmean(out["r2_null"]) < 0.0


def test_per_subject_targets_accepted():
    """Y may be (S, W, F) — the shape the eeg tier takes once per-subject NICE features
    are rebuilt (see the plan's Deferred section)."""
    emb, Y = _fake_data(seed=4, signal=2.0)
    Y3 = np.repeat(Y[None], S, axis=0) + 0.1 * np.random.default_rng(5).normal(size=(S, W, F))
    out = nested_ridge_r2(emb, Y3, n_blocks=NB)
    assert out["r2"].shape == (F,)
    assert np.isfinite(out["r2"]).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
