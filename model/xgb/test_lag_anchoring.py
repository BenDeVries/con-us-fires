"""Origin-anchoring test for the xgb design: no feature may see past the forecast origin.

The space-time expansion is only admissible in the bake-off because every lag it adds is
indexed from the origin `o`, never from the target month `m = o + h`. A target-anchored
lag `t{k}` with `k < h` sits *after* the origin and is not available at forecast time --
the mistake is easy to make because `model/starima.py`, whose lag layout this block
mirrors, is a 1-step nowcast and does index from the target.

Two controls, in the style of `model/diagnostics.py`:

  null      corrupt the panel strictly after the origin -> every feature column must be
            bitwise unchanged, and only the target `y` may move.
  positive  corrupt the panel at the origin month itself -> the history columns must move,
            proving the null control is sensitive rather than vacuously passing.

Runs on the test split at its earliest origin (2018-05, the last train month), so the
train-span baselines `county_mean_y` / `county_occ_rate` are outside the null's corruption
window and stay clean.

    conda run -n fire-xgb python -m model.xgb.test_lag_anchoring
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import features as F

CUTOFF = pd.Timestamp("2018-05-01")      # last train month; the earliest test origin
SPEC = dict(lookback=48, horizon=12, max_origins=1,
            max_spatial=3, max_temporal=47, n_nbr_pcs=4)


def _build(corrupt_from: pd.Timestamp | None, inclusive: bool):
    """Assemble the test split, optionally corrupting fire history from a month onward."""
    orig = F._load_full_panel

    def patched():
        full, ranges = orig()
        if corrupt_from is not None:
            sel = (full["date"] >= corrupt_from) if inclusive else (full["date"] > corrupt_from)
            full.loc[sel, F.TARGET_COL] = 0.5
            full.loc[sel, F.GATE_COL] = 1.0
        return full, ranges

    F._load_full_panel = patched
    try:
        return F.build_dataset(want_splits=("test",), **SPEC).splits["test"]
    finally:
        F._load_full_panel = orig


def main() -> int:
    clean = _build(None, False)
    print(f"design: {clean['X'].shape[1]} columns, {len(clean['y']):,} rows "
          f"(origin {clean['meta'].origin_date.iloc[0].date()})")

    failures = []

    # --- null control: nothing after the origin may reach a feature -------------
    after = _build(CUTOFF, inclusive=False)
    if not (after["y"] != clean["y"]).any():
        failures.append("null control never landed: corrupting months > origin left y unchanged")
    diff = [c for c in clean["X"].columns
            if not clean["X"][c].equals(after["X"][c])]
    if diff:
        failures.append(f"{len(diff)} feature(s) depend on months after the origin: "
                        f"{diff[:12]}{' ...' if len(diff) > 12 else ''}")

    # --- positive control: the same columns must react to history at the origin --
    at = _build(CUTOFF, inclusive=True)
    moved = {c for c in clean["X"].columns if not clean["X"][c].equals(at["X"][c])}
    # Only lag-0 cells may react: the corruption starts at the origin month itself.
    expect = {"y_o", "occ_o", "y_roll3", "nb_y_roll12",
              "st_y_s1_t0", "st_occ_s1_t0", "st_y_s3_t0"}
    stale = sorted(c for c in moved if (m := F._ST_RE.match(c)) and int(m.group(2)) > 0)
    if stale:
        failures.append(f"lag>0 columns moved when only the origin month changed: {stale[:12]}")
    missing = sorted(expect - moved)
    if missing:
        failures.append(f"positive control is insensitive; expected these to move when the "
                        f"origin month is corrupted: {missing}")
    else:
        print(f"positive control: {len(moved)} columns move when month {CUTOFF.date()} changes")

    for f in failures:
        print(f"FAIL: {f}")
    if failures:
        return 1
    print(f"PASS: all {clean['X'].shape[1]} columns are origin-anchored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
