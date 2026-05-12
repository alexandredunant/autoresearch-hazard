#!/home/adunant/miniconda3/envs/hazard_agent/bin/python
"""
1 — Build the model-ready feature matrix for each process.

Inputs (per process, written by 0_extract_precip.py):
  output/model_arrays/<process>/lag.npy     (n, MAX_LAG + 1) float32, oldest-first
  output/model_arrays/<process>/events.csv  identifiers + raw static predictors

Features (all swept k = 1..MAX_WINDOW so the EBM picks the right horizon):
  cum_k      = sum(lag_0..lag_{k-1})                                  k = 1..60
  max_k      = max(lag_0..lag_{k-1})                                  k = 1..60
  cum_norm_k = cum_k / tp_su_mean_annual_mean_cerra                   k = 1..60
  max_norm_k = max_k / tp_su_mean_annual_mean_cerra                   k = 1..60
  slope_k    = LS slope of daily precip over last k days,             k = 2..60
               time runs oldest -> event so positive = intensifying
  + static numerics from ProcessConfig (tp_su_mean_annual_mean_cerra is
    used only as the normalizer for cum_norm_k / max_norm_k, never as a
    standalone feature, since it is constant per basin and contributes
    no temporal discrimination relevant to early warning)
  + one-hot categoricals from ProcessConfig (loud failure if levels disagree)

Outputs (per process):
  X.npy             (n, n_features) float32
  y.npy             (n,) int64
  feature_names.csv single column "name", one row per X column
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from utils_data import DATASET_DIR, PROCESS_CONFIGS, ProcessConfig

MAX_WINDOW = 60

# Per-process top-3 lags per precipitation sub-family, picked from the
# paper_basin_cv importance ranking (see 2d_importance_family.py output).
# `cum_*` and `max_*` are intentionally excluded — `cum_norm_*` and `max_norm_*`
# carry the same information normalised by basin climatology and consistently
# outrank their unnormalised counterparts.
SELECTED_LAGS: dict[str, dict[str, list[int]]] = {
    p: {
        "cum_norm": [1, 7, 15, 30],         # event-day + weekly / fortnightly / monthly wetness
        "max_norm": [7, 15, 30],            # peak single day over each window (max_norm_1 == cum_norm_1)
        "slope":    [2, 3, 4, 5, 6, 7],     # daily rate-of-change over the past week
    }
    for p in ("slides", "flows", "falls")
}
log = logging.getLogger(__name__)


def cumulative_windows(seq: np.ndarray) -> np.ndarray:
    return np.cumsum(seq[:, :MAX_WINDOW], axis=1).astype(np.float32)


def max_windows(seq: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(seq[:, :MAX_WINDOW], axis=1).astype(np.float32)


def slope_windows(seq: np.ndarray) -> np.ndarray:
    n = seq.shape[0]
    out = np.empty((n, MAX_WINDOW - 1), dtype=np.float32)
    for k in range(2, MAX_WINDOW + 1):
        y = seq[:, :k][:, ::-1]
        x = np.arange(k, dtype=np.float32)
        x_c = x - x.mean()
        y_c = y - y.mean(axis=1, keepdims=True)
        out[:, k - 2] = ((y_c * x_c).sum(axis=1) / float((x_c ** 2).sum())).astype(np.float32)
    return out


def cum_norm_windows(cum: np.ndarray, map_mm: np.ndarray) -> np.ndarray:
    return (cum / map_mm[:, None]).astype(np.float32)


def max_norm_windows(mx: np.ndarray, map_mm: np.ndarray) -> np.ndarray:
    return (mx / map_mm[:, None]).astype(np.float32)


def build_one(config: ProcessConfig) -> dict:
    process_dir = DATASET_DIR / config.key
    seq = np.load(process_dir / "lag.npy")[:, ::-1].astype(np.float32)  # recent-first
    events = pd.read_csv(process_dir / "events.csv", parse_dates=["date"])
    if len(events) != len(seq):
        raise ValueError(f"[{config.key}] events.csv ({len(events)}) != lag.npy ({len(seq)})")

    # Derive cyclic day-of-year encoding if missing — events.csv is read raw,
    # so add_cyclic_doy() is not called. Compute from `doy` or fall back to date.
    if "doy_sin" not in events.columns or "doy_cos" not in events.columns:
        if "doy" in events.columns:
            doy = pd.to_numeric(events["doy"], errors="raise").to_numpy()
        else:
            doy = pd.to_datetime(events["date"]).dt.dayofyear.to_numpy()
        rad = 2.0 * np.pi * doy / 365.25
        events["doy_sin"] = np.sin(rad).astype(np.float32)
        events["doy_cos"] = np.cos(rad).astype(np.float32)

    cum = cumulative_windows(seq)
    mx = max_windows(seq)
    sl = slope_windows(seq)

    # Normalize cumulative and max windows by basin climatology so the model
    # sees "fraction of mean annual precipitation" rather than absolute mm.
    # MAP itself is not used as a feature (constant per basin).
    map_mm = pd.to_numeric(
        events["tp_su_mean_annual_mean_cerra"], errors="raise",
    ).astype(np.float32).to_numpy()
    if not np.isfinite(map_mm).all() or (map_mm <= 0).any():
        raise ValueError(
            f"[{config.key}] tp_su_mean_annual_mean_cerra has non-finite or "
            f"non-positive values; cannot normalize"
        )
    cum_norm = cum_norm_windows(cum, map_mm)
    max_norm = max_norm_windows(mx, map_mm)

    if config.key not in SELECTED_LAGS:
        raise KeyError(f"[{config.key}] missing entry in SELECTED_LAGS")
    sel = SELECTED_LAGS[config.key]
    cum_norm_lags = sorted(sel["cum_norm"])
    max_norm_lags = sorted(sel["max_norm"])
    slope_lags = sorted(sel["slope"])

    cum_norm_sub = cum_norm[:, [k - 1 for k in cum_norm_lags]]
    max_norm_sub = max_norm[:, [k - 1 for k in max_norm_lags]]
    slope_sub = sl[:, [k - 2 for k in slope_lags]]
    cum_norm_names = [f"cum_norm_{k}" for k in cum_norm_lags]
    max_norm_names = [f"max_norm_{k}" for k in max_norm_lags]
    slope_names = [f"slope_{k}" for k in slope_lags]

    base_numeric = list(config.numeric_features)
    numeric_arr = events[base_numeric].apply(pd.to_numeric, errors="raise").astype(np.float32).to_numpy()

    cat_cols = list(config.categorical_features)
    cat_df = events[cat_cols].fillna("__MISSING__").astype(str)
    encoder = OneHotEncoder(
        sparse_output=False,
        dtype=np.float32,
        handle_unknown="error",
        drop="if_binary",
        feature_name_combiner=lambda col, lvl: f"{col}__{lvl}",
    ).fit(cat_df)
    cat_arr = encoder.transform(cat_df)
    expected = encoder.get_feature_names_out(cat_cols).tolist()

    X = np.concatenate(
        [cum_norm_sub, max_norm_sub, slope_sub, numeric_arr, cat_arr], axis=1,
    )

    if not np.isfinite(X).all():
        raise ValueError(f"[{config.key}] X contains non-finite values")

    feature_names = (
        cum_norm_names + max_norm_names + slope_names + base_numeric + expected
    )

    y = pd.to_numeric(events[config.label_col], errors="raise").astype(np.int64).to_numpy() # binary label per event

    np.save(process_dir / "X.npy", X)
    np.save(process_dir / "y.npy", y)
    pd.DataFrame({"name": feature_names}).to_csv(process_dir / "feature_names.csv", index=False)

    return {
        "process": config.key, "rows": X.shape[0], "n_features": X.shape[1],
        "n_cum_norm": len(cum_norm_names), "n_max_norm": len(max_norm_names),
        "n_slope": len(slope_names),
        "n_numeric": len(base_numeric), "n_onehot": len(expected),
        "positives": int(y.sum()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.time()
    summary = [build_one(cfg) for cfg in tqdm(PROCESS_CONFIGS, desc="Building features")]
    pd.DataFrame(summary).to_csv(DATASET_DIR / "features_summary.csv", index=False)
    log.info("Done in %.1fs -> %s", time.time() - t0, DATASET_DIR / "features_summary.csv")


if __name__ == "__main__":
    main()
