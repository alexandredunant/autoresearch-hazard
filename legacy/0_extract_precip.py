#!/home/adunant/miniconda3/envs/hazard_agent/bin/python
"""
0 — Extract daily basin precipitation lags for every labelled event.

For each event in df_slides/df_flows/df_falls, walks back MAX_LAG days and pulls
daily basin-mean precipitation from CERRA-Land via exact_extract zonal stats.
Events whose lookup returns NaN (basin outside the CERRA finite mask) are
dropped LOUDLY (count + basin list logged, list also written to disk).

Per-process outputs under output/model_arrays/<process>/:
  lag.npy                (n, MAX_LAG + 1) float32, oldest-first [lag_60..lag_0]
  events.csv             cat, basin_id, date, year, label + raw static predictors
  dropped_precip_nan.csv events lost to NaN precipitation (diagnostic)

Plus output/model_arrays/extract_summary.csv (one row per process).
"""

from __future__ import annotations

import logging
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_data import (
    BASINS_GEOJSON_FILE, DATASET_DIR, GEOMETRY_DIR,
    PROCESS_CONFIGS, ProcessConfig,
    read_event_geojson,
)
from utils_extraction import build_lag_matrix, extract_zonal_precipitation

MAX_LAG = 60
MAX_ROWS = int(os.environ["EXTRACT_MAX_ROWS"]) if os.environ.get("EXTRACT_MAX_ROWS") else None
LAG_COLS = [f"lag_{k}" for k in range(MAX_LAG + 1)]
LAG_COLS_OLDEST_FIRST = [f"lag_{k}" for k in reversed(range(MAX_LAG + 1))]

log = logging.getLogger(__name__)


def attach_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Join CERRA-recomputed annual climatology by basin_id.

    Replaces the original ``tp_su_mean_annual_mean`` carried in the GeoJSON
    with ``tp_su_mean_annual_mean_cerra`` (cached by 0_extract_basin_climatology.py).
    Fails loud if any event basin lacks a climatology row.
    """
    clim_path = DATASET_DIR / "basin_climatology.csv"
    if not clim_path.exists():
        raise FileNotFoundError(
            f"Missing {clim_path}; run 0_extract_basin_climatology.py first"
        )
    clim = pd.read_csv(clim_path)[["basin_id", "tp_su_mean_annual_mean_cerra"]]
    clim["basin_id"] = clim["basin_id"].astype(int)
    out = df.merge(clim, on="basin_id", how="left")
    missing = out["tp_su_mean_annual_mean_cerra"].isna()
    if missing.any():
        bad = sorted(out.loc[missing, "basin_id"].unique().tolist())
        raise ValueError(
            f"{int(missing.sum())} events have no climatology "
            f"({len(bad)} basins, first 10: {bad[:10]})"
        )
    return out


def required_dates_and_basins(
    frames: list[tuple[ProcessConfig, pd.DataFrame]],
) -> tuple[list[pd.Timestamp], set[int]]:
    dates: set = set()
    basins: set = set()
    for _, df in frames:
        basins.update(pd.to_numeric(df["basin_id"], errors="raise").astype(int).unique())
        for dt in pd.to_datetime(df["date"]).dt.normalize().unique():
            for lag in range(MAX_LAG + 1):
                dates.add(dt - pd.Timedelta(days=lag))
    return sorted(dates), basins


def load_basins(required: set[int]) -> gpd.GeoDataFrame:
    basins = gpd.read_file(GEOMETRY_DIR / BASINS_GEOJSON_FILE)
    basins["basin_id"] = pd.to_numeric(basins["cat"], errors="raise").astype(int)
    basins = basins.loc[basins["basin_id"].isin(required)].copy()
    missing = sorted(required.difference(set(basins["basin_id"])))
    if missing:
        raise ValueError(
            f"{len(missing)} event basin IDs missing from {BASINS_GEOJSON_FILE}: {missing[:10]}"
        )
    return basins


def save_process(config: ProcessConfig, samples: pd.DataFrame) -> dict:
    process_dir = DATASET_DIR / config.key
    process_dir.mkdir(parents=True, exist_ok=True)
    n_before = len(samples)

    nan_mask = samples[LAG_COLS].isna().any(axis=1)
    dropped = samples.loc[nan_mask].copy()
    dropped[["cat", "basin_id", "date", "year", config.label_col]].to_csv(
        process_dir / "dropped_precip_nan.csv", index=False,
    )
    samples = samples.loc[~nan_mask].reset_index(drop=True)
    if len(dropped):
        log.warning(
            "[%s] dropped %s/%s rows with NaN precipitation (%s basins outside CERRA mask)",
            config.key, len(dropped), n_before, dropped["basin_id"].nunique(),
        )

    lag = samples[LAG_COLS_OLDEST_FIRST].to_numpy(dtype=np.float32)
    if not np.isfinite(lag).all():
        raise ValueError(f"[{config.key}] lag matrix has non-finite values after NaN drop")
    np.save(process_dir / "lag.npy", lag)

    # tp_su_mean_annual_mean_cerra is kept in events.csv as a normalizer (denominator
    # for cum_norm_k / max_norm_k in 1_build_features.py), not as a model feature.
    static_cols = list(config.numeric_features) + list(config.categorical_features)
    events = samples[[
        "cat", "basin_id", "date", "year", config.label_col,
        "tp_su_mean_annual_mean_cerra", *static_cols,
    ]].copy()
    for col in (config.label_col, "cat", "year"):
        events[col] = pd.to_numeric(events[col], errors="raise").astype(int)
    events.to_csv(process_dir / "events.csv", index=False)

    return {
        "process": config.key, "rows": len(samples),
        "positives": int(samples[config.label_col].sum()),
        "dropped_nan_rows": int(len(dropped)),
        "dropped_nan_basins": int(dropped["basin_id"].nunique()) if len(dropped) else 0,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.time()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    frames = [
        (cfg, attach_climatology(read_event_geojson(cfg, max_rows=MAX_ROWS)))
        for cfg in tqdm(PROCESS_CONFIGS, desc="Reading events")
    ]

    dates, basin_ids = required_dates_and_basins(frames)
    log.info(
        "Need %s basin IDs x %s lag dates (%s to %s)",
        len(basin_ids), len(dates), dates[0].date(), dates[-1].date(),
    )

    basins = load_basins(basin_ids)
    log.info("Loaded %s basin polygons", len(basins))

    precip_df = extract_zonal_precipitation(basins, dates)
    log.info("Extracted %s basin-date precipitation rows", len(precip_df))

    summary = [
        save_process(cfg, build_lag_matrix(df, precip_df, MAX_LAG))
        for cfg, df in tqdm(frames, desc="Building lag matrices")
    ]
    pd.DataFrame(summary).to_csv(DATASET_DIR / "extract_summary.csv", index=False)
    log.info("Done in %.1fs -> %s", time.time() - t0, DATASET_DIR / "extract_summary.csv")


if __name__ == "__main__":
    main()
