#!/usr/bin/env python3
"""Prepare basin-level feature matrices for the autoresearch loop.

Default mode is intentionally fast and reproducible: it uses the static/event
features already present in the process GeoJSONs plus any existing lag matrices
under output/model_arrays/<process>/. Pass --extract-cerra to rebuild lag
matrices directly from the CERRA NetCDF files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, parse_processes
from src.data import load_basins, load_events, normalize_event_frame
from src.features import build_feature_matrix, save_artifact
from src.precip import load_existing_lags
from utils_extraction import build_lag_matrix, extract_zonal_precipitation

log = logging.getLogger("prepare")


def sample_positions(events: pd.DataFrame, label_col: str, max_rows: int | None, seed: int) -> np.ndarray:
    if not max_rows or len(events) <= max_rows:
        return np.arange(len(events))
    y = pd.to_numeric(events[label_col], errors="raise").astype(int).to_numpy()
    if len(np.unique(y)) == 2 and np.bincount(y).min() >= 2:
        pos, _ = train_test_split(
            np.arange(len(events)),
            train_size=max_rows,
            stratify=y,
            random_state=seed,
        )
        return np.sort(pos)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(events), size=max_rows, replace=False))


def existing_events_and_lags(
    spec,
    *,
    source_dir: Path,
    max_lag: int,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    events_path = source_dir / spec.key / "events.csv"
    lag = load_existing_lags(spec.key, source_dir=source_dir, max_lag=max_lag)
    if lag is None or not events_path.exists():
        return None
    events = pd.read_csv(events_path, parse_dates=["date"])
    events = normalize_event_frame(events, spec)
    if len(events) != len(lag):
        raise ValueError(f"{spec.key}: {events_path} rows do not match lag.npy rows")
    return events, lag


def extract_lags_for_events(
    config,
    events: pd.DataFrame,
    *,
    max_lag: int,
) -> np.ndarray:
    basins = load_basins(config)
    needed = set(events["basin_id"].astype(int).unique())
    basins = basins.loc[basins["basin_id"].isin(needed)].copy()
    missing = needed.difference(set(basins["basin_id"].astype(int)))
    if missing:
        raise ValueError(f"Missing basin geometries for ids: {sorted(missing)[:10]}")

    dates: set[pd.Timestamp] = set()
    for date in pd.to_datetime(events["date"]).dt.normalize().unique():
        for lag in range(max_lag + 1):
            dates.add(pd.Timestamp(date) - pd.Timedelta(days=lag))
    precip = extract_zonal_precipitation(basins, sorted(dates))
    with_lags = build_lag_matrix(events, precip, max_lag)
    lag_cols = [f"lag_{i}" for i in range(max_lag + 1)]
    lag = with_lags[lag_cols].to_numpy(dtype=np.float32)
    if np.isnan(lag).any():
        raise ValueError("CERRA extraction produced NaN lag values")
    return lag


def prepare_one(config, spec, args) -> dict:
    source_dir = Path(args.existing_lag_dir)
    lag_recent: np.ndarray | None = None
    existing = None if args.ignore_existing_lags else existing_events_and_lags(
        spec, source_dir=source_dir, max_lag=args.max_lag
    )
    if existing is not None:
        events, lag_recent = existing
        source = "existing_lags"
        pos = sample_positions(events, spec.label_col, args.max_rows, config.random_state)
        events = events.iloc[pos].reset_index(drop=True)
        lag_recent = lag_recent[pos]
    else:
        events = load_events(config, spec)
        pos = sample_positions(events, spec.label_col, args.max_rows, config.random_state)
        events = events.iloc[pos].reset_index(drop=True)
        source = "geojson_legacy"
        if args.extract_cerra:
            lag_recent = extract_lags_for_events(config, events, max_lag=args.max_lag)
            source = "cerra_extracted"

    X, y, metadata, feature_names, schema = build_feature_matrix(
        events, spec, lag_recent=lag_recent, max_lag=args.max_lag
    )
    schema["source"] = source
    schema["rows"] = int(len(y))
    schema["positives"] = int(y.sum())
    out_dir = Path(args.out) / spec.key
    save_artifact(out_dir, X=X, y=y, metadata=metadata, feature_names=feature_names, schema=schema)
    return {
        "process": spec.key,
        "source": source,
        "rows": int(X.shape[0]),
        "positives": int(y.sum()),
        "n_features": int(X.shape[1]),
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/processes.yml")
    parser.add_argument("--process", default=None, help="Comma-separated process keys or 'all'.")
    parser.add_argument("--out", default="artifacts/features")
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test row cap per process.")
    parser.add_argument("--extract-cerra", action="store_true", help="Rebuild lag matrices from CERRA NetCDF files.")
    parser.add_argument("--ignore-existing-lags", action="store_true")
    parser.add_argument("--existing-lag-dir", default="output/model_arrays")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(args.config)
    args.max_lag = config.max_lag if args.max_lag is None else args.max_lag
    Path(args.out).mkdir(parents=True, exist_ok=True)

    summary = [prepare_one(config, spec, args) for spec in parse_processes(config, args.process)]
    summary_path = Path(args.out) / "summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
