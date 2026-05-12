from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import ProjectConfig
from utils_extraction import extract_zonal_precipitation, build_lag_matrix


def lag_columns(max_lag: int) -> list[str]:
    return [f"lag_{i}" for i in range(max_lag + 1)]


def required_max_lag(feature_names: list[str], default: int = 0) -> int:
    max_lag = default
    for name in feature_names:
        match = re.search(r"_(\d+)$", name)
        if match:
            max_lag = max(max_lag, int(match.group(1)))
    return max_lag


def load_existing_lags(process_key: str, *, source_dir: Path, max_lag: int) -> np.ndarray | None:
    path = source_dir / process_key / "lag.npy"
    if not path.exists():
        return None
    lag = np.load(path).astype(np.float32)
    if lag.shape[1] < max_lag + 1:
        raise ValueError(f"{path} has {lag.shape[1]} lag columns; need at least {max_lag + 1}")
    # Old extraction stores oldest-first [lag_N..lag_0]; the feature layer uses recent-first.
    return lag[:, ::-1][:, : max_lag + 1]


def extract_prediction_lags(
    config: ProjectConfig,
    basins: gpd.GeoDataFrame,
    *,
    prediction_date: pd.Timestamp,
    max_lag: int,
) -> np.ndarray:
    dates = [pd.Timestamp(prediction_date) - pd.Timedelta(days=i) for i in range(max_lag + 1)]
    work = basins.copy()
    work["basin_id"] = pd.to_numeric(work["basin_id"], errors="raise").astype(int)
    precip = extract_zonal_precipitation(work, dates)
    rows = pd.DataFrame({
        "basin_id": work["basin_id"].astype(int).to_numpy(),
        "date": pd.Timestamp(prediction_date),
    })
    with_lags = build_lag_matrix(rows, precip, max_lag)
    lag = with_lags[lag_columns(max_lag)].to_numpy(dtype=np.float32)
    if np.isnan(lag).any():
        raise ValueError("Prediction lag extraction produced NaN values")
    return lag
