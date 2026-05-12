from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.config import ProcessSpec, ProjectConfig


def configure_proj_data() -> None:
    """Point GDAL/rasterio at a usable PROJ database in mixed Conda/pip envs."""
    rasterio_spec = importlib.util.find_spec("rasterio")
    if rasterio_spec and rasterio_spec.origin:
        candidate = Path(rasterio_spec.origin).resolve().parent / "proj_data"
        if (candidate / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(candidate)
            os.environ["PROJ_LIB"] = str(candidate)
            return
    try:
        import pyproj
    except ImportError:
        return
    proj_data = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = proj_data
    os.environ["PROJ_LIB"] = proj_data


configure_proj_data()
warnings.filterwarnings("ignore", message="Neither osr.UseExceptions.*", category=FutureWarning)


def add_cyclic_doy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        date = pd.to_datetime(out["date"])
        doy = date.dt.dayofyear.to_numpy()
    elif "doy" in out.columns:
        doy = pd.to_numeric(out["doy"], errors="raise").to_numpy()
    else:
        return out
    rad = 2.0 * np.pi * doy / 365.25
    out["doy_sin"] = np.sin(rad).astype(np.float32)
    out["doy_cos"] = np.cos(rad).astype(np.float32)
    return out


def normalize_event_frame(gdf: gpd.GeoDataFrame | pd.DataFrame, spec: ProcessSpec) -> pd.DataFrame:
    out = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")).copy()
    required = [spec.label_col, spec.basin_id_col, "date", "year"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"{spec.key}: missing required columns {missing}")
    out["process"] = spec.key
    out["basin_id"] = pd.to_numeric(out[spec.basin_id_col], errors="raise").astype(int)
    out["cat"] = out["basin_id"]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["year"] = pd.to_numeric(out["year"], errors="raise").astype(int)
    out[spec.label_col] = pd.to_numeric(out[spec.label_col], errors="raise").astype(int)
    out = add_cyclic_doy(out)
    return out


def load_events(config: ProjectConfig, spec: ProcessSpec, max_rows: int | None = None) -> pd.DataFrame:
    path = config.data_dir / spec.event_file
    if not path.exists():
        raise FileNotFoundError(f"{spec.key}: event file not found: {path}")
    rows = slice(0, max_rows) if max_rows else None
    return normalize_event_frame(gpd.read_file(path, rows=rows), spec)


def load_basins(config: ProjectConfig) -> gpd.GeoDataFrame:
    if not config.basins_file.exists():
        raise FileNotFoundError(f"Basins file not found: {config.basins_file}")
    basins = gpd.read_file(config.basins_file)
    if "cat" not in basins.columns:
        raise ValueError(f"{config.basins_file} must contain a 'cat' basin id column")
    basins["basin_id"] = pd.to_numeric(basins["cat"], errors="raise").astype(int)
    return basins


def bolzano_basins(config: ProjectConfig) -> gpd.GeoDataFrame:
    basins = load_basins(config)
    if "BZ" not in basins.columns:
        raise ValueError(f"{config.basins_file} must contain a 'BZ' Bolzano flag column")
    mask = pd.to_numeric(basins["BZ"], errors="coerce").fillna(0).astype(int) == 1
    return basins.loc[mask].copy()


def latest_cerra_date(config: ProjectConfig) -> pd.Timestamp:
    files = sorted(config.cerra_dir.glob("tp_cerra_*_alps_lonlat.nc"))
    if not files:
        raise FileNotFoundError(f"No CERRA files found under {config.cerra_dir}")
    latest_file = files[-1]
    with xr.open_dataset(latest_file) as ds:
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        if time_name not in ds.coords:
            raise KeyError(f"{latest_file} has no time/valid_time coordinate")
        return pd.to_datetime(ds[time_name].values).max().normalize()
