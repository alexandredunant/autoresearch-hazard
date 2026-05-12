"""Zonal statistics and lag-matrix generation from NetCDF precipitation data."""

from __future__ import annotations

import importlib.util
import os
import time
import logging
import warnings
from pathlib import Path


def configure_proj_data() -> None:
    """Use a PROJ database compatible with GDAL-backed imports."""
    rasterio_spec = importlib.util.find_spec("rasterio")
    if rasterio_spec and rasterio_spec.origin:
        rasterio_proj = Path(rasterio_spec.origin).resolve().parent / "proj_data"
        if (rasterio_proj / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(rasterio_proj)
            os.environ["PROJ_LIB"] = str(rasterio_proj)
            return
    try:
        import pyproj
    except ImportError:
        return
    proj_data = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = proj_data
    os.environ["PROJ_LIB"] = proj_data


configure_proj_data()
warnings.filterwarnings(
    "ignore",
    message="Neither osr.UseExceptions.*",
    category=FutureWarning,
)

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from exactextract import exact_extract
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from utils_data import CERRA_DIR

log = logging.getLogger(__name__)

# Year-level parallelism for extract_zonal_precipitation. 1 = serial fallback.
EXTRACT_N_WORKERS = int(os.environ.get("EXTRACT_N_WORKERS", "-1"))

def find_cerra_file(year: int) -> Path:
    """Return the yearly CERRA file from the current EUSALP reprojection layout."""
    path = CERRA_DIR / f"tp_cerra_{year}_alps_lonlat.nc"
    if path.exists():
        return path
    raise FileNotFoundError(f"CERRA EUSALP reprojection file missing for {year}: {path}")


def load_cerra_year(year: int) -> xr.DataArray:
    """Open a single CERRA year file eagerly (no dask)."""
    path = find_cerra_file(year)
    ds = xr.open_dataset(path)
    variable = "tp" if "tp" in ds.data_vars else "P_H"
    if variable not in ds.data_vars:
        raise KeyError(f"Could not find precipitation variable in {path}. Available: {list(ds.data_vars)}")
    da = ds[variable]
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    # Drop any singleton non-spatial/time dims (e.g. height, pressure) that some
    # files include — exact_extract requires per-band 2D arrays.
    extra_dims = [d for d in da.dims if d not in ("time", "latitude", "longitude")]
    if extra_dims:
        da = da.squeeze(extra_dims, drop=True)
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    return da

def _exact_extract_wide_to_long(wide: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Convert exact_extract band columns back to one row per basin-date."""
    band_cols = [c for c in wide.columns if c != "basin_id" and "mean" in c]
    if len(band_cols) != len(dates):
        raise ValueError(
            f"Expected {len(dates)} exact_extract mean columns, got {len(band_cols)}: {band_cols[:5]}"
        )

    date_by_band = dict(zip(band_cols, dates))
    long = wide.melt(
        id_vars=["basin_id"],
        value_vars=band_cols,
        var_name="band",
        value_name="P_H_mm",
    )
    long["date"] = long["band"].map(date_by_band)
    precip_df = long[["basin_id", "date", "P_H_mm"]].copy()
    precip_df["basin_id"] = precip_df["basin_id"].astype(int)
    precip_df["date"] = pd.to_datetime(precip_df["date"]).dt.normalize()
    return precip_df

def _extract_one_year(
    year: int, dates: list[pd.Timestamp], basin_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Open one CERRA year file and run exact_extract over all required dates."""
    da = load_cerra_year(year)
    subset = da.sel(time=dates, method="nearest").load()
    # exact_extract requires 2D rasters; wrap each timestep as a separate
    # variable in a Dataset (zero-padded names keep band order = date order).
    # `reset_coords("time", drop=True)` avoids a MergeError from conflicting
    # scalar time coords across slices.
    subset_ds = xr.Dataset({
        f"b{i:04d}": subset.isel(time=i).reset_coords("time", drop=True)
        for i in range(subset.sizes["time"])
    })
    wide = exact_extract(
        subset_ds, basin_gdf, ["mean"], output="pandas", include_cols=["basin_id"]
    )
    da.close()
    return _exact_extract_wide_to_long(wide, dates)


def extract_zonal_precipitation(
    basin_gdf: gpd.GeoDataFrame,
    required_dates: list[pd.Timestamp],
    *,
    n_workers: int = EXTRACT_N_WORKERS,
) -> pd.DataFrame:
    """Extract area-weighted mean precipitation for given basins and dates.

    Years are independent (each opens its own NetCDF, runs its own exact_extract)
    and are dispatched in parallel via joblib. Set `n_workers=1` for serial.
    """
    required_dates = sorted(pd.to_datetime(required_dates).normalize().unique())
    log.info(
        "Extracting zonal mean for %s basins x %s dates (%s values), n_workers=%s",
        len(basin_gdf), len(required_dates),
        len(basin_gdf) * len(required_dates), n_workers,
    )

    dates_series = pd.Series(required_dates)
    by_year = [
        (int(year), list(group))
        for year, group in dates_series.groupby(dates_series.dt.year)
    ]

    total_t0 = time.time()
    with tqdm_joblib(tqdm(total=len(by_year), desc="Years done")):
        frames = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_extract_one_year)(year, dates, basin_gdf)
            for year, dates in by_year
        )

    precip_df = pd.concat(frames, ignore_index=True)
    log.info("Zonal stats complete in %.1fs", time.time() - total_t0)
    return precip_df

def build_lag_matrix(
    df: pd.DataFrame,
    precip_df: pd.DataFrame,
    max_lag: int,
) -> pd.DataFrame:
    """Build a matrix of lag_0..lag_N columns for each row in df (vectorized)."""
    out = df.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["basin_id"] = pd.to_numeric(out["basin_id"], errors="raise").astype(int)

    precip = precip_df.copy()
    precip["basin_id"] = precip["basin_id"].astype(int)
    precip["date"] = pd.to_datetime(precip["date"]).dt.normalize()
    lookup = precip.set_index(["basin_id", "date"])["P_H_mm"]

    lag_cols = {}
    for k in tqdm(range(max_lag + 1), desc="Building lag windows"):
        target_dates = out["date"] - pd.Timedelta(days=k)
        idx = pd.MultiIndex.from_arrays([out["basin_id"].to_numpy(), target_dates])
        lag_cols[f"lag_{k}"] = lookup.reindex(idx).to_numpy()

    lag_df = pd.DataFrame(lag_cols, index=out.index)
    if lag_df.isna().any().any():
        log.warning("Some lag values are NaN - missing precipitation data in lookup table.")

    return pd.concat([out, lag_df], axis=1)
