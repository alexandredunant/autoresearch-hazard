"""Process configs, path constants, and event-frame loading."""

from __future__ import annotations

import importlib.util
import json
import os
import warnings
from dataclasses import dataclass
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

import geopandas as gpd
import numpy as np
import pandas as pd


# ---- Path constants ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
GEOMETRY_DIR = DATA_DIR
DATASET_DIR = SCRIPT_DIR / "output" / "model_arrays"
EVAL_DIR = SCRIPT_DIR / "output" / "eval"
OUTPUT_DIR = SCRIPT_DIR / "output"
CERRA_DIR = SCRIPT_DIR / "CERRA_LAND_EUSALP_reproj"
BASINS_GEOJSON_FILE = "basins_with_BZ.geojson"


# ---- Process configuration --------------------------------------------------
@dataclass(frozen=True)
class ProcessConfig:
    key: str
    name: str
    geojson_file: str
    label_col: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


PROCESS_CONFIGS: list[ProcessConfig] = [
    ProcessConfig(
        key="slides", name="Slide-type",
        geojson_file="df_slides.geojson", label_col="SL01",
        numeric_features=(
            "SL_Slope", "doy_sin", "doy_cos",
        ),
        categorical_features=("SL_dominant_landcover", "SL_dominant_lithology"),
    ),
    ProcessConfig(
        key="flows", name="Flow-type",
        geojson_file="df_flows.geojson", label_col="DF01",
        numeric_features=(
            "DF_Slope", "DF_CI_mean", "doy_sin", "doy_cos",
        ),
        categorical_features=("DF_dominant_landcover", "DF_dominant_lithology"),
    ),
]

PROCESS_KEYS = [cfg.key for cfg in PROCESS_CONFIGS]
PROCESS_ORDER = [cfg.name for cfg in PROCESS_CONFIGS]
PROCESS_COLORS = {"Slide-type": "#56b899", "Flow-type": "#9052b6"}

# ---- Loaders & frame normalisation -----------------------------------------
def load_any_data(path: Path, config: ProcessConfig | None = None) -> pd.DataFrame:
    """Load tabular data from GeoJSON, Parquet, or CSV. Normalize if config provided."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    p = str(path)
    if p.endswith(".geojson"):
        df = gpd.read_file(path)
    elif p.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif p.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path}")
    
    if config:
        return normalize_event_frame(df, config)
    return df


def add_cyclic_doy(df: pd.DataFrame) -> pd.DataFrame:
    """Encode day-of-year as (sin, cos) so the model sees December and January as adjacent."""
    if "doy" not in df.columns:
        return df
    out = df.copy()
    rad = 2.0 * np.pi * pd.to_numeric(out["doy"], errors="raise") / 365.25
    out["doy_sin"] = np.sin(rad)
    out["doy_cos"] = np.cos(rad)
    return out


def normalize_event_frame(df: pd.DataFrame, config: ProcessConfig) -> pd.DataFrame:
    """Cast cat/year/label to int and date to midnight so joins line up exactly."""
    out = df.copy()
    if "geometry" in out.columns:
        out = pd.DataFrame(out.drop(columns="geometry"))
    
    for col in [config.label_col, "cat", "year"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    
    out["basin_id"] = out["cat"]
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    
    return add_cyclic_doy(out)


def read_event_geojson(config: ProcessConfig, *, max_rows: int | None = None) -> pd.DataFrame:
    """Read and normalize a process GeoJSON file."""
    path = DATA_DIR / config.geojson_file
    df = gpd.read_file(path, rows=max_rows) if max_rows else gpd.read_file(path)
    return normalize_event_frame(df, config)


def read_tabular(config: ProcessConfig, *, max_rows: int | None = None) -> pd.DataFrame:
    """tabular.csv is the per-process feature table written by 00_extract_model_arrays.py."""
    path = DATASET_DIR / config.key / "tabular.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    if max_rows:
        df = df.head(max_rows).copy()
    return normalize_event_frame(df, config)


def load_bolzano_basin_ids() -> set[int]:
    """basins_with_BZ.geojson tags each basin with BZ=1 if it lies in Bolzano."""
    gdf = gpd.read_file(GEOMETRY_DIR / BASINS_GEOJSON_FILE)
    gdf["cat"] = pd.to_numeric(gdf["cat"], errors="raise").astype(int)
    bz = pd.to_numeric(gdf["BZ"], errors="coerce").fillna(0).astype(int) == 1
    return set(gdf.loc[bz, "cat"].astype(int))


def build_feature_frame(
    df: pd.DataFrame, config: ProcessConfig
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Returns the feature matrix (numeric + categorical-as-strings), label vector, and feature names."""
    numeric = list(config.numeric_features)
    categorical = list(config.categorical_features)
    
    feature_names = numeric + categorical
    X = df[feature_names].copy()
    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="raise")
    for col in categorical:
        X[col] = X[col].fillna("__MISSING__").astype(str)
    
    y = pd.to_numeric(df[config.label_col], errors="raise").astype(int).to_numpy()
    return X, y, feature_names


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
