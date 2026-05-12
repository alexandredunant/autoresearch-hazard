#!/home/adunant/miniconda3/envs/hazard_agent/bin/python
"""
0 — CERRA annual precipitation climatology per basin (cached).

For every basin in basins_with_BZ.geojson, computes the multi-year mean of
ANNUAL precipitation totals (mm/year) from CERRA-Land, over the years covered
by labelled events (union of df_slides/df_flows/df_falls). This produces a
climatology that is independent of which basins ever had events, so the same
value can be used at prediction time for basins outside the training set.

Method:
    annual_total[year, basin] = exact_extract( da.sum(time), basins )["mean"]
    climatology[basin]        = mean over years of annual_total[year, basin]

Outputs:
    output/model_arrays/basin_climatology.csv
        cat, basin_id, tp_su_mean_annual_mean_cerra, n_years
    output/model_arrays/basin_climatology_comparison.csv
        per-basin diff vs. the original tp_su_mean_annual_mean carried in
        the event GeoJSONs (audit trail; not consumed downstream)
"""

from __future__ import annotations

import logging
import time

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from utils_data import (
    BASINS_GEOJSON_FILE, DATASET_DIR, GEOMETRY_DIR,
    PROCESS_CONFIGS, read_event_geojson,
)
from utils_extraction import EXTRACT_N_WORKERS, load_cerra_year

log = logging.getLogger(__name__)

# Flag basins where recomputed climatology disagrees strongly with the carried
# original value (sanity check; downstream uses the CERRA value regardless).
DIFF_ABS_FLAG_MM = 100.0
DIFF_REL_FLAG = 0.2


def event_year_range() -> list[int]:
    """Union of event years across all three process files."""
    years: set[int] = set()
    for cfg in PROCESS_CONFIGS:
        df = read_event_geojson(cfg)
        years.update(pd.to_numeric(df["year"], errors="raise").astype(int).unique())
    return sorted(years)


def load_all_basins() -> gpd.GeoDataFrame:
    """All basins (not just event basins) so prediction-time basins are covered."""
    gdf = gpd.read_file(GEOMETRY_DIR / BASINS_GEOJSON_FILE)
    gdf["basin_id"] = pd.to_numeric(gdf["cat"], errors="raise").astype(int)
    return gdf


def annual_total_one_year(year: int, basins: gpd.GeoDataFrame) -> pd.DataFrame:
    """Sum daily precip over a year, then exact_extract a per-basin mean."""
    da = load_cerra_year(year)
    annual = da.sum("time")
    annual = annual.rio.write_crs("EPSG:4326").rio.set_spatial_dims(
        x_dim="longitude", y_dim="latitude"
    )
    wide = exact_extract(
        annual, basins, ["mean"], output="pandas", include_cols=["basin_id"],
    )
    da.close()
    mean_col = next(c for c in wide.columns if "mean" in c)
    return pd.DataFrame({
        "basin_id": wide["basin_id"].astype(int),
        "year": int(year),
        "annual_mm": pd.to_numeric(wide[mean_col], errors="raise").astype(float),
    })


def build_original_lookup() -> pd.DataFrame:
    """Union of (basin_id, tp_su_mean_annual_mean) across all three event files."""
    frames = []
    for cfg in PROCESS_CONFIGS:
        df = read_event_geojson(cfg)
        if "tp_su_mean_annual_mean" not in df.columns:
            continue
        frames.append(df[["basin_id", "tp_su_mean_annual_mean"]].copy())
    if not frames:
        return pd.DataFrame(columns=["basin_id", "original"])
    combined = pd.concat(frames, ignore_index=True)
    combined["basin_id"] = combined["basin_id"].astype(int)
    combined["tp_su_mean_annual_mean"] = pd.to_numeric(
        combined["tp_su_mean_annual_mean"], errors="raise",
    )
    grouped = combined.groupby("basin_id")["tp_su_mean_annual_mean"]
    inconsistent = grouped.nunique()
    bad = inconsistent[inconsistent > 1].index.tolist()
    if bad:
        log.warning(
            "%s basins have inconsistent original tp_su_mean_annual_mean across "
            "process files (e.g. %s); using basin mean", len(bad), bad[:5],
        )
    return grouped.mean().reset_index().rename(
        columns={"tp_su_mean_annual_mean": "original"}
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0 = time.time()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    years = event_year_range()
    log.info("Event-year range: %s..%s (%s years)", years[0], years[-1], len(years))

    basins = load_all_basins()
    log.info("Loaded %s basin polygons from %s", len(basins), BASINS_GEOJSON_FILE)

    with tqdm_joblib(tqdm(total=len(years), desc="Annual totals")):
        per_year = Parallel(n_jobs=EXTRACT_N_WORKERS, backend="loky")(
            delayed(annual_total_one_year)(y, basins) for y in years
        )
    annual_df = pd.concat(per_year, ignore_index=True)

    clim = (
        annual_df.groupby("basin_id")["annual_mm"]
        .agg(tp_su_mean_annual_mean_cerra="mean", n_years="count")
        .reset_index()
    )
    clim["cat"] = clim["basin_id"]
    clim = clim[["cat", "basin_id", "tp_su_mean_annual_mean_cerra", "n_years"]]
    clim_path = DATASET_DIR / "basin_climatology.csv"
    clim.to_csv(clim_path, index=False)
    log.info(
        "Climatology: %s basins, mean=%.1f mm/yr, range=[%.0f, %.0f] -> %s",
        len(clim),
        clim["tp_su_mean_annual_mean_cerra"].mean(),
        clim["tp_su_mean_annual_mean_cerra"].min(),
        clim["tp_su_mean_annual_mean_cerra"].max(),
        clim_path,
    )

    original = build_original_lookup()
    cmp = clim.merge(original, on="basin_id", how="left")
    cmp["diff"] = cmp["tp_su_mean_annual_mean_cerra"] - cmp["original"]
    cmp["abs_diff"] = cmp["diff"].abs()
    cmp["rel_diff"] = cmp["abs_diff"] / cmp["original"]
    cmp["flag_large"] = (cmp["abs_diff"] > DIFF_ABS_FLAG_MM) | (
        cmp["rel_diff"] > DIFF_REL_FLAG
    )
    cmp_cols = [
        "cat", "basin_id", "original", "tp_su_mean_annual_mean_cerra",
        "diff", "abs_diff", "rel_diff", "flag_large", "n_years",
    ]
    cmp_path = DATASET_DIR / "basin_climatology_comparison.csv"
    cmp[cmp_cols].to_csv(cmp_path, index=False)

    # QGIS-friendly enriched basins layer (climatology + comparison + geometry).
    basins_qgis = basins[["basin_id", "geometry"]].merge(
        cmp[cmp_cols], on="basin_id", how="left",
    )
    gj_path = DATASET_DIR / "basin_climatology.geojson"
    basins_qgis.to_file(gj_path, driver="GeoJSON")
    log.info("Wrote QGIS layer (%s polygons) -> %s", len(basins_qgis), gj_path)

    overlap = cmp["original"].notna()
    log.info(
        "Comparison: %s/%s basins overlap with original; %s flagged "
        "(|diff|>%s mm or rel>%.0f%%) -> %s",
        int(overlap.sum()), len(cmp),
        int(cmp.loc[overlap, "flag_large"].sum()),
        DIFF_ABS_FLAG_MM, DIFF_REL_FLAG * 100, cmp_path,
    )
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
