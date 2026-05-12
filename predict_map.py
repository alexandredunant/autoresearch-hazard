#!/usr/bin/env python3
"""Export Bolzano basin-level susceptibility predictions from the best model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import bolzano_basins, latest_cerra_date
from src.features import build_prediction_frame
from src.precip import extract_prediction_lags, required_max_lag
from utils_metrics import assign_warning_level, compute_warning_thresholds


def top_importances(metrics: dict, n: int = 5) -> str:
    items = sorted(metrics.get("importances", {}).items(), key=lambda kv: abs(kv[1]), reverse=True)
    return "; ".join(f"{name}:{score:.4g}" for name, score in items[:n])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/processes.yml")
    parser.add_argument("--model", default="artifacts/models/best_model.pkl")
    parser.add_argument("--date", default="latest", help="'latest' or YYYY-MM-DD")
    parser.add_argument("--out", default="artifacts/maps/bolzano_latest_predictions.geojson")
    parser.add_argument("--max-basins", type=int, default=None, help="Smoke-test cap for prediction export.")
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.model, "rb") as handle:
        bundle = pickle.load(handle)

    prediction_date = latest_cerra_date(config) if args.date == "latest" else pd.Timestamp(args.date).normalize()
    basins = bolzano_basins(config)
    if args.max_basins:
        basins = basins.head(args.max_basins).copy()
    max_lag = max(
        required_max_lag(item["features"], default=0)
        for item in bundle["processes"].values()
    )
    lag_recent = None
    if max_lag > 0:
        lag_recent = extract_prediction_lags(
            config, basins, prediction_date=prediction_date, max_lag=max_lag
        )

    outputs = []
    for process_key, item in bundle["processes"].items():
        process = config.processes[process_key]
        features = item["features"]
        X_pred = build_prediction_frame(
            basins,
            process,
            item["schema"],
            prediction_date=prediction_date,
            selected_features=features,
            lag_recent=lag_recent,
        )
        prob = item["model"].predict_proba(X_pred.to_numpy(dtype=np.float32))[:, 1]
        prevalence = item["metrics"]["primary"].get("prevalence", float(np.mean(prob)))
        t1, t2, t3 = compute_warning_thresholds(float(prevalence))
        out = basins[["basin_id", "geometry"]].copy()
        out["process"] = process_key
        out["process_name"] = process.display_name
        out["prediction_date"] = str(prediction_date.date())
        out["probability"] = prob.astype(float)
        out["warning_level"] = [assign_warning_level(float(p), t1, t2, t3) for p in prob]
        out["model_val_pr_auc"] = float(item["metrics"]["primary"]["pr_auc"])
        out["top_terms"] = top_importances(item["metrics"])
        outputs.append(out)

    gdf = gpd.GeoDataFrame(pd.concat(outputs, ignore_index=True), crs=basins.crs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Wrote {out_path} ({len(gdf)} rows, date={prediction_date.date()})")


if __name__ == "__main__":
    main()
