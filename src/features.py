from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ProcessSpec


DYNAMIC_FAMILIES = ("cum", "max", "cum_norm", "max_norm", "slope")
NULL_FEATURE_NAME = "__null_feature__"


def safe_token(value: object) -> str:
    text = str(value).strip() if value is not None else "__MISSING__"
    text = text or "__MISSING__"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def slope_windows(lag_recent: np.ndarray, max_lag: int) -> np.ndarray:
    n = lag_recent.shape[0]
    out = np.zeros((n, max_lag - 1), dtype=np.float32)
    for k in range(2, max_lag + 1):
        y = lag_recent[:, :k][:, ::-1]
        x = np.arange(k, dtype=np.float32)
        x_centered = x - x.mean()
        y_centered = y - y.mean(axis=1, keepdims=True)
        out[:, k - 2] = (y_centered * x_centered).sum(axis=1) / float((x_centered**2).sum())
    return out


def dynamic_features_from_lags(
    lag_recent: np.ndarray,
    map_mm: np.ndarray,
    *,
    max_lag: int,
) -> pd.DataFrame:
    if lag_recent.shape[1] < max_lag + 1:
        raise ValueError(f"Need lag_0..lag_{max_lag}; got {lag_recent.shape[1]} columns")
    lag_recent = lag_recent[:, : max_lag + 1].astype(np.float32)
    windows = np.arange(1, max_lag + 1)
    cum = np.cumsum(lag_recent[:, :max_lag], axis=1).astype(np.float32)
    mx = np.maximum.accumulate(lag_recent[:, :max_lag], axis=1).astype(np.float32)
    denom = np.maximum(map_mm.astype(np.float32), 1e-6)
    data: dict[str, np.ndarray] = {}
    for idx, k in enumerate(windows):
        data[f"cum_{k}"] = cum[:, idx]
        data[f"max_{k}"] = mx[:, idx]
        data[f"cum_norm_{k}"] = cum[:, idx] / denom
        data[f"max_norm_{k}"] = mx[:, idx] / denom
    slopes = slope_windows(lag_recent, max_lag)
    for k in range(2, max_lag + 1):
        data[f"slope_{k}"] = slopes[:, k - 2]
    return pd.DataFrame(data)


def _normalizer(frame: pd.DataFrame) -> np.ndarray:
    if "tp_su_mean_annual_mean_cerra" in frame.columns:
        col = "tp_su_mean_annual_mean_cerra"
    elif "tp_su_mean_annual_mean" in frame.columns:
        col = "tp_su_mean_annual_mean"
    else:
        raise ValueError("Missing precipitation climatology normalizer")
    values = pd.to_numeric(frame[col], errors="raise").astype(np.float32).to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{col} contains non-finite or non-positive values")
    return values


def _numeric_frame(frame: pd.DataFrame, spec: ProcessSpec) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in spec.numeric_features:
        if col not in frame.columns:
            raise ValueError(f"{spec.key}: missing numeric feature {col}")
        data[col] = pd.to_numeric(frame[col], errors="raise").astype(np.float32).to_numpy()
    return pd.DataFrame(data)


def _legacy_precip_frame(frame: pd.DataFrame, spec: ProcessSpec) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for col in spec.legacy_precip_features:
        if col in frame.columns:
            data[f"legacy_{col}"] = pd.to_numeric(frame[col], errors="raise").astype(np.float32).to_numpy()
    return pd.DataFrame(data)


def _categorical_frame(frame: pd.DataFrame, spec: ProcessSpec) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    data: dict[str, np.ndarray] = {}
    levels_by_col: dict[str, list[str]] = {}
    for col in spec.categorical_features:
        if col not in frame.columns:
            raise ValueError(f"{spec.key}: missing categorical feature {col}")
        values = frame[col].fillna("__MISSING__").astype(str)
        levels = sorted(values.unique().tolist())
        levels_by_col[col] = levels
        for level in levels:
            name = f"cat__{col}__{safe_token(level)}"
            data[name] = (values == level).astype(np.float32).to_numpy()
    return pd.DataFrame(data), levels_by_col


def build_feature_matrix(
    events: pd.DataFrame,
    spec: ProcessSpec,
    *,
    lag_recent: np.ndarray | None,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    numeric = _numeric_frame(events, spec)
    legacy = _legacy_precip_frame(events, spec)
    categorical, levels = _categorical_frame(events, spec)
    parts = [numeric, legacy, categorical]
    if lag_recent is not None:
        parts.append(dynamic_features_from_lags(lag_recent, _normalizer(events), max_lag=max_lag))
    X_df = pd.concat(parts, axis=1)
    if not np.isfinite(X_df.to_numpy(dtype=np.float32)).all():
        bad = X_df.columns[~np.isfinite(X_df.to_numpy(dtype=np.float32)).all(axis=0)].tolist()
        raise ValueError(f"{spec.key}: feature matrix contains non-finite values in {bad[:10]}")
    y = pd.to_numeric(events[spec.label_col], errors="raise").astype(np.int8).to_numpy()
    metadata_cols = [c for c in ("process", "basin_id", "cat", "date", "year") if c in events.columns]
    metadata = events[metadata_cols].copy()
    schema = {
        "process": spec.key,
        "label_col": spec.label_col,
        "numeric_features": list(spec.numeric_features),
        "categorical_features": list(spec.categorical_features),
        "categorical_levels": levels,
        "legacy_precip_features": list(spec.legacy_precip_features),
        "max_lag": max_lag,
    }
    return X_df.to_numpy(dtype=np.float32), y, metadata, X_df.columns.tolist(), schema


def save_artifact(
    out_dir: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
    schema: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "matrix.npz", X=X, y=y, feature_names=np.array(feature_names, dtype=object))
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True))


def load_artifact(process_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    matrix_path = process_dir / "matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing prepared matrix: {matrix_path}. Run prepare.py first.")
    data = np.load(matrix_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    feature_names = data["feature_names"].tolist()
    metadata = pd.read_csv(process_dir / "metadata.csv", parse_dates=["date"])
    schema = json.loads((process_dir / "schema.json").read_text())
    return X, y, metadata, feature_names, schema


def feature_family(name: str) -> str:
    if name == NULL_FEATURE_NAME:
        return "null"
    if name in ("doy_sin", "doy_cos"):
        return "seasonality"
    if name.startswith("cat__"):
        return "categorical"
    if name.startswith("legacy_"):
        return "legacy_precip"
    for family in ("cum_norm", "max_norm", "cum", "max", "slope"):
        if name.startswith(f"{family}_"):
            return family
    return "static_numeric"


def feature_window(name: str) -> int | None:
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else None


def select_feature_names(feature_names: list[str], recipe: dict[str, Any]) -> list[str]:
    available = set(feature_names)
    include_exact = [f for f in recipe.get("include_features", []) if f in available]
    selected: list[str] = []
    families = recipe.get("families", {})
    windows = recipe.get("windows", {})
    if include_exact:
        selected.extend(include_exact)
    else:
        for name in feature_names:
            family = feature_family(name)
            if not families.get(family, False):
                continue
            allowed_windows = windows.get(family)
            if allowed_windows and feature_window(name) not in set(int(v) for v in allowed_windows):
                continue
            selected.append(name)
    for name in recipe.get("add_features", []):
        if name in available and name not in selected:
            selected.append(name)
    excludes = set(recipe.get("exclude_features", []))
    exclude_families = set(recipe.get("exclude_families", []))
    selected = [
        name for name in selected
        if name not in excludes and feature_family(name) not in exclude_families
    ]
    if not selected:
        if recipe.get("allow_no_features", False) or recipe.get("null_baseline", False):
            return [NULL_FEATURE_NAME]
        raise ValueError("Feature recipe selected zero available features")
    return selected


def selected_matrix(X: np.ndarray, feature_names: list[str], selected: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in selected if name != NULL_FEATURE_NAME and name not in index]
    if missing:
        raise KeyError(f"Selected features not present in matrix: {missing[:10]}")
    columns = []
    for name in selected:
        if name == NULL_FEATURE_NAME:
            columns.append(np.zeros((X.shape[0], 1), dtype=np.float32))
        else:
            columns.append(X[:, [index[name]]].astype(np.float32))
    return np.hstack(columns).astype(np.float32)


def build_prediction_frame(
    basins: pd.DataFrame,
    spec: ProcessSpec,
    schema: dict[str, Any],
    *,
    prediction_date: pd.Timestamp,
    selected_features: list[str],
    lag_recent: np.ndarray | None,
) -> pd.DataFrame:
    frame = basins.copy()
    frame["date"] = pd.Timestamp(prediction_date)
    frame["doy_sin"] = np.sin(2.0 * np.pi * frame["date"].dt.dayofyear / 365.25).astype(np.float32)
    frame["doy_cos"] = np.cos(2.0 * np.pi * frame["date"].dt.dayofyear / 365.25).astype(np.float32)

    data: dict[str, np.ndarray] = {}
    for col in schema["numeric_features"]:
        if col not in frame.columns:
            raise ValueError(f"{spec.key}: Bolzano basins missing numeric feature {col}")
        data[col] = pd.to_numeric(frame[col], errors="raise").astype(np.float32).to_numpy()

    if lag_recent is not None:
        max_lag = min(int(schema["max_lag"]), int(lag_recent.shape[1] - 1))
        dyn = dynamic_features_from_lags(lag_recent, _normalizer(frame), max_lag=max_lag)
        data.update({name: dyn[name].to_numpy(dtype=np.float32) for name in dyn.columns})
        for legacy in schema["legacy_precip_features"]:
            match = re.search(r"(\d+)$", legacy)
            if match:
                k = int(match.group(1))
                data[f"legacy_{legacy}"] = lag_recent[:, :k].sum(axis=1).astype(np.float32)

    for col, levels in schema["categorical_levels"].items():
        if col not in frame.columns:
            raise ValueError(f"{spec.key}: Bolzano basins missing categorical feature {col}")
        values = frame[col].fillna("__MISSING__").astype(str)
        for level in levels:
            name = f"cat__{col}__{safe_token(level)}"
            data[name] = (values == level).astype(np.float32).to_numpy()

    pred = pd.DataFrame(data)
    if NULL_FEATURE_NAME in selected_features:
        pred[NULL_FEATURE_NAME] = np.zeros(len(frame), dtype=np.float32)
    missing = [name for name in selected_features if name not in pred.columns]
    if missing:
        needs_lags = [name for name in missing if feature_family(name) in set(DYNAMIC_FAMILIES) | {"legacy_precip"}]
        if needs_lags and lag_recent is None:
            raise ValueError(f"{spec.key}: selected dynamic features require precipitation lags: {needs_lags[:10]}")
        raise ValueError(f"{spec.key}: cannot build selected features for map: {missing[:10]}")
    return pred[selected_features].astype(np.float32)
