#!/usr/bin/env python3
"""Train and evaluate the current editable EBM experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import load_config, parse_processes
from src.features import load_artifact, select_feature_names, selected_matrix

FIT_HEARTBEAT_SECONDS = 30.0


def load_experiment(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_ebm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_config: dict[str, Any],
    seed: int,
    label: str = "fit",
):
    model = ExplainableBoostingClassifier(
        feature_names=feature_names,
        random_state=seed,
        n_jobs=-1,
        **model_config,
    )
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(FIT_HEARTBEAT_SECONDS):
            print(
                f"[train.py] still fitting {label}: rows={len(y)} features={len(feature_names)}",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"[train.py] fitting {label}: rows={len(y)} features={len(feature_names)}",
        file=sys.stderr,
        flush=True,
    )
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        model.fit(X, y)
    finally:
        stop.set()
        thread.join(timeout=1.0)
    print(f"[train.py] finished {label}", file=sys.stderr, flush=True)
    return model


def scalar_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = {"pr_auc": float("nan"), "roc_auc": float("nan"), "prevalence": float(np.mean(y_true))}
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    return out


def random_split_eval(X, y, feature_names, model_config, seed: int, label: str) -> tuple[dict[str, float], Any]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    model = fit_ebm(X[train_idx], y[train_idx], feature_names, model_config, seed, label)
    prob = model.predict_proba(X[val_idx])[:, 1]
    metrics = scalar_metrics(y[val_idx], prob)
    metrics.update({"n_train": int(len(train_idx)), "n_val": int(len(val_idx))})
    return metrics, model


def temporal_audit_eval(X, y, metadata, feature_names, model_config, seed: int, label: str) -> dict[str, float]:
    years = metadata["year"].astype(int).to_numpy()
    unique_years = np.array(sorted(pd.unique(years)))
    if len(unique_years) < 4:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n_train": 0, "n_val": 0}
    split_year = unique_years[int(np.floor(len(unique_years) * 0.8))]
    train_idx = np.flatnonzero(years < split_year)
    val_idx = np.flatnonzero(years >= split_year)
    if len(train_idx) < 10 or len(val_idx) < 10 or len(np.unique(y[val_idx])) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    model = fit_ebm(X[train_idx], y[train_idx], feature_names, model_config, seed, label)
    metrics = scalar_metrics(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
    metrics.update({"n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "split_year": int(split_year)})
    return metrics


def spatial_audit_eval(X, y, metadata, feature_names, model_config, seed: int, label: str) -> dict[str, float]:
    groups = metadata["basin_id"].astype(int).to_numpy()
    if len(np.unique(groups)) < 3:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n_train": 0, "n_val": 0}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    if len(np.unique(y[val_idx])) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    model = fit_ebm(X[train_idx], y[train_idx], feature_names, model_config, seed, label)
    metrics = scalar_metrics(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
    metrics.update({"n_train": int(len(train_idx)), "n_val": int(len(val_idx))})
    return metrics


def importances(model) -> dict[str, float]:
    try:
        data = model.explain_global().data()
        return {str(name): float(score) for name, score in zip(data["names"], data["scores"])}
    except Exception:
        return {}


def train_one(config, process, experiment, args) -> tuple[dict[str, Any], dict[str, Any]]:
    X_full, y, metadata, feature_names, schema = load_artifact(Path(args.features) / process.key)
    selected = select_feature_names(feature_names, experiment.FEATURE_RECIPE)
    X = selected_matrix(X_full, feature_names, selected)
    random_metrics, split_model = random_split_eval(
        X, y, selected, experiment.MODEL_CONFIG, config.random_state, f"{process.key} random split"
    )
    temporal_metrics = temporal_audit_eval(
        X, y, metadata, selected, experiment.MODEL_CONFIG, config.random_state, f"{process.key} temporal audit"
    ) if args.with_audit else {}
    spatial_metrics = spatial_audit_eval(
        X, y, metadata, selected, experiment.MODEL_CONFIG, config.random_state, f"{process.key} spatial audit"
    ) if args.with_audit else {}
    final_model = fit_ebm(X, y, selected, experiment.MODEL_CONFIG, config.random_state, f"{process.key} full fit")
    result = {
        "process": process.key,
        "display_name": process.display_name,
        "primary": random_metrics,
        "audit_temporal": temporal_metrics,
        "audit_spatial": spatial_metrics,
        "n_rows": int(len(y)),
        "n_positive": int(y.sum()),
        "n_features": int(len(selected)),
        "features": selected,
        "importances": importances(split_model),
    }
    bundle_item = {
        "model": final_model,
        "features": selected,
        "schema": schema,
        "all_feature_names": feature_names,
        "metrics": result,
    }
    return result, bundle_item


def weighted_mean(results: list[dict[str, Any]], weights: dict[str, float]) -> float:
    vals = []
    wts = []
    for result in results:
        val = result["primary"]["pr_auc"]
        if np.isfinite(val):
            vals.append(float(val))
            wts.append(float(weights.get(result["process"], 1.0)))
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=wts))


def write_report(path: Path, output: dict[str, Any]) -> None:
    lines = [
        "# Current Experiment",
        "",
        f"Primary val_pr_auc: {output['val_pr_auc']:.6f}",
        "",
        "## Processes",
    ]
    for item in output["process_results"]:
        lines.append(
            f"- {item['process']}: PR-AUC={item['primary']['pr_auc']:.6f}, "
            f"ROC-AUC={item['primary']['roc_auc']:.6f}, features={item['n_features']}"
        )
    lines += ["", "## Rationale", output["rationale"].strip()]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/processes.yml")
    parser.add_argument("--process", default=None)
    parser.add_argument("--features", default="artifacts/features")
    parser.add_argument("--experiment", default="experiment.py")
    parser.add_argument("--out", default="artifacts/run_current")
    parser.add_argument("--with-audit", action="store_true", help="Run slower temporal and spatial audit fits.")
    parser.add_argument("--no-audit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.no_audit:
        args.with_audit = False

    config = load_config(args.config)
    processes = parse_processes(config, args.process)
    experiment = load_experiment(Path(args.experiment))

    available = {}
    for process in processes:
        _, _, _, feature_names, _ = load_artifact(Path(args.features) / process.key)
        available[process.key] = feature_names
    experiment.validate(available)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    bundle = {
        "config_path": args.config,
        "experiment_path": args.experiment,
        "feature_recipe": experiment.FEATURE_RECIPE,
        "model_config": experiment.MODEL_CONFIG,
        "rationale": experiment.EXPERIMENT_RATIONALE,
        "processes": {},
    }
    for process in processes:
        result, bundle_item = train_one(config, process, experiment, args)
        results.append(result)
        bundle["processes"][process.key] = bundle_item

    val = weighted_mean(results, experiment.PROCESS_WEIGHTS)
    output = {
        "val_pr_auc": round(val, 6),
        "process_results": results,
        "model_config": experiment.MODEL_CONFIG,
        "feature_recipe": experiment.FEATURE_RECIPE,
        "process_weights": experiment.PROCESS_WEIGHTS,
        "rationale": experiment.EXPERIMENT_RATIONALE,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    with (out_dir / "model_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)
    write_report(out_dir / "summary.md", output)
    print(json.dumps(output))
    print(f"val_pr_auc: {output['val_pr_auc']:.6f}")


if __name__ == "__main__":
    main()
