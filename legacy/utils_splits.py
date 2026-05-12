"""Train/test split generation for the operational Bolzano EWS.

Two schemes are produced and concatenated:
  * operational_bz_spatial — hold out a subset of Bolzano basins; train on the
    rest of the Alpine pool (including non-held-out BZ basins).
  * operational_bz_rolling — train on all Alpine rows from years before the
    test year; test on Bolzano rows in the test year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


@dataclass
class Split:
    # Outer evaluation fold:
    # - train_idx: rows available for model development within this fold
    # - test_idx: rows held out for final fold evaluation
    # Validation is NOT stored here; it is carved later from train_idx by
    # train_validation_indices(...).
    scheme: str
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def _spatial_splits(
    df: pd.DataFrame,
    bz_basin_ids: set[int],
    y: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[Split]:
    # StratifiedGroupKFold over Bolzano basins keeps each basin intact within
    # a fold while balancing the positive count across folds. Threshold-anchored
    # metrics (F2, precision, recall) are prevalence-sensitive, so balanced
    # test prevalence is required for honest fold-level scores.
    basin_ids = df["cat"].astype(int)  # one basin ID per row in the full dataset
    bz_df = df.loc[basin_ids.isin(bz_basin_ids), ["cat"]].copy()  # Bolzano-only subset
    bz_df["basin_id"] = bz_df["cat"].astype(int)  # explicit group label for grouped CV
    bz_df["y"] = y[bz_df.index].astype(int)  # labels aligned to the Bolzano subset rows

    n_unique_bz = bz_df["basin_id"].nunique()
    n_splits_eff = min(n_splits, n_unique_bz)  # cannot have more grouped folds than unique basins

    splits: list[Split] = []
    for rep in range(n_repeats):
        # Repeat grouped K-fold with a different seed each time. With shuffle=True,
        # this gives a different assignment of whole basins to folds in each repeat.
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits_eff, shuffle=True, random_state=random_state + rep,
        )
        # The splitter object defines the grouped/stratified K-fold strategy,
        # but it does not return concrete train/test folds until we iterate over
        # sgkf.split(...). This loop walks through the n_splits_eff folds produced
        # for the current repeat, one held-out test fold at a time.
        for fold, (_, test_pos_in_bz_subset) in enumerate(
            # split(...) is run on the explicit Bolzano-only subset:
            # - X is a dummy array because StratifiedGroupKFold requires an X
            #   argument in its API, but this splitter uses only y and groups to
            #   decide the folds in this case
            # - bz_df["y"] is used for stratification
            # - bz_df["basin_id"] is used to keep each basin intact within a fold
            sgkf.split(
                np.zeros(len(bz_df), dtype=int),
                bz_df["y"].to_numpy(),
                groups=bz_df["basin_id"].to_numpy(),
            ),
            start=1,
        ):
            # 1. choose a subset of Bolzano basins as test basins
            test_rows_in_bz_subset = bz_df.iloc[test_pos_in_bz_subset]
            test_basins = test_rows_in_bz_subset["basin_id"].unique()
            # 2. put every row from those basins into test_idx
            test_mask = basin_ids.isin(test_basins).to_numpy()
            test_idx = np.flatnonzero(test_mask)
            # 3. put every row from all other basins into train_idx
            #    This means all non-Bolzano basins are always kept in training;
            #    only the held-out Bolzano basins vary across folds/repeats.
            train_idx = np.flatnonzero(~test_mask)
            splits.append(Split(
                scheme="operational_bz_spatial",
                fold_id=f"rep{rep + 1:02d}_fold{fold:02d}",
                train_idx=train_idx,
                test_idx=test_idx,
            ))
    return splits


def _rolling_splits(
    df: pd.DataFrame,
    bz_basin_ids: set[int],
    *,
    min_train_years: int,
    start_year: int | None,
    end_year: int | None,
    test_window_years: int = 1,
) -> list[Split]:
    # Expanding-window forward chaining over BZ years: train < first(W), test = W on BZ.
    # test_window_years > 1 groups consecutive BZ years into one test fold so rare
    # processes (flows/slides) accumulate enough positives for a defined AUROC.
    if test_window_years < 1:
        raise ValueError(f"test_window_years must be >= 1, got {test_window_years}")
    groups = df["cat"].astype(int).to_numpy()
    years = df["year"].astype(int).to_numpy()
    bz_mask = np.isin(groups, list(bz_basin_ids))
    bz_years = np.array(sorted(pd.unique(years[bz_mask])))

    splits: list[Split] = []
    for pos in range(min_train_years, len(bz_years), test_window_years):
        window = bz_years[pos : pos + test_window_years]
        if len(window) == 0:
            continue
        first_year = int(window[0])
        last_year = int(window[-1])
        if start_year is not None and first_year < start_year:
            continue
        if end_year is not None and last_year > end_year:
            continue
        train_idx = np.flatnonzero(years < first_year)
        test_idx = np.flatnonzero(bz_mask & np.isin(years, window))
        fold_id = (
            f"test_{first_year}" if len(window) == 1
            else f"test_{first_year}_{last_year}"
        )
        splits.append(Split(
            scheme="operational_bz_rolling",
            fold_id=fold_id,
            train_idx=train_idx,
            test_idx=test_idx,
        ))
    return splits


def build_operational_bz_splits(
    df: pd.DataFrame,
    bz_basin_ids: set[int],
    y: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    min_train_years: int = 5,
    start_year: int | None = None,
    end_year: int | None = None,
    random_state: int = 42,
    max_per_scheme: int | None = None,
    test_window_years: int = 1,
) -> list[Split]:
    # Always produce both schemes — we need both for the EWS evaluation.
    splits = _spatial_splits(
        df, bz_basin_ids, y,
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    )
    splits += _rolling_splits(
        df, bz_basin_ids,
        min_train_years=min_train_years, start_year=start_year, end_year=end_year,
        test_window_years=test_window_years,
    )
    if max_per_scheme and max_per_scheme > 0:
        # Keep at most N folds per scheme — handy for quick smoke tests.
        counts: dict[str, int] = {}
        kept: list[Split] = []
        for s in splits:
            counts.setdefault(s.scheme, 0)
            if counts[s.scheme] < max_per_scheme:
                kept.append(s)
                counts[s.scheme] += 1
        splits = kept
    return splits


def _validation_by_last_year(
    train_idx: np.ndarray, years: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the LAST training year as validation (temporal early-stop)."""
    train_years = years[train_idx]
    val_year = train_years.max()
    val_mask = train_years == val_year
    return train_idx[~val_mask], train_idx[val_mask]


def _validation_by_held_out_basins(
    train_idx: np.ndarray, groups: np.ndarray,
    *, val_fraction: float = 0.15, random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole basins as validation (~15%, spatial early-stop)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    sub_pos, val_pos = next(splitter.split(train_idx, groups=groups[train_idx]))
    return train_idx[sub_pos], train_idx[val_pos]


def train_validation_indices(
    split: Split, metadata: pd.DataFrame, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a (sub-train, validation) slice from a fold's training rows.

    The outer Split already defines train_idx vs test_idx. This helper takes
    only split.train_idx and divides it again into:
      - sub-train: rows used in model.fit(...)
      - validation: rows where we compute predicted scores/probabilities after
        fitting, then use those validation scores to choose the F2 threshold

    split.test_idx is left untouched and remains the final held-out test fold,
    where the already-chosen threshold is applied once for final evaluation.

    metadata is the row-aligned event table for the same samples as split.train_idx
    / split.test_idx (in this project: events.csv). We only use its columns
    needed to carve validation:
      - metadata["year"] for rolling folds
      - metadata["cat"] for spatial folds

    Rolling folds → hold out the LAST training year (temporal signal).
    Spatial folds → hold out ~15% of basins (spatial-transfer signal).
    """
    if split.scheme == "operational_bz_rolling":
        years = metadata["year"].astype(int).to_numpy()
        return _validation_by_last_year(split.train_idx, years)
    if split.scheme == "operational_bz_spatial":
        groups = metadata["cat"].astype(int).to_numpy()
        return _validation_by_held_out_basins(split.train_idx, groups, random_state=seed)
    raise ValueError(f"Unknown scheme: {split.scheme!r}")


def split_context_row(split: Split, y: np.ndarray) -> dict:
    return {
        "scheme": split.scheme,
        "fold_id": split.fold_id,
        "n_train": int(len(split.train_idx)),
        "n_test": int(len(split.test_idx)),
        "train_prevalence": float(np.mean(y[split.train_idx])) if len(split.train_idx) else np.nan,
        "test_prevalence": float(np.mean(y[split.test_idx])) if len(split.test_idx) else np.nan,
    }
