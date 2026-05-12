"""Editable experiment module for the Bolzano autoresearch loop.

opencode may edit this file. Core data loading, evaluation, and artifact
writing live elsewhere and should stay stable.
"""

# ============================================================
# EDITABLE SECTION — AUTONOMOUS AGENT MAY CHANGE THIS BLOCK
# ============================================================

FEATURE_RECIPE = {
    "families": {
        "static_numeric": False,
        "seasonality": False,
        "categorical": False,
        "legacy_precip": False,
        "cum": False,
        "max": False,
        "cum_norm": False,
        "max_norm": False,
        "slope": True,
    },
    "windows": {
        "slope": [2, 3, 7],
    },
    "include_features": [],
    "add_features": [],
    "exclude_features": [],
    "exclude_families": [],
}

MODEL_CONFIG = {
    "interactions": 0,
    "max_bins": 32,
    "learning_rate": 0.02,
    "outer_bags": 4,
    "validation_size": 0.15,
    "early_stopping_rounds": 25,
}

PROCESS_WEIGHTS = {
    "slides": 1.0,
    "flows": 1.0,
}

EXPERIMENT_RATIONALE = """
Progression baseline: slope-only rainfall intensification EBM with no interactions and minimal bagging. This intentionally starts from a small, interpretable model so later iterations can visibly add antecedent rainfall, static basin context, categorical geology/landcover context, and then interaction capacity only after each simpler stage has been measured.
"""

# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================


def validate(available_by_process: dict[str, list[str]]) -> None:
    if MODEL_CONFIG.get("outer_bags", 0) < 4:
        raise ValueError("outer_bags must be >= 4")
    if MODEL_CONFIG.get("learning_rate", 1.0) > 0.5:
        raise ValueError("learning_rate must be <= 0.5")
    if MODEL_CONFIG.get("validation_size", 0.0) <= 0:
        raise ValueError("validation_size must be positive")
    if not EXPERIMENT_RATIONALE.strip():
        raise ValueError("EXPERIMENT_RATIONALE must explain the hypothesis")
    for process, weight in PROCESS_WEIGHTS.items():
        if process not in available_by_process:
            raise ValueError(f"PROCESS_WEIGHTS references unknown process {process!r}")
        if float(weight) <= 0:
            raise ValueError(f"PROCESS_WEIGHTS[{process!r}] must be positive")
