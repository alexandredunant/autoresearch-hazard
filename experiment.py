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
        "slope": False,
    },
    "windows": {
        "cum_norm": [],
        "max_norm": [],
        "slope": [],
    },
    "include_features": [],
    "add_features": [],
    "exclude_features": [],
    "exclude_families": [],
    "allow_no_features": True,
}

MODEL_CONFIG = {
    "interactions": 0,
    "max_bins": 128,
    "learning_rate": 0.01,
    "outer_bags": 8,
    "validation_size": 0.15,
    "early_stopping_rounds": 25,
}

PROCESS_WEIGHTS = {
    "slides": 1.0,
    "flows": 1.0,
}

EXPERIMENT_RATIONALE = """
Fresh-run null baseline. All predictive feature families are disabled and the
feature selector is explicitly allowed to provide only the constant null feature.
This establishes the no-feature reference score before any hazard-informed
feature engineering or model tuning is attempted.
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
