"""Editable experiment module for the Bolzano autoresearch loop.

opencode may edit this file. Core data loading, evaluation, and artifact
writing live elsewhere and should stay stable.
"""

# ============================================================
# EDITABLE SECTION — AUTONOMOUS AGENT MAY CHANGE THIS BLOCK
# ============================================================

FEATURE_RECIPE = {
    "families": {
        "static_numeric": True,
        "seasonality": True,
        "categorical": True,
        "legacy_precip": True,
        "cum": False,
        "max": False,
        "cum_norm": True,
        "max_norm": True,
        "slope": True,
    },
    "windows": {
        "cum_norm": [1, 2, 3, 7, 15, 21, 30, 45, 60],
        "max_norm": [2, 3, 7, 15, 21, 30, 45, 60],
        "slope": [2, 3, 4, 5, 6, 7, 15, 30],
    },
    "include_features": [],
    "add_features": [],
    "exclude_features": [],
    "exclude_families": [],
}

MODEL_CONFIG = {
    "interactions": 8,
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
Test whether a larger but still interpretable EBM interaction budget improves
joint slide/flow susceptibility after the four-interaction model exposed useful
terrain/rainfall and rainfall-window modifiers. Allowing eight interactions
should give the model room to represent additional slope, basin morphology,
lithology, landcover, or seasonal modifiers without changing the feature recipe,
learning rate, bags, or process weights. This hypothesis is rejected if
validation PR-AUC does not exceed the current best score.
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
