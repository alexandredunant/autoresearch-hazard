from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "processes.yml"


@dataclass(frozen=True)
class ProcessSpec:
    key: str
    display_name: str
    event_file: str
    label_col: str
    basin_id_col: str
    bolzano_col: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    legacy_precip_features: tuple[str, ...]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    data_dir: Path
    basins_file: Path
    cerra_dir: Path
    artifact_dir: Path
    default_processes: tuple[str, ...]
    max_lag: int
    random_state: int
    processes: dict[str, ProcessSpec]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path = DEFAULT_CONFIG) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    project = payload.get("project", {})
    data_dir = _resolve(REPO_ROOT, project.get("data_dir", "data"))
    artifact_dir = _resolve(REPO_ROOT, project.get("artifact_dir", "artifacts"))
    basins_file = data_dir / project.get("basins_file", "basins_with_BZ.geojson")
    cerra_dir = _resolve(REPO_ROOT, project.get("cerra_dir", "CERRA_LAND_EUSALP_reproj"))

    processes: dict[str, ProcessSpec] = {}
    for key, item in (payload.get("processes") or {}).items():
        processes[key] = ProcessSpec(
            key=key,
            display_name=str(item["display_name"]),
            event_file=str(item["event_file"]),
            label_col=str(item["label_col"]),
            basin_id_col=str(item.get("basin_id_col", "cat")),
            bolzano_col=str(item.get("bolzano_col", "BZ")),
            numeric_features=tuple(item.get("numeric_features", [])),
            categorical_features=tuple(item.get("categorical_features", [])),
            legacy_precip_features=tuple(item.get("legacy_precip_features", [])),
        )

    return ProjectConfig(
        name=str(project.get("name", "bolzano_mass_movement_autoresearch")),
        data_dir=data_dir,
        basins_file=basins_file,
        cerra_dir=cerra_dir,
        artifact_dir=artifact_dir,
        default_processes=tuple(project.get("default_processes", processes.keys())),
        max_lag=int(project.get("max_lag", 60)),
        random_state=int(project.get("random_state", 42)),
        processes=processes,
    )


def parse_processes(config: ProjectConfig, value: str | None) -> list[ProcessSpec]:
    keys = config.default_processes if not value or value == "all" else tuple(
        item.strip() for item in value.split(",") if item.strip()
    )
    missing = [key for key in keys if key not in config.processes]
    if missing:
        raise KeyError(f"Unknown process key(s): {missing}. Available: {sorted(config.processes)}")
    return [config.processes[key] for key in keys]
