#!/usr/bin/env python3
"""Autonomous opencode loop for Bolzano mass-movement susceptibility."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.discovery import build_discovery_context, initialize_hypotheses
from src.features import load_artifact
from src.config import load_config, parse_processes


EXPERIMENTS = Path("artifacts/experiments/experiments.tsv")
BEST_SCORE = Path("artifacts/experiments/best_score.txt")
BEST_MODEL = Path("artifacts/models/best_model.pkl")
DONE_FLAG = Path(".autoresearch_done")


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def git_ok() -> bool:
    return (Path(".git") / "HEAD").exists()


def ensure_git_repo() -> None:
    if not git_ok():
        run(["git", "init"])
    # Commit a baseline if the repository has no commits yet. Data and heavy
    # artifacts are kept out by .gitignore.
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], text=True, capture_output=True)
    if head.returncode != 0:
        run(["git", "add", "."])
        status = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if status.returncode != 0:
            run([
                "git", "-c", "user.name=autoresearch",
                "-c", "user.email=autoresearch@example.invalid",
                "commit", "-m", "Initialize Bolzano autoresearch pipeline",
            ])


def current_commit() -> str:
    result = run(["git", "rev-parse", "--short", "HEAD"], capture=True)
    return result.stdout.strip()


def best_score() -> float:
    try:
        return float(BEST_SCORE.read_text().strip())
    except Exception:
        return float("-inf")


def append_experiment(row: dict) -> None:
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not EXPERIMENTS.exists()
    with EXPERIMENTS.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("iteration\tcommit\tval_pr_auc\tstatus\tprocess_scores\trationale\n")
        handle.write(
            f"{row['iteration']}\t{row['commit']}\t{row['val_pr_auc']:.6f}\t"
            f"{row['status']}\t{row['process_scores']}\t{row['rationale']}\n"
        )


def feature_inventory(config, processes, features_dir: Path) -> str:
    lines = []
    for process in processes:
        try:
            _, y, _, names, schema = load_artifact(features_dir / process.key)
        except FileNotFoundError:
            lines.append(f"- {process.key}: missing prepared features")
            continue
        lines.append(
            f"- {process.key}: rows={schema.get('rows', len(y))}, positives={int(y.sum())}, "
            f"features={len(names)}, source={schema.get('source', 'unknown')}"
        )
        lines.append("  sample features: " + ", ".join(names[:35]))
    return "\n".join(lines)


def recent_history(limit: int = 12) -> str:
    if not EXPERIMENTS.exists():
        return "(none)"
    lines = EXPERIMENTS.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-limit:])


def build_prompt(args, config, processes) -> str:
    program = Path(args.program).read_text(errors="ignore")
    experiment = Path(args.experiment).read_text(errors="ignore")
    discovery = build_discovery_context(max_chars=5000)
    inventory = feature_inventory(config, processes, Path(args.features))
    return f"""
You are the code-editing researcher inside an autoresearch loop.

Follow this protocol exactly:

{program}

## Current best score
{best_score()}

## Prepared feature inventory
{inventory}

## Recent experiment history
{recent_history()}

## Discovery/RAG context
{discovery}

## Current editable experiment.py
```python
{experiment}
```

Edit only {args.experiment}. Propose one experiment by changing FEATURE_RECIPE,
MODEL_CONFIG, PROCESS_WEIGHTS, and EXPERIMENT_RATIONALE. Do not edit train.py,
prepare.py, src/, data files, or artifacts.
"""


def invoke_opencode(args, prompt: str) -> None:
    cmd = ["opencode", "run"]
    if args.model:
        cmd += ["--model", args.model]
    cmd.append(prompt)
    run(cmd, check=False)


def run_train(args) -> dict:
    cmd = [
        sys.executable, "train.py",
        "--config", args.config,
        "--process", args.process,
        "--features", args.features,
        "--experiment", args.experiment,
        "--out", "artifacts/run_current",
    ]
    if args.with_audit:
        cmd.append("--with-audit")
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"train.py did not emit JSON. stdout:\n{result.stdout}")


def commit_accept(iteration: int, score: float) -> str:
    run(["git", "add", "experiment.py", str(EXPERIMENTS), str(BEST_SCORE)])
    run([
        "git", "-c", "user.name=autoresearch",
        "-c", "user.email=autoresearch@example.invalid",
        "commit", "-m", f"Accept experiment {iteration}: val_pr_auc={score:.6f}",
    ])
    return current_commit()


def revert_experiment() -> None:
    run(["git", "checkout", "--", "experiment.py"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/processes.yml")
    parser.add_argument("--process", default="slides,flows")
    parser.add_argument("--features", default="artifacts/features")
    parser.add_argument("--experiment", default="experiment.py")
    parser.add_argument("--program", default="program.md")
    parser.add_argument("--model", default=None, help="opencode provider/model, e.g. openai/gpt-5.1")
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--with-audit", action="store_true", help="Run slower temporal/spatial audit fits in every iteration.")
    parser.add_argument("--skip-agent", action="store_true", help="Evaluate the current experiment.py without opencode edits.")
    args = parser.parse_args()

    ensure_git_repo()
    initialize_hypotheses(Path("artifacts/discovery/hypotheses.jsonl"))
    config = load_config(args.config)
    processes = parse_processes(config, args.process)
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.max_iters + 1):
        if DONE_FLAG.exists():
            print("Done flag present; stopping.")
            break
        print(f"--- iteration {iteration} ---")
        if not args.skip_agent:
            invoke_opencode(args, build_prompt(args, config, processes))
        try:
            result = run_train(args)
        except Exception as exc:
            append_experiment({
                "iteration": iteration,
                "commit": current_commit(),
                "val_pr_auc": 0.0,
                "status": "crash",
                "process_scores": "crash",
                "rationale": str(exc).replace("\t", " ")[:240],
            })
            revert_experiment()
            continue

        score = float(result["val_pr_auc"])
        process_scores = ",".join(
            f"{item['process']}={item['primary']['pr_auc']:.6f}"
            for item in result["process_results"]
        )
        old_best = best_score()
        status = "keep" if score > old_best else "discard"
        append_experiment({
            "iteration": iteration,
            "commit": current_commit(),
            "val_pr_auc": score,
            "status": status,
            "process_scores": process_scores,
            "rationale": result.get("rationale", "").replace("\t", " ").replace("\n", " ")[:240],
        })
        if status == "keep":
            BEST_SCORE.write_text(f"{score:.6f}\n")
            shutil.copy2("artifacts/run_current/model_bundle.pkl", BEST_MODEL)
            commit = commit_accept(iteration, score)
            print(f"KEEP {score:.6f} > {old_best:.6f}; commit={commit}")
        else:
            revert_experiment()
            print(f"DISCARD {score:.6f} <= {old_best:.6f}")


if __name__ == "__main__":
    main()
