# autoresearch

This is an experiment to have the LLM do its own research.

The agent is an autonomous code-editing researcher for the current hazard
prediction repository. The protocol should encourage discovery, not prescribe a
fixed scientific story. The hazard, processes, features, and plausible
mechanisms must be inferred from the repository and `literature/`, then tested
empirically.

The primary metric is `val_pr_auc`. Higher is better. A candidate is considered
an improvement only if it beats the current best score by at least `0.001`.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, for example
   `may13`. The branch `autoresearch/<tag>` must not already exist. This is a
   fresh run.
2. **Create the branch**: create `autoresearch/<tag>` from the current main
   development branch.
3. **Clear prior generated state**: a fresh run must not reuse old discovery,
   search, model, map, smoke, or run outputs. Remove stale generated artifacts
   before building any new context:

   ```bash
   rm -rf artifacts/discovery artifacts/rag artifacts/geoevolve_storage artifacts/maps artifacts/reports artifacts/run_current artifacts/run_smoke artifacts/run_smoke_check artifacts/features_smoke artifacts/models run.log
   mkdir -p artifacts/models artifacts/experiments
   ```

   Keep reusable input data such as `artifacts/features/` and `literature/`.
   Do not read stale files from removed artifact directories as evidence.
4. **Read the in-scope files**: the repo is small. Read these files for full
   context:
   - `README.md` for repository context and the active train command.
   - `prepare.py` for fixed preparation and data assumptions. Do not modify.
   - `train.py` for the deterministic evaluator and output format. Do not
     modify.
   - `experiment.py` for the editable experiment surface.
   - `configs/` and `src/` as needed to understand available processes,
     features, and model constraints.
5. **Inventory prepared data and features**: before proposing any experiment,
   identify what data is available and where it is stored. This is an inventory
   step, not a hypothesis source by itself.

   - `configs/processes.yml` defines the configured process keys, source data
     file names, label columns, and configured feature lists.
   - `artifacts/features/summary.csv` summarizes the prepared feature artifacts.
   - `artifacts/features/<process>/matrix.npz` stores the prepared matrix,
     labels, and feature-name array for each process.
   - `artifacts/features/<process>/metadata.csv` stores row metadata used by
     the evaluator and audits.
   - `artifacts/features/<process>/schema.json` stores the prepared schema,
     including available numeric, categorical, legacy, and lag-derived feature
     inventories.
   - `src/features.py` defines how `FEATURE_RECIPE` selects from those available
     feature names.

   Useful neutral inventory commands:

   ```bash
   sed -n '1,220p' configs/processes.yml
   find artifacts/features -maxdepth 2 -type f | sort
   cat artifacts/features/summary.csv
   cat artifacts/features/<process>/schema.json
   sed -n '1,5p' artifacts/features/<process>/metadata.csv
   python - <<'PY'
   import numpy as np
   from pathlib import Path
   for path in sorted(Path("artifacts/features").glob("*/matrix.npz")):
       data = np.load(path, allow_pickle=True)
       print(path.parent.name, data["X"].shape, data["y"].shape, len(data["feature_names"]))
       print("  first features:", data["feature_names"][:20].tolist())
   PY
   ```

   Do not treat feature names or schema groups as recommendations. Use the
   inventory only to understand the editable surface and to avoid impossible
   experiments. The scientific meaning and candidate mechanisms must come from
   `literature/`, repository context, and measured results.
6. **Understand the hazard from literature**: inspect `literature/`, then build
   and read discovery context before proposing modeling ideas:

   ```bash
   python rag.py context --broad-literature --auto-web-on-plateau --geoevolve-outside --openalex-crossref
   sed -n '1,220p' artifacts/discovery/context.md
   ```

   Use this to understand the hazard and candidate mechanisms. Do not treat
   `program.md` as the source of hazard-specific hypotheses.
7. **Verify data exists**: confirm that every process used by the active train
   command has `matrix.npz`, `metadata.csv`, and `schema.json` under
   `artifacts/features/<process>/`. If any required artifact is missing, tell
   the human which `prepare.py` command is needed.
8. **Initialize the experiment ledger**: create or reset
   `artifacts/experiments/experiments.tsv` for the fresh run with a header row:

   ```text
   iteration	commit	val_pr_auc	delta	status	process_scores	description
   ```

   Initialize `artifacts/experiments/best_score.txt` only after the first
   baseline run.
9. **First run is the null baseline**: the first experiment must use no
   predictive features. Configure `experiment.py` for a no-feature/null baseline
   before any feature engineering or model tuning. In this repo that means all
   feature families are disabled and `FEATURE_RECIPE["allow_no_features"]` is
   set to `True`, which gives the evaluator only a constant null feature.
10. **Confirm and go**: after setup is confirmed, begin the autonomous loop.

## Experimentation

Each experiment is a single coherent idea implemented by editing
`experiment.py`.

**What you CAN do:**
- Modify only the editable experiment constants in `experiment.py`.
- Change feature selection, model configuration, process weights, and the
  experiment rationale when they serve one coherent hypothesis.
- Use existing code paths, artifacts, and installed dependencies.
- Use `literature/` and `rag.py` to generate new hypotheses, especially after
  near-misses or plateaus.

**What you CANNOT do:**
- Modify `prepare.py`, `train.py`, `predict_map.py`, `run_loop.py`, `src/`,
  `configs/`, `legacy/`, data files, or the evaluation harness during an
  experiment.
- Install new packages or add dependencies.
- Move the goalposts by changing the metric, split, or evaluator.
- Re-run an already tried experiment, even if it has a new rationale.

**Simplicity criterion**: all else being equal, simpler is better. Added
complexity must earn its keep. A small metric gain from a simpler recipe is
valuable; a small gain from an opaque or brittle recipe is suspect. However, the
branch advances only when the primary metric improves by at least `0.001`.

## Output Format

The train command writes artifacts under `artifacts/run_current/` and prints a
final line:

```text
val_pr_auc: <float>
```

Redirect full training output to `run.log`; do not let training output flood the
agent context. Extract the key line after the run:

```bash
grep "^val_pr_auc:" run.log
```

Use `artifacts/run_current/metrics.json` or `artifacts/run_current/summary.md`
to read per-process scores and the exact experiment configuration.

## Logging Results

When an experiment finishes, append one tab-separated row to
`artifacts/experiments/experiments.tsv`.

The TSV has these columns:

```text
iteration	commit	val_pr_auc	delta	status	process_scores	description
```

1. `iteration`: monotonically increasing integer.
2. `commit`: short candidate commit hash, 7 characters.
3. `val_pr_auc`: achieved score, or `0.000000` for crashes.
4. `delta`: `val_pr_auc - previous_best`, or `0.000000` for crashes.
5. `status`: `keep`, `discard`, or `crash`.
6. `process_scores`: compact process-level scores, for example
   `a=0.812345,b=0.845678`; use the active process names.
7. `description`: short text describing exactly what changed. Avoid commas if a
   downstream tool may parse the file loosely.

The ledger is also the duplicate-avoidance memory. It must be inspected before
every candidate is created.

## Duplicate Avoidance

Repetition is the main failure mode. Before editing:

1. Read `artifacts/experiments/experiments.tsv`.
2. Inspect recent accepted commits with `git log --oneline`.
3. Compare the candidate idea against all previous descriptions and the current
   `experiment.py`.
4. If the candidate is equivalent to a previous experiment in feature recipe,
   model configuration, process weighting, or rationale, do not run it.
5. If unsure whether an idea is meaningfully new, assume it is already tried and
   choose a different axis.

Every candidate rationale must contain enough detail to identify the tested
change later. Generic rationales such as "try more interactions" are not
acceptable without the exact changed values and mechanism being tested.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/may13`.

LOOP FOREVER:

1. Look at the git state: current branch, current commit, and any dirty files.
   Save this commit as the candidate's start commit. Never reset behind the
   latest accepted improvement.
2. Read the current best score from `artifacts/experiments/best_score.txt`, if
   it exists.
3. Read the experiment ledger and reject duplicate ideas before editing.
4. Re-read discovery context, rebuilding it with `rag.py` when the ledger shows
   plateaus, near-misses, or exhausted local ideas.
5. Tune `experiment.py` with one novel experimental idea.
6. Commit the candidate edit:

   ```bash
   git add experiment.py
   git commit -m "Candidate experiment: <short-description>"
   ```

7. Run the experiment using the active train command from `README.md` and the
   current repository configuration. Redirect all output:

   ```bash
   python -u train.py <active-args> > run.log 2>&1
   ```

8. Read the result:

   ```bash
   grep "^val_pr_auc:" run.log
   ```

   If the grep output is empty, read the crash:

   ```bash
   tail -n 50 run.log
   ```

9. Record the row in `artifacts/experiments/experiments.tsv`.
10. If the run crashed:
    - Fix and rerun only if the bug is trivial and does not change the idea.
    - Otherwise write down the crash row, reset the branch to the start commit
      for the candidate, re-append the crash row to the ledger, and move on to a
      different idea.
11. If `val_pr_auc >= previous_best + 0.001`:
    - Mark status `keep`.
    - Copy `artifacts/run_current/model_bundle.pkl` to
      `artifacts/models/best_model.pkl`.
    - Update `artifacts/experiments/best_score.txt`.
    - Amend or follow up the candidate commit so the accepted `experiment.py`,
      best-score update, and ledger row are locked together.
    - This accepted commit is now the new starting point. Future experiments
      must build on it.
12. If `val_pr_auc < previous_best + 0.001`:
    - Write down a `discard` row, including near-misses.
    - Reset the branch back to the starting commit for the candidate.
    - Re-append the discard row to the ledger after the reset, but do not keep
      the experiment edit.

The branch must never be reset behind the latest accepted improvement unless
the human explicitly asks for a rewind. Improvements are locked and built upon.

## Timeout

An experiment should finish within the expected train budget for this repo. If a
run exceeds 20 minutes, kill it, mark it as `discard` or `crash` depending on
the evidence in `run.log`, reset the candidate edit, and move on.

Do not simplify a candidate just because it is slower unless it exceeds this
limit or fails to complete.

## Crashes

Crashes are data. If a candidate crashes because of a typo, missing import, or
other trivial implementation mistake, fix the same idea and rerun. If the idea
is structurally incompatible with the evaluator or available data, log `crash`,
reset the candidate, and try a different idea.

Use `0.000000` for `val_pr_auc` and `delta` in crash rows.

## NEVER STOP

Once the experiment loop has begun, do not pause to ask the human whether to
continue. The human may be away and expects autonomous work until interrupted.

If ideas run low, re-read the in-scope files, rebuild discovery context, inspect
near-misses, look for simplifications, and search for genuinely new axes. Do
not repeat prior experiments.
