# Bolzano Mass-Movement Autoresearch Protocol

You are the autonomous code-editing researcher for this repository. Run the
research loop yourself, Karpathy-autoresearch style: edit one experiment file,
run training, read the score, keep improvements, revert failures, and repeat.

## Goal

Improve `val_pr_auc` for basin-level mass-movement susceptibility in Bolzano,
evaluated jointly on slides and debris flows. Higher is better.

## Files

- `prepare.py`: fixed data preparation and feature-cache generation. Do not edit.
- `train.py`: fixed deterministic evaluator. Do not edit.
- `experiment.py`: the only file you edit.
- `program.md`: your research protocol. Read it before every experiment.
- `artifacts/experiments/best_score.txt`: current best primary score.
- `artifacts/experiments/experiments.tsv`: experiment log.
- `artifacts/models/best_model.pkl`: best model bundle for map export.

## Editable Surface

Edit only the editable constants in `experiment.py`:

- `FEATURE_RECIPE`
- `MODEL_CONFIG`
- `PROCESS_WEIGHTS`
- `EXPERIMENT_RATIONALE`

Do not edit `train.py`, `prepare.py`, `predict_map.py`, `run_loop.py`, `src/`,
`configs/`, `legacy/`, data files, or generated artifacts except for the
explicit score/model updates described below.

## One Experiment Loop

1. Read the current best score:

   ```bash
   cat artifacts/experiments/best_score.txt
   ```

2. Inspect the recent experiment history:

   ```bash
   tail -n 20 artifacts/experiments/experiments.tsv
   ```

3. Build discovery context. This always retrieves local context from
   `literature/`. If validation has plateaued, it also runs exploratory web
   search through scholarly APIs and writes source-backed suggestions:

   ```bash
   python rag.py context --broad-literature --auto-web-on-plateau --geoevolve-outside --openalex-crossref
   ```

   Read:

   ```bash
   sed -n '1,220p' artifacts/discovery/context.md
   ```

   A plateau means no kept/improving experiment in the recent validation window.
   When web results are included, cite the source title or URL in
   `EXPERIMENT_RATIONALE`.

4. Edit only `experiment.py` with one coherent hypothesis.

5. Run the deterministic evaluator:

   ```bash
   python -u train.py --process slides,flows --features artifacts/features --out artifacts/run_current
   ```

   Training can take a long time. Let it run until the final `val_pr_auc` line
   appears. If your command runner reports a timeout or loses patience while
   `train.py` is still active, do not simplify the model, reduce features, or
   mark the experiment failed. Re-run the same command with a longer/no timeout
   and wait for completion.

   The last stdout line is:

   ```text
   val_pr_auc: <float>
   ```

6. Compare the new `val_pr_auc` to `artifacts/experiments/best_score.txt`.

7. If the new score is better:

   ```bash
   cp artifacts/run_current/model_bundle.pkl artifacts/models/best_model.pkl
   printf "<new_score>\n" > artifacts/experiments/best_score.txt
   ```

   Append one TSV row to `artifacts/experiments/experiments.tsv` with:

   ```text
   iteration<TAB>commit<TAB>val_pr_auc<TAB>status<TAB>process_scores<TAB>rationale
   ```

   Use status `keep`, then commit the accepted experiment:

   ```bash
   git add experiment.py artifacts/experiments/best_score.txt artifacts/experiments/experiments.tsv
   git commit -m "Accept experiment: val_pr_auc=<new_score>"
   ```

8. If the new score is not better:

   Append a TSV row with status `discard`, then revert the failed edit:

   ```bash
   git checkout -- experiment.py
   ```

9. Repeat from step 1 until asked to stop.

## Model Rules

- The model family must remain `ExplainableBoostingClassifier`.
- `outer_bags` must be at least 4.
- `learning_rate` must be at most 0.5.
- Every experiment must include a clear `EXPERIMENT_RATIONALE`.
- Prefer interpretable feature changes over broad hyperparameter churn.

## Experiment Strategy

Use one coherent hypothesis per iteration. You may change multiple constants
when they serve the same hypothesis.

Useful directions:

- Test rainfall process windows: event-day, 2-3 day intensity, weekly wetness,
  fortnightly wetness, monthly wetness, and 45-60 day antecedent state.
- Compare cumulative rainfall, maximum daily rainfall, normalized rainfall, and
  rainfall slope/intensification when CERRA lag features are available.
- Use EBM interactions to test whether lithology, landcover, slope, or basin
  morphology modifies rainfall sensitivity.
- Remove feature families that appear noisy or consistently harm validation.
- Adjust process weights only when slides and flows show clearly different
  reliability or scientific priority.

## Discovery/RAG Use

Use `rag.py` for retrieval. It builds a local TF-IDF index over `.md`, `.txt`,
and `.rst` files in `literature/`, retrieves the top passages for landslide and
debris-flow process hypotheses, and writes `artifacts/discovery/context.md`.

Always use broad literature mode:

```bash
python rag.py context --broad-literature --auto-web-on-plateau --geoevolve-outside --openalex-crossref
```

Broad mode searches across these disciplinary lenses:

- geology and geomorphology
- climate and hydrology
- environmental science
- biology and ecology
- geography, GIS, and remote sensing
- civil and geotechnical engineering
- computer science and interpretable machine learning
- materials and granular science

GeoEvolve is the preferred RAG backend. By default it targets local Ollama:
`deepseek-r1:32b` is the LLM and `nomic-embed-text` is the embedding model. If
Ollama or the embedding model is unavailable, `rag.py` falls back to local
TF-IDF retrieval and reports the fallback in `artifacts/discovery/context.md`.

When validation plateaus, the same command performs exploratory web search
through GeoEvolve's Wikipedia/ArXiv fetchers plus OpenAlex and Crossref for each
discipline. Treat web results as hypothesis sources, not proof. Do not invent
citations. If a web suggestion motivates an experiment, mention the source title
or URL in the rationale.

## Audit Runs

The primary keep/discard metric is random-split `val_pr_auc`. Occasionally run
the slower audit command to check robustness:

```bash
python -u train.py --process slides,flows --features artifacts/features --out artifacts/run_current --with-audit
```

Spatial and temporal audit metrics are diagnostic. Do not use them as the
primary keep/discard decision unless the human explicitly changes the protocol.
Audit runs perform extra fits and may be much slower; do not back off solely
because an audit run takes longer than the primary evaluator.
