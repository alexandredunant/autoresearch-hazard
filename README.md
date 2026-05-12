# Hazard Autoresearch

This repository contains a Karpathy-autoresearch-style workflow for basin-level
mass-movement susceptibility in Bolzano. The first supported processes are
slides and debris flows.

## Core Workflow

Prepare feature matrices from the current GeoJSON fields:

```bash
python prepare.py --process slides,flows --out artifacts/features --ignore-existing-lags
```

Prepare feature matrices with CERRA lag features:

```bash
python prepare.py --process slides,flows --out artifacts/features --extract-cerra
```

Manually train the current experiment once:

```bash
python -u train.py --process slides,flows --features artifacts/features --out artifacts/run_current
```

The last line is:

```text
val_pr_auc: <float>
```

Run slower temporal and spatial audit fits:

```bash
python -u train.py --process slides,flows --features artifacts/features --out artifacts/run_current --with-audit
```

## RAG And Plateau Search

Build retrieval context from local notes/papers in `literature/`:

```bash
python rag.py context
```

Run local retrieval and automatically trigger exploratory web search when the
recent validation history has plateaued:

```bash
python rag.py context --broad-literature --auto-web-on-plateau --geoevolve-outside --openalex-crossref
```

Force exploratory web search regardless of plateau state:

```bash
python rag.py context --broad-literature --web --geoevolve-outside --openalex-crossref --query "landslide debris flow antecedent rainfall lithology interactions Alps"
```

Broad literature mode expands the query across geology/geomorphology,
climate/hydrology, environmental science, biology/ecology, geography and remote
sensing, civil/geotechnical engineering, computer science/ML, and
materials/granular science.

`rag.py` uses GeoEvolve as the preferred RAG backend. By default it targets a
local Ollama setup with `deepseek-r1:32b` for the LLM and `nomic-embed-text` for
embeddings. DeepSeek is used for reasoning, but RAG still needs a separate
embedding model. If Ollama or the embedding model is unavailable, `rag.py` falls
back to local TF-IDF retrieval and still writes `artifacts/discovery/context.md`.

To use API-hosted GeoEvolve defaults instead, pass `--geoevolve-backend default`
and configure one of GeoEvolve's supported providers, for example
`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`.

The output for opencode is written to:

```text
artifacts/discovery/context.md
```

## Running Autoresearch With opencode

Start opencode in this repo and tell it:

```text
Read program.md and run the autonomous experiment loop. Edit only experiment.py.
```

opencode should then:

1. Read `program.md`.
2. Edit `experiment.py`.
3. Run `python rag.py context --broad-literature --auto-web-on-plateau --geoevolve-outside --openalex-crossref`.
4. Run `python -u train.py --process slides,flows --features artifacts/features --out artifacts/run_current`.
5. Read the final `val_pr_auc`.
6. Keep the edit if it improves `artifacts/experiments/best_score.txt`.
7. Revert `experiment.py` if it does not improve.
8. Commit accepted experiments.

`train.py` emits progress heartbeats on stderr while an EBM fit is still
running. If opencode's command runner reports a timeout before the final
`val_pr_auc` line appears, re-run the same training command with a longer/no
timeout and wait for completion; do not reduce the model or discard the
experiment just because the command runner timed out.

`run_loop.py` remains as an optional helper, but it is not the default workflow.
The intended loop manager is opencode itself.

## Export Map

Export the latest Bolzano basin prediction map from the current best model:

```bash
python predict_map.py --model artifacts/models/best_model.pkl --out artifacts/maps/bolzano_latest_predictions.geojson
```

## Editable Surface

The autonomous agent edits only `experiment.py`. The fixed preparation,
training/evaluation, map export, and utilities live in `prepare.py`, `train.py`,
`predict_map.py`, and `src/`.

The protocol for generated experiments is in `program.md`.

Older scripts from previous project versions are archived under `legacy/` for
reference. They are not part of the active loop.

## Tracking

Accepted experiments are committed by opencode. The small score registry is
tracked in `artifacts/experiments/`. Large local data, feature matrices, model
bundles, and map outputs are ignored by Git.
