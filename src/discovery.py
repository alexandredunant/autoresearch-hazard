from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_HYPOTHESES = [
    {
        "hypothesis": "Antecedent rainfall over weekly to monthly windows may improve susceptibility beyond event-day rainfall.",
        "feature_families": ["cum_norm", "max_norm"],
        "evidence": "exploratory",
    },
    {
        "hypothesis": "Rapid rainfall intensification before the event may separate debris-flow triggering from slide triggering.",
        "feature_families": ["slope"],
        "evidence": "exploratory",
    },
    {
        "hypothesis": "Lithology and landcover modify rainfall sensitivity, so EBM interaction terms may be useful.",
        "feature_families": ["categorical", "cum_norm", "max_norm"],
        "evidence": "exploratory",
    },
    {
        "hypothesis": "Seasonality can proxy snowmelt, vegetation state, and reporting/trigger regimes.",
        "feature_families": ["seasonality"],
        "evidence": "exploratory",
    },
]


def iter_text_sources(literature_dir: Path):
    for path in sorted(literature_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(errors="ignore").strip()
        if text:
            yield path, text


def build_discovery_context(
    *,
    literature_dir: Path = Path("literature"),
    hypotheses_path: Path = Path("artifacts/discovery/hypotheses.jsonl"),
    rag_context_path: Path = Path("artifacts/discovery/context.md"),
    max_chars: int = 6000,
) -> str:
    chunks: list[str] = []
    if rag_context_path.exists():
        text = rag_context_path.read_text(errors="ignore").strip()
        if text:
            chunks.append("## Retrieved RAG Context\n" + text[:max_chars])
    if hypotheses_path.exists():
        recent = hypotheses_path.read_text(errors="ignore").splitlines()[-20:]
        if recent:
            chunks.append("## Recorded Process Hypotheses\n" + "\n".join(recent))
    if literature_dir.exists():
        snippets = []
        for path, text in iter_text_sources(literature_dir):
            snippets.append(f"### {path}\n{text[:1200]}")
            if sum(len(s) for s in snippets) > max_chars:
                break
        if snippets:
            chunks.append("## Local Literature/Notes\n" + "\n\n".join(snippets))
    if not chunks:
        chunks.append("## Default Exploratory Hypotheses\n" + "\n".join(json.dumps(x) for x in DEFAULT_HYPOTHESES))
    return "\n\n".join(chunks)[:max_chars]


def initialize_hypotheses(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8") as handle:
        for item in DEFAULT_HYPOTHESES:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/discovery/hypotheses.jsonl")
    args = parser.parse_args()
    initialize_hypotheses(Path(args.out))
    print(f"initialized {args.out}")


if __name__ == "__main__":
    main()
