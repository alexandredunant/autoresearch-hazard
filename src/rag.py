from __future__ import annotations

import argparse
import html
import json
import re
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.checkpoint.memory import MemorySaver
    from geoevolve.geo_knowledge_rag import (
        GeoKnowledgeRAG,
        add_arxiv_papers,
        add_wiki_pages,
        direct_add_document_to_db,
    )
    from geoevolve.llm import get_llm
except Exception:
    Document = None
    Chroma = None
    OpenAIEmbeddings = None
    RecursiveCharacterTextSplitter = None
    MemorySaver = None
    GeoKnowledgeRAG = None
    add_arxiv_papers = None
    add_wiki_pages = None
    direct_add_document_to_db = None
    get_llm = None


DEFAULT_QUERY = (
    "landslide debris flow susceptibility antecedent rainfall lithology "
    "land cover interactions Alps Bolzano"
)

DOMAIN_PROFILES: dict[str, str] = {
    "geology_geomorphology": (
        "landslide debris flow geomorphology lithology structural geology "
        "weathering regolith sediment connectivity hillslope failure Alps"
    ),
    "climate_hydrology": (
        "extreme precipitation antecedent rainfall soil moisture snowmelt "
        "rain on snow hydrological triggering debris flow landslide Alps"
    ),
    "environmental_science": (
        "land use change forest disturbance wildfire erosion soil degradation "
        "vegetation cover environmental drivers slope instability"
    ),
    "biology_ecology": (
        "root reinforcement vegetation ecology forest structure bioengineering "
        "soil stability plant traits slope failure landslide"
    ),
    "geography_remote_sensing": (
        "GIS remote sensing terrain indices susceptibility mapping spatial "
        "autocorrelation landscape metrics landslide debris flow"
    ),
    "civil_geotechnical_engineering": (
        "geotechnical engineering slope stability shear strength pore pressure "
        "infiltration unsaturated soil debris flow landslide hazard"
    ),
    "computer_science_ml": (
        "interpretable machine learning feature interactions causal discovery "
        "spatiotemporal modeling rare event prediction landslide susceptibility"
    ),
    "materials_granular_science": (
        "granular flow rheology soil mechanics particle size friction cohesion "
        "material properties debris flow initiation landslide"
    ),
}

DEFAULT_CONTEXT = [
    {
        "title": "Antecedent rainfall windows",
        "text": (
            "Explore short intensity windows, weekly wetness, monthly wetness, "
            "and normalized rainfall relative to basin climatology."
        ),
        "source": "default_hypothesis",
        "url": "",
        "kind": "default",
        "domain": "climate_hydrology",
    },
    {
        "title": "Process interactions",
        "text": (
            "Explore whether lithology, landcover, slope, or basin morphology "
            "modifies rainfall response differently for slides and flows."
        ),
        "source": "default_hypothesis",
        "url": "",
        "kind": "default",
        "domain": "geology_geomorphology",
    },
]


@dataclass
class Chunk:
    title: str
    text: str
    source: str
    url: str = ""
    kind: str = "local"
    score: float = 0.0
    domain: str = ""


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, *, words: int = 220, overlap: int = 45) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    step = max(1, words - overlap)
    for start in range(0, len(tokens), step):
        part = tokens[start : start + words]
        if len(part) >= 20:
            chunks.append(" ".join(part))
        if start + words >= len(tokens):
            break
    return chunks


def iter_local_documents(literature_dir: Path) -> Iterable[tuple[Path, str]]:
    if not literature_dir.exists():
        return
    for path in sorted(literature_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".rst"}:
            continue
        text = path.read_text(errors="ignore").strip()
        if text:
            yield path, clean_text(text)


def build_local_chunks(literature_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, text in iter_local_documents(literature_dir) or []:
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append(Chunk(
                title=f"{path.name} chunk {i + 1}",
                text=chunk,
                source=str(path),
                kind="local",
                domain="local",
            ))
    if not chunks:
        chunks = [Chunk(**item) for item in DEFAULT_CONTEXT]
    return chunks


def geoevolve_available() -> bool:
    return bool(Document and GeoKnowledgeRAG and direct_add_document_to_db)


def initialize_geoevolve_rag(args: argparse.Namespace) -> tuple[object | None, str | None]:
    if not geoevolve_available():
        return None, "GeoEvolve or LangChain document classes are not importable."
    try:
        if args.geoevolve_backend == "ollama":
            if not all([Chroma, OpenAIEmbeddings, RecursiveCharacterTextSplitter, MemorySaver, get_llm]):
                return None, "Ollama GeoEvolve backend requires langchain_chroma, langchain_openai, langchain_text_splitters, and langgraph."
            rag = GeoKnowledgeRAG.__new__(GeoKnowledgeRAG)
            rag.llm = get_llm(
                model=args.ollama_llm_model,
                base_url=args.ollama_base_url,
                source="Custom",
                api_key=args.ollama_api_key,
                temperature=0.2,
            )
            rag.embeddings = OpenAIEmbeddings(
                model=args.ollama_embedding_model,
                base_url=args.ollama_base_url,
                api_key=args.ollama_api_key,
                check_embedding_ctx_length=False,
            )
            rag.splitter = RecursiveCharacterTextSplitter(
                chunk_size=args.geoevolve_chunk_size,
                chunk_overlap=args.geoevolve_chunk_overlap,
            )
            rag.db = Chroma(
                collection_name=args.geoevolve_collection,
                embedding_function=rag.embeddings,
                persist_directory=args.geoevolve_persist,
            )
            rag.memory = MemorySaver()
            rag.retriever = rag.db.as_retriever(search_kwargs={"k": args.top_k})
            return rag, None
        rag = GeoKnowledgeRAG(
            persist_dir=args.geoevolve_persist,
            rag_embedding_model_name=args.geoevolve_embedding_model,
            rag_llm_model_name=args.geoevolve_llm_model,
            collection_name=args.geoevolve_collection,
            chunk_size=args.geoevolve_chunk_size,
            chunk_overlap=args.geoevolve_chunk_overlap,
        )
        return rag, None
    except Exception as exc:
        return None, f"GeoEvolve initialization failed: {type(exc).__name__}: {exc}"


def add_chunks_to_geoevolve(rag: object, chunks: list[Chunk]) -> str | None:
    try:
        for chunk in chunks:
            direct_add_document_to_db(
                rag=rag,
                knowledge=chunk.text,
                title=chunk.title,
                category=chunk.domain or chunk.kind or "local",
                max_length=4000,
            )
        return None
    except Exception as exc:
        return f"GeoEvolve local indexing failed: {type(exc).__name__}: {exc}"


def retrieve_geoevolve(rag: object, query: str, *, top_k: int) -> tuple[list[Chunk], str | None]:
    try:
        rag.retriever = rag.db.as_retriever(search_kwargs={"k": top_k})
        docs = rag.retriever.invoke(query)
        chunks = []
        for rank, doc in enumerate(docs, start=1):
            meta = dict(getattr(doc, "metadata", {}) or {})
            title = str(meta.get("name") or meta.get("title") or f"GeoEvolve result {rank}")
            source = str(meta.get("source") or meta.get("category") or "geoevolve_chroma")
            chunks.append(Chunk(
                title=title,
                text=clean_text(getattr(doc, "page_content", "")),
                source=source,
                kind="geoevolve",
                domain=str(meta.get("category") or "geoevolve"),
                score=0.0,
            ))
        return chunks, None
    except Exception as exc:
        return [], f"GeoEvolve retrieval failed: {type(exc).__name__}: {exc}"


def fetch_geoevolve_outside_knowledge(
    rag: object,
    query_plan: list[tuple[str, str]],
    *,
    geo_knowledge_dir: Path,
    max_arxiv_papers: int,
) -> list[str]:
    errors = []
    if not (add_wiki_pages and add_arxiv_papers):
        return ["GeoEvolve outside-knowledge helpers are not importable."]
    geo_knowledge_dir.mkdir(parents=True, exist_ok=True)
    for domain, query in query_plan:
        try:
            add_wiki_pages(
                topic=query,
                category=domain,
                rag=rag,
                geo_knowledge_dir=str(geo_knowledge_dir),
            )
        except Exception as exc:
            errors.append(f"GeoEvolve Wikipedia/{domain}: {type(exc).__name__}: {exc}")
        try:
            add_arxiv_papers(
                query=query,
                max_results=max_arxiv_papers,
                category=domain,
                rag=rag,
                geo_knowledge_dir=str(geo_knowledge_dir),
            )
        except Exception as exc:
            errors.append(f"GeoEvolve ArXiv/{domain}: {type(exc).__name__}: {exc}")
    return errors


def load_geoevolve_saved_knowledge(geo_knowledge_dir: Path) -> list[Chunk]:
    chunks = []
    if not geo_knowledge_dir.exists():
        return chunks
    for path in sorted(geo_knowledge_dir.rglob("*.txt")):
        text = clean_text(path.read_text(errors="ignore"))
        if not text:
            continue
        domain = path.parent.name
        for i, part in enumerate(chunk_text(text)):
            chunks.append(Chunk(
                title=f"{path.stem} chunk {i + 1}",
                text=part,
                source=str(path),
                kind="geoevolve_saved",
                domain=domain,
            ))
    return chunks


def write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    if not path.exists():
        return chunks
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        chunks.append(Chunk(**json.loads(line)))
    return chunks


def retrieve(chunks: list[Chunk], query: str, *, top_k: int) -> list[Chunk]:
    if not chunks:
        return []
    corpus = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    matrix = vectorizer.fit_transform(corpus)
    qvec = vectorizer.transform([query])
    scores = cosine_similarity(qvec, matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    out = []
    for idx in order:
        chunk = Chunk(**asdict(chunks[int(idx)]))
        chunk.score = float(scores[int(idx)])
        out.append(chunk)
    return out


def parse_experiments(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["iteration", "val_pr_auc", "status", "rationale"])
    return pd.read_csv(path, sep="\t")


def plateau_status(path: Path, *, window: int = 5, min_delta: float = 1e-6) -> dict:
    df = parse_experiments(path)
    if len(df) <= window:
        return {
            "plateau": False,
            "reason": f"Need more than {window} logged experiments to detect a plateau.",
            "n_experiments": int(len(df)),
        }
    vals = pd.to_numeric(df["val_pr_auc"], errors="coerce")
    statuses = df["status"].fillna("").astype(str)
    previous_best = float(vals.iloc[:-window].max())
    recent_best = float(vals.iloc[-window:].max())
    recent_keeps = int((statuses.iloc[-window:] == "keep").sum())
    plateau = recent_keeps == 0 and recent_best <= previous_best + min_delta
    reason = (
        f"No kept experiments in last {window}; recent_best={recent_best:.6f}, "
        f"previous_best={previous_best:.6f}."
        if plateau else
        f"Recent window still has improvement signal; recent_best={recent_best:.6f}, "
        f"previous_best={previous_best:.6f}, recent_keeps={recent_keeps}."
    )
    return {
        "plateau": bool(plateau),
        "reason": reason,
        "n_experiments": int(len(df)),
        "window": int(window),
        "previous_best": previous_best,
        "recent_best": recent_best,
        "recent_keeps": recent_keeps,
    }


def abstract_from_openalex(inv: dict | None) -> str:
    if not inv:
        return ""
    positions = []
    for word, locs in inv.items():
        for loc in locs:
            positions.append((int(loc), word))
    return " ".join(word for _, word in sorted(positions))


def search_openalex(query: str, *, rows: int, timeout: float, domain: str = "") -> list[Chunk]:
    params = {
        "search": query,
        "filter": "type:article",
        "per-page": rows,
        "sort": "relevance_score:desc",
    }
    url = "https://api.openalex.org/works?" + urlencode(params)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "bolzano-autoresearch-rag/1.0"})
    response.raise_for_status()
    payload = response.json()
    chunks = []
    for item in payload.get("results", []):
        title = clean_text(item.get("display_name") or "Untitled")
        abstract = clean_text(abstract_from_openalex(item.get("abstract_inverted_index")))
        venue = item.get("primary_location", {}).get("source", {}) or {}
        year = item.get("publication_year")
        doi = item.get("doi") or item.get("id") or ""
        text = " ".join(part for part in [
            f"{title}.",
            f"Year: {year}." if year else "",
            f"Venue: {venue.get('display_name')}." if venue.get("display_name") else "",
            abstract,
        ] if part)
        if text.strip():
            chunks.append(Chunk(title=title, text=text, source="OpenAlex", url=doi, kind="web", domain=domain))
    return chunks


def search_crossref(query: str, *, rows: int, timeout: float, domain: str = "") -> list[Chunk]:
    params = {"query": query, "rows": rows, "select": "title,abstract,DOI,URL,published-print,published-online,container-title"}
    url = "https://api.crossref.org/works?" + urlencode(params)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "bolzano-autoresearch-rag/1.0"})
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    chunks = []
    for item in items:
        title = clean_text(" ".join(item.get("title") or ["Untitled"]))
        abstract = clean_text(item.get("abstract") or "")
        venue = clean_text(" ".join(item.get("container-title") or []))
        doi = item.get("DOI")
        url_out = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
        text = " ".join(part for part in [
            f"{title}.",
            f"Venue: {venue}." if venue else "",
            abstract,
        ] if part)
        if text.strip():
            chunks.append(Chunk(title=title, text=text, source="Crossref", url=url_out, kind="web", domain=domain))
    return chunks


def selected_domain_profiles(domains: str | None) -> dict[str, str]:
    if not domains:
        return dict(DOMAIN_PROFILES)
    keys = [item.strip() for item in domains.split(",") if item.strip()]
    unknown = [key for key in keys if key not in DOMAIN_PROFILES]
    if unknown:
        raise KeyError(f"Unknown RAG domain(s): {unknown}. Available: {sorted(DOMAIN_PROFILES)}")
    return {key: DOMAIN_PROFILES[key] for key in keys}


def build_query_plan(
    base_query: str,
    *,
    broad_literature: bool,
    domains: str | None,
) -> list[tuple[str, str]]:
    if not broad_literature:
        return [("core_mass_movement", base_query)]
    return [
        (domain, f"{base_query} {terms}")
        for domain, terms in selected_domain_profiles(domains).items()
    ]


def exploratory_web_search(
    query_plan: list[tuple[str, str]],
    *,
    rows_per_query: int = 4,
    max_results: int = 24,
    timeout: float = 12.0,
) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    errors: list[str] = []
    for domain, query in query_plan:
        for name, fn in (("OpenAlex", search_openalex), ("Crossref", search_crossref)):
            try:
                chunks.extend(fn(query, rows=rows_per_query, timeout=timeout, domain=domain))
            except Exception as exc:
                errors.append(f"{name}/{domain}: {type(exc).__name__}: {exc}")
    seen = set()
    unique = []
    for chunk in chunks:
        key = (chunk.title.lower(), chunk.url.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique[:max_results], errors


def format_context(
    *,
    query: str,
    query_plan: list[tuple[str, str]],
    plateau: dict,
    local: list[Chunk],
    web: list[Chunk],
    web_errors: list[str],
    backend: str,
    backend_notes: list[str],
) -> str:
    lines = [
        "# Discovery Context",
        "",
        f"Query: `{query}`",
        "",
        "## Plateau Status",
        "",
        f"- Plateau detected: `{plateau.get('plateau')}`",
        f"- Reason: {plateau.get('reason')}",
        f"- Retrieval backend: `{backend}`",
        "",
        "## Literature Query Plan",
        "",
    ]
    for domain, planned_query in query_plan:
        lines += [
            f"- `{domain}`: {planned_query}",
        ]
    lines += [
        "",
        "## Local Retrieval",
        "",
    ]
    if backend_notes:
        lines += ["## Retrieval Notes", ""]
        lines.extend(f"- {note}" for note in backend_notes)
        lines.append("")
    for i, chunk in enumerate(local, start=1):
        lines += [
            f"### Local {i}: {chunk.title}",
            f"- Source: `{chunk.source}`",
            f"- Domain: `{chunk.domain or 'local'}`",
            f"- Score: {chunk.score:.4f}",
            "",
            textwrap.shorten(chunk.text, width=1200, placeholder=" ..."),
            "",
        ]
    if web:
        lines += ["## Exploratory Web Search", ""]
        for i, chunk in enumerate(web, start=1):
            lines += [
                f"### Web {i}: {chunk.title}",
                f"- Domain: `{chunk.domain or 'unknown'}`",
                f"- Source: {chunk.source}",
                f"- URL: {chunk.url or '(none)'}",
                "",
                textwrap.shorten(chunk.text, width=1300, placeholder=" ..."),
                "",
            ]
    elif plateau.get("plateau"):
        lines += ["## Exploratory Web Search", "", "No web results were retrieved.", ""]
    if web_errors:
        lines += ["## Web Search Errors", ""]
        lines.extend(f"- {err}" for err in web_errors)
        lines.append("")
    return "\n".join(lines)


def run_context(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    chunks = build_local_chunks(Path(args.literature))
    write_jsonl(index_path, chunks)
    query = args.query or DEFAULT_QUERY
    plateau = plateau_status(Path(args.experiments), window=args.plateau_window)
    query_plan = build_query_plan(
        query,
        broad_literature=args.broad_literature,
        domains=args.domains,
    )
    retrieval_query = " ".join(planned_query for _, planned_query in query_plan)
    backend_notes: list[str] = []
    backend_used = "tfidf"
    rag = None
    if args.backend in {"auto", "geoevolve"}:
        rag, note = initialize_geoevolve_rag(args)
        if note:
            backend_notes.append(note)
            if args.backend == "geoevolve":
                raise RuntimeError(note)
        if rag is not None:
            err = add_chunks_to_geoevolve(rag, chunks)
            if err:
                backend_notes.append(err)
                if args.backend == "geoevolve":
                    raise RuntimeError(err)
            else:
                local, err = retrieve_geoevolve(rag, retrieval_query, top_k=args.top_k)
                if err:
                    backend_notes.append(err)
                    if args.backend == "geoevolve":
                        raise RuntimeError(err)
                elif local:
                    backend_used = "geoevolve"
                else:
                    backend_notes.append("GeoEvolve returned no documents; using TF-IDF fallback.")
    if backend_used != "geoevolve":
        local = retrieve(chunks, retrieval_query, top_k=args.top_k)
    should_search_web = args.web or (args.auto_web_on_plateau and plateau.get("plateau"))
    web: list[Chunk] = []
    errors: list[str] = []
    if should_search_web:
        if rag is not None and args.geoevolve_outside:
            errors.extend(fetch_geoevolve_outside_knowledge(
                rag,
                query_plan,
                geo_knowledge_dir=Path(args.geo_knowledge_dir),
                max_arxiv_papers=args.geoevolve_max_arxiv_papers,
            ))
            saved = load_geoevolve_saved_knowledge(Path(args.geo_knowledge_dir))
            if saved:
                write_jsonl(Path(args.geoevolve_saved_index), saved)
                web.extend(retrieve(saved, retrieval_query, top_k=min(args.web_max_results, len(saved))))
        if args.openalex_crossref:
            extra_web, extra_errors = exploratory_web_search(
                query_plan,
                rows_per_query=args.web_rows_per_domain,
                max_results=args.web_max_results,
                timeout=args.timeout,
            )
            web.extend(extra_web)
            errors.extend(extra_errors)
        deduped_web = []
        seen_web = set()
        for chunk in web:
            key = (chunk.title.lower(), chunk.url.lower(), chunk.source.lower())
            if key in seen_web:
                continue
            seen_web.add(key)
            deduped_web.append(chunk)
        web = deduped_web[:args.web_max_results]
    context = format_context(
        query=query,
        query_plan=query_plan,
        plateau=plateau,
        local=local,
        web=web,
        web_errors=errors,
        backend=backend_used,
        backend_notes=backend_notes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(context, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"plateau: {plateau.get('plateau')}")
    print(f"web_results: {len(web)}")
    print("domains: " + ",".join(domain for domain, _ in query_plan))
    print(f"backend: {backend_used}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GeoEvolve-backed RAG plus plateau-triggered broad literature search.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ctx = sub.add_parser("context", help="Build discovery context for the next experiment.")
    ctx.add_argument("--query", default=None)
    ctx.add_argument("--literature", default="literature")
    ctx.add_argument("--experiments", default="artifacts/experiments/experiments.tsv")
    ctx.add_argument("--index", default="artifacts/rag/local_chunks.jsonl")
    ctx.add_argument("--out", default="artifacts/discovery/context.md")
    ctx.add_argument("--top-k", type=int, default=6)
    ctx.add_argument("--plateau-window", type=int, default=5)
    ctx.add_argument("--backend", choices=["auto", "geoevolve", "tfidf"], default="auto")
    ctx.add_argument("--geoevolve-backend", choices=["default", "ollama"], default="ollama")
    ctx.add_argument("--geoevolve-persist", default="artifacts/geoevolve_storage")
    ctx.add_argument("--geo-knowledge-dir", default="artifacts/geoevolve_knowledge")
    ctx.add_argument("--geoevolve-saved-index", default="artifacts/rag/geoevolve_saved_chunks.jsonl")
    ctx.add_argument("--geoevolve-embedding-model", default="text-embedding-3-small")
    ctx.add_argument("--geoevolve-llm-model", default="gpt-4o-mini")
    ctx.add_argument("--geoevolve-collection", default="bolzano_geo_knowledge")
    ctx.add_argument("--geoevolve-chunk-size", type=int, default=300)
    ctx.add_argument("--geoevolve-chunk-overlap", type=int, default=50)
    ctx.add_argument("--geoevolve-outside", action="store_true", help="Use GeoEvolve Wikipedia/ArXiv fetchers during web search.")
    ctx.add_argument("--geoevolve-max-arxiv-papers", type=int, default=1)
    ctx.add_argument("--ollama-base-url", default="http://localhost:11434/v1")
    ctx.add_argument("--ollama-api-key", default="ollama")
    ctx.add_argument("--ollama-llm-model", default="deepseek-r1:32b")
    ctx.add_argument("--ollama-embedding-model", default="nomic-embed-text")
    ctx.add_argument(
        "--broad-literature",
        action="store_true",
        help="Expand retrieval/web search across geology, climate, environment, biology, geography, engineering, computer science, and materials science.",
    )
    ctx.add_argument(
        "--domains",
        default=None,
        help="Comma-separated subset of broad domains. Use with --broad-literature.",
    )
    ctx.add_argument("--auto-web-on-plateau", action="store_true")
    ctx.add_argument("--web", action="store_true", help="Force exploratory web search.")
    ctx.add_argument("--openalex-crossref", action="store_true", help="Use OpenAlex/Crossref in addition to GeoEvolve outside search.")
    ctx.add_argument("--web-rows-per-domain", type=int, default=3)
    ctx.add_argument("--web-max-results", type=int, default=24)
    ctx.add_argument("--timeout", type=float, default=12.0)
    ctx.set_defaults(func=run_context)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
