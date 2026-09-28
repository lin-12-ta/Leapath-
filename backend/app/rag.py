"""Small semantic retrieval index for the curated career corpus."""
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import settings


@dataclass(frozen=True)
class Career:
    title: str
    description: str
    skills: str
    work_style: str
    job_paths: str = ""

CAREER_SOURCE = Path(__file__).with_name("careers.md")


def _load_careers(path: Path = CAREER_SOURCE) -> list[Career]:
    """Load career records from Markdown sections with labeled fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Career knowledge source not found: {path}")

    records: list[Career] = []
    title: str | None = None
    fields: dict[str, str] = {}

    def save_current() -> None:
        if title is None:
            return
        records.append(Career(
            title=title,
            description=fields.get("What they do", ""),
            skills=fields.get("Skills", ""),
            work_style=" ".join(filter(None, (
                fields.get("Who thrives here", ""),
                fields.get("Work environment", ""),
            ))),
            job_paths=fields.get("Outlook", ""),
        ))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            save_current()
            title = heading.group(1).strip()
            fields = {}
            continue
        labeled = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", line)
        if title and labeled:
            fields[labeled.group(1).strip()] = labeled.group(2).strip()

    save_current()
    if not records:
        raise ValueError(f"No career entries found in Markdown source: {path}")
    return records


CAREERS = _load_careers()

EMBEDDING_MODEL = settings.openrouter_embedding_model
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "career_embeddings.json"
_cache_lock = threading.Lock()
EMBEDDING_MAX_ATTEMPTS = 2


def _document_text(career: Career) -> str:
    return (f"Career: {career.title}. Description: {career.description} "
            f"Skills: {career.skills}. Work style: {career.work_style}. "
            f"Career outlook and progression: {career.job_paths}")


def _corpus_fingerprint() -> str:
    content = EMBEDDING_MODEL + "\n" + "\n".join(_document_text(c) for c in CAREERS)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _hashed_embed(text: str) -> list[float]:
    """Offline fallback; OpenRouter semantic embeddings are preferred when configured."""
    dimensions = 384
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    features = tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        vector[value % dimensions] += 1.0 if (value >> 12) & 1 else -1.0
    return _normalize(vector)


def _openrouter_embeddings(inputs: str | list[str]) -> list[list[float]]:
    """Get one or more dense vectors from OpenRouter with transient-error backoff."""
    last_error: Exception | None = None
    for attempt in range(EMBEDDING_MAX_ATTEMPTS):
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBEDDING_MODEL, "input": inputs},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            vectors = [item.get("embedding") for item in sorted(data, key=lambda item: item.get("index", 0))]
            if not vectors or any(not isinstance(vector, list) or not vector for vector in vectors):
                raise ValueError("OpenRouter returned no embeddings")
            dimensions = len(vectors[0])
            if any(len(vector) != dimensions for vector in vectors):
                raise ValueError("OpenRouter returned embeddings with inconsistent dimensions")
            return [_normalize([float(value) for value in vector]) for vector in vectors]
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_error = exc
            retryable = True
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            retryable = status in {408, 409, 425, 429} or status >= 500
        if not retryable or attempt == EMBEDDING_MAX_ATTEMPTS - 1:
            raise last_error
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("OpenRouter embedding attempts exhausted")


def _read_cached_documents(fingerprint: str) -> list[list[float]] | None:
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if payload.get("model") != EMBEDDING_MODEL or payload.get("fingerprint") != fingerprint:
            return None
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(CAREERS):
            return None
        if any(not isinstance(v, list) or not v or not all(isinstance(x, (int, float)) for x in v) for v in vectors):
            return None
        return [_normalize([float(x) for x in vector]) for vector in vectors]
    except (OSError, ValueError, TypeError):
        return None


def _embed_documents_openrouter() -> list[list[float]]:
    fingerprint = _corpus_fingerprint()
    cached = _read_cached_documents(fingerprint)
    if cached is not None:
        print(f"[RAG] loaded {len(cached)} career vectors from cache ({EMBEDDING_MODEL})", flush=True)
        return cached

    vectors = _openrouter_embeddings([_document_text(career) for career in CAREERS])
    if len(vectors) != len(CAREERS):
        raise ValueError("OpenRouter returned an unexpected number of career embeddings")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "model": EMBEDDING_MODEL,
        "fingerprint": fingerprint,
        "titles": [career.title for career in CAREERS],
        "vectors": vectors,
    }), encoding="utf-8")
    temporary.replace(CACHE_PATH)
    print(f"[RAG] embedded {len(vectors)} career documents ({EMBEDDING_MODEL})", flush=True)
    return vectors


def _embed_query_openrouter(query: str) -> list[float]:
    embeddings = _openrouter_embeddings(query)
    if len(embeddings) != 1:
        raise ValueError("OpenRouter returned an unexpected number of query embeddings")
    return embeddings[0]


def _semantic_vectors(query: str) -> tuple[list[float], list[list[float]], str]:
    if not settings.openrouter_api_key:
        docs = [_hashed_embed(_document_text(career)) for career in CAREERS]
        return _hashed_embed(query), docs, "offline_hash"
    with _cache_lock:
        try:
            docs = _embed_documents_openrouter()
            query_vector = _embed_query_openrouter(query)
            if len(query_vector) != len(docs[0]):
                raise ValueError("OpenRouter query/document embedding dimensions do not match")
            return query_vector, docs, "openrouter"
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "unknown")
            print(f"[RAG] OpenRouter embeddings unavailable; using offline vectors (error_type={type(exc).__name__} status={status})", flush=True)
            docs = [_hashed_embed(_document_text(career)) for career in CAREERS]
            return _hashed_embed(query), docs, "offline_hash_fallback"


def retrieve(query: str, k: int = 3) -> list[dict]:
    query_vector, document_vectors, mode = _semantic_vectors(query)
    scored = [
        (sum(a * b for a, b in zip(query_vector, vector)), career)
        for career, vector in zip(CAREERS, document_vectors)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    print(f"[RAG] mode={mode} results={min(k, len(scored))} top_score={scored[0][0]:.4f}" if scored else f"[RAG] mode={mode} results=0", flush=True)
    return [{
        "title": career.title,
        "description": career.description,
        "skills": career.skills,
        "work_style": career.work_style,
        "job_paths": career.job_paths,
        "score": round(score, 4),
    } for score, career in scored[:k]]
