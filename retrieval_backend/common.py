"""Shared constants and historical text processing for retrieval."""

from __future__ import annotations

import html
import re
import unicodedata
from functools import lru_cache
from typing import Any

DEFAULT_QDRANT_URL = "http://localhost:6333"

TOP_N = 100
K_MAX = 10000

BM25_COLLECTION = "datasets_bm25"
HARRIER_COLLECTION = "datasets_harrier"
QWEN_COLLECTION = "datasets_qwen"

BM25_TITLE_VECTOR = "bm25_title"
BM25_DESC_VECTOR = "bm25_desc"
HARRIER_TITLE_VECTOR = "dense_title_harrier"
HARRIER_DESC_VECTOR = "dense_desc_harrier"
QWEN_TITLE_VECTOR = "dense_title_qwen"
QWEN_DESC_VECTOR = "dense_desc_qwen"

SPARSE_MODEL_NAME = "Qdrant/bm25"
HARRIER_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = "Retrieve semantically similar text"

MODEL_MAX_TOKENS = 512
CHARS_PER_TOKEN = 4
MAX_BM25_CHARS = 20000
MAX_CELL_CHARS = 500
MAX_CONTENT_TEXT_CHARS = 100_000
MAX_CONTENT_LINES = 25

DATASET_FIELDS = (
    "id",
    "title",
    "description",
    "header",
    "content",
    "metadato_fileName",
    "resource_fileName",
)

BM25_PREPROCESSING: dict[str, Any] = {
    "enabled": True,
    "lowercase": True,
    "strip_html": True,
    "strip_urls": True,
    "remove_accents": False,
    "remove_numbers": False,
    "token_min_len": 2,
    "stopwords": {
        "enabled": True,
        "language": "spanish",
        "backend": "spacy",
        "spacy_model": "es_core_news_lg",
        "path": None,
        "extra": [],
    },
    "morphology": {
        "mode": "lemma",
        "language": "spanish",
        "backend": "spacy",
        "spacy_model": "es_core_news_lg",
    },
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+", re.UNICODE)


def document_dense_preprocess(text: Any) -> str:
    """Historical QdrantIndexer dense document normalization.

    The historical dense indexer preserved the original casing for title and
    description embeddings. It only normalized line breaks/spacing and applied
    model-length clipping later.
    """
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def query_dense_preprocess(text: Any) -> str:
    """Historical OpenDataSearch dense query normalization."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(
        r"([!?])[!?]+",
        lambda match: "".join(sorted(set(match.group(0)), key=match.group(0).index)),
        text,
    )
    text = re.sub(r"#{2,}", "#", text)
    text = re.sub(r"\s{1,10}([.,:;!?\)\]])", r"\1", text)
    text = re.sub(r"(?<=\d)(°)(?=\w)", r"\1 ", text)
    text = re.sub(r"(?<!\d)([.,:;!?)])(?=[^\s.,:;!?)])", r"\1 ", text)
    text = re.sub(r"(\w)([¿¡#(])", r"\1 \2", text)
    text = re.sub(r"([¿¡#(])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clip_for_model(text: Any) -> str:
    """Historical approximate model-token clipping."""
    text = "" if text is None else str(text)
    max_chars = MODEL_MAX_TOKENS * CHARS_PER_TOKEN
    return text[:max_chars] if len(text) > max_chars else text


def clip_for_bm25(text: Any) -> str:
    """Historical sparse-text clipping."""
    text = "" if text is None else str(text)
    return text[:MAX_BM25_CHARS]


def _strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _load_stopwords_from_file(path: str | None) -> set[str]:
    out: set[str] = set()
    if not path:
        return out
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().lower()
            if token and not token.startswith("#"):
                out.add(token)
    return out


@lru_cache(maxsize=16)
def _get_nltk_stopwords(language: str) -> set[str]:
    try:
        from nltk.corpus import stopwords
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("NLTK is required for NLTK stopwords.") from exc
    try:
        return set(stopwords.words(language))
    except LookupError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("NLTK stopwords corpus is missing.") from exc


@lru_cache(maxsize=8)
def _get_spacy_model(model_name: str):
    try:
        import spacy
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(f"spaCy is required for model {model_name!r}.") from exc
    try:
        return spacy.load(model_name, disable=["parser", "ner", "textcat"])
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(f"Could not load spaCy model {model_name!r}.") from exc


@lru_cache(maxsize=16)
def _get_nltk_stemmer(language: str):
    try:
        from nltk.stem.snowball import SnowballStemmer
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("NLTK is required for stemming.") from exc
    return SnowballStemmer(language)


@lru_cache(maxsize=8)
def _get_nltk_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("NLTK is required for lemmatization.") from exc
    return WordNetLemmatizer()


def _resolve_stopwords(text: str, stopword_config: dict[str, Any]) -> str:
    if not stopword_config.get("enabled", False):
        return text

    backend = str(stopword_config.get("backend", "auto")).lower()
    language = str(stopword_config.get("language", "spanish")).lower()
    spacy_model = str(stopword_config.get("spacy_model", "es_core_news_sm")).strip()
    stop_set: set[str] = set()

    if backend in {"auto", "nltk"}:
        try:
            stop_set = _get_nltk_stopwords(language)
        except Exception:
            if backend == "nltk":
                raise

    if not stop_set and backend in {"auto", "spacy"}:
        nlp = _get_spacy_model(spacy_model)
        stop_set = set(getattr(nlp.Defaults, "stop_words", set()))

    stop_set |= {
        str(item).strip().lower()
        for item in stopword_config.get("extra", [])
        if str(item).strip()
    }
    stop_set |= _load_stopwords_from_file(stopword_config.get("path"))

    if not stop_set:
        return text

    tokens = _TOKEN_RE.findall(text)
    return " ".join(token for token in tokens if token and token.lower() not in stop_set)


def _apply_morphology(text: str, morphology_config: dict[str, Any]) -> str:
    mode = str(morphology_config.get("mode", "none")).lower()
    if mode == "none":
        return text

    language = str(morphology_config.get("language", "spanish")).lower()
    backend = str(morphology_config.get("backend", "auto")).lower()
    spacy_model = str(morphology_config.get("spacy_model", "es_core_news_sm")).strip()
    tokens = [token for token in _TOKEN_RE.findall(text) if token]
    if not tokens:
        return ""

    if mode == "stem":
        if backend in {"auto", "nltk"}:
            try:
                stemmer = _get_nltk_stemmer(language)
                return " ".join(stemmer.stem(token) for token in tokens)
            except Exception:
                if backend == "nltk":
                    raise
        raise RuntimeError("No stemming backend is available.")

    if mode == "lemma":
        if backend in {"auto", "spacy"}:
            try:
                nlp = _get_spacy_model(spacy_model)
                doc = nlp(" ".join(tokens))
                return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)
            except Exception:
                if backend == "spacy":
                    raise

        if backend in {"auto", "nltk"} and language == "english":
            lemmatizer = _get_nltk_lemmatizer()
            return " ".join(lemmatizer.lemmatize(token) for token in tokens)

        raise RuntimeError("No lemmatization backend is available.")

    raise ValueError(f"Unsupported morphology mode: {mode}")


def preprocess_bm25_text(text: Any) -> str:
    """Historical BM25 preprocessing used by indexing and lexical search."""
    text = clip_for_bm25(text)
    config = BM25_PREPROCESSING
    if not config.get("enabled", False):
        return text

    text = html.unescape(str(text or ""))
    if config.get("strip_html", True):
        text = _HTML_TAG_RE.sub(" ", text)
    if config.get("strip_urls", True):
        text = _URL_RE.sub(" ", text)

    text = document_dense_preprocess(text)
    if config.get("lowercase", True):
        text = text.lower()
    if config.get("remove_accents", False):
        text = _strip_accents(text)

    tokens = _TOKEN_RE.findall(text)
    if config.get("remove_numbers", False):
        tokens = [token for token in tokens if not token.isdigit()]

    min_len = int(config.get("token_min_len", 1))
    if min_len > 1:
        tokens = [token for token in tokens if len(token) >= min_len]

    text = " ".join(tokens)
    text = _resolve_stopwords(text, config.get("stopwords", {}))
    text = _apply_morphology(text, config.get("morphology", {}))
    text = document_dense_preprocess(text)
    if config.get("lowercase", True):
        text = text.lower()
    return clip_for_bm25(text)


def query_prompt() -> str:
    """Historical prompt prefix passed to SentenceTransformer.encode."""
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: "


def profile_config(profile: str) -> dict[str, str]:
    """Return the collection and vector names for one public profile."""
    if profile == "bm25":
        return {
            "collection": BM25_COLLECTION,
            "title_vector": BM25_TITLE_VECTOR,
            "description_vector": BM25_DESC_VECTOR,
        }
    if profile == "harrier":
        return {
            "collection": HARRIER_COLLECTION,
            "title_vector": HARRIER_TITLE_VECTOR,
            "description_vector": HARRIER_DESC_VECTOR,
            "model": HARRIER_MODEL_NAME,
        }
    if profile == "qwen":
        return {
            "collection": QWEN_COLLECTION,
            "title_vector": QWEN_TITLE_VECTOR,
            "description_vector": QWEN_DESC_VECTOR,
            "model": QWEN_MODEL_NAME,
        }
    raise ValueError(f"Unknown profile: {profile}")
