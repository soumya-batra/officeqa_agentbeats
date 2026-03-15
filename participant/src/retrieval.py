import re
from pathlib import Path

from corpus_loader import CorpusDocument, load_corpus
from models import RetrievedContext


TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
TABLE_ROW_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)
STOPWORDS = {
    "what",
    "were",
    "was",
    "the",
    "for",
    "and",
    "from",
    "with",
    "that",
    "this",
    "into",
    "millions",
    "million",
    "dollars",
    "nominal",
    "calendar",
    "year",
}
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 600


def _question_tokens(question: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(question)
        if match.group(0).lower() not in STOPWORDS
    }


def _iter_chunks(content: str) -> list[str]:
    if not content:
        return []

    chunks: list[str] = []
    table_blocks = [match.group(0) for match in TABLE_ROW_PATTERN.finditer(content)]
    if table_blocks:
        chunks.extend(_expand_table_blocks(content))

    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(content), step):
        chunk = content[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = f"{chunk[:250]}::{chunk[-250:]}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _expand_table_blocks(content: str) -> list[str]:
    lines = content.splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        start = max(0, index - 4)
        end = index
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1
        block = "\n".join(lines[start : min(len(lines), end + 4)]).strip()
        if block:
            blocks.append(block)
        index = end
    return blocks


def _score_text(question: str, text: str, *, source_name: str = "", hint_bonus: float = 0.0) -> float:
    years = set(YEAR_PATTERN.findall(question))
    tokens = _question_tokens(question)
    text_tokens = _question_tokens(text)
    text_years = set(YEAR_PATTERN.findall(text))
    lowered = text.lower()

    score = 0.0
    score += len(tokens & text_tokens) * 3.0
    score += len(years & text_years) * 10.0
    score += 4.0 if "|" in text else 0.0
    score += 3.0 if "<table>" in lowered else 0.0
    score += 5.0 if "national defense" in lowered and "national defense" in question.lower() else 0.0
    score += 3.0 if "calendar year" in lowered and "calendar year" in question.lower() else 0.0
    score += 3.0 if "cash outgo" in lowered and "expenditures" in question.lower() else 0.0
    score += hint_bonus
    if source_name:
        score += 1.0
    return score


class Retriever:
    def __init__(self, corpus_dir, top_k: int = 3):
        self._documents = load_corpus(corpus_dir)
        self._top_k = top_k
        self._documents_by_name = {document.path.name: document for document in self._documents}

    def retrieve(self, question: str) -> list[RetrievedContext]:
        if not self._documents:
            return []

        years = set(YEAR_PATTERN.findall(question))
        tokens = _question_tokens(question)
        scored: list[tuple[float, CorpusDocument]] = []

        for document in self._documents:
            year_score = len(years & document.years) * 10
            token_score = len(tokens & document.tokens)
            score = float(year_score + token_score)
            if score <= 0:
                continue
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        candidate_documents = [document for _, document in scored[: max(self._top_k * 3, 5)]]
        return self._best_chunks(question, candidate_documents, per_document_limit=2)

    def retrieve_by_source_files(self, source_files: list[str], question: str) -> list[RetrievedContext]:
        documents: list[CorpusDocument] = []
        for source_file in source_files:
            document = self._documents_by_name.get(Path(source_file).name)
            if document is not None:
                documents.append(document)
        return self._best_chunks(question, documents, per_document_limit=4, hint_bonus=1000.0)

    def _best_chunks(
        self,
        question: str,
        documents: list[CorpusDocument],
        *,
        per_document_limit: int,
        hint_bonus: float = 0.0,
    ) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        for document in documents:
            chunk_scores: list[tuple[float, str]] = []
            for chunk in _iter_chunks(document.content):
                score = _score_text(question, chunk, source_name=document.path.name, hint_bonus=hint_bonus)
                if score <= 0:
                    continue
                chunk_scores.append((score, chunk))
            chunk_scores.sort(key=lambda item: item[0], reverse=True)
            for score, chunk in chunk_scores[:per_document_limit]:
                contexts.append(
                    RetrievedContext(
                        source=str(document.path),
                        content=chunk[:8000],
                        score=score,
                    )
                )
        contexts.sort(key=lambda context: context.score, reverse=True)
        return contexts[: max(self._top_k + 2, per_document_limit)]
