import json
import re
from pathlib import Path

from models import RetrievedContext


TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _page_id_candidates(page_number_hint: int) -> list[int]:
    return [max(0, page_number_hint - 1), page_number_hint, page_number_hint + 1]


def _question_tokens(question: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(question)}


def _score_page(question: str, content: str, page_id: int, candidate_pages: set[int]) -> float:
    lowered_question = question.lower()
    lowered_content = content.lower()
    score = 0.0
    score += len(_question_tokens(question) & _question_tokens(content)) * 2.0
    score += len(set(YEAR_PATTERN.findall(question)) & set(YEAR_PATTERN.findall(content))) * 8.0
    score += 5.0 if "<table>" in lowered_content else 0.0
    score += 5.0 if "national defense" in lowered_content and "national defense" in lowered_question else 0.0
    score += 4.0 if "calendar year" in lowered_content and "calendar year" in lowered_question else 0.0
    score += 4.0 if "cash outgo" in lowered_content and "expenditures" in lowered_question else 0.0
    score += 25.0 if page_id in candidate_pages else 0.0
    return score


def load_page_contexts(
    json_dir: Path | None,
    source_files: list[str],
    source_pages: list[int],
    question: str,
    top_k: int = 5,
) -> list[RetrievedContext]:
    if json_dir is None or not source_files:
        return []

    candidate_pages = set()
    for page in source_pages:
        candidate_pages.update(_page_id_candidates(page))

    scored_contexts: list[RetrievedContext] = []
    for source_file in source_files:
        json_path = json_dir / source_file.replace(".txt", ".json")
        if not json_path.exists():
            continue

        with json_path.open() as handle:
            data = json.load(handle)

        elements = data.get("document", {}).get("elements", [])
        page_snippets: dict[int, list[str]] = {}
        for element in elements:
            content = (element.get("content") or "").strip()
            if not content:
                continue
            for bbox in element.get("bbox") or []:
                page_id = bbox.get("page_id")
                page_snippets.setdefault(page_id, []).append(content)
                break

        for page_id, snippets in sorted(page_snippets.items()):
            content = "\n\n".join(snippets[:8])[:8000]
            score = _score_page(question, content, page_id, candidate_pages)
            if score <= 0:
                continue
            scored_contexts.append(
                RetrievedContext(
                    source=f"{json_path.name}#page_id={page_id}",
                    content=content,
                    score=1000.0 + score,
                )
            )
    scored_contexts.sort(key=lambda context: context.score, reverse=True)
    return scored_contexts[:top_k]
