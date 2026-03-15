import ast
import re
from dataclasses import dataclass


SOURCE_FILES_PATTERN = re.compile(r"Relevant source files:\s*(.+)", re.IGNORECASE)
SOURCE_DOCS_PATTERN = re.compile(r"Relevant source documents:\s*(.+)", re.IGNORECASE)
PAGE_PATTERN = re.compile(r"[?&]page=(\d+)")


@dataclass(frozen=True)
class SourceHints:
    source_files: list[str]
    source_docs: list[str]
    source_pages: list[int]


def parse_source_hints(question_text: str) -> SourceHints:
    source_files: list[str] = []
    source_docs: list[str] = []
    source_pages: list[int] = []

    files_match = SOURCE_FILES_PATTERN.search(question_text)
    if files_match:
        source_files = _parse_hint_values(files_match.group(1))

    docs_match = SOURCE_DOCS_PATTERN.search(question_text)
    if docs_match:
        source_docs = _parse_hint_values(docs_match.group(1))
        for doc in source_docs:
            page_match = PAGE_PATTERN.search(doc)
            if page_match:
                source_pages.append(int(page_match.group(1)))

    return SourceHints(
        source_files=source_files,
        source_docs=source_docs,
        source_pages=source_pages,
    )


def _parse_hint_values(raw_value: str) -> list[str]:
    text = raw_value.strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip().strip("'\"")]
