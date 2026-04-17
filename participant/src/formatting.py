import re

from models import SolverResult


NUMBER_TOKEN_PATTERN = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
ORDINAL_WORDS = ("first", "second", "third", "fourth", "fifth")

# Patterns for detecting expected answer units from the question text.
# Order matters: more specific patterns first.
_UNIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:in|report\b.*?\bin)\s+trillions?\s+of\b", re.I), "trillion"),
    (re.compile(r"(?:in|report\b.*?\bin)\s+billions?\s+of\b", re.I), "billion"),
    (re.compile(r"(?:in|report\b.*?\bin)\s+millions?\s+of\b", re.I), "million"),
    (re.compile(r"(?:in|report\b.*?\bin)\s+thousands?\s+of\b", re.I), "thousand"),
    (re.compile(r"report\b.*?\bmillions?\s+of\s+dollars", re.I), "million"),
    (re.compile(r"report\b.*?\bbillions?\s+of\s+dollars", re.I), "billion"),
]


def _extract_expected_unit(question: str) -> str | None:
    """Return the unit keyword the question says the answer should be in, or None."""
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(question):
            return unit
    return None


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _strip_markdown(text: str) -> str:
    stripped = re.sub(r"</?[^>]+>", "", text)
    stripped = stripped.replace("**", "").replace("__", "").replace("`", "")
    stripped = stripped.replace("$", "")
    stripped = stripped.replace("\\[", "").replace("\\]", "")
    stripped = re.sub(r"\s+", " ", stripped.replace("\n", "\n")).strip()
    return stripped


def _expects_list(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in (
            "square brackets",
            "enclosed brackets",
            "inside square brackets",
            "report your answer as [",
            "output the answers in format [",
            "comma-separated values enclosed in brackets",
            "comma separated list",
            "comma-separated list",
            "as a list [",
            "as a list starting",
            "return as [",
        )
    )


def _expected_list_count(question: str) -> int | None:
    lowered = question.lower()
    explicit = re.search(r"containing\s+(\d+)\s+numbers", lowered)
    if explicit:
        return int(explicit.group(1))
    if "slope and intercept" in lowered:
        return 2
    if "first value as the slope" in lowered:
        return 3
    if "increased or decreased" in lowered:
        return 3
    ordinal_count = sum(1 for word in ORDINAL_WORDS if word in lowered)
    if ordinal_count >= 2:
        return ordinal_count
    if "and what percent" in lowered:
        return 2
    return None


def _question_wants_percent_sign(question: str) -> bool:
    lowered = question.lower()
    if "as a decimal" in lowered or "decimal value" in lowered or "no percentage sign" in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "percent value",
            "as a percentage",
            "reported as a percent value",
            "expressed as a percent",
            "%",
        )
    )


def _extract_numeric_tokens(text: str, *, keep_years: bool = False) -> list[str]:
    tokens: list[str] = []
    for match in NUMBER_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        normalized = token.replace("$", "")
        bare = normalized.rstrip("%").replace(",", "")
        if not keep_years:
            try:
                value = float(bare)
            except ValueError:
                value = None
            if value is not None and value.is_integer() and 1900 <= value <= 2100:
                continue
        tokens.append(normalized)
    return tokens


def _extract_list_numeric_tokens(text: str) -> list[str]:
    """Extract numbers from list-formatted text, splitting on commas first.

    This avoids the regex greedily fusing comma-separated values like
    ``[10102000000,4.73]`` into a single token.  Years are kept because
    list answers may legitimately contain year values.
    """
    bracket_match = re.search(r"\[([^\]]*)\]", text)
    inner = bracket_match.group(1) if bracket_match else text

    # Prefer ', ' split (preserves thousand-separator commas within numbers).
    # Fall back to ',' when there are no spaces after commas.
    if ", " in inner:
        parts = inner.split(", ")
    else:
        parts = inner.split(",")

    tokens: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = NUMBER_TOKEN_PATTERN.search(part)
        if match:
            tokens.append(match.group(0).replace("$", ""))
    return tokens


def _extract_direction_token(text: str) -> str | None:
    match = re.search(r"\b(Increased|Decreased)\b", text, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).lower()
    return token.capitalize()


def _normalize_numeric_token(token: str, *, keep_percent: bool) -> str:
    stripped = token.replace("$", "").replace(",", "").strip()
    has_percent = stripped.endswith("%")
    number_text = stripped.rstrip("%")
    try:
        value = float(number_text)
    except ValueError:
        return stripped
    if value.is_integer():
        rendered = str(int(value))
    else:
        rendered = number_text
    if keep_percent and has_percent:
        return f"{rendered}%"
    return rendered


def _candidate_lines(text: str) -> list[str]:
    raw_lines = [line.strip(" -*") for line in text.splitlines()]
    return [line for line in raw_lines if line.strip()]


def _select_scalar_candidate(question: str, final_answer: str) -> str:
    lowered_question = question.lower()
    lines = _candidate_lines(final_answer)
    keyword_priority = [
        "final",
        "therefore",
        "thus",
        "answer",
        "difference",
        "change",
        "value",
        "amount",
        "norm",
        "forecast error",
        "z-score",
        "range",
        "volatility",
    ]
    for keyword in keyword_priority:
        for line in reversed(lines):
            if keyword in line.lower() and _extract_numeric_tokens(line):
                return line
    for line in reversed(lines):
        if _extract_numeric_tokens(line):
            return line
    if any(token in lowered_question for token in ("difference", "change", "norm")):
        sentences = re.split(r"(?<=[.!?])\s+", final_answer)
        for sentence in reversed(sentences):
            if _extract_numeric_tokens(sentence):
                return sentence
    return final_answer


def _maybe_append_unit(answer: str, question: str) -> str:
    """Benchmark expects bare numbers without unit suffixes — return as-is."""
    return answer


def canonicalize_final_answer(question: str, final_answer: str) -> str:
    cleaned = _strip_markdown(final_answer)
    if not cleaned:
        return final_answer.strip()

    is_list = _expects_list(question)
    # Also treat as list if the model returned bracket-enclosed content with commas
    if not is_list and re.search(r"\[.+,.+\]", cleaned):
        is_list = True

    if is_list:
        raw_numeric_tokens = _extract_list_numeric_tokens(cleaned)
        keep_percent = _question_wants_percent_sign(question) or any(token.endswith("%") for token in raw_numeric_tokens)
        numeric_tokens = [_normalize_numeric_token(token, keep_percent=keep_percent) for token in raw_numeric_tokens]
        count = _expected_list_count(question)
        items: list[str] = []
        if count is not None:
            items.extend(numeric_tokens[:count])
        else:
            items.extend(numeric_tokens)
        if "increased or decreased" in question.lower():
            direction = _extract_direction_token(cleaned)
            if direction is not None:
                items = items[:2] + [direction]
        if items:
            return "[" + ", ".join(items) + "]"

    keep_percent = _question_wants_percent_sign(question)
    candidate = _select_scalar_candidate(question, cleaned)
    tokens = _extract_numeric_tokens(candidate)
    if not tokens and candidate != cleaned:
        tokens = _extract_numeric_tokens(cleaned)
    if not tokens:
        return _maybe_append_unit(cleaned, question)

    if len(tokens) == 1:
        result = _normalize_numeric_token(tokens[0], keep_percent=keep_percent)
        return _maybe_append_unit(result, question)

    lowered_candidate = candidate.lower()
    if "about" in lowered_candidate or "(" in lowered_candidate:
        result = _normalize_numeric_token(tokens[0], keep_percent=keep_percent)
        return _maybe_append_unit(result, question)

    if any(token in question.lower() for token in ("norm", "forecast error", "z-score", "volatility", "range")):
        result = _normalize_numeric_token(tokens[-1], keep_percent=keep_percent)
        return _maybe_append_unit(result, question)

    result = _normalize_numeric_token(tokens[0], keep_percent=keep_percent)
    return _maybe_append_unit(result, question)


_ANSWER_PHRASE_RE = re.compile(
    r"(?:(?:the\s+)?(?:final\s+)?answer\s+is|total\s+(?:is|=|:)|therefore|thus)[:\s]*"
    r"(-?\$?[\d,]+\.?\d*%?)",
    re.IGNORECASE,
)

_LAST_NUMBER_RE = re.compile(r"-?\$?[\d,]+\.?\d*%?")


def _is_year(token: str) -> bool:
    bare = token.replace(",", "").rstrip("%").replace(".", "").lstrip("-")
    if not bare.isdigit():
        return False
    try:
        return 1900 <= int(bare) <= 2100
    except ValueError:
        return False


def _fallback_extract_answer(text: str) -> str:
    """Try to pull a numeric answer from free text when FINAL_ANSWER tag is missing."""
    m = _ANSWER_PHRASE_RE.search(text)
    if m:
        return m.group(1).replace("$", "")
    numbers = [n.replace("$", "") for n in _LAST_NUMBER_RE.findall(text) if not _is_year(n)]
    if numbers:
        return numbers[-1]
    return ""


def ensure_structured_response(raw_response: str) -> tuple[str, str]:
    reasoning = extract_tag(raw_response, "REASONING")
    final_answer = extract_tag(raw_response, "FINAL_ANSWER")

    cleaned = raw_response.strip()
    if not final_answer:
        if reasoning:
            final_answer = _fallback_extract_answer(reasoning)
        if not final_answer:
            final_answer = _fallback_extract_answer(cleaned)
        if not final_answer:
            final_answer = cleaned or "Unable to determine"
    if not reasoning:
        reasoning = "Derived from the solver pipeline output."
    return reasoning, final_answer


def render_solver_result(result: SolverResult) -> str:
    reasoning = result.reasoning.strip() or "Derived from the solver pipeline output."
    final_answer = result.final_answer.strip() or "Unable to determine"
    return (
        "<REASONING>\n"
        f"{reasoning}\n"
        "</REASONING>\n"
        "<FINAL_ANSWER>\n"
        f"{final_answer}\n"
        "</FINAL_ANSWER>"
    )
