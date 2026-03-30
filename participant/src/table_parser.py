import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser

from models import RetrievedContext
from normalize import parse_number


CELL_SPLIT_PATTERN = re.compile(r"\s{2,}|\t+|\s+\|\s+")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
STOPWORDS = {
    "what",
    "was",
    "were",
    "the",
    "for",
    "and",
    "from",
    "between",
    "during",
    "total",
    "value",
    "amount",
    "percent",
    "change",
}
HEADER_LABELS = {
    "fiscal year or month",
    "calendar year or month",
    "calendar year",
    "fiscal year",
    "month",
}


class HtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
            self._in_cell = True
            attr_map = dict(attrs)
            try:
                self._colspan = max(1, int(attr_map.get("colspan", "1")))
            except ValueError:
                self._colspan = 1

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._in_cell:
            cell_value = unescape("".join(self._cell)).strip()
            for _ in range(self._colspan):
                self._row.append(cell_value)
            self._in_cell = False
            self._colspan = 1
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


@dataclass(frozen=True)
class ParsedTableRow:
    label: str
    values_by_year: dict[str, float] = field(default_factory=dict)
    source: str = ""


def _clean_label(label: str) -> str:
    return label.strip().strip("|").strip()


def _split_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in CELL_SPLIT_PATTERN.split(line.strip()) if cell.strip()]
    return cells if len(cells) > 1 else []


def _question_keywords(question: str) -> set[str]:
    exclusion_tokens = _question_exclusion_keywords(question)
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z]{3,}", question)
        if token.lower() not in STOPWORDS and token.lower() not in exclusion_tokens
    }


def _question_exclusion_keywords(question: str) -> set[str]:
    patterns = [
        r"excluding\s+([^.;,\n]+)",
        r"exclude\s+([^.;,\n]+)",
        r"without\s+([^.;,\n]+)",
        r"should(?:\s+not|n't|n’t)\s+contain\s+([^.;,\n]+)",
        r"should(?:\s+not|n't|n’t)\s+include\s+([^.;,\n]+)",
        r"do(?:\s+not|n't|n’t)\s+include\s+([^.;,\n]+)",
    ]
    tokens: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            tokens.update(
                token.lower()
                for token in re.findall(r"[a-zA-Z]{3,}", match.group(1))
                if token.lower() not in STOPWORDS
            )
    return tokens


def _extract_rows(text: str) -> list[ParsedTableRow]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    rows: list[ParsedTableRow] = []

    if "<table>" in text.lower():
        parser = HtmlTableParser()
        parser.feed(text)
        rows.extend(_extract_matrix_rows(parser.rows, source=""))

    rows.extend(_extract_pipe_rows(lines))

    for index, line in enumerate(lines):
        header_cells = _split_cells(line)
        if not header_cells:
            continue
        header_years = [cell for cell in header_cells if YEAR_PATTERN.fullmatch(cell)]
        if len(header_years) < 2:
            continue

        for candidate in lines[index + 1 : index + 8]:
            cells = _split_cells(candidate)
            if len(cells) < len(header_years) + 1:
                continue
            label = cells[0]
            label = _clean_label(label)
            values: dict[str, float] = {}
            for year, cell in zip(header_years, cells[1:], strict=False):
                number = parse_number(cell)
                if number is not None:
                    values[year] = number
            if values:
                rows.append(ParsedTableRow(label=label, values_by_year=values, source="",))
    return rows


def _extract_pipe_rows(lines: list[str]) -> list[ParsedTableRow]:
    table_blocks: list[list[list[str]]] = []
    current_block: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if any(cell.strip("-:") for cell in cells):
                current_block.append(cells)
            continue
        if current_block:
            table_blocks.append(current_block)
            current_block = []
    if current_block:
        table_blocks.append(current_block)

    parsed: list[ParsedTableRow] = []
    for block in table_blocks:
        parsed.extend(_extract_matrix_rows(block, source=""))
    return parsed


def _extract_matrix_rows(table_rows: list[list[str]], *, source: str) -> list[ParsedTableRow]:
    if not table_rows:
        return []

    header_years: list[str] = []
    for row in table_rows[:4]:
        years = [cell for cell in row if YEAR_PATTERN.fullmatch(cell)]
        if len(years) > len(header_years):
            header_years = years

    parsed: list[ParsedTableRow] = []
    if header_years:
        for row in table_rows[1:]:
            if len(row) < 2:
                continue
            label = row[0].strip()
            label = _clean_label(label)
            if _is_header_label(label):
                continue
            values_by_year: dict[str, float] = {}
            numeric_cells = [parse_number(cell) for cell in row[1:]]
            numeric_values = [value for value in numeric_cells if value is not None]
            if len(numeric_values) < len(header_years):
                continue
            for year, value in zip(header_years, numeric_values[-len(header_years):], strict=False):
                values_by_year[year] = value
            if values_by_year:
                parsed.append(ParsedTableRow(label=label, values_by_year=values_by_year, source=source))

    if table_rows:
        parsed.extend(_extract_column_series(table_rows, source=source))
    return parsed


def _extract_column_series(table_rows: list[list[str]], *, source: str) -> list[ParsedTableRow]:
    header = table_rows[0]
    if len(header) < 3:
        return []
    if _has_month_series_rows(table_rows[1:]):
        return _extract_month_series(table_rows, source=source)
    year_rows = [row for row in table_rows[1:] if row and YEAR_PATTERN.fullmatch(row[0].strip())]
    if len(year_rows) < 2:
        return []

    parsed: list[ParsedTableRow] = []
    for column_index, label in enumerate(header[1:], start=1):
        clean_label = label.strip()
        clean_label = _clean_label(clean_label)
        if _is_header_label(clean_label) or not clean_label:
            continue
        values_by_year: dict[str, float] = {}
        for row in year_rows:
            if column_index >= len(row):
                continue
            value = parse_number(row[column_index])
            if value is None:
                continue
            values_by_year[row[0].strip()] = value
        if values_by_year:
            parsed.append(ParsedTableRow(label=clean_label, values_by_year=values_by_year, source=source))
    return parsed


def _has_month_series_rows(rows: list[list[str]]) -> bool:
    for row in rows:
        if not row:
            continue
        label = row[0].strip()
        if _normalize_month(label) is not None:
            return True
        if re.search(r"((?:19|20)\d{2})\s*[-/ ]\s*([A-Za-z]+)", label) or re.search(r"([A-Za-z]+)\s+((?:19|20)\d{2})", label):
            return True
    return False


def _extract_month_series(table_rows: list[list[str]], *, source: str) -> list[ParsedTableRow]:
    header = table_rows[0]
    if len(header) < 2:
        return []

    parsed: list[ParsedTableRow] = []
    current_year: str | None = None
    period_rows: list[tuple[str, list[str]]] = []
    first_month_like_index = None
    year_only_before_month = 0
    for index, row in enumerate(table_rows[1:]):
        if not row:
            continue
        label = row[0].strip()
        if _normalize_month(label) is not None or re.search(r"((?:19|20)\d{2})\s*[-/ ]\s*([A-Za-z]+)", label) or re.search(
            r"([A-Za-z]+)\s+((?:19|20)\d{2})",
            label,
        ):
            first_month_like_index = index
            break
        if YEAR_PATTERN.fullmatch(label):
            year_only_before_month += 1
    single_leading_year = first_month_like_index is not None and year_only_before_month == 1

    for index, row in enumerate(table_rows[1:]):
        if not row:
            continue
        first = row[0].strip()
        period_key, current_year = _parse_period_key(first, current_year, treat_year_as_january=single_leading_year and index == 0)
        if period_key is None:
            continue
        period_rows.append((period_key, row))

    if len(period_rows) < 2:
        return []

    for column_index, label in enumerate(header[1:], start=1):
        clean_label = label.strip()
        if _is_header_label(clean_label) or not clean_label:
            continue
        values_by_year: dict[str, float] = {}
        for period_key, row in period_rows:
            if column_index >= len(row):
                continue
            value = parse_number(row[column_index])
            if value is None:
                continue
            values_by_year[period_key] = value
        if values_by_year:
            parsed.append(ParsedTableRow(label=clean_label, values_by_year=values_by_year, source=source))
    return parsed


def _parse_period_key(label: str, current_year: str | None, *, treat_year_as_january: bool = False) -> tuple[str | None, str | None]:
    stripped = label.strip()
    if not stripped:
        return None, current_year

    if YEAR_PATTERN.fullmatch(stripped):
        return (f"{stripped}-01" if treat_year_as_january else stripped), stripped

    explicit = re.search(r"((?:19|20)\d{2})\s*[-/ ]\s*([A-Za-z]+)", stripped)
    if explicit:
        month = _normalize_month(explicit.group(2))
        if month is None:
            return None, current_year
        year = explicit.group(1)
        return f"{year}-{_month_number(month):02d}", year

    reverse = re.search(r"([A-Za-z]+)\s+((?:19|20)\d{2})", stripped)
    if reverse:
        month = _normalize_month(reverse.group(1))
        if month is None:
            return None, current_year
        year = reverse.group(2)
        return f"{year}-{_month_number(month):02d}", year

    month = _normalize_month(stripped)
    if month is not None and current_year is not None:
        return f"{current_year}-{_month_number(month):02d}", current_year
    return None, current_year


def _month_number(month: str) -> int:
    month_order = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    return month_order[month]


def find_calendar_year_total(question: str, contexts: list[RetrievedContext]) -> tuple[float, str] | None:
    lowered = question.lower()
    if "calendar year" not in lowered and "calendar months" not in lowered and "individual calendar months" not in lowered:
        return None

    years = YEAR_PATTERN.findall(question)
    if len(years) != 1:
        return None
    target_year = years[0]
    keywords = _question_keywords(question)

    for context in contexts:
        for rows in _extract_tables(context.content):
            if len(rows) < 3:
                continue
            result = _find_calendar_year_total_in_rows(rows, target_year, keywords)
            if result is not None:
                return result

    return None


def _extract_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    if "<table>" in text.lower():
        parser = HtmlTableParser()
        parser.feed(text)
        if parser.rows:
            tables.append(parser.rows)

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    current_block: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if any(cell.strip("-:") for cell in cells):
                current_block.append(cells)
            continue
        if current_block:
            tables.append(current_block)
            current_block = []
    if current_block:
        tables.append(current_block)
    return tables


def _find_calendar_year_total_in_rows(
    rows: list[list[str]],
    target_year: str,
    keywords: set[str],
) -> tuple[float, str] | None:
    year_header = rows[0]
    month_header = rows[1] if len(rows) > 1 else []
    explicit_indices = [
        index
        for index, (year, month) in enumerate(zip(year_header, month_header, strict=False))
        if year == target_year and _is_month(month)
    ]
    rolling_month_indices = _rolling_calendar_indices(month_header)

    for row in rows[2:]:
        if not row:
            continue
        label = row[0].strip()
        if _is_header_label(label):
            continue
        overlap = len(keywords & _question_keywords(label))
        if overlap < 2:
            continue

        if len(explicit_indices) >= 12:
            values = _extract_row_values(row, explicit_indices)
            if len(values) >= 12:
                return sum(values), f"Summed {target_year} monthly values from row '{label}'"

        if rolling_month_indices is not None and target_year in " ".join(year_header):
            values = _extract_row_values(row, rolling_month_indices)
            if len(values) >= 12:
                return sum(values), f"Summed inferred {target_year} monthly values from row '{label}'"
    return None


def _extract_row_values(row: list[str], indices: list[int]) -> list[float]:
    values: list[float] = []
    for index in indices:
        if index >= len(row):
            continue
        value = parse_number(row[index])
        if value is not None:
            values.append(value)
    return values


def _rolling_calendar_indices(month_header: list[str]) -> list[int] | None:
    normalized = [_normalize_month(value) for value in month_header]
    month_indices = [index for index, month in enumerate(normalized) if month]
    month_names = [normalized[index] for index in month_indices]
    if month_names != [
        "dec",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]:
        return None
    return month_indices[1:]


def _is_month(value: str) -> bool:
    return _normalize_month(value) is not None


def _normalize_month(value: str) -> str | None:
    normalized = value.strip().lower().rstrip(".")
    mapping = {
        "jan": "jan",
        "january": "jan",
        "feb": "feb",
        "february": "feb",
        "mar": "mar",
        "march": "mar",
        "apr": "apr",
        "april": "apr",
        "may": "may",
        "jun": "jun",
        "june": "jun",
        "jul": "jul",
        "july": "jul",
        "aug": "aug",
        "august": "aug",
        "sept": "sep",
        "sep": "sep",
        "september": "sep",
        "oct": "oct",
        "october": "oct",
        "nov": "nov",
        "november": "nov",
        "dec": "dec",
        "deco": "dec",
        "december": "dec",
    }
    if normalized in mapping:
        return mapping[normalized]
    return None


def _is_header_label(label: str) -> bool:
    normalized = label.strip().lower()
    return not normalized or normalized == "nan" or normalized in HEADER_LABELS


def find_relevant_row(question: str, contexts: list[RetrievedContext]) -> ParsedTableRow | None:
    ranked = rank_relevant_rows(question, contexts, limit=1)
    return ranked[0] if ranked else None


def rank_relevant_rows(question: str, contexts: list[RetrievedContext], *, limit: int = 5) -> list[ParsedTableRow]:
    keywords = _question_keywords(question)
    exclusion_keywords = _question_exclusion_keywords(question)
    years = set(YEAR_PATTERN.findall(question))
    ranked: list[tuple[int, ParsedTableRow]] = []

    for context in contexts:
        for row in _extract_rows(context.content):
            overlap = len(keywords & _question_keywords(row.label))
            year_overlap = len(years & set(row.values_by_year))
            exclusion_overlap = len(exclusion_keywords & _question_keywords(row.label))
            score = overlap * 10 + year_overlap - exclusion_overlap * 20
            if any(term in question.lower() for term in {"month", "months", "calendar months", "monthly"}):
                monthly_keys = [key for key in row.values_by_year if re.fullmatch(r"(?:19|20)\d{2}-\d{2}", key)]
                score += 20 if len(monthly_keys) >= 2 else -10
            if row.label != _clean_label(row.label):
                score -= 25
            if "country" in question.lower():
                score += 8 if _looks_like_country_label(row.label) else -8
            if any(term in question.lower() for term in {"department", "administration", "agency", "bureau"}):
                score += 6 if _looks_like_institution_label(row.label) else -4
            score += int(context.score // 100)
            if overlap > 0 or year_overlap > 0:
                ranked.append((score, ParsedTableRow(label=row.label, values_by_year=row.values_by_year, source=context.source)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    deduped: list[ParsedTableRow] = []
    seen: set[str] = set()
    for _, row in ranked:
        key = row.label.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


def _looks_like_aggregate_label(label: str) -> bool:
    lowered = label.lower().strip()
    markers = {
        "total",
        "grand total",
        "other",
        "all other",
        "total asia",
        "total europe",
        "total foreign countries",
        "official institutions",
        "liabilities",
        "claims",
        "organizations",
        "miscellaneous",
    }
    return lowered.endswith(":") or any(marker in lowered for marker in markers)


def _looks_like_country_label(label: str) -> bool:
    lowered = label.lower().strip()
    if _looks_like_aggregate_label(label):
        return False
    if any(char.isdigit() for char in label):
        return False
    if len(lowered.split()) > 4:
        return False
    return not any(term in lowered for term in {"banks", "liabilities", "claims", "reported", "institutions", "organizations"})


def _looks_like_institution_label(label: str) -> bool:
    lowered = label.lower()
    return any(term in lowered for term in {"department", "administration", "agency", "bureau", "service", "judiciary"})


def reformat_tables_in_context(text: str) -> str:
    """Reformat pipe-delimited tables into explicit row-value format.

    Transforms:
        | Category | 1939 | 1940 | 1941 |
        |----------|------|------|------|
        | National defense | 1,075 | 1,657 | 6,301 |

    Into:
        TABLE:
        [headers: Category | 1939 | 1940 | 1941]
        National defense → 1939: 1,075 | 1940: 1,657 | 1941: 6,301
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Detect start of pipe table
        if line.startswith("|") and line.endswith("|") and line.count("|") >= 3:
            # Collect all pipe-table lines
            table_lines: list[str] = []
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("|") and stripped.endswith("|"):
                    table_lines.append(stripped)
                    i += 1
                else:
                    break

            # Parse the table
            reformatted = _reformat_pipe_table(table_lines)
            if reformatted:
                result_lines.append(reformatted)
            else:
                # Fallback: keep original
                result_lines.extend(table_lines)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


def _reformat_pipe_table(table_lines: list[str]) -> str | None:
    """Reformat a list of pipe-delimited table lines into explicit format."""
    if len(table_lines) < 2:
        return None

    # Parse cells from each row
    parsed_rows: list[list[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # Skip separator rows (---|---|---)
        if all(re.fullmatch(r"[-:]+", c) for c in cells if c):
            continue
        parsed_rows.append(cells)

    if len(parsed_rows) < 2:
        return None

    headers = parsed_rows[0]
    data_rows = parsed_rows[1:]

    lines = [f"[headers: {' | '.join(headers)}]"]
    for row in data_rows:
        if not row:
            continue
        label = row[0] if row else ""
        values = row[1:] if len(row) > 1 else []
        if not label.strip():
            continue
        # Pair values with headers
        pairs = []
        for j, val in enumerate(values):
            if val.strip() and j + 1 < len(headers):
                pairs.append(f"{headers[j + 1]}: {val}")
        if pairs:
            lines.append(f"  {label} → {' | '.join(pairs)}")
        else:
            lines.append(f"  {label}")

    return "\n".join(lines) if len(lines) > 1 else None
