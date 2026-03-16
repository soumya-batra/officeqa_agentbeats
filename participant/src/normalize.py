import re


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_numeric_text(value: str) -> str:
    return value.replace("\u2212", "-").replace("−", "-").strip()


def parse_number(value: str) -> float | None:
    normalized = normalize_numeric_text(value)
    match = NUMBER_PATTERN.search(normalized)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


