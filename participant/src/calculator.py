import math
import re
from dataclasses import dataclass


MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_RANGE_PATTERN = re.compile(
    r"from\s+([A-Za-z]+)\s+((?:19|20)\d{2})\s+to\s+([A-Za-z]+)\s+((?:19|20)\d{2})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CalculationResult:
    value: float
    explanation: str
    formatted_answer: str | None = None


def _ordered_series(series: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(series.items(), key=lambda item: _period_sort_key(item[0]))


def _period_sort_key(key: str) -> tuple[int, int]:
    if re.fullmatch(r"(?:19|20)\d{2}-\d{2}", key):
        year, month = key.split("-", 1)
        return int(year), int(month)
    if YEAR_PATTERN.fullmatch(key):
        return int(key), 0
    return 0, 0


def _annual_series(series: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in series.items() if YEAR_PATTERN.fullmatch(key)}


def _monthly_series(series: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in series.items() if re.fullmatch(r"(?:19|20)\d{2}-\d{2}", key)}


def _question_years(question: str) -> list[str]:
    return YEAR_PATTERN.findall(question)


def _year_totals(series: dict[str, float], years: list[str]) -> dict[str, float]:
    monthly = _monthly_series(series)
    totals: dict[str, float] = {}
    for year in years:
        values = [value for key, value in monthly.items() if key.startswith(f"{year}-")]
        if values:
            totals[year] = sum(values)
    return totals


def _month_range_keys(question: str) -> tuple[str, str] | None:
    match = MONTH_RANGE_PATTERN.search(question)
    if not match:
        return None
    start_month = MONTH_NAME_TO_NUMBER.get(match.group(1).lower().rstrip("."))
    end_month = MONTH_NAME_TO_NUMBER.get(match.group(3).lower().rstrip("."))
    if start_month is None or end_month is None:
        return None
    return (
        f"{match.group(2)}-{start_month:02d}",
        f"{match.group(4)}-{end_month:02d}",
    )


def _range_values(question: str, series: dict[str, float]) -> list[float]:
    monthly = _monthly_series(series)
    month_range = _month_range_keys(question)
    if month_range is not None and monthly:
        start_key, end_key = month_range
        return [
            value
            for key, value in _ordered_series(monthly)
            if _period_sort_key(start_key) <= _period_sort_key(key) <= _period_sort_key(end_key)
        ]

    years = _question_years(question)
    if monthly and years and any(term in question.lower() for term in {"calendar months", "monthly", "month"}):
        totals = _year_totals(series, years)
        if any(term in question.lower() for term in {"total sum", "sum values", "percent change", "difference"}):
            return [totals[year] for year in years if year in totals]

    annual = _annual_series(series)
    if years:
        values = [annual[year] for year in years if year in annual]
        if values:
            return values
    return [value for _, value in _ordered_series(annual or monthly)]


def _regression_inputs(question: str, series: dict[str, float]) -> tuple[list[float], list[float], int] | None:
    annual = _annual_series(series)
    if not annual:
        totals = _year_totals(series, _question_years(question))
        annual = totals
    if len(annual) < 2:
        return None

    years = sorted(int(year) for year in annual)
    base_year = years[0]
    explicit_base = re.search(r'(?:treat|setting)\s+t?\s*=\s*0\s+in\s+(?:fy|fiscal year|year)?\s*((?:19|20)\d{2})', question, re.IGNORECASE)
    if explicit_base:
        base_year = int(explicit_base.group(1))
    x_values = [year - base_year for year in years]
    y_values = [annual[str(year)] for year in years]
    return x_values, y_values, base_year


def _ols(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=False))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    slope = numerator / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _format_list(values: list[float], decimals: int) -> str:
    rendered: list[str] = []
    for value in values:
        if abs(value - round(value)) < 1e-9:
            rendered.append(str(int(round(value))))
        else:
            rendered.append(f"{value:.{decimals}f}")
    return f"[{', '.join(rendered)}]"


def calculate_from_series(question: str, series: dict[str, float]) -> CalculationResult | None:
    if len(series) < 1:
        return None

    lowered = question.lower()
    ordered = _ordered_series(series)
    exact_years = _question_years(question)
    monthly = _monthly_series(series)

    if "regression" in lowered or "ordinary least squares" in lowered or "linear trend" in lowered:
        regression_inputs = _regression_inputs(question, series)
        if regression_inputs is None:
            return None
        x_values, y_values, base_year = regression_inputs
        slope, intercept = _ols(x_values, y_values)
        predict_match = re.search(r"(?:predict|project).+?((?:19|20)\d{2})", question, re.IGNORECASE)
        if "slope and intercept" in lowered and predict_match is None:
            return CalculationResult(
                value=slope,
                explanation="Computed ordinary least squares slope and intercept",
                formatted_answer=_format_list([slope, intercept], 3),
            )
        if predict_match:
            target_year = int(predict_match.group(1))
            predicted = slope * (target_year - base_year) + intercept
            if "actual" in lowered and ("forecast error" in lowered or "projected" in lowered):
                actual = _annual_series(series).get(str(target_year))
                if actual is not None:
                    error = actual - predicted if "actual -" in lowered else predicted - actual
                    return CalculationResult(
                        value=error,
                        explanation=f"Computed regression forecast error for {target_year}",
                    )
            if "inside square brackets" in lowered or "report all values inside square brackets" in lowered:
                return CalculationResult(
                    value=predicted,
                    explanation=f"Computed regression projection for {target_year}",
                    formatted_answer=_format_list([slope, intercept, predicted], 2),
                )
            return CalculationResult(
                value=predicted,
                explanation=f"Computed regression projection for {target_year}",
            )

    values = _range_values(question, series)
    if not values:
        return None

    if "geometric mean" in lowered:
        positive_values = [value for value in values if value > 0]
        if not positive_values:
            return None
        geometric_mean = math.exp(sum(math.log(value) for value in positive_values) / len(positive_values))
        return CalculationResult(
            value=geometric_mean,
            explanation=f"Computed geometric mean across {len(positive_values)} values",
        )

    if "coefficient of variation" in lowered:
        mean_value = sum(values) / len(values)
        if mean_value == 0:
            return None
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        stddev = math.sqrt(variance)
        return CalculationResult(
            value=(stddev / mean_value) * 100,
            explanation=f"Computed coefficient of variation across {len(values)} values",
        )

    if "standard deviation" in lowered:
        divisor = len(values) - 1 if "sample" in lowered and len(values) > 1 else len(values)
        if divisor <= 0:
            return None
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / divisor
        return CalculationResult(
            value=math.sqrt(variance),
            explanation=f"Computed {'sample' if 'sample' in lowered else 'population'} standard deviation across {len(values)} values",
        )

    if "percent change" in lowered or ("percent" in lowered and "change" in lowered):
        if len(values) < 2:
            return None
        start_value, end_value = values[0], values[-1]
        if start_value == 0:
            return None
        value = ((end_value - start_value) / start_value) * 100
        return CalculationResult(
            value=abs(value) if "absolute percent change" in lowered else value,
            explanation=f"Computed percent change using {start_value} and {end_value}",
        )

    if "difference" in lowered or "change in" in lowered:
        if len(values) < 2:
            return None
        difference = values[-1] - values[0]
        return CalculationResult(
            value=abs(difference) if "absolute difference" in lowered else difference,
            explanation=f"Computed difference using {values[0]} and {values[-1]}",
        )

    if "average" in lowered or "mean" in lowered:
        avg = sum(values) / len(values)
        return CalculationResult(
            value=avg,
            explanation=f"Computed arithmetic mean across {len(values)} values",
        )

    if monthly and exact_years and any(term in lowered for term in {"total sum", "calendar months", "monthly"}):
        totals = _year_totals(series, exact_years)
        if len(exact_years) == 1 and exact_years[0] in totals:
            year = exact_years[0]
            return CalculationResult(
                value=totals[year],
                explanation=f"Summed monthly values for {year}",
            )

    direct_lookup_blockers = {
        "month",
        "months",
        "average",
        "mean",
        "difference",
        "change",
        "regression",
        "standard deviation",
        "coefficient of variation",
        "geometric mean",
        "predict",
        "project",
    }
    if len(exact_years) == 1 and exact_years[0] in series and not any(token in lowered for token in direct_lookup_blockers):
        year = exact_years[0]
        return CalculationResult(
            value=series[year],
            explanation=f"Selected the value for {year} directly from the parsed table row",
        )

    if len(ordered) == 1 and not any(token in lowered for token in direct_lookup_blockers):
        year, value = ordered[0]
        return CalculationResult(
            value=value,
            explanation=f"Selected the only available value from {year}",
        )

    return None
