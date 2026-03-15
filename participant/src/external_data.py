import csv
from pathlib import Path


class CPIData:
    def __init__(self, path: Path | None):
        self._values: dict[str, float] = {}
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                year = str(row.get("year", "")).strip()
                value = str(row.get("cpi", "")).strip()
                if not year or not value:
                    continue
                try:
                    self._values[year] = float(value)
                except ValueError:
                    continue

    def adjust(self, amount: float, from_year: str, to_year: str) -> float | None:
        from_cpi = self._values.get(from_year)
        to_cpi = self._values.get(to_year)
        if from_cpi in (None, 0) or to_cpi is None:
            return None
        return amount * (to_cpi / from_cpi)
