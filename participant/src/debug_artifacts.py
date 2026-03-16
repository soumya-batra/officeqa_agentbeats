import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_debug_artifact(output_dir: Path | None, artifact_id: str, payload: dict[str, Any]) -> None:
    if output_dir is None or not artifact_id:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{artifact_id}.json"
    path.write_text(
        json.dumps(_normalize(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_context_snapshot(contexts: list) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for context in contexts:
        snapshots.append(
            {
                "source": getattr(context, "source", ""),
                "score": getattr(context, "score", 0.0),
                "content_preview": (getattr(context, "content", "") or "")[:800],
            }
        )
    return snapshots


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    return value
