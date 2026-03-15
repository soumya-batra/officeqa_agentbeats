#!/usr/bin/env python3
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SCENARIO_PATH = Path("scenario.toml")
COMPOSE_PATH = Path("docker-compose.yml")
A2A_SCENARIO_PATH = Path("a2a-scenario.toml")
OUTPUT_DIR = Path("output")


def load_scenario():
    with SCENARIO_PATH.open("rb") as f:
        return tomllib.load(f)


def resolve_env(env: dict) -> dict:
    resolved_env = {}
    for key, value in env.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved_env[key] = os.environ.get(env_var, "")
        else:
            resolved_env[key] = value
    return resolved_env


def build_compose(scenario: dict) -> dict:
    services: dict[str, dict] = {}
    participants = scenario.get("participants", [])
    run_id = scenario.get("config", {}).get("run_id") or os.environ.get(
        "RUN_ID",
        datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"),
    )

    green = scenario.get("green_agent", {})
    green_env = green.get("env", {})
    services["judge"] = {
        "image": green.get("image", "ghcr.io/OWNER/officeqa-judge:latest"),
        "container_name": "officeqa-judge",
        "command": ["--host", "0.0.0.0", "--port", "9009", "--card-url", "http://judge:9009"],
        "ports": ["9009:9009"],
        "environment": {
            "PYTHONUNBUFFERED": "1",
            **green_env,
        },
        "healthcheck": {
            "test": ["CMD", "curl", "-f", "http://localhost:9009/.well-known/agent-card.json"],
            "interval": "5s",
            "timeout": "3s",
            "retries": 10,
            "start_period": "30s",
        },
        "depends_on": {},
        "networks": ["agentnet"],
    }

    for index, participant in enumerate(participants):
        name = participant.get("name", f"participant_{index}")
        host_port = 9019 + index
        participant_env = participant.get("env", {})
        resolved_env = resolve_env(participant_env)
        volumes = [f"./{OUTPUT_DIR}:/app/output"]
        for env_key in ("CORPUS_DIR", "CPI_DATA_PATH"):
            mounted_path = resolved_env.get(env_key)
            if mounted_path:
                volumes.append(f"{mounted_path}:{mounted_path}:ro")
        corpus_dir = resolved_env.get("CORPUS_DIR")
        parsed_json_dir = resolved_env.get("PARSED_JSON_DIR")
        if not parsed_json_dir and corpus_dir:
            corpus_path = Path(corpus_dir)
            candidate = corpus_path.parent / "jsons" if corpus_path.name == "transformed" else corpus_path.parent / "jsons"
            if candidate.exists():
                parsed_json_dir = str(candidate)
        if parsed_json_dir:
            if "PARSED_JSON_DIR" not in participant_env:
                participant_env = {**participant_env, "PARSED_JSON_DIR": parsed_json_dir}
            volumes.append(f"{parsed_json_dir}:{parsed_json_dir}:ro")
        if "DEBUG_OUTPUT_DIR" not in participant_env:
            participant_env = {**participant_env, "DEBUG_OUTPUT_DIR": f"/app/output/solver_debug/{run_id}"}
        if "WRITE_DEBUG_ARTIFACTS" not in participant_env:
            participant_env = {**participant_env, "WRITE_DEBUG_ARTIFACTS": "true"}
        if "LLM_CACHE_PATH" not in participant_env:
            participant_env = {**participant_env, "LLM_CACHE_PATH": "/app/output/llm_cache.json"}
        services[name] = {
            "image": participant.get("image", "ghcr.io/OWNER/officeqa-agent:latest"),
            "container_name": name.replace("_", "-"),
            "command": ["--host", "0.0.0.0", "--port", "9009", "--card-url", f"http://{name}:9009"],
            "ports": [f"{host_port}:9009"],
            "environment": {
                "PYTHONUNBUFFERED": "1",
                **participant_env,
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", "http://localhost:9009/.well-known/agent-card.json"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 10,
                "start_period": "30s",
            },
            "networks": ["agentnet"],
        }
        if volumes:
            services[name]["volumes"] = volumes
        services["judge"]["depends_on"][name] = {"condition": "service_healthy"}

    services["agentbeats-client"] = {
        "image": "ghcr.io/agentbeats/agentbeats-client:v1.0.0",
        "container_name": "agentbeats-client",
        "volumes": [
            f"./{A2A_SCENARIO_PATH.name}:/app/scenario.toml",
            f"./{OUTPUT_DIR}:/app/output",
        ],
        "command": ["scenario.toml", "output/results.json"],
        "depends_on": {
            "judge": {"condition": "service_healthy"},
            **{
                participant.get("name", f"participant_{index}"): {"condition": "service_healthy"}
                for index, participant in enumerate(participants)
            },
        },
        "networks": ["agentnet"],
    }

    return {
        "services": services,
        "networks": {"agentnet": {"driver": "bridge"}},
    }


def write_a2a_scenario(scenario: dict) -> None:
    lines = ['[green_agent]', 'endpoint = "http://judge:9009"', ""]

    for index, participant in enumerate(scenario.get("participants", [])):
        role = participant.get("name", f"participant_{index}")
        lines.extend(
            [
                "[[participants]]",
                f'role = "{role}"',
                f'endpoint = "http://{role}:9009"',
                "",
            ]
        )

    lines.append("[config]")
    for key, value in scenario.get("config", {}).items():
        if isinstance(value, str):
            rendered = f'"{value}"'
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    lines.append("")

    A2A_SCENARIO_PATH.write_text("\n".join(lines), encoding="utf-8")


def generate_files(scenario: dict) -> None:
    compose = build_compose(scenario)
    COMPOSE_PATH.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")
    write_a2a_scenario(scenario)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Generated {COMPOSE_PATH.name} and {A2A_SCENARIO_PATH.name}")


if __name__ == "__main__":
    generate_files(load_scenario())
