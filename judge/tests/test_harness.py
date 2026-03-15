import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generate_compose
from agent import OfficeQAAgent


def test_build_compose_includes_participant_dependencies_and_runner():
    scenario = {
        "green_agent": {"image": "judge-image", "env": {"LOG_LEVEL": "INFO"}},
        "participants": [
            {
                "name": "officeqa_agent",
                "image": "participant-image",
                "env": {"OPENAI_API_KEY": "${OPENAI_API_KEY}"},
            }
        ],
        "config": {"num_questions": 1, "difficulty": "all", "tolerance": 0.0},
    }

    compose = generate_compose.build_compose(scenario)

    assert "judge" in compose["services"]
    assert "officeqa_agent" in compose["services"]
    assert "agentbeats-client" in compose["services"]
    assert compose["services"]["judge"]["depends_on"] == {
        "officeqa_agent": {"condition": "service_healthy"}
    }
    assert compose["services"]["agentbeats-client"]["depends_on"]["judge"] == {
        "condition": "service_healthy"
    }
    assert compose["services"]["agentbeats-client"]["depends_on"]["officeqa_agent"] == {
        "condition": "service_healthy"
    }


def test_generate_files_writes_compose_and_a2a_scenario(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scenario.toml").write_text(
        """
[green_agent]
image = "judge-image"

[[participants]]
name = "officeqa_agent"
image = "participant-image"

[config]
num_questions = 2
difficulty = "easy"
tolerance = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    generate_compose.generate_files(generate_compose.load_scenario())

    compose = yaml.safe_load((tmp_path / "docker-compose.yml").read_text(encoding="utf-8"))
    a2a_scenario = (tmp_path / "a2a-scenario.toml").read_text(encoding="utf-8")

    assert "agentbeats-client" in compose["services"]
    assert 'role = "officeqa_agent"' in a2a_scenario
    assert 'endpoint = "http://officeqa_agent:9009"' in a2a_scenario
    assert "num_questions = 2" in a2a_scenario


def test_single_question_timeout_returns_failed_result():
    class SlowMessenger:
        async def talk_to_agent(self, **_kwargs):
            await asyncio.sleep(0.05)
            return "<FINAL_ANSWER>42</FINAL_ANSWER>"

    agent = OfficeQAAgent(messenger=SlowMessenger())
    question = {
        "uid": "q1",
        "question": "What is the answer?",
        "answer": "42",
        "difficulty": "easy",
    }

    result = asyncio.run(
        agent._evaluate_single_question(
            question,
            agent_url="http://officeqa_agent:9009",
            tolerance=0.0,
            question_timeout=0,
        )
    )

    assert result.is_correct is False
    assert "timed out" in result.predicted
    assert "No FINAL_ANSWER tags found" in result.rationale
