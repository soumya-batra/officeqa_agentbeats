"""A2A server for OfficeQA purple agent."""
from __future__ import annotations

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="OfficeQA Purple Agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    skill = AgentSkill(
        id="officeqa_rag",
        name="OfficeQA Treasury-Bulletin QA",
        description="Answers questions about U.S. Treasury Bulletins (1939-2025) using BM25 retrieval + GPT 5.4",
        tags=["document-qa", "financial", "rag"],
        examples=["What were the total expenditures for U.S national defense in calendar year 1940?"],
    )

    agent_card = AgentCard(
        name="OfficeQA Purple Agent",
        description="RAG agent for OfficeQA benchmark using BM25 retrieval + GPT 5.4 with code interpreter",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        skills=[skill],
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
    )

    handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
