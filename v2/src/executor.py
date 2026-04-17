"""A2A executor for OfficeQA purple agent."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message, Part, TaskState, TaskStatus, TaskStatusUpdateEvent, TextPart, UnsupportedOperationError,
)

from .agent import OfficeQAAgent
from .corpus import Corpus

logger = logging.getLogger(__name__)

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class Executor(AgentExecutor):
    def __init__(self) -> None:
        self._cache_dir = Path(os.environ.get("CORPUS_CACHE_DIR", "/data/corpus"))
        self._agent: OfficeQAAgent | None = None
        self._init_lock = asyncio.Lock()

    async def _ensure_agent(self) -> OfficeQAAgent:
        if self._agent is not None:
            return self._agent
        async with self._init_lock:
            if self._agent is None:
                corpus = await asyncio.to_thread(Corpus.load, self._cache_dir)
                self._agent = OfficeQAAgent(corpus=corpus)
        return self._agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if not message or not message.parts:
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text="Retrieving and reasoning..."))],
                    ),
                ),
                final=False,
            )
        )

        question_text = ""
        for part in message.parts:
            root = part.root if hasattr(part, "root") else part
            if isinstance(root, TextPart):
                question_text = root.text
                break

        try:
            agent = await self._ensure_agent()
            result = await asyncio.to_thread(agent.answer_question, question_text)
            response = result.raw_response
        except Exception as e:
            logger.exception("Agent failure: %s", e)
            response = f"<REASONING>Agent error: {e}</REASONING>\n<FINAL_ANSWER>0</FINAL_ANSWER>"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=response))],
                    ),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
