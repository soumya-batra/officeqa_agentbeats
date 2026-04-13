"""OfficeQA reasoning pipeline: Analyze → Retrieve → Answer → Refine."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .corpus import Corpus, Chunk

logger = logging.getLogger(__name__)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
TOP_K = int(os.environ.get("TOP_K", "22"))
TOP_K_REFINE = int(os.environ.get("TOP_K_REFINE", "18"))
MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", "1800"))

ANALYZE_SYSTEM = """You convert a U.S. Treasury Bulletin question into a retrieval plan.
You never try to answer the question here.

The corpus is monthly Treasury Bulletins from 1939-2025, each ~150 KB of text including tables.

Return a compact JSON object with keys:
  "years": list[int] — all years the question depends on (include ±1 for fiscal years)
  "query": string — keyword-rich search query, ~20-40 words, emphasising table headers, metric names, dates
  "expect_unit": string — e.g. "millions of dollars", "percent", "year"
  "expect_type": string — one of "number", "percent", "year", "text", "date", "list"
Return ONLY JSON, no prose, no markdown fence."""

ANSWER_SYSTEM = """You are an analyst answering questions about U.S. Treasury Bulletins (1939-2025).
You are given retrieved excerpts containing the ground truth — prefer them over prior knowledge.

PRECISION RULES:
- Copy figures from excerpts with every digit intact. Never round unless asked.
- If the question asks "in millions" and the excerpt says "in thousands", divide by 1000.
- For PERCENT answers: if excerpt shows "0.1234" and question asks for percent, multiply by 100.
- Fiscal year: FY N = July 1 of N-1 to June 30 of N (pre-1977), Oct 1 of N-1 to Sep 30 of N (after).
- For LIST answers: emit values comma-separated in square brackets, e.g. [0.096, -184.143].

FORMATTING RULES:
- Emit exactly one value (or one list). Multiple candidate numbers will AUTO-FAIL.
- No units suffix unless the answer is a text string. Use bare number.
- Never output "no answer found" or empty tags — give your best numeric guess.
- Use code_interpreter for ALL arithmetic: ratios, percentages, sums, differences, rounding.

Respond in this exact format:
<REASONING>
[brief step-by-step: which excerpt, which table row/column, what arithmetic]
</REASONING>
<FINAL_ANSWER>
[the single canonical value, nothing else]
</FINAL_ANSWER>"""

REFINE_SYSTEM = """You previously drafted an answer. Re-read the new excerpts and submit a final answer
using the same <REASONING>/<FINAL_ANSWER> format. You may revise the draft."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_REASONING_RE = re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?%?")


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _years_in(text: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(text)]


def _extract_reasoning(text: str) -> str | None:
    m = _REASONING_RE.search(text or "")
    return m.group(1).strip() if m else None


def _extract_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text or "")
    return m.group(1).strip() if m else None


def _missing_info(text: str) -> bool:
    fa = _extract_final_answer(text) or ""
    rt = _extract_reasoning(text) or ""
    flags = ["cannot find", "not found", "not available", "unable to", "no data",
             "insufficient", "not enough", "do not have", "missing"]
    haystack = (fa + " " + rt).lower()
    return any(f in haystack for f in flags) or not fa.strip()


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        body = c.text[:MAX_CHARS_PER_CHUNK]
        if len(c.text) > MAX_CHARS_PER_CHUNK:
            body += " ...[truncated]"
        parts.append(f"--- EXCERPT {i} ({c.filename}) ---\n{body}")
    return "\n\n".join(parts)


def _normalize_response(text: str) -> str:
    if not text:
        return "<REASONING>(no reasoning)</REASONING>\n<FINAL_ANSWER>0</FINAL_ANSWER>"

    reasoning = _extract_reasoning(text)
    final = _extract_final_answer(text)

    if reasoning is None and final is None:
        nums = _NUM_RE.findall(text)
        final = nums[-1] if nums else text.strip()[:200]
        reasoning = text.strip()[:2000]
    if reasoning is None:
        reasoning = "(not provided)"
    if not final or not final.strip():
        final = "0"

    # De-hedge: if multiple numbers in final answer (not a list), keep first non-year
    is_list = bool(re.search(r"\[.*?,.*?\]", final))
    if not is_list:
        nums = [n for n in _NUM_RE.findall(final) if re.search(r"\d", n)]
        unique = []
        for n in nums:
            c = n.replace(",", "")
            if c and c not in unique:
                unique.append(c)
        if len(unique) > 1:
            non_years = [n for n in unique if not (1900 <= float(n.rstrip("%")) <= 2100 and float(n.rstrip("%")) == int(float(n.rstrip("%"))))]
            final = (non_years[0] if non_years else unique[0])

    return f"<REASONING>\n{reasoning}\n</REASONING>\n<FINAL_ANSWER>\n{final.strip()}\n</FINAL_ANSWER>"


@dataclass
class AgentResult:
    reasoning: str
    final_answer: str
    raw_response: str


class OfficeQAAgent:
    def __init__(self, corpus: Corpus, client: OpenAI | None = None) -> None:
        self.corpus = corpus
        self.client = client or OpenAI()

    def _respond(self, system: str, user: str, effort: str | None = None) -> str:
        eff = effort or REASONING_EFFORT
        try:
            tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            resp = self.client.responses.create(
                model=MODEL,
                instructions=system,
                input=[{"role": "user", "content": user}],
                reasoning={"effort": eff},
                tools=tools,
            )
            return resp.output_text or ""
        except Exception:
            logger.exception("OpenAI API call failed")
            raise

    def analyze(self, question: str) -> dict[str, Any]:
        try:
            raw = self._respond(ANALYZE_SYSTEM, question, effort="low")
        except Exception:
            raw = ""
        plan = _extract_json(raw) or {}
        years = set(plan.get("years") or [])
        years.update(_years_in(question))
        widened = set(years)
        for y in list(years):
            widened.update({y - 1, y + 1})
        plan["years"] = sorted(widened)
        if not plan.get("query"):
            plan["query"] = question
        return plan

    def retrieve(self, query: str, years: list[int] | None, k: int) -> list[Chunk]:
        hits = self.corpus.search(query, top_k=k, year_filter=years or None)
        return [c for c, _ in hits]

    def answer(self, question: str, chunks: list[Chunk], plan: dict[str, Any]) -> str:
        context = _format_chunks(chunks)
        user = (
            f"QUESTION:\n{question}\n\n"
            f"EXPECTED ANSWER UNIT: {plan.get('expect_unit', 'unknown')}\n"
            f"EXPECTED ANSWER TYPE: {plan.get('expect_type', 'unknown')}\n\n"
            f"RETRIEVED EXCERPTS ({len(chunks)}):\n{context}\n"
        )
        return self._respond(ANSWER_SYSTEM, user)

    def refine(self, question: str, prior: str, chunks: list[Chunk], plan: dict[str, Any]) -> str:
        context = _format_chunks(chunks)
        user = (
            f"QUESTION:\n{question}\n\n"
            f"EXPECTED ANSWER UNIT: {plan.get('expect_unit', 'unknown')}\n"
            f"EXPECTED ANSWER TYPE: {plan.get('expect_type', 'unknown')}\n\n"
            f"YOUR PRIOR DRAFT (may be wrong — verify against new excerpts):\n{prior}\n\n"
            f"NEW RETRIEVED EXCERPTS ({len(chunks)}):\n{context}\n"
        )
        return self._respond(REFINE_SYSTEM, user, effort="high")

    def answer_question(self, question: str) -> AgentResult:
        plan = self.analyze(question)
        logger.info("Plan: years=%s query=%r", plan.get("years"), plan.get("query", "")[:80])

        chunks = self.retrieve(plan["query"], plan.get("years"), TOP_K)
        logger.info("Retrieved %d chunks", len(chunks))

        draft = self.answer(question, chunks, plan)

        # Refine if the draft admits missing info
        if _missing_info(draft):
            reasoning_text = _extract_reasoning(draft) or draft
            draft_final = _extract_final_answer(draft) or ""
            extra_query = plan["query"] + " " + reasoning_text[:800] + " " + draft_final
            second = self.retrieve(extra_query, plan.get("years"), TOP_K_REFINE)
            seen = {c.chunk_id for c in chunks[:6]}
            new_chunks = [c for c in second if c.chunk_id not in seen][:TOP_K_REFINE]
            combined = chunks[:6] + new_chunks
            draft = self.refine(question, draft, combined, plan)

        normalized = _normalize_response(draft)
        return AgentResult(
            reasoning=_extract_reasoning(normalized) or "",
            final_answer=_extract_final_answer(normalized) or "",
            raw_response=normalized,
        )
