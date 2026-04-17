"""OfficeQA agent: analyze → BM25 retrieve → answer → optional refine."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .corpus import Corpus, Chunk

logger = logging.getLogger(__name__)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
TOP_K = int(os.environ.get("TOP_K", "22"))
TOP_K_REFINE = int(os.environ.get("TOP_K_REFINE", "18"))
MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", "2500"))

ANALYZE_SYSTEM = """You convert a U.S. Treasury Bulletin question into a retrieval plan.
Do NOT answer the question. Just plan retrieval.

Return a compact JSON object with keys:
  "years": list[int] — all years the question depends on (include ±1 for fiscal years)
  "query": string — keyword-rich search query, ~20-40 words, table headers, metric names, dates
Return ONLY JSON, no prose, no markdown fence."""

ANSWER_SYSTEM = """You are an analyst answering questions about U.S. Treasury Bulletins (1939-2025).
You are given retrieved excerpts containing the ground truth — prefer them over prior knowledge.

PRECISION RULES:
- Copy figures from excerpts with every digit intact. Never round unless asked.
- If the question asks "in millions" and the excerpt says "in thousands", divide by 1000.
- For PERCENT answers: if excerpt shows "0.1234" and question asks for percent, multiply by 100.
- Fiscal year: FY N = July 1 of N-1 to June 30 of N (pre-1977), Oct 1 of N-1 to Sep 30 of N (after).
- For LIST answers: emit values comma-separated in square brackets with a space after each comma, e.g. [0.096, -184.143].

ROUNDING RULES (CRITICAL):
- If the question says "rounded to N decimal places" or "to the nearest thousandth/hundredth", you MUST round to exactly N decimal places. E.g. "rounded to 2 decimal places" → output 1.67, not 1.667857.
- If the question says "rounded to the nearest integer/whole number", round to 0 decimal places.
- If no rounding is specified, preserve full precision from the source.

SIGN RULES (CRITICAL):
- When computing a CHANGE (B − A), DIFFERENCE, or GROWTH RATE, keep the negative sign if B < A.
- When the question says "by how much did X decrease", the answer should be NEGATIVE if it decreased.
- NEVER drop a negative sign. If your computation gives -118255.5, output -118255.5, not 118255.5.
- Double-check: does your sign match the direction described in the question?

PERCENT RULES:
- If the question asks for a percentage or percent value, include the % sign in your answer.
- If the question says "as a percentage" or "expressed as percent", include %.
- If the question says "as a decimal", do NOT include %.

FORMATTING RULES:
- Emit exactly one value (or one list). Multiple candidate numbers will AUTO-FAIL.
- No units suffix (no "million", "dollars") unless the answer is a text string. Use bare number.
- Never output "no answer found" — give your best numeric guess.
- Use code_interpreter for ALL arithmetic. Show final value clearly with print().
- If a web_search tool is available, use it to look up facts not in the retrieved context (e.g. historical dates, bureau names, exchange rates, CPI values).

You MUST always end with both tags:
<REASONING>
[brief: which excerpt, which row/column, what arithmetic]
</REASONING>
<FINAL_ANSWER>
[bare number or value only]
</FINAL_ANSWER>"""

REFINE_SYSTEM = """You previously drafted an answer but it may be incomplete or wrong.
Re-read the new excerpts and submit a corrected final answer using the same format.
Pay special attention to: correct sign (negative/positive), correct units, exact digits, rounding."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_REASONING_RE = re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?%?")
_ANSWER_PHRASE_RE = re.compile(
    r"(?:(?:the\s+)?(?:final\s+)?answer\s+is|total\s+(?:is|=|:)|therefore|thus)[:\s]*"
    r"(-?\$?[\d,]+\.?\d*%?)",
    re.IGNORECASE,
)


def _years_in(text: str) -> list[int]:
    years = set(int(y) for y in _YEAR_RE.findall(text))
    widened = set(years)
    for y in years:
        widened.update({y - 1, y + 1})
    return sorted(widened)


def _extract_json(text: str) -> dict[str, Any] | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_reasoning(text: str) -> str | None:
    m = _REASONING_RE.search(text or "")
    return m.group(1).strip() if m else None


def _extract_final_answer(text: str) -> str | None:
    m = _FINAL_RE.search(text or "")
    return m.group(1).strip() if m else None


def _fallback_extract_answer(text: str) -> str:
    """Try to pull a numeric answer from free text when FINAL_ANSWER tag is missing."""
    m = _ANSWER_PHRASE_RE.search(text)
    if m:
        return m.group(1).replace("$", "")
    numbers = [n.replace("$", "") for n in _NUM_RE.findall(text) if not _is_year_token(n)]
    if numbers:
        return numbers[-1]
    return ""


def _is_year_token(token: str) -> bool:
    bare = token.replace(",", "").rstrip("%").replace(".", "").lstrip("-")
    if not bare.isdigit():
        return False
    try:
        return 1900 <= int(bare) <= 2100
    except ValueError:
        return False


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

    reasoning = _REASONING_RE.search(text or "")
    final = _FINAL_RE.search(text or "")

    r = reasoning.group(1).strip() if reasoning else "(not provided)"
    f = final.group(1).strip() if final else ""

    # Fallback: if no FINAL_ANSWER tag, try to extract from reasoning or raw text
    if not f:
        if reasoning:
            f = _fallback_extract_answer(reasoning.group(1))
        if not f:
            f = _fallback_extract_answer(text)
        if not f:
            f = "0"

    if not reasoning and not final:
        r = text.strip()[:2000]

    # De-hedge: if multiple numbers (not a list), keep first non-year
    is_list = bool(re.search(r"\[.*?,.*?\]", f))
    if not is_list:
        nums = [n for n in _NUM_RE.findall(f) if re.search(r"\d", n)]
        unique = []
        for n in nums:
            c = n.replace(",", "")
            if c and c not in unique:
                unique.append(c)
        if len(unique) > 1:
            non_years = [n for n in unique if not _is_year(n)]
            f = non_years[0] if non_years else unique[0]

    # Fix list formatting: ensure "space after comma" in bracket lists
    f = f.strip()
    if re.match(r"^\[.*\]$", f):
        f = re.sub(r",\s*", ", ", f)

    return f"<REASONING>\n{r}\n</REASONING>\n<FINAL_ANSWER>\n{f}\n</FINAL_ANSWER>"


def _is_year(s: str) -> bool:
    try:
        v = float(s.rstrip("%"))
        return 1900 <= v <= 2100 and v == int(v)
    except Exception:
        return False


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
        t0 = time.time()
        try:
            tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            if os.environ.get("ENABLE_WEB_SEARCH", "").lower() in ("1", "true", "yes"):
                tools.append({"type": "web_search"})
            resp = self.client.responses.create(
                model=MODEL,
                instructions=system,
                input=[{"role": "user", "content": user}],
                reasoning={"effort": eff},
                tools=tools,
            )
            raw = resp.output_text or ""
            logger.info("LLM call: %.1fs, len=%d", time.time() - t0, len(raw))
            return raw
        except Exception:
            logger.exception("OpenAI API failed (%.1fs)", time.time() - t0)
            raise

    def _analyze(self, question: str) -> tuple[list[int], str]:
        years = _years_in(question)
        query = question
        try:
            t0 = time.time()
            resp = self.client.responses.create(
                model=MODEL,
                instructions=ANALYZE_SYSTEM,
                input=[{"role": "user", "content": question}],
                reasoning={"effort": "low"},
            )
            raw = resp.output_text or ""
            logger.info("Analyze: %.1fs", time.time() - t0)
            plan = _extract_json(raw)
            if plan:
                if plan.get("years"):
                    extra = set(plan["years"])
                    for y in list(extra):
                        extra.update({y - 1, y + 1})
                    years = sorted(set(years) | extra)
                if plan.get("query"):
                    query = plan["query"]
        except Exception as e:
            logger.warning("Analyze failed: %s", e)
        return years, query

    def answer_question(self, question: str) -> AgentResult:
        # 1. Analyze (fast, effort=low)
        years, query = self._analyze(question)
        logger.info("Years: %s, Query: %.80s", years, query)

        # 2. BM25 retrieve
        t0 = time.time()
        hits = self.corpus.search(query, top_k=TOP_K, year_filter=years or None)
        chunks = [c for c, _ in hits]
        logger.info("Retrieved %d chunks in %.1fs", len(chunks), time.time() - t0)

        # 3. Answer (main call)
        context = _format_chunks(chunks)
        user_prompt = f"QUESTION:\n{question}\n\nRETRIEVED EXCERPTS ({len(chunks)}):\n{context}\n"
        try:
            draft = self._respond(ANSWER_SYSTEM, user_prompt)
        except Exception as e:
            draft = f"<REASONING>API error: {e}</REASONING>\n<FINAL_ANSWER>0</FINAL_ANSWER>"

        # 4. Refine if answer admits missing info
        if _missing_info(draft):
            logger.info("Low confidence, refining...")
            try:
                reasoning_text = _extract_reasoning(draft) or draft
                draft_final = _extract_final_answer(draft) or ""
                extra_query = query + " " + reasoning_text[:800] + " " + draft_final
                second_hits = self.corpus.search(extra_query, top_k=TOP_K_REFINE, year_filter=years or None)
                seen = {c.chunk_id for c in chunks[:6]}
                new_chunks = [c for c in [h for h, _ in second_hits] if c.chunk_id not in seen][:TOP_K_REFINE]
                combined = chunks[:6] + new_chunks

                refine_context = _format_chunks(combined)
                refine_prompt = (
                    f"QUESTION:\n{question}\n\n"
                    f"YOUR PRIOR DRAFT:\n{draft}\n\n"
                    f"NEW RETRIEVED EXCERPTS ({len(combined)}):\n{refine_context}\n"
                )
                draft = self._respond(REFINE_SYSTEM, refine_prompt, effort="high")
            except Exception as e:
                logger.warning("Refine failed: %s", e)

        normalized = _normalize_response(draft)
        r = _extract_reasoning(normalized)
        f = _extract_final_answer(normalized)
        logger.info("Final answer: %s", (f or "NONE")[:100])

        return AgentResult(
            reasoning=r or "",
            final_answer=f or "0",
            raw_response=normalized,
        )
