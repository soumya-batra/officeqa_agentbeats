"""OfficeQA agent: extract years → BM25 retrieve → single GPT 5.4 call."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from .corpus import Corpus, Chunk

logger = logging.getLogger(__name__)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")
TOP_K = int(os.environ.get("TOP_K", "22"))
MAX_CHARS_PER_CHUNK = int(os.environ.get("MAX_CHARS_PER_CHUNK", "1800"))

SYSTEM_PROMPT = """You are an analyst answering questions about U.S. Treasury Bulletins (1939-2025).
You are given retrieved excerpts containing the ground truth — prefer them over prior knowledge.

PRECISION RULES:
- Copy figures from excerpts with every digit intact. Never round unless asked.
- If the question asks "in millions" and the excerpt says "in thousands", divide by 1000.
- For PERCENT answers: if excerpt shows "0.1234" and question asks for percent, multiply by 100.
- Fiscal year: FY N = July 1 of N-1 to June 30 of N (pre-1977), Oct 1 of N-1 to Sep 30 of N (after).
- For LIST answers: emit values comma-separated in square brackets.
- Preserve the sign of computed values. Keep negative sign if the result is negative.
- Use code_interpreter for ALL arithmetic: sums, percentages, regressions, rounding.

FORMATTING RULES:
- Emit exactly one value (or one list). Multiple candidate numbers will AUTO-FAIL.
- No units suffix. Use bare number.
- Never output "no answer found" — give your best numeric guess.

You MUST always end with both tags:
<REASONING>
[brief: which excerpt, which row/column, what arithmetic]
</REASONING>
<FINAL_ANSWER>
[bare number or value only]
</FINAL_ANSWER>"""

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_REASONING_RE = re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:[,\d]*\d)?(?:\.\d+)?%?")


def _years_in(text: str) -> list[int]:
    """Extract years from text and widen by ±1 for fiscal year robustness."""
    years = set(int(y) for y in _YEAR_RE.findall(text))
    widened = set(years)
    for y in years:
        widened.update({y - 1, y + 1})
    return sorted(widened)


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

    if not reasoning and not final:
        nums = _NUM_RE.findall(text)
        f = nums[-1] if nums else text.strip()[:200]
        r = text.strip()[:2000]

    if not f:
        f = "0"

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

    return f"<REASONING>\n{r}\n</REASONING>\n<FINAL_ANSWER>\n{f.strip()}\n</FINAL_ANSWER>"


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

    def answer_question(self, question: str) -> AgentResult:
        # 1. Extract years directly from question text (no LLM call)
        years = _years_in(question)
        logger.info("Years: %s", years)

        # 2. BM25 retrieve (no LLM call)
        hits = self.corpus.search(question, top_k=TOP_K, year_filter=years or None)
        chunks = [c for c, _ in hits]
        logger.info("Retrieved %d chunks", len(chunks))

        # 3. Single GPT 5.4 call with code_interpreter
        context = _format_chunks(chunks)
        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED EXCERPTS ({len(chunks)}):\n{context}\n"
        )

        try:
            tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            resp = self.client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": user_prompt}],
                reasoning={"effort": REASONING_EFFORT},
                tools=tools,
            )
            raw = resp.output_text or ""
        except Exception as e:
            logger.exception("OpenAI API call failed: %s", e)
            raw = f"<REASONING>API error: {e}</REASONING>\n<FINAL_ANSWER>0</FINAL_ANSWER>"

        normalized = _normalize_response(raw)
        r = _REASONING_RE.search(normalized)
        f = _FINAL_RE.search(normalized)

        return AgentResult(
            reasoning=r.group(1).strip() if r else "",
            final_answer=f.group(1).strip() if f else "0",
            raw_response=normalized,
        )
