import json
import logging
import re

from config import SolverConfig
from debug_artifacts import build_context_snapshot, timestamp_utc, write_debug_artifact
from external_data import CPIData
from formatting import canonicalize_final_answer, ensure_structured_response
from json_source import load_page_contexts
from llm import LLMClient
from models import RetrievedContext, SolverResult
from faiss_retriever import FaissRetriever
from source_hints import parse_source_hints
from table_parser import find_calendar_year_total, reformat_tables_in_context


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM_PROMPT = """You are a search query optimizer for US Treasury Bulletin documents.

Given a user question, decompose it into 1-3 retrieval queries optimized for document search.
For simple lookups, output one query.  For multi-step questions, output sub-queries that each
target a specific piece of information needed to answer the overall question.

Rules per query:
- 10-20 words
- Resolve implicit time references (e.g., "1 year before World War 2" -> 1938)
- Convert fiscal year notation (FY1934 -> fiscal year 1934)
- Preserve table names, row labels, column names, specific years, dollar amounts
- Strip procedural filler ("what was", "find the", "calculate")

Output JSON (no markdown fences):
{
  "queries": ["<query 1>", "<query 2 if needed>"]
}"""

SYSTEM_PROMPT = """You answer questions about U.S. Treasury Bulletin documents using retrieved context.

Rules:
- Use code_execution for ALL table parsing and arithmetic. Never eyeball numbers from tables.
- Extract figures from the retrieved context. Use web search only for external data (CPI, exchange rates).
- Fiscal year (pre-1977): July 1 – June 30. Calendar year: Jan 1 – Dec 31.
- Preserve full precision; round only at the final step if instructed.
- Match exact table name, row label, and column. Do not substitute similar categories.
- Page number questions: report the physical page number (small integer), not section codes.

Output format (keep visible output SHORT):
<REASONING>
[1-2 sentence summary]
</REASONING>
<FINAL_ANSWER>
[bare number or value only — no units, no prose, no markdown]
</FINAL_ANSWER>
"""

PLANNER_SYSTEM_PROMPT = """You are an expert question analyst for U.S. Treasury Bulletin research.

Given a question, produce a structured analysis that will guide retrieval and answering.

Output JSON (no markdown fences):
{
  "data_points": ["<specific data point 1 needed>", "<data point 2>"],
  "table_names": ["<exact table name if mentioned, e.g. FFO-3, FFO-5>"],
  "time_periods": ["<resolved time period, e.g. fiscal year 1934, February 2012, calendar year 1981>"],
  "constraints": ["<inclusion/exclusion constraint 1>", "<constraint 2>"],
  "answer_type": "<scalar|list|page_number|text>",
  "unit": "<expected unit if specified, e.g. millions of dollars, percentage points, billions>",
  "extra_queries": ["<additional retrieval query if the question needs data not obvious from the main question>"]
}

Rules:
- Resolve ALL implicit time references (e.g. "1 year before WW2" -> "1938", "FY2013 budget proposal release month" -> "February 2012")
- Extract EVERY inclusion/exclusion constraint (e.g. "excluding territories", "shouldn't contain revolving funds", "net of refunds")
- Identify exact table names if the question references one (FFO-3, FFO-5, OFS-1, etc.)
- Keep data_points specific: not "some value" but "total interest-bearing U.S. public debt September 1989"
- extra_queries should target data the main question text wouldn't directly match in search"""


_PAGE_NUMBER_RE = re.compile(
    r"\b(?:page\s*number|what\s+(?:is\s+)?(?:the\s+)?page|which\s+page)\b", re.I
)


def _parse_json_robust(raw: str) -> dict:
    """Parse JSON from LLM output, handling markdown fences and extra text."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object within the text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found", text, 0)


class OfficeQASolver:
    def __init__(self, config: SolverConfig | None = None, llm_client: LLMClient | None = None):
        self._config = config or SolverConfig.from_env()
        self._llm_client = llm_client or LLMClient(self._config)
        self._retriever = FaissRetriever(self._config.corpus_dir, self._config.faiss_index_dir, self._config.retrieval_top_k)
        self._cpi_data = CPIData(self._config.cpi_data_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def solve_question(self, question: str) -> SolverResult:
        question_uid = self._extract_question_uid(question)
        source_hints = parse_source_hints(question)
        core_question = self._extract_core_question(question)
        is_page_q = bool(_PAGE_NUMBER_RE.search(core_question))

        # ── Agent 1: PLANNER ─────────────────────────────────────────
        plan = self._plan_question(core_question)

        # ── Agent 2: RETRIEVER (multi-strategy) ─────────────────────
        sub_queries = self._decompose_question(core_question)
        # Merge planner's extra queries with decomposed queries
        if plan.get("extra_queries"):
            for eq in plan["extra_queries"]:
                if eq and eq not in sub_queries:
                    sub_queries.append(eq)
        # Also add table-name-targeted queries from the plan
        for table_name in plan.get("table_names", []):
            if table_name:
                table_query = f"{table_name} Treasury Bulletin table"
                if table_query not in sub_queries:
                    sub_queries.append(table_query)
        # For calendar-month-total questions, add month-specific queries
        month_queries = self._generate_month_queries(core_question, plan)
        if month_queries:
            sub_queries.extend(month_queries)
            print(f"[MONTH QUERIES] Added {len(month_queries)} month-specific queries")

        has_month_queries = bool(month_queries)
        contexts = self._collect_multi_query_contexts(sub_queries, source_hints, is_page_q, has_month_queries)

        self._log_chunks(contexts)
        logger.info(
            "Solving question with source_files=%s source_pages=%s contexts=%s plan=%s",
            source_hints.source_files,
            source_hints.source_pages,
            [c.source for c in contexts],
            plan,
        )

        base_debug_payload = {
            "timestamp_utc": timestamp_utc(),
            "question_uid": question_uid,
            "question_prompt": question,
            "core_question": core_question,
            "plan": plan,
            "sub_queries": sub_queries,
            "source_hints": source_hints,
            "contexts": build_context_snapshot(contexts),
        }

        try:
            # ── Reformat tables for LLM readability ──────────────────
            contexts = self._reformat_contexts(contexts)

            # ── Table extraction ─────────────────────────────────────
            table_data = self._extract_table_data(core_question, contexts)
            if table_data:
                print(f"[TABLE EXTRACT]\n{table_data}")

            # ── Agent 3: SOLVER ──────────────────────────────────────
            prompt = self._build_prompt(core_question, contexts, is_page_q, table_data=table_data, plan=plan)
            raw_response = self._llm_client.complete(system_prompt=SYSTEM_PROMPT, prompt=prompt)
            reasoning, final_answer = ensure_structured_response(raw_response)

            # ── Post-process ─────────────────────────────────────────
            final_answer = canonicalize_final_answer(question, final_answer)

            result = SolverResult(
                final_answer=final_answer,
                reasoning=reasoning,
                retrieved_contexts=contexts,
                raw_response=raw_response,
            )
            self._write_debug_artifact(
                question_uid or self._fallback_artifact_id(core_question),
                {
                    **base_debug_payload,
                    "route": "model",
                    "final_answer": final_answer,
                    "reasoning": reasoning,
                    "raw_response": raw_response,
                },
            )
            return result

        except Exception as exc:
            self._write_debug_artifact(
                question_uid or self._fallback_artifact_id(core_question),
                {
                    **base_debug_payload,
                    "route": "error",
                    "error": str(exc),
                },
            )
            raise

    # ------------------------------------------------------------------
    # Agent 1: Planner
    # ------------------------------------------------------------------

    def _plan_question(self, core_question: str) -> dict:
        """Use a cheap model to produce a structured plan from the question."""
        try:
            raw = self._llm_client.complete_cheap(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                prompt=core_question,
            )
            plan = _parse_json_robust(raw)
            print(f"[PLAN] {json.dumps(plan, indent=2)}")
            return plan
        except Exception as exc:
            logger.warning("Planner failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Month-specific retrieval for calendar year questions
    # ------------------------------------------------------------------

    _MONTHS = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

    @staticmethod
    def _generate_month_queries(question: str, plan: dict) -> list[str]:
        """For questions about 'all individual calendar months in YEAR',
        generate month-specific retrieval queries to ensure full coverage.

        Only triggers for precise patterns with 1-2 explicit years to avoid
        flooding retrieval with hundreds of queries.
        """
        lowered = question.lower()
        # Only trigger for the most specific patterns
        if "all individual calendar months" not in lowered:
            return []

        # Extract years that appear RIGHT AFTER "months in"
        # e.g., "all individual calendar months in 1953"
        year_matches = re.findall(r"calendar months in (\d{4})", lowered)
        if not year_matches or len(year_matches) > 2:
            return []

        # Extract the topic
        topic = ""
        if plan.get("data_points"):
            topic = plan["data_points"][0]
        else:
            for phrase in ["national defense", "budget expenditures", "public debt",
                           "receipts", "outlays", "interest"]:
                if phrase in lowered:
                    topic = phrase
                    break

        queries = []
        for year in set(year_matches):
            for month in OfficeQASolver._MONTHS:
                queries.append(f"{topic} {month} {year} Treasury Bulletin")
        return queries

    # ------------------------------------------------------------------
    # Question decomposition (multi-query)
    # ------------------------------------------------------------------

    def _decompose_question(self, core_question: str) -> list[str]:
        """Use a cheap model to decompose the question into 1-3 retrieval sub-queries."""
        try:
            raw = self._llm_client.complete_cheap(
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                prompt=core_question,
            )
            parsed = _parse_json_robust(raw)
            queries = parsed.get("queries") or [core_question]
            # Sanity: at most 3 sub-queries, each non-empty
            queries = [q.strip() for q in queries[:3] if q.strip()]
            if not queries:
                queries = [core_question]
            print(f"[DECOMPOSE] sub-queries: {queries}")
            return queries
        except Exception as exc:
            logger.warning("Decompose failed, using raw question: %s", exc)
            return [core_question]

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _collect_multi_query_contexts(
        self,
        queries: list[str],
        source_hints,
        is_page_question: bool,
        has_month_queries: bool = False,
    ) -> list[RetrievedContext]:
        """Retrieve for each sub-query and merge via deduplication + score."""
        all_contexts: list[RetrievedContext] = []
        seen_queries: set[str] = set()

        for query in queries:
            if query in seen_queries:
                continue
            seen_queries.add(query)
            all_contexts.extend(self._collect_single_query_contexts(query, source_hints))

        # For page-number questions, also load JSON page contexts if available
        if is_page_question and source_hints.source_files:
            all_contexts.extend(
                load_page_contexts(
                    self._config.parsed_json_dir,
                    source_hints.source_files,
                    source_hints.source_pages,
                    queries[0],
                    top_k=10,
                )
            )

        limit = self._context_limit(source_hints, has_month_queries=has_month_queries)
        return self._dedup_and_rank(all_contexts, limit)

    def _collect_single_query_contexts(self, query: str, source_hints) -> list[RetrievedContext]:
        """Retrieve contexts for a single query string."""
        # Only compute embedding if FAISS index is available
        query_vec = None
        if self._retriever._index is not None:
            query_vec = self._retriever._embed_query(query)
        contexts: list[RetrievedContext] = []
        if source_hints.source_files:
            contexts.extend(
                self._retriever.retrieve_by_source_files(source_hints.source_files, query, query_vec)
            )
            contexts.extend(
                load_page_contexts(
                    self._config.parsed_json_dir,
                    source_hints.source_files,
                    source_hints.source_pages,
                    query,
                    top_k=8,
                )
            )
        contexts.extend(self._retriever.retrieve(query, query_vec))
        return contexts

    def _context_limit(self, source_hints, has_month_queries: bool = False) -> int:
        base = self._config.retrieval_top_k
        if has_month_queries:
            return max(base + 15, 30)  # Need more context for monthly data
        if source_hints.source_files:
            return max(base + 10, 20)
        return base + 5

    @staticmethod
    def _dedup_and_rank(
        contexts: list[RetrievedContext], limit: int
    ) -> list[RetrievedContext]:
        deduped: dict[tuple, RetrievedContext] = {}
        for ctx in contexts:
            key = (ctx.source, ctx.content[:500])
            existing = deduped.get(key)
            if existing is None or ctx.score > existing.score:
                deduped[key] = ctx
        return sorted(deduped.values(), key=lambda c: c.score, reverse=True)[:limit]

    @staticmethod
    def _merge_contexts(
        existing: list[RetrievedContext], new: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        deduped: dict[tuple, RetrievedContext] = {}
        for ctx in existing:
            key = (ctx.source, ctx.content[:500])
            deduped[key] = ctx
        for ctx in new:
            key = (ctx.source, ctx.content[:500])
            if key not in deduped or ctx.score > deduped[key].score:
                deduped[key] = ctx
        return sorted(deduped.values(), key=lambda c: c.score, reverse=True)


    # ------------------------------------------------------------------
    # Table reformatting (make tables LLM-readable)
    # ------------------------------------------------------------------

    @staticmethod
    def _reformat_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Reformat pipe tables in retrieved contexts to explicit row-value format."""
        reformatted = []
        for ctx in contexts:
            try:
                new_content = reformat_tables_in_context(ctx.content)
                reformatted.append(RetrievedContext(
                    content=new_content,
                    source=ctx.source,
                    score=ctx.score,
                ))
            except Exception:
                reformatted.append(ctx)
        return reformatted

    # ------------------------------------------------------------------
    # Table extraction (structured data for LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_table_data(question: str, contexts: list[RetrievedContext]) -> str:
        """Pre-extract structured table data from retrieved contexts.

        Only extracts calendar year totals (precise computation).
        General row ranking is too noisy and can mislead the LLM.
        """
        lines: list[str] = []

        # Try calendar year total extraction (handles monthly sum questions)
        try:
            cal_result = find_calendar_year_total(question, contexts)
            if cal_result is not None:
                total, explanation = cal_result
                lines.append(f"HINT — Calendar year total (sum of monthly values): {total:,.2f}")
                lines.append(f"  ({explanation})")
                lines.append("  Verify this against the source context before using.")
                lines.append("")
        except Exception:
            pass

        return "\n".join(lines) if lines else ""


    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, contexts: list[RetrievedContext], is_page_question: bool = False, table_data: str = "", plan: dict | None = None) -> str:
        lines = [
            "Solve the following OfficeQA question using the best available evidence.",
            "",
            "FINAL_ANSWER rules:",
            "- Output ONLY the bare number or value. No units (no 'million', 'dollars', etc.), no prose, no markdown.",
            "- If the question says 'in millions of dollars', the answer is just the number (e.g., '507' not '507 million').",
            "- Use square brackets only if the question explicitly asks for a bracketed list.",
        ]
        if is_page_question:
            lines.extend([
                "",
                "PAGE NUMBER GUIDANCE:",
                "- The answer is the physical page number printed on the page (a small integer, usually 1-100).",
                "- Do NOT report bulletin section codes (like 'B-462-B') or index numbers.",
                "- Look for the page number at the top or bottom of the page content.",
            ])
        # Inject plan data_points as row-label guidance
        if plan and plan.get("data_points"):
            lines.extend(["", "EXACT DATA TO FIND (match these labels precisely in the tables):"])
            for dp in plan["data_points"]:
                lines.append(f"  - {dp}")
        if plan and plan.get("constraints"):
            lines.extend(["", "QUESTION CONSTRAINTS:"])
            for c in plan["constraints"]:
                lines.append(f"  - {c}")
        lines.extend([
            "",
            "Question:",
            question,
        ])
        if table_data:
            lines.extend([
                "",
                "STRUCTURED TABLE DATA (pre-extracted from retrieved context — use these values when they match your query):",
                table_data,
            ])
        if contexts:
            lines.extend(["", "Retrieved context:"])
            for index, context in enumerate(contexts, start=1):
                lines.extend([
                    f"[Source {index}] {context.source} (score={context.score:.1f})",
                    context.content,
                    "",
                ])
        else:
            lines.extend([
                "",
                "No local corpus context was retrieved. If tools are enabled, search carefully.",
            ])
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _extract_core_question(self, question: str) -> str:
        lines: list[str] = []
        for line in question.splitlines():
            stripped = line.strip()
            if not stripped:
                if lines:
                    break
                continue
            lowered = stripped.lower()
            if lowered.startswith("question uid:"):
                continue
            if lowered.startswith("use the officeqa corpus") or lowered.startswith("relevant source "):
                break
            lines.append(stripped)
        return " ".join(lines).strip() or question.strip()

    def _extract_question_uid(self, question: str) -> str:
        for line in question.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("question uid:"):
                return stripped.split(":", 1)[1].strip()
        return ""

    def _write_debug_artifact(self, artifact_id: str, payload: dict) -> None:
        if not self._config.write_debug_artifacts:
            return
        write_debug_artifact(self._config.debug_output_dir, artifact_id, payload)

    def _fallback_artifact_id(self, core_question: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", core_question.lower()).strip("_")
        return normalized[:80] or "question"

    @staticmethod
    def _log_chunks(contexts: list[RetrievedContext]) -> None:
        print(f"\n[RETRIEVAL] {len(contexts)} chunks retrieved")
        for i, ctx in enumerate(contexts[:5], start=1):
            print(f"  [{i}] {ctx.source} (score={ctx.score:.2f}) {ctx.content[:100]}...")

