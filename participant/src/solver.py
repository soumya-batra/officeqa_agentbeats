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

# Kimi K2.5-fast has max_seq_len=131_083. Reserve ~31K for system prompt,
# user instructions, tool-call round-trips, and model output (max_tokens=12K).
# Leaves ~100K for retrieved contexts. Estimate 1 token ≈ 3 chars (conservative
# — English averages ~4, so this over-counts and under-fills the budget).
PROMPT_TOKEN_BUDGET = 100_000


def _count_tokens(text: str) -> int:
    return len(text) // 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions about U.S. Treasury Bulletin documents using retrieved context.

Rules:
- Use the retrieved context to answer. When data is ambiguous or partially available, reason through it rather than giving up.
- Use the execute_python tool for any non-trivial arithmetic (sums, percentages, regressions, KL divergence, rounding to N places). Do not perform multi-step math in prose.
- Fiscal year (pre-1977): July 1 – June 30. Calendar year: Jan 1 – Dec 31.
- Preserve full precision; round only at the final step if instructed.
- Match exact table name, row label, and column. Do not substitute similar categories or sum unrelated columns.
- If a web_search tool is available, use it to look up facts not in the retrieved context (e.g. historical dates, bureau names, exchange rates).
- Preserve the sign of computed values. If computing a change (B − A), difference, or growth rate, keep the negative sign if the result is negative.
- Output in the exact format requested by the question, if specified.

You MUST always end your response with both tags below — never omit FINAL_ANSWER:
<REASONING>
[1-2 sentence summary of how you derived the answer]
</REASONING>
<FINAL_ANSWER>
[bare number or value only — no units, no prose, no markdown]
</FINAL_ANSWER>
"""

_PAGE_NUMBER_RE = re.compile(
    r"\b(?:page\s*number|what\s+(?:is\s+)?(?:the\s+)?page|which\s+page)\b", re.I
)


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

        # ── Retrieval ────────────────────────────────────────────────
        sub_queries = [core_question]
        contexts = self._collect_multi_query_contexts(sub_queries, source_hints, is_page_q)

        self._log_chunks(contexts)
        logger.info(
            "Solving question with source_files=%s source_pages=%s contexts=%s",
            source_hints.source_files,
            source_hints.source_pages,
            [c.source for c in contexts],
        )

        base_debug_payload = {
            "timestamp_utc": timestamp_utc(),
            "question_uid": question_uid,
            "question_prompt": question,
            "core_question": core_question,
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

            # ── Solver ───────────────────────────────────────────────
            prompt = self._build_prompt(core_question, contexts, is_page_q, table_data=table_data)
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
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _collect_multi_query_contexts(
        self,
        queries: list[str],
        source_hints,
        is_page_question: bool,
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

        limit = self._context_limit(source_hints)
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

    def _context_limit(self, source_hints) -> int:
        base = self._config.retrieval_top_k
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

    def _build_prompt(self, question: str, contexts: list[RetrievedContext], is_page_question: bool = False, table_data: str = "") -> str:
        lines = [
            "Solve the following OfficeQA question using the best available evidence.",
            "",
            "FINAL_ANSWER rules:",
            "- Output ONLY the bare number or value. No units (no 'million', 'dollars', etc.), no prose, no markdown.",
            "- Pay close attention to the units the question asks for and ensure your answer is in those units.",
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
            header_tokens = _count_tokens("\n".join(lines))
            budget = PROMPT_TOKEN_BUDGET - header_tokens
            lines.extend(["", "Retrieved context:"])
            kept = 0
            for index, context in enumerate(contexts, start=1):
                block = f"[Source {index}] {context.source} (score={context.score:.1f})\n{context.content}\n"
                block_tokens = _count_tokens(block)
                if block_tokens > budget:
                    break
                lines.extend([
                    f"[Source {index}] {context.source} (score={context.score:.1f})",
                    context.content,
                    "",
                ])
                budget -= block_tokens
                kept += 1
            if kept < len(contexts):
                print(f"[PROMPT TRIM] kept top {kept}/{len(contexts)} contexts to fit token budget", flush=True)
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

