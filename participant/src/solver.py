import json
import logging
import re

from config import SolverConfig
from debug_artifacts import build_context_snapshot, timestamp_utc, write_debug_artifact
from external_data import CPIData
from formatting import canonicalize_final_answer, ensure_structured_response
from json_source import load_page_contexts
from llm import LLMClient
from models import SolverResult
from faiss_retriever import FaissRetriever
from source_hints import parse_source_hints


logger = logging.getLogger(__name__)

PREPROCESS_SYSTEM_PROMPT = """You are a search query optimizer for US Treasury Bulletin documents.

Given a user question, output a short retrieval query optimized for document search:
- 10-20 words
- Resolves implicit time references (e.g., "1 year before World War 2" -> 1938)
- Converts fiscal year notation (FY1934 -> fiscal year 1934)
- Preserves table names, row labels, column names, specific years, dollar amounts, and named entities
- Strips procedural filler ("what was", "find the", "calculate")

Output in JSON format:
{
  "retrieval_query": "<your optimized query>"
}"""

SYSTEM_PROMPT = """You are answering questions from Treasury documents using retrieved context.

Rules:
- Always answer using the provided context.
- If the question involves numerical reasoning, aggregation, comparison, or multi-step calculations, use the python tool.
- Do not perform arithmetic mentally when precise computation is required.
- For simple lookups, answer directly without using tools.
- When using the python tool, show the final computed answer clearly.

## Source priority
1. Always retrieve figures directly from the Treasury Bulletin documents using file search.
2. Use web search only for external data explicitly required by the question (e.g. CPI indices, exchange rates).
3. Never rely on memorized values — always verify against the source documents.

## Calendar year vs fiscal year
- Fiscal year: July 1 through June 30 (e.g. fiscal year 1940 = July 1939 – June 1940).
- Calendar year: January 1 through December 31 of that year.

## Precision
- Preserve full numerical precision throughout; only round at the final step as instructed.
- Pay close attention to the exact table name, row label, and column in the question — do not substitute a similar-sounding category.


Return your response in the following required format:
<REASONING>
[steps and calculations]
</REASONING>
<FINAL_ANSWER>
[canonical answer only: no prose, no markdown, no explanation]
</FINAL_ANSWER>
"""

class OfficeQASolver:
    def __init__(self, config: SolverConfig | None = None, llm_client: LLMClient | None = None):
        self._config = config or SolverConfig.from_env()
        self._llm_client = llm_client or LLMClient(self._config)
        self._retriever = FaissRetriever(self._config.corpus_dir, self._config.faiss_index_dir, self._config.retrieval_top_k)
        self._cpi_data = CPIData(self._config.cpi_data_path)

    def _preprocess_question(self, core_question: str) -> tuple[str, str]:
        """Call a cheap LLM to get an optimized retrieval query.

        Returns (retrieval_query, _). Falls back to core_question on error.
        """
        try:
            raw = self._llm_client.complete_cheap(
                system_prompt=PREPROCESS_SYSTEM_PROMPT,
                prompt=core_question,
            )
            parsed = json.loads(raw)
            retrieval_query = parsed.get("retrieval_query") or core_question
            print(f"[PREPROCESS] retrieval_query: {retrieval_query}")
            return retrieval_query, retrieval_query
        except Exception as exc:
            logger.warning("Preprocess failed, using raw question: %s", exc)
            return core_question, core_question

    def solve_question(self, question: str) -> SolverResult:
        question_uid = self._extract_question_uid(question)
        source_hints = parse_source_hints(question)
        core_question = self._extract_core_question(question)
        retrieval_query, _ = self._preprocess_question(core_question)
        contexts = self._collect_contexts(retrieval_query, source_hints)
        print("\n" + "="*80)
        print(f"RETRIEVED CHUNKS ({len(contexts)} total):")
        print("-"*80)
        for i, ctx in enumerate(contexts, start=1):
            print(f"[Chunk {i}] source={ctx.source}  score={ctx.score:.2f}")
            print(ctx.content)
            print("-"*40)
        print("="*80 + "\n")
        logger.info(
            "Solving question with source_files=%s source_pages=%s contexts=%s",
            source_hints.source_files,
            source_hints.source_pages,
            [context.source for context in contexts],
        )
        base_debug_payload = {
            "timestamp_utc": timestamp_utc(),
            "question_uid": question_uid,
            "question_prompt": question,
            "core_question": core_question,
            "retrieval_query": retrieval_query,
            "source_hints": source_hints,
            "contexts": build_context_snapshot(contexts),
        }
        try:
            prompt = self._build_prompt(core_question, contexts)
            raw_response = self._llm_client.complete(system_prompt=SYSTEM_PROMPT, prompt=prompt)
            reasoning, final_answer = ensure_structured_response(raw_response)
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

    def _collect_contexts(self, retrieval_query: str, source_hints) -> list:
        query_vec = self._retriever._embed_query(retrieval_query)
        contexts = []
        if source_hints.source_files:
            contexts.extend(self._retriever.retrieve_by_source_files(source_hints.source_files, retrieval_query, query_vec))  # type: ignore[arg-type]
            contexts.extend(
                load_page_contexts(
                    self._config.parsed_json_dir,
                    source_hints.source_files,
                    source_hints.source_pages,
                    retrieval_query,
                    top_k=8,
                )
            )
        contexts.extend(self._retriever.retrieve(retrieval_query, query_vec))
        limit = max(self._config.retrieval_top_k + 7, 10) if source_hints.source_files else self._config.retrieval_top_k + 3
        deduped = {}
        for context in contexts:
            key = (context.source, context.content[:500])
            existing = deduped.get(key)
            if existing is None or context.score > existing.score:
                deduped[key] = context
        return sorted(deduped.values(), key=lambda context: context.score, reverse=True)[:limit]

    def _build_prompt(self, question: str, contexts) -> str:
        lines = [
            "Solve the following OfficeQA question using the best available evidence.",
            "",
            "FINAL_ANSWER rules:",
            "- If the answer is a single scalar, output only that scalar.",
            "- Do not add prose, bullets, markdown, or explanations inside FINAL_ANSWER.",
            "- Use square brackets only if the question explicitly asks for a bracketed list.",
            "",
            "Question:",
            question,
        ]
        if contexts:
            lines.extend(["", "Retrieved context:"])
            for index, context in enumerate(contexts, start=1):
                lines.extend(
                    [
                        f"[Source {index}] {context.source} (score={context.score:.1f})",
                        context.content,
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "",
                    "No local corpus context was retrieved. If tools are enabled, search carefully.",
                ]
            )
        return "\n".join(lines).strip()
