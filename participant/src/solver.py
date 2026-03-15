import logging
import re

from calculator import calculate_from_series
from config import SolverConfig
from debug_artifacts import build_context_snapshot, timestamp_utc, write_debug_artifact
from external_data import CPIData
from formatting import canonicalize_final_answer, ensure_structured_response
from json_source import load_page_contexts
from llm import LLMClient
from models import QuestionAnalysis, SolverResult
from retrieval import Retriever
from source_hints import parse_source_hints
from table_parser import find_calendar_year_total, find_relevant_row


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a grounded OfficeQA solving agent.

Use retrieved Treasury Bulletin context when provided. Prefer exact figures from the documents over guesswork.
If calculation is required, explain the arithmetic clearly and preserve full precision until the final step.

Return your response in the following required format:
<REASONING>
[steps and calculations]
</REASONING>
<FINAL_ANSWER>
[canonical answer only: no prose, no markdown, no explanation]
</FINAL_ANSWER>
"""

MATH_KEYWORDS = {
    "calculate",
    "change",
    "difference",
    "average",
    "mean",
    "correlation",
    "regression",
    "variance",
    "standard deviation",
    "percent",
    "ratio",
    "forecast",
}
EXTERNAL_DATA_KEYWORDS = {"inflation", "cpi", "exchange rate", "exchange rates"}
MONTH_NAME_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
SAFE_MONTHLY_TOTAL_TOKENS = {
    "individual calendar months",
    "monthly values",
    "calendar months",
    "reported values for all individual calendar months",
}
COMPLEX_DETERMINISTIC_TOKENS = {
    "average",
    "mean",
    "geometric mean",
    "weighted average",
    "difference",
    "change",
    "ratio",
    "regression",
    "ordinary least squares",
    "ols",
    "correlation",
    "variance",
    "standard deviation",
    "coefficient of variation",
    "cagr",
    "compound annual growth rate",
    "predict",
    "project",
    "forecast",
    "highest",
    "lowest",
    "maximum",
    "minimum",
    "page number",
    "issue date",
    "list",
    "square brackets",
    "comma-separated",
    "euclidean norm",
    "fisher",
    "box-cox",
    "duration",
    "denomination",
    "absolute percentage points",
    "contribution",
    "slope",
    "intercept",
    "which",
    "on which",
}


class OfficeQASolver:
    def __init__(self, config: SolverConfig | None = None, llm_client: LLMClient | None = None):
        self._config = config or SolverConfig.from_env()
        self._llm_client = llm_client or LLMClient(self._config)
        self._retriever = Retriever(self._config.corpus_dir, self._config.retrieval_top_k)
        self._cpi_data = CPIData(self._config.cpi_data_path)

    def solve_question(self, question: str) -> SolverResult:
        question_uid = self._extract_question_uid(question)
        source_hints = parse_source_hints(question)
        core_question = self._extract_core_question(question)
        analysis = self._analyze_question(core_question)
        contexts = self._collect_contexts(core_question, source_hints) if analysis.needs_retrieval else []
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
            "source_hints": source_hints,
            "analysis": analysis,
            "contexts": build_context_snapshot(contexts),
        }
        try:
            deterministic = self._solve_deterministically(core_question, analysis, contexts)
            if deterministic is not None:
                logger.info("Answered deterministically with %s", deterministic.final_answer)
                deterministic = SolverResult(
                    final_answer=canonicalize_final_answer(question, deterministic.final_answer),
                    reasoning=deterministic.reasoning,
                    analysis=deterministic.analysis,
                    retrieved_contexts=deterministic.retrieved_contexts,
                    raw_response=deterministic.raw_response,
                )
                self._write_debug_artifact(
                    question_uid or self._fallback_artifact_id(core_question),
                    {
                        **base_debug_payload,
                        "route": "deterministic",
                        "final_answer": deterministic.final_answer,
                        "reasoning": deterministic.reasoning,
                    },
                )
                return deterministic

            prompt = self._build_prompt(core_question, analysis, contexts)
            logger.info("Falling back to model for question")
            raw_response = self._llm_client.complete(system_prompt=SYSTEM_PROMPT, prompt=prompt)
            reasoning, final_answer = ensure_structured_response(raw_response)
            final_answer = canonicalize_final_answer(question, final_answer)
            result = SolverResult(
                final_answer=final_answer,
                reasoning=reasoning,
                analysis=analysis,
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

    def _collect_contexts(self, question: str, source_hints) -> list:
        contexts = []
        if source_hints.source_files:
            contexts.extend(self._retriever.retrieve_by_source_files(source_hints.source_files, question))
            contexts.extend(
                load_page_contexts(
                    self._config.parsed_json_dir,
                    source_hints.source_files,
                    source_hints.source_pages,
                    question,
                    top_k=8,
                )
            )
        contexts.extend(self._retriever.retrieve(question))
        limit = max(self._config.retrieval_top_k + 7, 10) if source_hints.source_files else self._config.retrieval_top_k + 3
        deduped = {}
        for context in contexts:
            key = (context.source, context.content[:500])
            existing = deduped.get(key)
            if existing is None or context.score > existing.score:
                deduped[key] = context
        return sorted(deduped.values(), key=lambda context: context.score, reverse=True)[:limit]

    def _solve_deterministically(self, question: str, analysis: QuestionAnalysis, contexts) -> SolverResult | None:
        year_total = find_calendar_year_total(question, contexts)
        if year_total is not None:
            final_answer = self._format_numeric_answer(question, year_total[0])
            return SolverResult(
                final_answer=final_answer,
                reasoning=year_total[1],
                analysis=analysis,
                retrieved_contexts=contexts,
                raw_response="",
            )

        if not self._question_allows_safe_row_deterministic(question):
            return None

        row = find_relevant_row(question, contexts)
        if row is None:
            return None

        if self._question_is_safe_monthly_total(question):
            calculation = calculate_from_series(question, row.values_by_year)
            if calculation is None:
                return None
            final_answer = calculation.formatted_answer or self._format_numeric_answer(question, calculation.value)
            return SolverResult(
                final_answer=final_answer,
                reasoning=f"{calculation.explanation}. Source row: {row.label}",
                analysis=analysis,
                retrieved_contexts=contexts,
                raw_response="",
            )

        if self._question_is_safe_annual_lookup(question, row.values_by_year):
            year = re.findall(r"\b(?:19|20)\d{2}\b", question)[0]
            final_answer = self._format_numeric_answer(question, row.values_by_year[year])
            return SolverResult(
                final_answer=final_answer,
                reasoning=f"Selected the value for {year} directly from the parsed table row. Source row: {row.label}",
                analysis=analysis,
                retrieved_contexts=contexts,
                raw_response="",
            )

        adjusted = self._maybe_adjust_for_inflation(question, row.values_by_year)
        if adjusted is not None:
            final_answer = self._format_numeric_answer(question, adjusted[0])
            return SolverResult(
                final_answer=final_answer,
                reasoning=adjusted[1],
                analysis=analysis,
                retrieved_contexts=contexts,
                raw_response="",
            )
        return None

    def _question_allows_safe_row_deterministic(self, question: str) -> bool:
        lowered = question.lower()
        if any(token in lowered for token in COMPLEX_DETERMINISTIC_TOKENS):
            return False
        return self._question_is_safe_monthly_total(question) or self._question_has_single_annual_target(question)

    def _question_is_safe_monthly_total(self, question: str) -> bool:
        lowered = question.lower()
        if "difference" in lowered or "change" in lowered:
            return False
        if len(re.findall(r"\b(?:19|20)\d{2}\b", question)) != 1:
            return False
        return any(token in lowered for token in SAFE_MONTHLY_TOTAL_TOKENS)

    def _question_has_single_annual_target(self, question: str) -> bool:
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        if len(years) != 1:
            return False
        if MONTH_NAME_PATTERN.search(question):
            return False
        return True

    def _question_is_safe_annual_lookup(self, question: str, series: dict[str, float]) -> bool:
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        return len(years) == 1 and years[0] in series and self._question_has_single_annual_target(question)

    def _analyze_question(self, question: str) -> QuestionAnalysis:
        lowered = question.lower()
        needs_calculation = any(keyword in lowered for keyword in MATH_KEYWORDS)
        needs_external_data = any(keyword in lowered for keyword in EXTERNAL_DATA_KEYWORDS)
        has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", question))
        category = "calculation" if needs_calculation else "lookup"
        if needs_external_data:
            category = "external-data"
        return QuestionAnalysis(
            category=category,
            needs_retrieval=True if has_year or "treasury" in lowered else True,
            needs_calculation=needs_calculation,
            needs_external_data=needs_external_data,
        )

    def _maybe_adjust_for_inflation(self, question: str, series: dict[str, float]) -> tuple[float, str] | None:
        lowered = question.lower()
        if "inflation" not in lowered and "cpi" not in lowered:
            return None
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        if len(years) < 2:
            return None
        from_year, to_year = years[0], years[-1]
        amount = series.get(from_year)
        if amount is None:
            return None
        adjusted = self._cpi_data.adjust(amount, from_year=from_year, to_year=to_year)
        if adjusted is None:
            return None
        explanation = f"Adjusted {amount} from {from_year} dollars to {to_year} dollars using CPI data"
        return adjusted, explanation

    def _format_numeric_answer(self, question: str, value: float) -> str:
        lowered = question.lower()
        wants_percent_sign = any(
            token in lowered
            for token in (
                "percent value",
                "as a percentage",
                "reported as a percent value",
                "expressed as a percent",
                "%",
            )
        ) and "decimal value" not in lowered and "no percentage sign" not in lowered
        if wants_percent_sign:
            return f"{value:.2f}%"
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"

    def _build_prompt(self, question: str, analysis: QuestionAnalysis, contexts) -> str:
        lines = [
            "Solve the following OfficeQA question using the best available evidence.",
            f"Question category: {analysis.category}",
            f"Requires calculation: {'yes' if analysis.needs_calculation else 'no'}",
            f"Requires external data: {'yes' if analysis.needs_external_data else 'no'}",
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
