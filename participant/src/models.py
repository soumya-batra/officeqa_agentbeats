from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedContext:
    source: str
    content: str
    score: float


@dataclass(frozen=True)
class QuestionAnalysis:
    category: str
    needs_retrieval: bool
    needs_calculation: bool
    needs_external_data: bool


@dataclass(frozen=True)
class SolverResult:
    final_answer: str
    reasoning: str
    analysis: QuestionAnalysis
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)
    raw_response: str = ""
