# OfficeQA Agent — Kimi K2.5-fast with FAISS+BM25 Hybrid Retrieval

A purple agent submission for the [OfficeQA AgentBeats benchmark](https://agentbeats.dev), answering complex questions about U.S. Treasury Bulletin documents (1939-2025). Built on Kimi K2.5-fast via Nebius, with a baked-in FAISS+BM25 hybrid retrieval pipeline and local tool execution.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Purple Agent (Participant)                │
│                                                              │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │ server.py│───>│   solver.py    │───>│     llm.py       │  │
│  │ A2A API  │    │ Orchestration  │    │ Kimi K2.5-fast   │  │
│  └──────────┘    └───────┬────────┘    │ + tool loop      │  │
│                          │             └────────┬─────────┘  │
│                          v                      │            │
│                 ┌────────────────┐     ┌────────v─────────┐  │
│                 │faiss_retriever │     │ execute_python   │  │
│                 │ FAISS+BM25+RRF│     │ web_search       │  │
│                 └────────────────┘     │ (Tavily)         │  │
│                          │             └──────────────────┘  │
│                          v                                   │
│                 ┌────────────────┐    ┌──────────────────┐   │
│                 │  table_parser  │    │   formatting.py  │   │
│                 │  Reformat for  │    │  Canonicalize    │   │
│                 │  LLM reading   │    │  final answer    │   │
│                 └────────────────┘    └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### Hybrid Retrieval (FAISS + BM25 + Year-Era Filtering)
- **FAISS semantic search** using Qwen3-Embedding-8B (via Nebius) for dense retrieval
- **BM25 keyword search** using bm25s with Snowball stemming for sparse retrieval
- **Year-era filtering**: Detects years in the question and narrows FAISS search to chunks from matching Treasury Bulletin publication years, with content-level year indexing for precision
- **Reciprocal Rank Fusion (RRF)** merges FAISS, BM25, and year-era results with adaptive weighting:
  - When year-era hits exist: 25% global FAISS / 10% BM25 / 25% wide-era / 40% near-era
  - Otherwise: 50% FAISS / 50% BM25
- **Source-hint retrieval**: When the judge provides source file hints, retrieval is scoped to those specific bulletins before falling back to global search
- Corpus and FAISS index are **baked into the Docker image** at build time for zero cold-start latency

### Kimi K2.5-fast via Nebius with Local Tool Execution
- Uses [Kimi K2.5-fast](https://platform.moonshot.ai/) hosted on Nebius AI Studio
- **Local sandboxed Python execution** (`execute_python` tool) — the model writes and runs Python code for arithmetic, statistics, regressions, and rounding rather than computing in prose
- **Shared sandbox namespace** across tool calls within a single question, so intermediate variables persist
- **Web search** via Tavily API for external facts (historical dates, exchange rates) not in the corpus
- **Kimi native tool-call parsing**: Handles Kimi's non-standard `<|tool_call_begin|>` markup when it doesn't populate the OpenAI-standard `tool_calls` field
- Up to 5 tool-call round-trips per question, with empty-response retry logic (2 retries)

### Answer Post-Processing (`formatting.py`)
- **Sign preservation**: Negative lookahead regex in `_candidate_lines` prevents stripping leading minus signs from negative numbers (e.g., `-0.356`, `-75`)
- **List detection and formatting**: Recognizes when questions expect bracketed list answers (`[a, b, c]`) and extracts/normalizes numeric tokens accordingly
- **Scalar candidate selection**: Prioritizes lines containing keywords like "final", "therefore", "answer" when extracting the numeric answer from model output
- **Percent handling**: Detects whether the question expects a `%` sign and normalizes accordingly
- **Fallback extraction**: When the model omits `<FINAL_ANSWER>` tags, regex-based extraction finds the answer from phrases like "the answer is..." or the last number in the response

### Robustness
- **Retry logic**: Retries only on Nebius 429 (rate limit) and 400 (connection error) with exponential backoff (3s, 8s). Never retries 504 timeouts or context-length blowouts
- **Token budget management**: Estimated 100K token budget for retrieved context (conservative 1 token = 3 chars), with automatic truncation of lower-ranked chunks
- **Table reformatting**: Pipe-delimited tables in the corpus are converted to explicit row-value format for better LLM comprehension
- **LLM response cache**: Caches LLM responses to disk, skipping empty responses to avoid poisoning the cache

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEBIUS_API_KEY` | Nebius AI Studio API key | required |
| `NEBIUS_MODEL` | Model identifier | `nebius/moonshotai/Kimi-K2.5-fast` |
| `NEBIUS_BASE_URL` | Nebius API endpoint | `https://api.studio.nebius.com/v1/` |
| `TAVILY_API_KEY` | Tavily API key for web search | optional |
| `CORPUS_DIR` | Path to Treasury Bulletin `.txt` files | `/app/corpus/transformed` |
| `FAISS_INDEX_DIR` | Path to pre-built FAISS index | `/app/faiss_index` |
| `RETRIEVAL_TOP_K` | Number of chunks to retrieve | `25` |

### Assessment Parameters (`a2a-scenario.toml`)

| Parameter | Description | Values |
|-----------|-------------|--------|
| `num_questions` | Number of questions to evaluate | 1-246 |
| `difficulty` | Question difficulty filter | `"easy"`, `"hard"`, `"all"` |
| `tolerance` | Numerical matching tolerance | `0.0` (exact) |
| `max_concurrent` | Parallel questions per shard | `10` |

## Running Locally

### Prerequisites
- Docker
- Nebius API key (for Kimi K2.5-fast)
- Tavily API key (optional, for web search)

### Quick Test (3 questions)
```bash
# Set up environment
echo "NEBIUS_API_KEY=<your-key>" > .env
echo "TAVILY_API_KEY=<your-key>" >> .env

# Build and run
docker compose up --abort-on-container-exit
cat output/results.json
```

### Full Evaluation (246 questions)
Update `a2a-scenario.toml`:
```toml
[config]
num_questions = 246
difficulty = "all"
tolerance = 0.0
max_concurrent = 10
```

Then:
```bash
docker compose up --abort-on-container-exit
```

### Building the Docker Image
```bash
./build.sh  # stages corpus + FAISS index, builds linux/amd64 image
```

The build script bakes the Treasury Bulletin corpus and pre-built FAISS index into the image so retrieval works without external storage at runtime.

## Dataset

The [OfficeQA Dataset](https://github.com/databricks/officeqa) is publicly available:
- **Questions**: [officeqa_full.csv](https://github.com/databricks/officeqa/blob/main/officeqa_full.csv)
- **Corpus**: [treasury_bulletins_parsed/](https://github.com/databricks/officeqa/tree/main/treasury_bulletins_parsed)

## License

Apache 2.0
