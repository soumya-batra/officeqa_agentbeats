# OfficeQA Purple Agent

A purple agent for the [OfficeQA benchmark](https://github.com/databricks/officeqa) on the [AgentBeats](https://agentbeats.dev) platform. Answers complex questions about U.S. Treasury Bulletin documents (1939-2025) using retrieval-augmented generation.

## How It Works

1. **Plan** — Analyzes the question to identify data points, table names, time periods, and constraints
2. **Decompose** — Breaks the question into 1-3 focused retrieval queries
3. **Retrieve** — Hybrid FAISS (semantic) + BM25 (keyword) search over 150K corpus chunks
4. **Solve** — GPT 5.4 with code interpreter parses tables, runs calculations, and produces the answer
5. **Format** — Canonicalizes the answer for judge compatibility

## Score

| Config | Score | Accuracy |
|--------|-------|----------|
| GPT 5.4 + FAISS/BM25 | 97/246 | 39.4% |

## Project Structure

```
participant/src/
  solver.py          # Main solving pipeline (plan -> decompose -> retrieve -> solve -> format)
  faiss_retriever.py # Hybrid FAISS + BM25 retrieval
  llm.py             # LLM client (OpenAI, Gemini, Anthropic)
  config.py          # Environment-based configuration
  formatting.py      # Final answer normalization
  server.py          # A2A server
  executor.py        # A2A request handling
judge/src/           # Green agent (evaluator) — loads questions, scores answers
amber-manifest.json  # Amber manifest for AgentBeats Quick Submit
Dockerfile.officeqa-agent  # Production Docker image (corpus + FAISS baked in)
build.sh             # Build script for production image
```

## Quick Start (Local)

```bash
git clone https://github.com/soumya-batra/officeqa_agentbeats.git
cd officeqa_agentbeats

# Install dependencies
uv sync --extra judge --extra participant --extra dev

# Configure
cat > .env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-5.4
ENABLE_WEB_SEARCH=false
REASONING_EFFORT=medium
RETRIEVAL_TOP_K=15
CORPUS_DIR=/path/to/treasury_bulletins_parsed/transformed
FAISS_INDEX_DIR=/path/to/faiss_index
EOF

# Generate compose and run
set -a && source .env && set +a
python generate_compose.py
docker compose up --abort-on-container-exit --exit-code-from agentbeats-client

# Check results
python3 -c "import json; d=json.load(open('output/results.json'))['results'][0]; print(f\"{d['correct_answers']}/{d['total_questions']} ({d['accuracy']*100:.1f}%)\")"
```

## Leaderboard Submission

The agent is deployed via [AgentBeats Quick Submit](https://agentbeats.dev/agentbeater/officeqa/submit):

- **Docker image**: `ghcr.io/zaidishahbaz1/officeqa-agent:latest`
- **Manifest**: `https://raw.githubusercontent.com/soumya-batra/officeqa_agentbeats/main/amber-manifest.json`
- **Required secret**: `openai_api_key`

### Building the Production Image

The production image bakes in the corpus and pre-built indexes for instant startup:

```bash
# Stage files
cp -r /path/to/treasury_bulletins_parsed/transformed .docker-corpus
mkdir -p .docker-faiss/openai
cp faiss_index/openai/{index.faiss,chunks.json,bm25.pkl,year_index.json} .docker-faiss/openai/

# Build and push (must be linux/amd64 for GitHub Actions runners)
docker buildx build --platform linux/amd64 -f Dockerfile.officeqa-agent \
  -t ghcr.io/zaidishahbaz1/officeqa-agent:latest --push .

# Clean up
rm -rf .docker-corpus .docker-faiss
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai`, `gemini`, or `anthropic` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Model name | `gpt-5.4` |
| `REASONING_EFFORT` | Reasoning level for GPT-5 | `medium` |
| `ENABLE_WEB_SEARCH` | Web search for external data | `false` |
| `RETRIEVAL_TOP_K` | Number of chunks to retrieve | `15` |
| `CORPUS_DIR` | Path to Treasury Bulletin `.txt` files | - |
| `FAISS_INDEX_DIR` | Path to pre-built FAISS/BM25 indexes | - |

## Architecture

```
Question
  |
  v
[Planner] --> structured plan (data points, constraints, time periods)
  |
  v
[Decomposer] --> 1-3 retrieval sub-queries
  |
  v
[FAISS + BM25 Retriever] --> top-K relevant chunks from 150K corpus
  |
  v
[GPT 5.4 + code_interpreter] --> reasoning + answer
  |
  v
[Formatter] --> canonicalized final answer
```

## Dataset

The [OfficeQA Dataset](https://github.com/databricks/officeqa) is publicly available:
- **Questions**: [officeqa.csv](https://github.com/databricks/officeqa/blob/main/officeqa.csv) (246 questions)
- **Source Documents**: [treasury_bulletins_parsed](https://github.com/databricks/officeqa/tree/main/treasury_bulletins_parsed) (698 bulletins, 1939-2025)

## License

- **Code**: Apache 2.0
- **Dataset**: CC-BY-SA 4.0
