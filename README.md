# OfficeQA Purple Agent

A purple agent for the [OfficeQA benchmark](https://github.com/databricks/officeqa) on the [AgentBeats](https://agentbeats.dev) platform. Answers questions about U.S. Treasury Bulletin documents (1939-2025) using retrieval-augmented generation.

## How It Works

1. **Analyze**: a fast LLM call (reasoning_effort=low) extracts target years and a keyword-rich search query from the question.
2. **Retrieve**: BM25 search over ~70K chunks (CHUNK_CHARS=3000, overlap=400) with year-based filtering returns the top 22 excerpts.
3. **Answer**: GPT 5.4 with `code_interpreter` reads the excerpts and produces a structured response wrapped in `<REASONING>` and `<FINAL_ANSWER>` tags.
4. **Refine**: if the draft admits it could not find the data, a second retrieval + LLM call (effort=high) attempts recovery.
5. **Normalize**: output canonicalization fixes list formatting (space after comma), de-hedges multi-number answers, and falls back to regex extraction if the tags are missing.

The prompt has explicit rules for rounding, sign preservation, percent formatting, and unit conversion (millions vs thousands).

## Project Structure

```
v2/                           # The production agent (current)
  src/
    server.py                 # A2A server (port 8080)
    executor.py               # A2A request handling + lazy corpus init
    agent.py                  # Analyze -> retrieve -> answer -> refine pipeline
    corpus.py                 # Corpus download + chunking + BM25
  Dockerfile
  requirements.txt

amber-manifest.json           # AgentBeats manifest (v0.2.0)

participant/                  # Legacy v1 agent (pre-v2-manifest pivot)
judge/                        # Local green agent for dev testing
```

## Leaderboard Submission

We use AgentBeats Quick Submit at https://agentbeats.dev/agentbeater/officeqa/submit.

**Docker image:** `ghcr.io/zaidishahbaz1/officeqa-agent:latest`

**Manifest URL:** `https://raw.githubusercontent.com/soumya-batra/officeqa_agentbeats/main/amber-manifest.json`

**Quick Submit config:**

| Field | Value |
|-------|-------|
| OPENAI_API_KEY | your OpenAI key |
| OPENAI_MODEL | `gpt-5.4` |
| REASONING_EFFORT | `high` |
| Config JSON | `{"max_concurrent": 1}` |

`max_concurrent=1` is important. It runs questions sequentially so each call gets its full time budget and avoids 504 timeouts from `code_interpreter`.

## Building the Docker Image

```bash
cd v2
docker buildx build --platform linux/amd64 \
  -t ghcr.io/zaidishahbaz1/officeqa-agent:latest \
  --push .
```

The image must be `linux/amd64` because the platform runs on GitHub Actions amd64 runners.

The image does NOT bake in the corpus. It downloads the Treasury Bulletin zip from GitHub on first request, then builds the BM25 index in memory (about 30 seconds). This keeps the image small (about 200MB).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required |
| `OPENAI_MODEL` | `gpt-5.4-mini` | OpenAI model |
| `REASONING_EFFORT` | `medium` | `low`, `medium`, or `high` |
| `ENABLE_WEB_SEARCH` | `false` | Enable OpenAI web_search tool |
| `TOP_K` | `22` | BM25 results per query |
| `TOP_K_REFINE` | `18` | BM25 results for refine pass |
| `MAX_CHARS_PER_CHUNK` | `2500` | Max chars per excerpt in prompt |
| `CORPUS_CACHE_DIR` | `/data/corpus` | Where the corpus zip is cached |

## Dataset

[OfficeQA Dataset](https://github.com/databricks/officeqa):
- **Questions**: `officeqa.csv` (246 questions, easy + hard difficulty)
- **Corpus**: 697 U.S. Treasury Bulletins (1939-2025)

## License

- Code: Apache 2.0
- Dataset: CC-BY-SA 4.0
