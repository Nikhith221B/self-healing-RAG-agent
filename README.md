# Self-Healing RAG Agent

Agentic self-healing RAG backend with a critic-based feedback loop and retry mechanism. The system retrieves document chunks, generates a grounded answer, critiques grounding, reformulates the query when needed, and refuses when evidence is insufficient.

Built a production-style Self-Healing RAG API using FastAPI, LangGraph, ChromaDB, and Gemini that retrieves context, generates answers, critiques grounding, and retries with reformulated queries when responses are weak or hallucinated.

> This is **not** classical RL (PPO/DQN). It is a **feedback-driven agentic workflow** inspired by RLHF-style evaluation. Optional **offline learning** uses logged runs (`logs/runs.jsonl`) to analyze accept/refuse rates and improve over time.

## Why this project matters

Simple RAG often hallucinates or answers from weak retrieval. This pipeline **self-corrects within a request**: a critic checks grounding, the graph **retries retrieval** with a rewritten query, or **refuses** instead of inventing facts.

## Architecture

```text
User question
     │
     ▼
┌─────────────┐
│  Retrieve   │◄──────────────────┐
└──────┬──────┘                   │
       ▼                          │
┌─────────────┐                   │
│  Generate   │                   │
└──────┬──────┘                   │
       ▼                          │
┌─────────────┐     retry         │
│   Critic    ├───────────────────┤
└──────┬──────┘                   │
       │ accept                    │
       ▼              rewrite query│
  Final answer              ┌─────┴─────┐
       │                    │  Rewrite  │
 refuse (no evidence)       └───────────┘
       ▼
  Refusal message
```

## Tech stack

- **FastAPI** + Uvicorn — HTTP API and web UI host
- **LangGraph** — cyclic agent workflow
- **ChromaDB** + Gemini embeddings — vector store
- **Gemini** (`google-genai`) — generation and critic
- **PDFPlumber** — PDF ingestion

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Set `GEMINI_API_KEY` in `.env`.

Index sample documents:

```bash
python ingest.py
```

## Run the product

```bash
python -m uvicorn main:app --reload
```

Windows helper script:

```powershell
.\scripts\run_dev.ps1
```

> Prefer `python -m uvicorn` over `uvicorn.exe` if your virtualenv was moved/copied.

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/app/ | **Web UI** (upload + ask) |
| http://127.0.0.1:8000/docs | Swagger API docs |
| http://127.0.0.1:8000/health | Health check |

> On Windows, prefer `python -m uvicorn` over `uvicorn.exe` if the venv was moved.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required for embeddings and LLM |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `CHROMA_PERSIST_DIR` | `vectorstore` | Chroma persist path |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Collection name |
| `LOG_DIR` | `logs` | Ask-run logs for offline learning |
| `ENABLE_BANDIT` | `false` | Epsilon-greedy top_k selection (offline learning) |
| `DEFAULT_TOP_K` | `5` | Retrieval top-k |
| `MAX_RETRIES` | `2` | Max critic-driven retries |

## API

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `POST /ingest`

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "files=@data/sample_docs/hr_policy.txt"
```

### `POST /ask`

Returns a `run_id` for human feedback.

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"How many paid leaves are available for full-time employees?\",\"top_k\":5,\"max_retries\":2}"
```

### `POST /feedback`

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\":\"<run_id from /ask>\",\"rating\":\"helpful\"}"
```

Ratings: `helpful` or `not_helpful`.

### Example accepted response

```json
{
  "answer": "Full-time employees are eligible for 18 paid leaves per year.",
  "final_status": "accepted",
  "retry_count": 0,
  "original_query": "How many paid leaves are available for full-time employees?",
  "query_history": ["How many paid leaves are available for full-time employees?"],
  "critic_feedback": {
    "grounded": true,
    "score": 0.91,
    "decision": "accept",
    "supported_chunks": ["hr_policy.txt_0"],
    "unsupported_claims": [],
    "reason": "Supported by the leave policy chunk."
  },
  "critic_history": [],
  "sources": []
}
```

### Example refused response

```json
{
  "answer": "I don't have enough information in the provided documents to answer this reliably.",
  "final_status": "refused",
  "retry_count": 2,
  "critic_feedback": {
    "grounded": false,
    "decision": "refuse",
    "reason": "No chunk contains the requested information."
  },
  "sources": []
}
```

## Demo testing

After `python ingest.py` (indexes all files in `data/sample_docs/`):

| Type | Question |
|------|----------|
| **Answerable** | How many paid leaves are available for full-time employees? |
| **Answerable** | What is the 401(k) match percentage? |
| **Partially answerable** | Summarize all employee benefits and remote work rules. |
| **Unanswerable** | What is the CEO's personal phone number? |

Use the web UI at `/app/` or `POST /ask`.

## Offline learning

Each `/ask` call appends a row to `logs/runs.jsonl` (question, status, retries, critic scores). Human feedback goes to `logs/feedback.jsonl`.

Analyze logs:

```bash
python -m learning.analyze_logs
```

Optional **epsilon-greedy bandit** for `top_k` (arms 3 / 5 / 8): set `ENABLE_BANDIT=true` in `.env`. State is stored in `logs/bandit_state.json`.

Use aggregates to tune prompts, `top_k`, or rewrite strategies over time. This is **self-improvement from feedback logs**, not PPO training.

## Smoke tests

```bash
python -m unittest tests.test_smoke -v
```

## Project completion checklist

- [ ] `python ingest.py` indexes all 3 files under `data/sample_docs/`
- [ ] `python -m unittest tests.test_smoke -v` passes
- [ ] UI demo at `/app/` (accept + refuse questions)
- [ ] `logs/runs.jsonl` grows after `/ask`
- [ ] Add screenshots to `screenshots/` (see `screenshots/README.md`)
- [ ] Push to GitHub — confirm `.env` is not committed

## CLI (without UI)

```bash
python ingest.py
python retriever.py
python llm.py
python graph.py
```

## Limitations

- Requires a valid **Gemini API key** and network access.
- Critic is an LLM — can mis-judge accept/refuse/retry.
- Single Chroma collection on disk — not multi-tenant production scale.
- No authentication on API endpoints by default.
- Per-request self-healing only; no automatic model fine-tuning.

## Future improvements

- Evaluation dataset and retrieval metrics
- Learned reward model or contextual bandit for retry decisions
- Frontend auth and multi-user workspaces
- Docker Compose deployment
- Human feedback on `/ask` outcomes

## Sample documents

- `data/sample_docs/hr_policy.txt`
- `data/sample_docs/benefits_policy.txt`
- `data/sample_docs/remote_work_policy.txt`
