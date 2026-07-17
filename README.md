# Pietro Ruzzante - Clarity Assessment -  Movie Recommendation Chatbot

An agentic movie recommendation chatbot with two specialized agents — trending movies
(TMDB) and Netflix catalog search (RAG) — routed by an LLM, built on LangGraph, with
a streaming CLI.

## Requirements

- Python 3.10+

## Setup

```bash
git clone https://github.com/pietroruzzante/clarity-assessment.git
cd clarity-assessment
python3.10 -m venv .venv          # or: python3 -m venv .venv, if python3 --version is already 3.10+
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Edit `.env` and set:

```
OPENAI_API_KEY=<your OpenAI gateway key>
TMDB_API_KEY=<your TMDB Read Access Token (v4 auth)>
```
The Netflix dataset (`data/titles.csv`, from [Kaggle](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies))
is already included in the repo, so no Kaggle account is needed.

## Run

From your terminal just run: 

```bash
moviebot
```

(`pip install -e .` registers this console command; if you skipped that step,
`python -m src.main` works identically.)

On first run, the Netflix catalog is embedded into a local Chroma index
(`./chroma_db`) — this takes 1-2 minutes and a few cents in embedding calls.
Subsequent runs reuse the cached index.

Type `exit` or `quit` to leave, or press Ctrl+C.

### Example queries

- "Which recent movie do you recommend to watch today?" — trending agent
- "What is the current best movie released lately?" — trending agent
- "What are some good Netflix nature documentaries?" — Netflix RAG agent
- "I want to see a romantic comedy movie. What do you recommend?" — Netflix RAG agent
- "What's the weather today?" — off-topic, politely declined

## Tests

```bash
pytest -q
```

All 12 tests run fully mocked — no API keys or network access required.

## Evaluation & Observability

```bash
python eval.py
```

Runs the 6 example queries from the assessment through the real graph, scores
each response with an LLM-as-judge (relevance 1-5, groundedness against the
actual TMDB/Netflix context the agent used), and writes `eval_results.md`
(a summary line + a per-query table) while also printing the summary to
stdout. Unlike `pytest`, this makes real OpenAI/TMDB API calls, so it needs a
valid `.env` — run it manually when you want a snapshot of response quality,
not on every commit.

Every node execution (router, trending agent, netflix agent) also appends one
JSON line to `traces.jsonl` — timestamp, `turn_id` (shared by all nodes in the
same user turn), node name, latency, the route decided, whether the router's
keyword fallback fired, whether the trending agent degraded (TMDB down), any
error, and token usage when available. Inspect it while chatting:

```bash
tail -f traces.jsonl        # watch traces live in another terminal
cat traces.jsonl | jq .     # pretty-print everything so far
```

## Project layout

```
src/
├── main.py           # CLI REPL, streaming output, error hiding
├── graph.py           # LangGraph state, router, orchestration
├── agent_trending.py   # Agent 1: TMDB trending movies
├── agent_netflix.py     # Agent 2: Netflix RAG
├── tmdb.py                # TMDB client with retry
├── rag.py                  # Netflix CSV ingestion + Chroma retriever
├── tracing.py                # structured JSONL tracing (traces.jsonl)
└── config.py                   # env settings, logging
tests/test_chatbot.py             # mocked pytest suite
eval.py                             # LLM-as-judge evaluation script (manual, real API calls)
```

See `REPORT.md` for design rationale, assumptions, and next steps.
