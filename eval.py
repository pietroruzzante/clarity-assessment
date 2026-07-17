"""Standalone LLM-as-judge evaluation script.

Runs the 6 example queries from the assessment through the real graph and
scores each response for relevance and groundedness. Makes real OpenAI/TMDB
API calls — run manually (`python eval.py`), not part of `pytest -q`.
"""
import json
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import AGENT_MODEL, OPENAI_API_KEY
from src.graph import build_graph
from src.rag import get_retriever
from src.tmdb import TMDBUnavailableError, get_trending_movies

QUERIES = [
    "Which recent movie do you recommend to watch today?",
    "What is the current best movie released lately?",
    "Is there a good movie to watch about superheroes lately?",
    "Find me a movie to watch on the TV about action and spies",
    "What are some good Netflix nature documentaries?",
    "I want to see a romantic comedy movie. What do you recommend?",
]


class JudgeScore(BaseModel):
    relevance: int
    grounded: bool
    notes: str


_judge_llm = ChatOpenAI(model=AGENT_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_structured_judge = _judge_llm.with_structured_output(JudgeScore)

JUDGE_PROMPT = (
    "You are grading a movie recommendation chatbot's response.\n\n"
    "User query: {query}\n\n"
    "Context the agent had access to (grounding source):\n{context}\n\n"
    "Agent's response:\n{response}\n\n"
    "Score:\n"
    "- relevance (1-5): does the response actually address the user's query?\n"
    "- grounded (true/false): are all movie/show titles cited in the response "
    "actually present in the context above? If any cited title is not in the "
    "context, grounded must be false.\n"
    "- notes: one short sentence justifying the scores."
)


def _grounding_context(route: str, query: str) -> str:
    """Fetch the same grounding source the agent used, so the judge doesn't guess."""
    if route == "trending":
        try:
            return json.dumps(get_trending_movies())
        except TMDBUnavailableError:
            return "(TMDB unavailable at evaluation time)"
    if route == "netflix_catalog":
        docs = get_retriever().invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    return "(off-topic query — no grounding context)"


def judge(query: str, route: str, response: str) -> JudgeScore:
    context = _grounding_context(route, query)
    prompt = JUDGE_PROMPT.format(query=query, context=context, response=response)
    return _structured_judge.invoke(prompt)


def run_eval() -> list[dict]:
    graph = build_graph()
    rows = []
    for query in QUERIES:
        state = {"messages": [HumanMessage(content=query)], "route": "netflix_catalog"}
        start = time.perf_counter()
        result = graph.invoke(state)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        route = result["route"]
        response = result["messages"][-1].content
        score = judge(query, route, response)

        rows.append(
            {
                "query": query,
                "route": route,
                "relevance": score.relevance,
                "grounded": score.grounded,
                "notes": score.notes,
                "latency_ms": latency_ms,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "avg_relevance": sum(r["relevance"] for r in rows) / len(rows),
        "pct_grounded": 100 * sum(r["grounded"] for r in rows) / len(rows),
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
    }


def write_report(rows: list[dict], path: str = "eval_results.md") -> None:
    summary = summarize(rows)
    lines = [
        f"# Evaluation Results ({datetime.now(timezone.utc).isoformat()})",
        "",
        f"Avg relevance: {summary['avg_relevance']:.2f}/5 · "
        f"Grounded: {summary['pct_grounded']:.0f}% · "
        f"Avg latency: {summary['avg_latency_ms']:.0f} ms",
        "",
        "| Query | Route | Relevance | Grounded | Latency (ms) | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['query']} | {r['route']} | {r['relevance']}/5 | "
            f"{'yes' if r['grounded'] else 'no'} | {r['latency_ms']:.0f} | {r['notes']} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run_eval()
    write_report(results)
    summary = summarize(results)
    print(f"Avg relevance: {summary['avg_relevance']:.2f}/5")
    print(f"Grounded: {summary['pct_grounded']:.0f}%")
    print(f"Avg latency: {summary['avg_latency_ms']:.0f} ms")
    print("Full results written to eval_results.md")
