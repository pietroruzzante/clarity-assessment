import json

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import AGENT_MODEL, OPENAI_API_KEY, logger
from src.tmdb import TMDBUnavailableError, get_trending_movies
from src.tracing import trace_node, usage_from_message

_llm = ChatOpenAI(model=AGENT_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)

SYSTEM_PROMPT = (
    "You are a movie expert. Given today's trending movies as JSON below, "
    "answer the user's question by recommending the most suitable ones. "
    "Always cite title, release year, and rating for each recommendation.\n\n"
    "Trending movies:\n{movies}"
)

FALLBACK_MESSAGE = (
    "I can't reach today's trending movies right now, please try again shortly "
    "— in the meantime I can recommend something from the Netflix catalog instead."
)


def trending_node(state: dict) -> dict:
    turn_id = state.get("turn_id", "unknown")
    with trace_node("trending_agent", turn_id) as trace:
        trace["route"] = "trending"
        try:
            movies = get_trending_movies()
        except TMDBUnavailableError:
            logger.warning("Trending agent degraded: TMDB unavailable.")
            trace["degraded"] = True
            return {"messages": state["messages"] + [AIMessage(content=FALLBACK_MESSAGE)]}

        system = SystemMessage(content=SYSTEM_PROMPT.format(movies=json.dumps(movies)))
        response = _llm.invoke([system] + state["messages"])
        trace["tokens"] = usage_from_message(response)
        return {"messages": state["messages"] + [response]}
