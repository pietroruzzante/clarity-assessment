import uuid
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agent_netflix import netflix_node
from src.agent_trending import trending_node
from src.config import OPENAI_API_KEY, ROUTER_MODEL, logger
from src.tracing import trace_node

Route = Literal["trending", "netflix_catalog", "off_topic"]

OFF_TOPIC_MESSAGE = (
    "I'm a movie recommendation assistant, so I can only help with movies and "
    "TV shows — try asking about trending releases or the Netflix catalog!"
)

TRENDING_KEYWORDS = ("trending", "today", "lately", "recent", "now playing", "new movie")
NETFLIX_KEYWORDS = ("netflix", "show", "series", "documentary", "documentaries")


class RouteDecision(BaseModel):
    route: Route


class ChatState(TypedDict):
    messages: list[BaseMessage]
    route: Route
    turn_id: str


_router_llm = ChatOpenAI(model=ROUTER_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_structured_router = _router_llm.with_structured_output(RouteDecision)

ROUTER_PROMPT = (
    "Classify the user's latest message into one of: 'trending' (current/recent "
    "movies, what's popular today), 'netflix_catalog' (Netflix movies/shows by "
    "genre, mood, or theme), or 'off_topic' (anything not about movie/show "
    "recommendations). The previous turn's topic was '{previous_route}'. A short "
    "follow-up (e.g. 'something funnier', 'anything else?', 'what about action') "
    "that doesn't introduce a new subject continues that same topic — classify it "
    "as '{previous_route}', not 'off_topic'."
)


def _keyword_fallback(query: str, previous_route: Route) -> Route:
    lowered = query.lower()
    if any(word in lowered for word in TRENDING_KEYWORDS):
        return "trending"
    if any(word in lowered for word in NETFLIX_KEYWORDS):
        return "netflix_catalog"
    return previous_route


def _classify_route(messages: list) -> RouteDecision:
    """Thin wrapper so tests can mock this without touching the pydantic LLM object."""
    return _structured_router.invoke(messages)


def router_node(state: ChatState) -> ChatState:
    query = state["messages"][-1].content
    previous_route = state.get("route") or "netflix_catalog"
    turn_id = state.get("turn_id") or str(uuid.uuid4())[:8]

    with trace_node("router", turn_id) as trace:
        try:
            system_prompt = ROUTER_PROMPT.format(previous_route=previous_route)
            decision = _classify_route(
                [{"role": "system", "content": system_prompt}] + state["messages"]
            )
            route = decision.route
        except Exception as exc:
            logger.warning("Router LLM call failed (%s), using keyword fallback.", exc)
            route = _keyword_fallback(query, previous_route)
            trace["fallback_triggered"] = True
        trace["route"] = route

    return {"messages": state["messages"], "route": route, "turn_id": turn_id}


def _off_topic_node(state: ChatState) -> ChatState:
    return {"messages": state["messages"] + [AIMessage(content=OFF_TOPIC_MESSAGE)]}


def _route_edge(state: ChatState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("router", router_node)
    graph.add_node("trending_agent", trending_node)
    graph.add_node("netflix_agent", netflix_node)
    graph.add_node("off_topic", _off_topic_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_edge,
        {
            "trending": "trending_agent",
            "netflix_catalog": "netflix_agent",
            "off_topic": "off_topic",
        },
    )
    graph.add_edge("trending_agent", END)
    graph.add_edge("netflix_agent", END)
    graph.add_edge("off_topic", END)

    return graph.compile()
