from langchain_core.messages import HumanMessage
from rich.console import Console

from src.config import logger
from src.graph import build_graph

console = Console()

WELCOME = """[bold cyan]Movie Recommendation Chatbot[/bold cyan]
Ask about trending movies or the Netflix catalog. Type 'exit' to quit.

Examples:
  - Which recent movie do you recommend to watch today?
  - What are some good Netflix nature documentaries?
"""

GENERIC_ERROR = "Something went wrong on my side, please try again."
AGENT_NODES = {"trending_agent", "netflix_agent", "off_topic"}


def _stream_turn(graph, state: dict) -> dict:
    """Stream agent tokens live, return the final graph state for the turn."""
    console.print("[dim]Bot:[/dim] ", end="")
    streamed_any = False
    final_state = state

    for mode, payload in graph.stream(state, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") in AGENT_NODES:
                token = getattr(chunk, "content", "")
                if token:
                    console.print(token, end="", style="green")
                    streamed_any = True
        elif mode == "updates":
            for node_state in payload.values():
                if "messages" in node_state:
                    final_state = node_state

    if not streamed_any:
        # Fixed responses (e.g. off-topic) don't stream token-by-token.
        console.print(final_state["messages"][-1].content, style="green")

    console.print()
    return final_state


def handle_query(graph, state: dict, query: str) -> dict:
    """Run one turn; on any unhandled error, hide it behind a generic message."""
    state["messages"] = state["messages"] + [HumanMessage(content=query)]
    try:
        final_state = _stream_turn(graph, state)
        state["messages"] = final_state["messages"]
        state["route"] = final_state.get("route", state.get("route"))
    except Exception:
        logger.exception("Unhandled error during turn.")
        console.print(f"[dim]Bot:[/dim] [red]{GENERIC_ERROR}[/red]")
    return state


def run():
    console.print(WELCOME)
    graph = build_graph()
    state = {"messages": [], "route": "netflix_catalog"}

    while True:
        try:
            query = console.input("[bold]You:[/bold] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if query.strip().lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not query.strip():
            continue

        state = handle_query(graph, state, query)


if __name__ == "__main__":
    run()
