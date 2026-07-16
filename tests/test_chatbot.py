"""Mocked test suite — no real network/API calls, no keys required."""
import requests
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import src.agent_netflix as agent_netflix
import src.graph as graph_mod
import src.main as main_mod
import src.rag as rag_mod
import src.tmdb as tmdb_mod
from src.tmdb import TMDBUnavailableError, get_trending_movies


# ---- Router ----------------------------------------------------------

EXAMPLE_QUERIES = {
    "Which recent movie do you recommend to watch today?": "trending",
    "What is the current best movie released lately?": "trending",
    "Is there a good movie to watch about superheroes lately?": "trending",
    "Find me a movie to watch on the TV about action and spies": "netflix_catalog",
    "What are some good Netflix nature documentaries?": "netflix_catalog",
    "I want to see a romantic comedy movie. What do you recommend?": "netflix_catalog",
}


def test_router_classifies_example_queries(mocker):
    for query, expected_route in EXAMPLE_QUERIES.items():
        mocker.patch.object(
            graph_mod,
            "_classify_route",
            return_value=graph_mod.RouteDecision(route=expected_route),
        )
        state = {"messages": [HumanMessage(content=query)], "route": "netflix_catalog"}
        result = graph_mod.router_node(state)
        assert result["route"] == expected_route


def test_router_off_topic(mocker):
    mocker.patch.object(
        graph_mod,
        "_classify_route",
        return_value=graph_mod.RouteDecision(route="off_topic"),
    )
    state = {"messages": [HumanMessage(content="what's the weather?")], "route": "netflix_catalog"}
    result = graph_mod.router_node(state)
    assert result["route"] == "off_topic"


def test_router_keyword_fallback_on_llm_failure(mocker):
    mocker.patch.object(graph_mod, "_classify_route", side_effect=RuntimeError("boom"))
    state = {"messages": [HumanMessage(content="what's trending today?")], "route": "netflix_catalog"}
    result = graph_mod.router_node(state)
    assert result["route"] == "trending"


def test_router_keyword_fallback_defaults_to_previous_route(mocker):
    mocker.patch.object(graph_mod, "_classify_route", side_effect=RuntimeError("boom"))
    state = {"messages": [HumanMessage(content="anything else?")], "route": "trending"}
    result = graph_mod.router_node(state)
    assert result["route"] == "trending"


def test_off_topic_node_returns_fixed_message():
    state = {"messages": [HumanMessage(content="hi")], "route": "off_topic"}
    result = graph_mod._off_topic_node(state)
    assert result["messages"][-1].content == graph_mod.OFF_TOPIC_MESSAGE


# ---- TMDB --------------------------------------------------------------

def test_get_trending_movies_parses_results(mocker):
    payload = {
        "results": [
            {
                "title": "Movie A",
                "overview": "desc",
                "release_date": "2026-01-01",
                "vote_average": 8.1,
                "popularity": 100.0,
            }
        ]
    }
    mock_response = mocker.Mock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    mocker.patch.object(tmdb_mod.requests, "get", return_value=mock_response)

    movies = get_trending_movies()
    assert movies == [
        {
            "title": "Movie A",
            "overview": "desc",
            "release_date": "2026-01-01",
            "rating": 8.1,
            "popularity": 100.0,
        }
    ]


def test_tmdb_retries_then_raises_unavailable(mocker):
    mocker.patch.object(
        tmdb_mod.requests, "get", side_effect=requests.RequestException("network down")
    )
    sleep_patch = mocker.patch("time.sleep", return_value=None)
    try:
        get_trending_movies()
        assert False, "expected TMDBUnavailableError"
    except TMDBUnavailableError:
        pass
    assert tmdb_mod.requests.get.call_count == 3


# ---- Netflix RAG agent ---------------------------------------------------

def test_netflix_node_uses_retrieved_documents(mocker):
    docs = [Document(page_content="Our Planet (SHOW): a nature documentary. Genres: Documentation")]
    mock_retriever = mocker.Mock()
    mock_retriever.invoke.return_value = docs
    mocker.patch.object(agent_netflix, "get_retriever", return_value=mock_retriever)

    mock_response = AIMessage(content="I recommend Our Planet.")
    mock_generate = mocker.patch.object(agent_netflix, "_generate_reply", return_value=mock_response)

    state = {"messages": [HumanMessage(content="good nature documentaries?")], "route": "netflix_catalog"}
    result = agent_netflix.netflix_node(state)

    called_messages = mock_generate.call_args[0][0]
    system_message = called_messages[0]
    assert "Our Planet" in system_message.content
    assert result["messages"][-1] == mock_response


def test_rag_ingestion_skipped_when_collection_populated(mocker):
    mocker.patch.object(rag_mod._store._collection, "count", return_value=42)
    add_documents = mocker.patch.object(rag_mod._store, "add_documents")

    rag_mod._ingest_if_needed()

    add_documents.assert_not_called()


# ---- Error hiding --------------------------------------------------------

def test_handle_query_hides_exception_from_user(mocker):
    fake_graph = mocker.Mock()
    fake_graph.stream.side_effect = RuntimeError("something broke deep in a node")
    mocker.patch.object(main_mod, "console")

    state = {"messages": [], "route": "netflix_catalog"}
    result = main_mod.handle_query(fake_graph, state, "recommend me something")

    printed_texts = [call.args[0] for call in main_mod.console.print.call_args_list if call.args]
    assert any(main_mod.GENERIC_ERROR in text for text in printed_texts)
    assert not any("RuntimeError" in text or "Traceback" in text for text in printed_texts)
