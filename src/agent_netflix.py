from langchain_core.messages import SystemMessage

from src.config import AGENT_MODEL, OPENAI_API_KEY
from src.rag import get_retriever
from langchain_openai import ChatOpenAI

_llm = ChatOpenAI(model=AGENT_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)

SYSTEM_PROMPT = (
    "You are a movie/show expert with access to the Netflix catalog. Recommend "
    "titles based only on the catalog excerpts below — do not invent titles that "
    "are not listed. Cite title, year, genre, and IMDb score when available.\n\n"
    "Catalog excerpts:\n{context}"
)


def _generate_reply(messages: list):
    """Thin wrapper so tests can mock this without touching the pydantic LLM object."""
    return _llm.invoke(messages)


def netflix_node(state: dict) -> dict:
    query = state["messages"][-1].content
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    system = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
    response = _generate_reply([system] + state["messages"])
    return {"messages": state["messages"] + [response]}
