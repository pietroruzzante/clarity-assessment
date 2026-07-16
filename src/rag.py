import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, TITLES_CSV, logger

_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
_store = Chroma(
    collection_name="netflix_titles",
    embedding_function=_embeddings,
    persist_directory=CHROMA_DIR,
)


def _ingest_if_needed() -> None:
    if _store._collection.count() > 0:
        logger.info("Chroma collection already populated, skipping ingestion.")
        return

    if not os.path.exists(TITLES_CSV):
        logger.error("Missing %s, cannot ingest Netflix catalog.", TITLES_CSV)
        return

    logger.info("Ingesting Netflix titles into Chroma (first run)...")
    df = pd.read_csv(TITLES_CSV)
    df = df.dropna(subset=["description"])

    documents = []
    for _, row in df.iterrows():
        text = f"{row.get('title', '')} ({row.get('type', '')}): {row.get('description', '')} Genres: {row.get('genres', '')}"
        metadata = {
            "title": str(row.get("title", "")),
            "type": str(row.get("type", "")),
            "release_year": str(row.get("release_year", "")),
            "genres": str(row.get("genres", "")),
            "imdb_score": str(row.get("imdb_score", "")),
        }
        documents.append(Document(page_content=text, metadata=metadata))

    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        _store.add_documents(documents[i : i + batch_size])
    logger.info("Ingested %d Netflix titles.", len(documents))


def get_retriever(k: int = 8):
    _ingest_if_needed()
    return _store.as_retriever(search_kwargs={"k": k})
