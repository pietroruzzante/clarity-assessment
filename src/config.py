import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Placeholder keeps client construction working at import time (e.g. for tests
# that mock network calls); a real .env value is required for actual runs.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "sk-placeholder"
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

ROUTER_MODEL = "gpt-4o-mini"
AGENT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_DIR = "./chroma_db"
TITLES_CSV = "data/titles.csv"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename="app.log",
)
logging.getLogger().handlers[0].setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("movie_chatbot")
