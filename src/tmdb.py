import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import TMDB_API_KEY, logger

TRENDING_URL = "https://api.themoviedb.org/3/trending/movie/day"


class TMDBUnavailableError(Exception):
    """Raised when TMDB can't be reached after retries."""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _fetch_trending() -> dict:
    response = requests.get(
        TRENDING_URL,
        headers={"Authorization": f"Bearer {TMDB_API_KEY}"},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


def get_trending_movies(limit: int = 15) -> list[dict]:
    """Return today's trending movies (title, overview, date, rating, popularity)."""
    try:
        data = _fetch_trending()
    except requests.RequestException as exc:
        logger.warning("TMDB unavailable after retries: %s", exc)
        raise TMDBUnavailableError from exc

    movies = []
    for item in data.get("results", [])[:limit]:
        movies.append(
            {
                "title": item.get("title"),
                "overview": item.get("overview"),
                "release_date": item.get("release_date"),
                "rating": item.get("vote_average"),
                "popularity": item.get("popularity"),
            }
        )
    return movies
