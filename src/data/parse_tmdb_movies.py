"""Parse additional movie data from TMDb API.

The main project dataset is loaded from Kaggle, but this script provides an independent
parsing step required for the course checkpoint. It collects popular movies from TMDb,
loads details for each movie and writes a CSV file that can be used as an auxiliary
source or for sanity checks against the main dataset.

Usage:
    export TMDB_API_KEY=<your_key>
    python -m src.data.parse_tmdb_movies --pages 3
    --output data/processed/parsed_tmdb_movies_sample.csv
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_BASE_URL = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT_SECONDS = 20
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.25


class TmdbClientError(RuntimeError):
    """Raised when the TMDb API request fails."""


def fetch_json(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch a JSON object from TMDb API."""
    query_params = {"api_key": api_key}
    if params:
        query_params.update(params)

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    response = session.get(url, params=query_params, timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise TmdbClientError(
            f"TMDb request failed: status={response.status_code}, url={response.url}, "
            f"body={response.text[:300]}"
        )
    return response.json()


def parse_movie_details(movie: dict[str, Any]) -> dict[str, Any]:
    """Convert TMDb movie details JSON into a flat row."""
    genres = movie.get("genres") or []
    production_companies = movie.get("production_companies") or []
    production_countries = movie.get("production_countries") or []
    spoken_languages = movie.get("spoken_languages") or []

    return {
        "tmdb_id": movie.get("id"),
        "title": movie.get("title"),
        "original_title": movie.get("original_title"),
        "release_date": movie.get("release_date"),
        "budget": movie.get("budget"),
        "revenue": movie.get("revenue"),
        "runtime": movie.get("runtime"),
        "popularity": movie.get("popularity"),
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "original_language": movie.get("original_language"),
        "overview": movie.get("overview"),
        "homepage": movie.get("homepage"),
        "genres": ", ".join(item.get("name", "") for item in genres if item.get("name")),
        "companies": ", ".join(
            item.get("name", "") for item in production_companies if item.get("name")
        ),
        "countries": ", ".join(
            item.get("name", "") for item in production_countries if item.get("name")
        ),
        "languages": ", ".join(
            item.get("english_name", "") for item in spoken_languages if item.get("english_name")
        ),
    }


def collect_movies(api_key: str, pages: int, language: str = "en-US") -> pd.DataFrame:
    """Collect movie details from TMDb discover endpoint."""
    rows: list[dict[str, Any]] = []

    with requests.Session() as session:
        for page in range(1, pages + 1):
            discover_data = fetch_json(
                session=session,
                endpoint="discover/movie",
                api_key=api_key,
                params={
                    "page": page,
                    "language": language,
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                },
            )
            for movie_short in discover_data.get("results", []):
                movie_id = movie_short.get("id")
                if movie_id is None:
                    continue
                details = fetch_json(
                    session=session,
                    endpoint=f"movie/{movie_id}",
                    api_key=api_key,
                    params={"language": language},
                )
                rows.append(parse_movie_details(details))
                time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    return pd.DataFrame(rows).drop_duplicates(subset=["tmdb_id"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse movie data from TMDb API.")
    parser.add_argument("--api-key", default=os.getenv("TMDB_API_KEY"), help="TMDb API key.")
    parser.add_argument("--pages", type=int, default=3, help="Number of discover pages to parse.")
    parser.add_argument(
        "--language",
        default="en-US",
        help="TMDb language code used for requests, for example en-US.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/parsed_tmdb_movies_sample.csv"),
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the parser and save CSV output."""
    args = parse_args()
    if not args.api_key:
        raise ValueError("TMDb API key is required. Set TMDB_API_KEY or pass --api-key explicitly.")
    if args.pages < 1:
        raise ValueError("--pages must be a positive integer.")

    movies = collect_movies(api_key=args.api_key, pages=args.pages, language=args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(args.output, index=False)
    print(f"Saved {len(movies)} rows to {args.output}")


if __name__ == "__main__":
    main()
