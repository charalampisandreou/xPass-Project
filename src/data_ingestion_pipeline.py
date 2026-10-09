"""
Downloads StatsBomb's open data (https://github.com/hudl/open-data) to disk.

The data is published as three kinds of JSON file, fetched in this order:

    competitions.json                          every competition and season available
    matches/<competition_id>/<season_id>.json  the matches in one season
    events/<match_id>.json                     every event in one match (passes, shots, ...)

Event files are saved as <target_dir>/<League>/<Season>/<match_id>.json. That layout is
what the later steps rely on to filter by league and season.
"""
import requests
import os
import json
import time

from src import console
from src.naming import names_match

BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/refs/heads/master/data"
REQUEST_TIMEOUT = 30  # seconds; stops a stalled connection from hanging the pipeline


def _get_json(url: str):
    """GETs a JSON file. Returns None on any network error or non-200 response."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return response.json()


def download_statsbomb_free_data(target_dir: str = "data/raw/statsbomb_free", league: str = None, season: str = None):
    """
    Downloads match event files for every competition-season, or for one league / season.

    Files that already exist are skipped, so this is cheap to re-run and only fetches new
    matches. A season or match that fails to download is reported and skipped.

    Returns a summary dict with the keys "seasons", "downloaded", "skipped" and "failed".

    Raises:
        ConnectionError: the competitions index couldn't be fetched (usually no internet).
        ValueError(message, available_names): `league` or `season` matched nothing. The
            second argument lists the valid names so the caller can show them.
    """
    console.info("Fetching the competitions index from StatsBomb...")
    competitions = _get_json(f"{BASE_URL}/competitions.json")

    if competitions is None:
        raise ConnectionError("Could not fetch the StatsBomb competitions index.")

    selected = [
        comp for comp in competitions
        if (not league or names_match(comp['competition_name'], league))
        and (not season or names_match(comp['season_name'], season))
    ]

    # Nothing matched: work out whether the league or the season was wrong, and list the valid options
    if not selected:
        if league and not any(names_match(c['competition_name'], league) for c in competitions):
            available = sorted({c['competition_name'] for c in competitions})
            raise ValueError(f"No StatsBomb league matches '{league}'.", available)
        available = sorted(c['season_name'] for c in competitions if names_match(c['competition_name'], league))
        raise ValueError(f"No StatsBomb season of {league} matches '{season}'.", available)

    console.success(f"{len(selected)} season(s) selected")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for comp in selected:

        competition_id = comp['competition_id']
        season_id = comp['season_id']

        # "/" can't appear in folder names, so "2015/2016" is stored as "2015-2016"
        competition_name = comp['competition_name'].replace("/", "-")
        season_name = comp['season_name'].replace("/", "-")
        label = f"{comp['competition_name']} {comp['season_name']}"

        season_dir = os.path.join(target_dir, competition_name, season_name)
        os.makedirs(season_dir, exist_ok=True)

        matches = _get_json(f"{BASE_URL}/matches/{competition_id}/{season_id}.json")

        if matches is None:
            console.warn(f"Skipping {label}: could not fetch the match list.")
            continue

        new_files = 0
        failed_ids = []

        for i, match in enumerate(matches, start=1):
            match_id = match['match_id']

            match_path = os.path.join(season_dir, f"{match_id}.json")

            if os.path.exists(match_path):
                total_skipped += 1
            else:
                events = _get_json(f"{BASE_URL}/events/{match_id}.json")

                if events is not None:
                    with open(match_path, 'w', encoding = 'utf-8') as f:
                        json.dump(events, f)
                    total_downloaded += 1
                    new_files += 1
                    # Short pause between downloads to be polite to GitHub's servers
                    time.sleep(0.05)
                else:
                    failed_ids.append(match_id)

            console.progress(i, len(matches), label)

        total_failed += len(failed_ids)
        status = f"{new_files} new" if new_files else "up to date"
        console.success(f"{label} · {len(matches)} matches ({status})")
        if failed_ids:
            console.warn(f"Skipping {len(failed_ids)} match(es) that failed to download: {', '.join(map(str, failed_ids))}.")

    return {
        "seasons": len(selected),
        "downloaded": total_downloaded,
        "skipped": total_skipped,
        "failed": total_failed,
    }
