"""
Step 1 · Data Ingestion

Downloads StatsBomb's free event data into data/raw/statsbomb_free/<League>/<Season>/,
one JSON file per match. Matches that are already on disk are skipped, so re-running
this step only fetches what's new.

If GitHub can't be reached but match files already exist locally, the step finishes
with a warning instead of failing, so the rest of the pipeline can still run offline.

Usage:
    python run_files/run_data_ingestion.py
    python run_files/run_data_ingestion.py --league "La Liga" --season "2015/2016"
"""
import argparse
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_ingestion_pipeline import download_statsbomb_free_data
from src import console


def parse_args():
    """Reads the optional --league / --season scope from the command line."""
    parser = argparse.ArgumentParser(description="Download StatsBomb open data (only new matches are fetched).")
    parser.add_argument("--league", help="Only download this league, e.g. \"La Liga\" (default: every competition)")
    parser.add_argument("--season", help="Only download this season of --league, e.g. \"2015/2016\"")
    args = parser.parse_args()

    if args.season and not args.league:
        parser.error("--season requires --league")

    return args


def main():
    args = parse_args()

    console.header("Data Ingestion · StatsBomb open data")

    target_dir = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb_free")
    console.info(f"Scope: {console.describe_scope(args.league, args.season)}")

    try:
        summary = download_statsbomb_free_data(target_dir, league=args.league, season=args.season)

    except ValueError as e:
        # The league / season didn't match anything; the error carries the valid names
        message, available = e.args
        console.error(message)
        console.listing(available, "Available")
        sys.exit(1)

    except ConnectionError as e:
        # Being offline is fine as long as there is already local data to work with
        local_files = glob.glob(os.path.join(target_dir, "**", "*.json"), recursive=True)
        if local_files:
            console.warn(f"{e} Continuing with {len(local_files):,} local match files.")
            console.done("Data ingestion skipped (offline)")
            return
        console.error(f"{e} Check your internet connection.")
        sys.exit(1)

    console.section("Summary")
    console.kv("Seasons checked", f"{summary['seasons']:,}")
    console.kv("Matches downloaded", f"{summary['downloaded']:,}")
    console.kv("Matches already present", f"{summary['skipped']:,}")
    if summary["failed"]:
        console.kv("Matches failed", f"{summary['failed']:,}")
    console.kv("Saved to", os.path.relpath(target_dir, PROJECT_ROOT))

    console.done("Data ingestion complete")


if __name__ == "__main__":
    main()
