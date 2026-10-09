"""
Step 2 · Data Analysis

Turns the raw StatsBomb match files into a single modelling dataset with one row per
pass: pitch coordinates, angle and distance, game state (clock and score difference),
pass height, body part, play pattern, pressure, and whether the pass was completed.

Matches are processed in parallel on all CPU cores but one. A match that fails to
parse is reported and skipped rather than stopping the whole run.

Output: data/processed/processed_passes_[<League>_<Season>_]<timestamp>.csv

Usage:
    python run_files/run_data_analysis.py
    python run_files/run_data_analysis.py --league "La Liga" --season "2015/2016"
"""
import argparse
import glob
import os
import sys
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_analysis_pipeline import process_match_json
from src.naming import find_subfolder, list_subfolders
from src import console


def process_file(file_path):
    """
    Processes one match file inside a worker process.

    Returns (file_name, match_df, error). Exceptions are caught and returned as text
    instead of raised, so one bad match can't bring down the whole worker pool.
    """
    file_name = os.path.basename(file_path)
    try:
        match_df = process_match_json(file_path)
        return file_name, match_df, None
    except Exception as e:
        return file_name, None, str(e)


def parse_args():
    """Reads the optional --league / --season scope from the command line."""
    parser = argparse.ArgumentParser(description="Extract pass features from StatsBomb match JSONs.")
    parser.add_argument("--league", help="Only process this league, e.g. \"La Liga\" (default: every match available)")
    parser.add_argument("--season", help="Only process this season of --league, e.g. \"2015/2016\"")
    args = parser.parse_args()

    if args.season and not args.league:
        parser.error("--season requires --league")

    return args


def main():
    args = parse_args()

    console.header("Data Analysis · StatsBomb JSON → pass features")

    json_directory = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb_free")
    output_directory = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_directory, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if not args.league:
        console.info(f"Scope: {console.describe_scope()}")
        output_path = os.path.join(output_directory, f"processed_passes_{date_str}.csv")
        json_pattern = os.path.join(json_directory, "**", "*.json")

    else:
        # Match the user's spelling ("la liga", "La-Liga") to the actual folder name on disk
        league_name = find_subfolder(json_directory, args.league)
        if league_name is None:
            console.error(f"No downloaded league matches '{args.league}'.")
            available = list_subfolders(json_directory)
            if available:
                console.listing(available, "Available")
            else:
                console.info("No leagues downloaded yet. Run run_data_ingestion.py first.")
            sys.exit(1)
        # Filename-friendly version of the league, e.g. "FA Women's Super League" -> "FA-Womens-Super-League"
        league_name_form = league_name.replace("'", "").replace(" ", "-")

        if args.season:
            league_dir = os.path.join(json_directory, league_name)
            season_name = find_subfolder(league_dir, args.season)
            if season_name is None:
                console.error(f"No downloaded season of {league_name} matches '{args.season}'.")
                console.listing(list_subfolders(league_dir), "Available")
                sys.exit(1)
            console.info(f"Scope: {console.describe_scope(league_name, season_name)}")
            output_path = os.path.join(output_directory, f"processed_passes_{league_name_form}_{season_name}_{date_str}.csv")
            json_pattern = os.path.join(json_directory, league_name, season_name, "*.json")

        else:
            console.info(f"Scope: {console.describe_scope(league_name)}")
            output_path = os.path.join(output_directory, f"processed_passes_{league_name_form}_all_seasons_{date_str}.csv")
            json_pattern = os.path.join(json_directory, league_name, "**", "*.json")


    json_files = glob.glob(json_pattern, recursive=True)

    if not json_files:
        console.error("No match files found for this scope. Run run_data_ingestion.py first.")
        sys.exit(1)

    # Use every core but one, so the machine stays responsive while the pool is running
    num_cores = max(1, mp.cpu_count() - 1)
    console.info(f"Found {len(json_files):,} match files · using {num_cores} processes")

    console.section("Processing matches")

    processed_matches_list = []
    total_files = len(json_files)
    failed = 0
    skipped = 0

    # Each worker has to start Python and import pandas first, so the first result takes a few seconds
    console.info(f"Starting {num_cores} worker processes...")
    console.progress(0, total_files, "waiting for the first match...")

    # imap_unordered hands back each match as soon as any worker finishes it,
    # which keeps the progress line moving steadily
    with mp.Pool(num_cores) as pool:
        for i, (file_name, match_df, error) in enumerate(pool.imap_unordered(process_file, json_files), start=1):

            if error:
                failed += 1
                console.warn(f"Skipping {file_name}: {error}")

            elif isinstance(match_df, pd.DataFrame) and not match_df.empty:
                processed_matches_list.append(match_df)

            else:
                skipped += 1
                console.warn(f"Skipping {file_name}: no usable passes.")

            console.progress(i, total_files, file_name)

    if processed_matches_list:
        console.section("Saving outputs")

        console.info(f"Combining {len(processed_matches_list):,} matches into one dataset...")
        start = time.time()
        master_df = pd.concat(processed_matches_list, ignore_index=True)
        console.success(f"Combined {len(master_df):,} passes ({time.time() - start:.1f}s)")

        console.info(f"Writing {len(master_df):,} rows to CSV (can take a few minutes on the full dataset)...")
        start = time.time()
        master_df.to_csv(output_path, index=False)
        console.success(f"Saved processed dataset: {os.path.basename(output_path)} ({time.time() - start:.1f}s)")

        console.section("Summary")
        console.kv("Matches processed", f"{len(processed_matches_list):,}/{total_files:,}")
        if failed:
            console.kv("Matches failed", f"{failed:,}")
        if skipped:
            console.kv("Matches empty", f"{skipped:,}")
        console.kv("Passes extracted", f"{len(master_df):,}")
        console.kv("Saved to", os.path.relpath(output_path, PROJECT_ROOT))

        console.done("Data analysis complete")

    else:
        console.error("No match produced usable passes. See the warnings above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
