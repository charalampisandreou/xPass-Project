"""
Step 6 · Visualizations

Draws four figures from the newest xP dataset:

    - a leaderboard table of the top players by pass value added
    - a risk vs. execution scatter (mean xP against actual completion %, per player)
    - a pass map for the best-performing player
    - a team quadrant chart (mean xP against CPOE, completion percentage over expected, per team)

The comparisons only mean something inside one league and season, so when the dataset
covers every league the step picks a scope first: from --league / --season if given,
otherwise by asking at the terminal. In unattended runs (CI, logs, scheduled jobs) it
skips the charts with a note instead of waiting for input.

Output: models/final/model_<timestamp>/visuals/*.png

Usage:
    python run_files/run_visualizations.py
    python run_files/run_visualizations.py --league "La Liga" --season "2015/2016"
"""
import argparse
import os
import sys
import glob
from datetime import datetime
import pandas as pd
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Make the `src` package importable when this file is run directly as a script
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualizations_pipeline import draw_leaderboard_table, plot_pass_risk_execution, plot_top_player_pass_map, plot_team_pass_quadrants
from src.post_analysis_pipeline import player_analysis, team_analysis
from src.naming import find_subfolder, list_subfolders
from src import console

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "statsbomb_free")
DATE_PATTERN = r'\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}'
DEFAULT_MIN_PASSES = 300   # minimum passes for a player to appear in the player charts
SMALL_SCOPE_MATCHES = 5    # warn below this many matches

def extract_tags(csv_path: str) -> str:
    """
    Returns the league/season part of an xP dataset's filename, used to name the figures.
    For example:

        xp_added_passes_La-Liga_2015-2016_09-10-2026_18-02-41.csv  ->  "La-Liga_2015-2016"

    Falls back to "all_matches" if nothing is left after removing the prefix and timestamp.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("xp_added_passes_"):
        base = base[len("xp_added_passes_"):]

    # Remove the trailing "_DD-MM-YYYY_HH-MM-SS" timestamp. The underscore is optional
    # because a file made without --league has nothing but the timestamp after its prefix.
    cleaned = re.sub(r'_?\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"


def covers_all_leagues(tag: str) -> bool:
    """
    True if the dataset was built without --league.

    Such datasets have no league in their filename tag: the tag is "all_matches", or for
    files made by older versions of the pipeline, just a leftover timestamp.
    """
    return tag == "all_matches" or re.fullmatch(DATE_PATTERN, tag) is not None


def parse_args():
    """Reads the optional --league / --season scope from the command line."""
    parser = argparse.ArgumentParser(description="Draw the xPass charts for one league / season.")
    parser.add_argument("--league", help="Only chart this league, e.g. \"La Liga\"")
    parser.add_argument("--season", help="Only chart this season of --league, e.g. \"2015/2016\"")
    args = parser.parse_args()

    if args.season and not args.league:
        parser.error("--season requires --league")

    return args


def resolve_scope(league_arg: str, season_arg: str):
    """
    Maps --league / --season to the matching folder names under data/raw.

    Returns (league, season), where season is None for "all seasons". Exits with a list
    of the valid names if either one doesn't match a downloaded folder.
    """
    league = find_subfolder(RAW_DIR, league_arg)
    if league is None:
        console.error(f"No downloaded league matches '{league_arg}'.")
        console.listing(list_subfolders(RAW_DIR), "Available")
        sys.exit(1)

    season = None
    if season_arg:
        season = find_subfolder(os.path.join(RAW_DIR, league), season_arg)
        if season is None:
            console.error(f"No downloaded season of {league} matches '{season_arg}'.")
            console.listing(list_subfolders(os.path.join(RAW_DIR, league)), "Available")
            sys.exit(1)

    return league, season


def ask_scope():
    """
    Interactive version of resolve_scope(): keeps asking until the answers match
    downloaded folders. A blank season means "all seasons". Returns (league, season).
    """
    while True:
        league = find_subfolder(RAW_DIR, console.ask("League (e.g. La Liga)"))
        if league:
            break
        console.warn("No downloaded league matches that name.")
        console.listing(list_subfolders(RAW_DIR), "Available")

    league_dir = os.path.join(RAW_DIR, league)
    while True:
        answer = console.ask("Season (e.g. 2015/2016, leave blank for all seasons)")
        if not answer:
            return league, None
        season = find_subfolder(league_dir, answer)
        if season:
            return league, season
        console.warn(f"No downloaded season of {league} matches that name.")
        console.listing(list_subfolders(league_dir), "Available")


def match_ids_in_scope(league: str, season: str) -> set:
    """
    Returns the match ids (as strings) that belong to a league and season.

    The processed data has no league or season column, but every raw file is stored as
    data/raw/statsbomb_free/<League>/<Season>/<match_id>.json, so the folder contents
    tell us exactly which matches are in scope.
    """
    pattern = os.path.join(RAW_DIR, league, season or "**", "*.json")
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(pattern, recursive=True)}


def deployed_model_label(model_dir: str):
    """
    Returns which model version a training run deployed ("Calibrated XGBoost" or
    "Raw XGBoost"), read from that run's metrics report, or None if there's no report.
    """
    reports = glob.glob(os.path.join(model_dir, "reports", "metrics_report_*.txt"))
    if not reports:
        return None
    with open(max(reports, key=os.path.getctime), encoding="utf-8") as f:
        for line in f:
            if line.startswith("Deployment Decision:"):
                decision = line.split(":", 1)[1].strip()
                return "Calibrated XGBoost" if decision == "Calibrated" else decision
    return None


def skip(reason: str) -> None:
    """Ends the step without drawing anything. This is not an error, so the pipeline carries on."""
    console.warn(reason)
    console.done("Visualizations skipped")


def main():
    args = parse_args()

    console.header("Visualizations · charts & pass maps")

    console.section("Loading datasets")

    # Start from the newest xP dataset rather than the newest model folder. If predictions were
    # run with an older model (--model), its results live in that older folder, not the newest one.
    pre_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*", "datasets", "pre analysis", "xp_added_passes_*.csv")
    pre_files = glob.glob(pre_pattern)

    if not pre_files:
        console.error("No xP datasets found in models/final. Run run_predictions.py first.")
        sys.exit(1)

    latest_pre = max(pre_files, key = os.path.getctime)
    # .../model_<timestamp>/datasets/pre analysis/<file>.csv -> .../model_<timestamp>
    latest_model_dir = os.path.dirname(os.path.dirname(os.path.dirname(latest_pre)))
    model_label = deployed_model_label(latest_model_dir)
    console.info(f"Model: {os.path.basename(latest_model_dir)}" + (f" ({model_label})" if model_label else ""))

    pre_df = pd.read_csv(latest_pre)
    console.info(f"Dataset: {os.path.basename(latest_pre)}")

    status_tag = extract_tags(latest_pre)

    # Comparing players across different leagues, eras and competitions isn't meaningful,
    # so a dataset that covers everything needs a scope before anything is drawn
    league = season = None
    if args.league:
        league, season = resolve_scope(args.league, args.season)

    elif covers_all_leagues(status_tag):
        console.section("Choosing a scope")
        console.info("This dataset covers every league. Charts are drawn for one league (and season) at a time.")

        no_terminal = "No terminal to ask for a league, so no charts were drawn. Re-run with --league to draw them."
        if not console.can_ask():
            skip(no_terminal)
            return

        try:
            if not console.ask_yes_no("Would you like to generate visualizations?"):
                skip("Visualizations skipped at your request.")
                return
            league, season = ask_scope()
        except EOFError:
            # Input ran out, e.g. it was redirected from an empty file or from NUL,
            # which Windows reports as a terminal even though nobody can answer
            print()
            skip(no_terminal)
            return

    if league:
        console.info(f"Scope: {console.describe_scope(league, season)}")

        match_ids = match_ids_in_scope(league, season)
        pre_df = pre_df[pre_df['match_id'].astype(str).isin(match_ids)]

        if pre_df.empty:
            console.error(f"No passes with xP from {console.describe_scope(league, season)} in this dataset. "
                          f"Was this league included when the model was run?")
            sys.exit(1)

        # Re-aggregate for this scope only, since the saved post-analysis files cover the whole dataset
        post_player_df = player_analysis(pre_df)
        post_team_df = team_analysis(pre_df)
        status_tag = f"{league}_{season or 'all-seasons'}".replace("'", "").replace(" ", "-")

    else:
        # The dataset is already one league / season, so reuse the summaries Step 5 saved for it
        post_player_pattern = os.path.join(latest_model_dir, "datasets", "post analysis", "player_analysis_*.csv")
        post_player_files = glob.glob(post_player_pattern)

        if not post_player_files:
            console.error(f"No player analysis found in {os.path.relpath(latest_model_dir, PROJECT_ROOT)}. Run run_post_analysis.py first.")
            sys.exit(1)

        latest_post_player = max(post_player_files, key = os.path.getctime)
        post_player_df = pd.read_csv(latest_post_player)
        console.info(f"Player analysis: {os.path.basename(latest_post_player)}")

        post_team_pattern = os.path.join(latest_model_dir, "datasets", "post analysis", "team_analysis_*.csv")
        post_team_files = glob.glob(post_team_pattern)

        if not post_team_files:
            console.error(f"No team analysis found in {os.path.relpath(latest_model_dir, PROJECT_ROOT)}. Run run_post_analysis.py first.")
            sys.exit(1)

        latest_post_team = max(post_team_files, key = os.path.getctime)
        post_team_df = pd.read_csv(latest_post_team)
        console.info(f"Team analysis: {os.path.basename(latest_post_team)}")

    console.success(f"Loaded {len(pre_df):,} passes · {len(post_player_df):,} players · {len(post_team_df):,} teams")

    n_matches = pre_df['match_id'].nunique()
    if n_matches < SMALL_SCOPE_MATCHES:
        console.warn(f"Only {n_matches} match(es) in this scope, so the charts are based on a small sample.")

    # 300 passes is a sensible bar for a full season, but small scopes (a single final, say)
    # can't reach it. In that case use half of the top passer's total, so the top passer
    # and the busiest players still qualify.
    min_passes = min(DEFAULT_MIN_PASSES, int(post_player_df['total_passes'].max() // 2))
    if min_passes < DEFAULT_MIN_PASSES:
        console.info(f"Minimum passes per player lowered to {min_passes} for this scope (normally {DEFAULT_MIN_PASSES}).")

    console.section("Drawing figures")

    fig_leaderboard_table = draw_leaderboard_table(post_player_df, min_passes=min_passes, model_label=model_label)
    console.success("Drew player leaderboard table")
    fig_plot_pass_risk_execution = plot_pass_risk_execution(post_player_df, min_passes=min_passes, model_label=model_label)
    console.success("Drew pass risk vs execution scatter")
    fig_plot_top_player_pass_map = plot_top_player_pass_map(pva_df=post_player_df, passes_df=pre_df, min_passes=min_passes)
    console.success("Drew top player pass map")
    fig_plot_team_pass_quadrants = plot_team_pass_quadrants(post_team_df)
    console.success("Drew team pass quadrants")

    # Figures go in the model's "visuals" folder: .../model_<timestamp>/visuals
    pre_analysis_dir = os.path.dirname(latest_pre)
    model_dataset_dir = os.path.dirname(pre_analysis_dir)
    model_output_dir = os.path.dirname(model_dataset_dir)
    output_dir = os.path.join(model_output_dir, "visuals")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    console.section("Saving outputs")

    leaderboard_output_path = os.path.join(output_dir, f"player_leaderboard_table_{status_tag}_{date_str}.png")
    fig_leaderboard_table.savefig(leaderboard_output_path)
    console.success(f"Saved player leaderboard table: {os.path.basename(leaderboard_output_path)}")

    pass_risk_output_path = os.path.join(output_dir, f"pass_risk_execution_scatter_{status_tag}_{date_str}.png")
    fig_plot_pass_risk_execution.savefig(pass_risk_output_path)
    console.success(f"Saved pass risk scatter: {os.path.basename(pass_risk_output_path)}")

    top_player_pass_map_output_path = os.path.join(output_dir, f"top_player_pass_map_{status_tag}_{date_str}.png")
    fig_plot_top_player_pass_map.savefig(top_player_pass_map_output_path)
    console.success(f"Saved top player pass map: {os.path.basename(top_player_pass_map_output_path)}")

    team_pass_quadrants_output_path = os.path.join(output_dir, f"team_pass_quadrants_{status_tag}_{date_str}.png")
    fig_plot_team_pass_quadrants.savefig(team_pass_quadrants_output_path)
    console.success(f"Saved team pass quadrants: {os.path.basename(team_pass_quadrants_output_path)}")

    console.section("Summary")
    if league:
        console.kv("Scope", console.describe_scope(league, season))
    console.kv("Figures", "4")
    console.kv("Saved to", os.path.relpath(output_dir, PROJECT_ROOT))

    console.done("Visualizations complete")

if __name__ == "__main__":
    main()
