import os
import sys
import glob
from datetime import datetime
import pandas as pd
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.visualizations_pipeline import draw_leaderboard_table, plot_pass_risk_execution, plot_top_player_pass_map

def extract_tags(csv_path: str) -> str:
    """
    Extracts the league/season identifier from input filenames.
    """
    filename = os.path.basename(csv_path)

    base = os.path.splitext(filename)[0]

    if base.startswith("xp_added_passes_"):
        base = base[len("xp_added_passes_"):]

    #Strip datetime pattern
    cleaned = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', '', base)

    return cleaned if cleaned else "all_matches"



def main():
    print("Starting Visualization Processing...\n")

    # Load the latest processed datasets
    print("Loading the latest datasets for visualization...")

    final_model_pattern = os.path.join(PROJECT_ROOT, "models", "final", "*")
    model_dirs = glob.glob(os.path.join(final_model_pattern))

    if not model_dirs:
        print("Error: No final model directories found in models/final.")
        return

    latest_model_dir = max(model_dirs, key=os.path.getctime)

    pre_pattern = os.path.join(latest_model_dir, "datasets", "pre analysis", "xp_added_passes_*.csv")
    pre_files = glob.glob(pre_pattern)
    
    if not pre_files:
        print("Error: No processed pass datasets found in models/final or data/processed directory.")
        return
    
    latest_pre = max(pre_files, key = os.path.getctime)
    pre_df = pd.read_csv(latest_pre)


    post_player_pattern = os.path.join(latest_model_dir, "datasets", "post analysis", "player_analysis_*.csv")
    post_player_files = glob.glob(post_player_pattern)

    if not post_player_files:
        print("Error: No post-analysis player datasets found in models/final directory.")
        return

    latest_post_player = max(post_player_files, key = os.path.getctime)
    post_player_df = pd.read_csv(latest_post_player)

    post_team_pattern = os.path.join(latest_model_dir, "datasets", "post analysis", "team_analysis_*.csv")
    post_team_files = glob.glob(post_team_pattern)

    if not post_team_files:
        print("Error: No post-analysis team datasets found in models/final directory.")
        return

    latest_post_team = max(post_team_files, key = os.path.getctime)
    post_team_df = pd.read_csv(latest_post_team)

    print("All datasets successfully loaded for visualization.")

    print("\nGenerating visualizations...\n")

    fig_leaderboard_table = draw_leaderboard_table(post_player_df)
    fig_plot_pass_risk_execution = plot_pass_risk_execution(post_player_df, min_passes=300)
    fig_plot_top_player_pass_map = plot_top_player_pass_map(pva_df=post_player_df, passes_df=pre_df)

    # Save visualizations to the output directory
    pre_analysis_dir = os.path.dirname(latest_pre)
    model_dataset_dir = os.path.dirname(pre_analysis_dir)
    output_dir = os.path.join(model_dataset_dir, "visuals")
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    status_tag = extract_tags(latest_pre)

    leaderboard_output_path = os.path.join(output_dir, f"player_leaderboard_table_{status_tag}_{date_str}.png")
    fig_leaderboard_table.savefig(leaderboard_output_path)

    pass_risk_output_path = os.path.join(output_dir, f"pass_risk_execution_scatter_{status_tag}_{date_str}.png")
    fig_plot_pass_risk_execution.savefig(pass_risk_output_path)

    top_player_pass_map_output_path = os.path.join(output_dir, f"top_player_pass_map_{status_tag}_{date_str}.png")
    fig_plot_top_player_pass_map.savefig(top_player_pass_map_output_path)

    print(f"\nVisualizations saved to: {output_dir}")

    print("\n[SUCCESS]: Visualization Processing Completed Successfully.")

if __name__ == "__main__":
    main()