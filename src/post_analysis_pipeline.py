"""
Per-player and per-team summaries of a pass dataset that already has xP and PVA columns.

Both summaries contain the same columns:

    total_passes              passes attempted
    completed_passes          passes completed
    total_xp                  sum of xP, i.e. how many completions the model expected
    total_pva                 sum of pass value added (completed_passes - total_xp)
    actual_completion_rate    completed_passes / total_passes
    expected_completion_rate  mean xP
    CPOE                      completion percentage over expected: actual minus expected rate.
                              Positive means the player or team completes more passes
                              than the difficulty of those passes would suggest.
"""
import pandas as pd

def player_analysis(df: pd.DataFrame):
    """Returns one row per player, sorted by passes attempted (most first)."""

    # Accept either column name, depending on how the pass dataset was built
    player_col = 'player_name' if 'player_name' in df.columns else 'player'

    player_df = df.groupby(player_col).agg(
        total_passes = ('pass_outcome', 'count'),
        completed_passes = ('pass_outcome', 'sum'),
        total_xp = ('xP', 'sum'),
        total_pva = ('PVA', 'sum'),
        actual_completion_rate = ('pass_outcome', 'mean'),
        expected_completion_rate = ('xP', 'mean')
    ).reset_index()

    player_df['actual_completion_rate'] = player_df['actual_completion_rate'].fillna(0)
    player_df['expected_completion_rate'] = player_df['expected_completion_rate'].fillna(0)

    # Completion percentage over expected (CPOE) = actual completion rate - expected completion rate
    player_df['CPOE'] = player_df['actual_completion_rate'] - player_df['expected_completion_rate']

    player_df = player_df.sort_values(by='total_passes', ascending=False).reset_index(drop=True)

    return player_df


def team_analysis(df: pd.DataFrame):
    """Returns one row per team, sorted by passes attempted (most first)."""

    # Accept either column name, depending on how the pass dataset was built
    team_col = 'team_name' if 'team_name' in df.columns else 'team'

    team_df = df.groupby(team_col).agg(
        total_passes = ('pass_outcome', 'count'),
        completed_passes = ('pass_outcome', 'sum'),
        total_xp = ('xP', 'sum'),
        total_pva = ('PVA', 'sum'),
        actual_completion_rate = ('pass_outcome', 'mean'),
        expected_completion_rate = ('xP', 'mean')
    ).reset_index()

    team_df['actual_completion_rate'] = team_df['actual_completion_rate'].fillna(0)
    team_df['expected_completion_rate'] = team_df['expected_completion_rate'].fillna(0)

    # Completion percentage over expected (CPOE) = actual completion rate - expected completion rate
    team_df['CPOE'] = team_df['actual_completion_rate'] - team_df['expected_completion_rate']

    team_df = team_df.sort_values(by='total_passes', ascending=False).reset_index(drop=True)

    return team_df
