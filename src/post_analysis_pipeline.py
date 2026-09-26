import pandas as pd

def player_analysis(df: pd.DataFrame):

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

    # Completion over expectation (COE) = Actual Completion Rate - Expected Completion Rate
    player_df['COE'] = player_df['actual_completion_rate'] - player_df['expected_completion_rate']

    player_df = player_df.sort_values(by='total_passes', ascending=False).reset_index(drop=True)

    return player_df


def team_analysis(df: pd.DataFrame):

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

    # Completion over expectation (COE) = Actual Completion Rate - Expected Completion Rate
    team_df['COE'] = team_df['actual_completion_rate'] - team_df['expected_completion_rate']

    team_df = team_df.sort_values(by='total_passes', ascending=False).reset_index(drop=True)

    return team_df
