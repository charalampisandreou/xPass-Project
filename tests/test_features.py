"""
Leakage checks for the xPass model: features must only use start-of-pass information,
and the train/test split must keep each match on one side.
"""
import numpy as np
import pandas as pd

from src.data_analysis_pipeline import angle_to_goal
from src.features import FEAT_TRUEBEST
from src.model_training_pipeline import grouped_train_test_split


def test_angle_to_goal_needs_no_end_location():
    df = pd.DataFrame({'start_x': [60.0, 120.0, 100.0], 'start_y': [40.0, 0.0, 60.0]})

    angles = angle_to_goal(df['start_x'], df['start_y'])

    assert 'end_x' not in df.columns and 'end_y' not in df.columns
    np.testing.assert_allclose(angles, [0.0, np.pi / 2, np.arctan2(-20, 20)])


def test_no_end_of_pass_features():
    leaky = {'pass_angle', 'pass_length', 'end_x', 'end_y'}
    assert leaky.isdisjoint(FEAT_TRUEBEST)


def test_split_is_disjoint_by_match_id():
    df = pd.DataFrame({
        'match_id': np.repeat(np.arange(20), 10),
        'pass_outcome': np.tile([0, 1], 100),
    })

    train_idx, test_idx = grouped_train_test_split(df, test_size=0.15, random_state=42)

    train_matches = set(df['match_id'].iloc[train_idx])
    test_matches = set(df['match_id'].iloc[test_idx])
    assert test_matches
    assert train_matches.isdisjoint(test_matches)
    assert len(train_idx) + len(test_idx) == len(df)
