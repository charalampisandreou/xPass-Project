"""
Applies a trained xPass model to a processed pass dataset.
"""
import pandas as pd
import joblib

from src.features import FEAT_TRUEBEST


def calculate_xp(df_path: str, model_path: str, features: list = FEAT_TRUEBEST):
    """
    Loads a processed pass dataset and a saved model, and returns the dataset with two new columns:

        xP    predicted probability that the pass is completed
        PVA   pass value added: pass_outcome (1 = completed, 0 = failed) minus xP

    `features` must be the same columns, in the same order, that the model was trained on.
    """
    df = pd.read_csv(df_path)
    model = joblib.load(model_path)

    X = df[features]
    # predict_proba returns [P(failed), P(completed)] for each pass; keep P(completed)
    df['xP'] = model.predict_proba(X)[:, 1]
    df['PVA'] = df['pass_outcome'] - df['xP']

    return df
