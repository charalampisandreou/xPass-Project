"""
Training, evaluation and diagnostics for the xPass (expected pass completion) model.

The model is an XGBoost classifier that predicts the probability that a pass is completed.
Two versions are trained on the same 85% training split:

    raw          XGBoost on its own
    calibrated   XGBoost wrapped in 5-fold, isotonic CalibratedClassifierCV, which
                 adjusts the probabilities so that, say, passes given 0.8 are
                 completed about 80% of the time

Both are scored on the 15% held-out test set, and the one with the lower log-loss is
saved. The diagnostics (reliability diagram and SHAP summary) go in a reports/ folder
next to the model.
"""
import os
import glob
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

from src.features import FEAT_TRUEBEST
from src import console



def save_metrics_report(
        model_name: str,
        models_dir: str,
        is_calibrated_deployed: bool,
        raw_logloss: float,
        cal_logloss: float,
        raw_brier: float,
        cal_brier: float,
        raw_rocauc: float,
        cal_rocauc: float,
        raw_testaccuracy: float,
        cal_testaccuracy: float,
        final_logloss: float,
        final_brier: float,
        final_rocauc: float,
        final_testaccuracy: float,
        features: list = FEAT_TRUEBEST
):
    """
    Writes a plain-text report comparing the raw and calibrated models on the test set,
    plus the metrics of the version that was deployed, to <models_dir>/reports/.
    """
    report_name = f"metrics_report_{model_name}.txt"
    report_path = os.path.join(models_dir, "reports", report_name)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = [
        f"Model Name: {model_name}",
        "="*50,
        "", "",
        f"Features used: {features}",
        "", "",
        "---Uncalibrated Baseline---",
        f"Test Accuracy: {raw_testaccuracy:.4f}",
        f"Log-Loss: {raw_logloss:.4f}",
        f"Brier Score: {raw_brier:.4f}",
        f"ROC AUC: {raw_rocauc:.4f}",
        "",
        "---Calibrated (Isotonic, 5-Fold CV)---",
        f"Test Accuracy: {cal_testaccuracy:.4f}",
        f"Log-Loss: {cal_logloss:.4f}",
        f"Brier Score: {cal_brier:.4f}",
        f"ROC AUC: {cal_rocauc:.4f}",
        "", "",
        "---FINAL MODEL---",
        f"Deployment Decision: {'Calibrated' if is_calibrated_deployed else 'Raw XGBoost'}",
        f"Final Test Accuracy: {final_testaccuracy:.4f}",
        f"Final Log-Loss: {final_logloss:.4f}",
        f"Final Brier Score: {final_brier:.4f}",
        f"Final ROC AUC: {final_rocauc:.4f}",
        ""
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    console.success(f"Saved metrics report: {os.path.basename(report_path)}")



# --- Reliability diagram ---

def fig_reliability_diagram(raw_cal_curve, cal_cal_curve, model_name, models_dir):
    """
    Plots predicted probability against the actual completion rate for both models.

    Each curve comes from sklearn's calibration_curve: test passes are grouped into
    probability bins, and each point shows a bin's mean prediction and the share of those
    passes that were actually completed. The closer a curve is to the diagonal, the
    better calibrated the model.
    """
    raw_prob_true = raw_cal_curve[0]
    raw_prob_pred = raw_cal_curve[1]

    cal_prob_true = cal_cal_curve[0]
    cal_prob_pred = cal_cal_curve[1]

    fig, ax = plt.subplots(figsize = (9, 9), dpi = 300)

    # The diagonal is perfect calibration: predicted probability == observed rate
    ax.plot([0, 1], [0, 1], 'k--', label = 'Perfectly Calibrated', alpha = 0.7)

    ax.plot(raw_prob_pred, raw_prob_true, 's-', color = 'blue', label = 'Uncalibrated')
    ax.plot(cal_prob_pred, cal_prob_true, 'o-', color = 'red', label = 'Calibrated')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_xlabel("Mean Predicted Probability (xPass)")
    ax.set_ylabel("Actual Pass Completion Rate")

    ax.grid(True, alpha = 0.3)
    fig.suptitle(
        "Model Reliability Diagram",
        color = 'black',
        fontsize = 16,
        fontweight = 'bold'
    )
    ax.set_title(
        f"Model Name: {model_name}",
        color = 'black',
        loc = 'center',
        style = 'italic',
        fontsize = 12,
        pad = 10
    )
    ax.legend(loc = 'lower right', fontsize = 12)

    ticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    fig_name = "reliability_diagram_" + model_name + ".png"
    fig_dir = os.path.join(models_dir, "reports", fig_name)
    if not os.path.exists(os.path.dirname(fig_dir)):
        os.makedirs(os.path.dirname(fig_dir))
    plt.savefig(fig_dir, bbox_inches = 'tight')

    console.success(f"Saved reliability diagram: {os.path.basename(fig_dir)}")

    plt.close()



    
def collapse_onehot_groups(shap_values, X: pd.DataFrame):
    """
    Merges the one-hot columns of each categorical feature back into one row of the SHAP plot.

    Without this, "Play Pattern" would appear as seven separate rows. SHAP values are
    additive, so the group's columns can simply be summed to get the whole feature's
    contribution. For the colour scale, each pass gets a category code: 0 for the
    baseline category and 1..n for the one-hot columns in order.

    Returns (values, data, feature_names) as plain arrays for shap.summary_plot.
    """
    vals = shap_values.values.copy()
    data = X.values.copy()
    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    # One entry per categorical feature: its one-hot columns, and the name of the
    # baseline category (the one that is all zeros because it has no column of its own)
    groups = {
        "Height": {
            "height_Low Pass": "Low Pass",
            "height_High Pass": "High Pass",
            "__baseline__": "Ground Pass"
        },
        "Body Part":{
            "body_part_Foot": "Foot",
            "body_part_Head": "Head",
            "body_part_Keeper Arm": "Keeper Arm",
            "__baseline__": "Other"
        },
        "Play Pattern":{
            "play_pattern_Regular Play": "Regular Play",
            "play_pattern_Kick Off": "Kick Off",
            "play_pattern_Throw In": "Throw In",
            "play_pattern_Free Kick": "Free Kick",
            "play_pattern_Goal Kick": "Goal Kick",
            "play_pattern_Keeper": "Keeper",
            "play_pattern_Corner": "Corner",
            "__baseline__": "Other"
        }
    }

    keep_mask = np.ones(len(cols), dtype = bool)
    new_cols = []
    new_vals = []
    new_data = []

    for label, mapping in groups.items():
        dummy_cols = [c for c in mapping if c!= "__baseline__"]
        group_idx = [col_idx[c] for c in dummy_cols if c in col_idx]
        if not group_idx:
            continue
        # These columns are replaced by the merged feature
        keep_mask[group_idx] = False

        group_data = data[:, group_idx]
        summed = vals[:, group_idx].sum(axis=1)

        # Category code per pass: 0 = baseline, 1..n = the one-hot columns in order
        active_idx = group_data.argmax(axis=1)          # position of the 1 in the row; meaningless if the row is all zeros
        is_baseline = group_data.sum(axis=1) == 0
        cat_code = np.where(is_baseline, 0, active_idx + 1)

        new_cols.append(label)
        new_vals.append(summed)
        new_data.append(cat_code.astype(float))

    # Untouched features first, then the merged categorical features
    remaining_cols = [c for i, c in enumerate(cols) if keep_mask[i]]
    remaining_vals = vals[:, keep_mask]
    remaining_data = data[:, keep_mask]

    final_vals = np.hstack([remaining_vals, np.array(new_vals).T]) if new_cols else remaining_vals
    final_data = np.hstack([remaining_data, np.array(new_data).T]) if new_cols else remaining_data
    final_cols = remaining_cols + new_cols if new_cols else remaining_cols

    return final_vals, final_data, final_cols

# --- SHAP summary ---

def fig_shap_diagram(base_model: XGBClassifier, models_dir: str, X_sample: pd.DataFrame, model_name: str,
                     collapse_categoricals: bool = True):
    """
    Saves a SHAP summary plot showing how much each feature pushes xP up or down.

    Each dot is one pass: its position is the feature's effect on that prediction, and
    its colour is the feature's value. One-hot groups are merged into single rows unless
    collapse_categoricals is False.
    """
    # Imported here because shap is slow to import and only needed for this plot
    import shap

    # Readable axis labels for the plot; any other column just has "_" replaced with spaces
    rename_map = {
        "start_x": "Start Position (X)",
        "start_y": "Start Position (Y)",
        "dist_to_goal": "Distance to Goal",
        "pass_angle": "Pass Angle",
        "under_pressure": "Under Pressure",
        "period": "Match Period",
        "half_percentage": "Time in Half (%)",
        "net_score": "Score Differential",
    }

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer(X_sample)

    fig = plt.figure(figsize = (10, 8), dpi = 300)

    if collapse_categoricals:
        final_vals, final_data, final_cols = collapse_onehot_groups(shap_values, X_sample)
        display_cols = [rename_map.get(c, c.replace("_", " ")) for c in final_cols]

        shap.summary_plot(
            final_vals,
            final_data,
            feature_names=display_cols,
            show=False,
            plot_size=None,
        )

    else:
        X_display = X_sample.copy()
        X_display.columns = [rename_map.get(c, c.replace("_", " ")) for c in X_sample.columns]

        shap.summary_plot(
            shap_values,
            X_display,
            show=False,
            plot_size=None,
        )


    fig.suptitle(
        "Feature Contribution & Impact (SHAP)",
        color = 'black',
        fontsize = 16,
        fontweight = 'bold',
        y = 1.03
    )

    plt.title(
        f"Model Name: {model_name}",
        color = 'black',
        loc = 'center',
        style = 'italic',
        fontsize = 12,
        pad = 10
    )

    fig_name = "shap_diagram_" + model_name + ".png"
    fig_dir = os.path.join(models_dir, "reports", fig_name)
    os.makedirs(os.path.dirname(fig_dir), exist_ok = True)

    plt.savefig(fig_dir, bbox_inches = 'tight')
    console.success(f"Saved SHAP diagram: {os.path.basename(fig_dir)}")

    plt.close()





# --- Training ---

def run_model_training(df: pd.DataFrame, model_name: str = 'xgb_model_draft', models_dir: str = "./models", 
                       force_calibration: bool = False, features: list = FEAT_TRUEBEST, run_diagnostics: bool = True):
    """
    Trains the raw and calibrated models, saves the better one and writes its reports.

    The version with the lower test log-loss is saved as <models_dir>/<model_name>.joblib;
    force_calibration=True always saves the calibrated one. run_diagnostics=False skips
    the reliability and SHAP plots, which take a while on large datasets.

    Returns (model, is_calibrated).
    """
    console.section(f"Training corpus · {len(df):,} passes")

    X = df[features]
    y = df['pass_outcome']

    # Hold out 15% for testing. Stratifying keeps the completed / failed ratio the same in
    # both splits, and the fixed random_state makes runs reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    console.listing(features, "Features")

    # 1. Raw XGBoost model
    console.info("Training standalone base XGBoost model...")

    xgb_standalone = XGBClassifier(
        n_estimators = 300,
        max_depth = 5,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        eval_metric = 'logloss',
        random_state = 42,
        n_jobs = -1
    )

    xgb_standalone.fit(X_train, y_train)


    # 2. Calibrated model: identical XGBoost settings, wrapped in 5-fold isotonic calibration.
    #    Each fold trains on 4/5 of the training data and fits the calibration on the remaining
    #    1/5, so the calibration never sees passes its model was trained on.
    console.info("Training 5-fold cross-validated calibrated model (isotonic)...")

    xgb_base_for_cal = XGBClassifier(
        n_estimators = 300,
        max_depth = 5,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        eval_metric = 'logloss',
        random_state = 42,
        n_jobs = -1
    )

    calibrated_model = CalibratedClassifierCV(
        estimator = xgb_base_for_cal,
        method = 'isotonic',
        cv = 5
    )

    calibrated_model.fit(X_train, y_train)


    # 3. Compare both models on the held-out test set. Log-loss and Brier score measure how
    #    good the probabilities are (lower is better). ROC AUC and accuracy are recorded too,
    #    but probabilities are what xP is built on.
    console.info("Evaluating both models on the hold-out test set...")
    raw_probs = xgb_standalone.predict_proba(X_test)[:, 1]
    cal_probs = calibrated_model.predict_proba(X_test)[:, 1]

    raw_preds = xgb_standalone.predict(X_test)
    cal_preds = calibrated_model.predict(X_test)

    raw_testaccuracy = accuracy_score(y_test, raw_preds)
    cal_testaccuracy = accuracy_score(y_test, cal_preds)

    raw_logloss = log_loss(y_test, raw_probs)
    cal_logloss = log_loss(y_test, cal_probs)

    raw_brier = brier_score_loss(y_test, raw_probs)
    cal_brier = brier_score_loss(y_test, cal_probs)

    raw_rocauc = roc_auc_score(y_test, raw_probs)
    cal_rocauc = roc_auc_score(y_test, cal_probs)

    console.section("Hold-out evaluation")
    console.kv("", f"{'Log-Loss':>10}  {'Brier':>8}")
    console.kv("Uncalibrated baseline", f"{raw_logloss:>10.4f}  {raw_brier:>8.4f}")
    console.kv("Calibrated (isotonic)", f"{cal_logloss:>10.4f}  {cal_brier:>8.4f}")


    # 4. Keep the calibrated model only if it has a strictly lower test log-loss (or is forced)
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, model_name + '.joblib')

    if cal_logloss < raw_logloss or force_calibration:
        console.section("Deployment decision")
        if cal_logloss < raw_logloss:
            console.info("Calibrated model has the lower test log-loss · deploying calibrated model")
        else:
            console.info("Calibration forced (force_calibration=True) · deploying calibrated model")
        joblib.dump(calibrated_model, artifact_path)
        console.success(f"Saved model: {os.path.basename(artifact_path)}")
        final_model = calibrated_model
        final_testaccuracy = cal_testaccuracy
        final_logloss = cal_logloss
        final_brier = cal_brier
        final_rocauc = cal_rocauc
        is_calibrated_deployed = True
    else:
        console.section("Deployment decision")
        console.info("Raw XGBoost has the lower (or equal) test log-loss · deploying it without calibration")
        joblib.dump(xgb_standalone, artifact_path)
        console.success(f"Saved model: {os.path.basename(artifact_path)}")
        final_model = xgb_standalone
        final_testaccuracy = raw_testaccuracy
        final_logloss = raw_logloss
        final_brier = raw_brier
        final_rocauc = raw_rocauc
        is_calibrated_deployed = False

    console.section("Final model · test metrics")
    console.kv("Accuracy", f"{final_testaccuracy:.4f}")
    console.kv("Log-Loss", f"{final_logloss:.4f}")
    console.kv("Brier Score", f"{final_brier:.4f}")
    console.kv("ROC AUC", f"{final_rocauc:.4f}")

    save_metrics_report(
        model_name=model_name,
        models_dir=models_dir,
        is_calibrated_deployed=is_calibrated_deployed,
        raw_logloss=raw_logloss,
        cal_logloss=cal_logloss,
        raw_brier=raw_brier,
        cal_brier=cal_brier,
        raw_rocauc=raw_rocauc,
        cal_rocauc=cal_rocauc,
        raw_testaccuracy=raw_testaccuracy,
        cal_testaccuracy=cal_testaccuracy,
        final_logloss=final_logloss,
        final_brier=final_brier,
        final_rocauc=final_rocauc,
        final_testaccuracy=final_testaccuracy,
    )


    if run_diagnostics:

        console.section("Model diagnostics")

        console.info("Drawing the reliability diagram...")
        raw_cal_curve = calibration_curve(y_test, raw_probs, n_bins = 10)
        cal_cal_curve = calibration_curve(y_test, cal_probs, n_bins = 10)
        fig_reliability_diagram(raw_cal_curve, cal_cal_curve, model_name, models_dir = models_dir)
        
        # SHAP is slow on large datasets, so explain a sample of at most 30,000 test passes
        X_shap = X_test.sample(n = min(30000, len(X_test)), random_state = 42)

        # TreeExplainer needs a plain XGBoost model. A calibrated model holds one XGBoost
        # model per CV fold, so the first fold's model stands in for the ensemble; its
        # feature effects are a close approximation of the deployed model's.
        if is_calibrated_deployed:
            base_estimator = final_model.calibrated_classifiers_[0].estimator
        else:
            base_estimator = final_model

        console.info(f"Calculating SHAP values for {len(X_shap):,} test passes...")
        fig_shap_diagram(base_estimator, models_dir, X_shap, model_name)

        console.success("Diagnostics complete")



    return final_model, is_calibrated_deployed