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

FEAT_TRUEBEST = [
    'start_x',
    'start_y',
    'pass_angle',
    'dist_to_goal',
    'height_Low Pass',
    'height_High Pass',
    'body_part_Foot',
    'body_part_Head',
    'body_part_Keeper Arm',
    'play_pattern_Regular Play',
    'play_pattern_Kick Off',
    'play_pattern_Throw In',
    'play_pattern_Free Kick',
    'play_pattern_Goal Kick',
    'play_pattern_Keeper',
    'play_pattern_Corner',
    'under_pressure'
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def fig_calibration_curve(raw_cal_curve, cal_cal_curve, model_name):
    raw_prob_true = raw_cal_curve[0]
    raw_prob_pred = raw_cal_curve[1]

    cal_prob_true = cal_cal_curve[0]
    cal_prob_pred = cal_cal_curve[1]

    fig, ax = plt.subplots(figsize = (9, 9), dpi = 300)

    # Draw the dotted diagonal
    ax.plot([0, 1], [0, 1], 'k--', label = 'Perfectly Calibrated', alpha = 0.7)

    # Plot the curves
    ax.plot(raw_prob_pred, raw_prob_true, 's-', color = 'blue', label = 'Uncalibrated')
    ax.plot(cal_prob_pred, cal_prob_true, 'o-', color = 'red', label = 'Calibrated')

    # Set the limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set the labels
    ax.set_xlabel("Mean Predicted Probability (xPass)")
    ax.set_ylabel("Actual Pass Completion Rate")

    #Add Grid, Title and Legend
    ax.grid(True, alpha = 0.3)
    ax.suptitle(
        "Model Reliability Diagram",
        color = 'black',
        loc= 'center',
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

    fig_name = "calibration_curve_" + model_name + ".png"
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures", "diagnostics", fig_name)
    if not os.path.exists(os.path.dirname(fig_dir)):
        os.makedirs(os.path.dirname(fig_dir))
    plt.savefig(fig_dir, bbox_inches = 'tight')

    plt.show()
    

def run_model_training(df: pd.DataFrame, model_name: str = 'xgb_model_draft', models_dir: str = "./models", force_calibration: bool = False, features: list = FEAT_TRUEBEST, run_diagnostics: bool = True):
    """
    Trains the xP model on the expanded dataset footprint. Automatically tests 
    Cross-Validated Calibration and deploys the best-performing version based on Test Log-Loss.
    """
    print(f"\n--- Initiating Training Corpus Ingestion ({len(df)} passes) ---")

    X = df[features]
    y = df['pass_outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nFeatures that will be used: {features}\n")

    # 1. Train the Pure Standalone Base Model
    print("Training Standalone Base XGBoost Model...")

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


    # 2. Train the 5-Fold Cross-Validated Calibrated Model
    print("Training 5-Fold Cross-Validated Calibrated Model (Isotonic)...")

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


    # 3. Evaluate both approaches on Hold-out Test Data
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

    print("\n=== EXPANDED DATASET HOLD-OUT EVALUATION ===")
    print(f"Uncalibrated Baseline -> Log-Loss: {raw_logloss:.4f} | Brier Score: {raw_brier:.4f}")
    print(f"Calibrated (Isotonic) -> Log-Loss: {cal_logloss:.4f} | Brier Score: {cal_brier:.4f}")


    # 4. Automated Decision Gate
    os.makedirs(models_dir, exist_ok=True)
    artifact_path = os.path.join(models_dir, model_name + '.joblib')

    # If calibration successfully lowers the loss, or if we force it, deploy calibration

    if cal_logloss < raw_logloss or force_calibration:
        print("\n[DEPLOYMENT DECISION]: Calibrated model outperformed or matched baseline. Saving Calibrated Wrapper.")
        joblib.dump(calibrated_model, artifact_path)
        final_model = calibrated_model
        final_testaccuracy = cal_testaccuracy
        final_logloss = cal_logloss
        final_brier = cal_brier
        final_rocauc = cal_rocauc
        is_calibrated_deployed = True
    else:
        print("\n[DEPLOYMENT DECISION]: Raw XGBoost preserved superior generalization. Dropping calibration layer.")
        joblib.dump(xgb_standalone, artifact_path)
        final_model = xgb_standalone
        final_testaccuracy = raw_testaccuracy
        final_logloss = raw_logloss
        final_brier = raw_brier
        final_rocauc = raw_rocauc
        is_calibrated_deployed = False

    print('Final XGBOOST Model Test Metrics:')
    print(f"Test Accuracy: {final_testaccuracy:.4f}")
    print(f"Test Log-Loss: {final_logloss:.4f}")
    print(f"Test Brier Score: {final_brier:.4f}")
    print(f"Test ROC AUC: {final_rocauc:.4f}")


    if run_diagnostics:
        raw_cal_curve = calibration_curve(y_test, raw_probs, n_bins = 10)
        cal_cal_curve = calibration_curve(y_test, cal_probs, n_bins = 10)
        fig_calibration_curve(raw_cal_curve, cal_cal_curve, model_name)
        



    
    

    return final_model, is_calibrated_deployed