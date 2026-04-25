"""
train.py — Train the earnings surprise prediction model.

Pipeline
────────
  1. Load data/features.csv + data/sentiment_scores.csv
  2. Temporal train / test split  (train < 2022, test 2022-2024)
  3. Train LightGBM classifier (with class-weight balancing)
  4. Train Logistic Regression baseline
  5. Evaluate: AUC-ROC, precision, recall, F1, betting accuracy at multiple thresholds
  6. Save plots: feature importance, calibration curve, ROC curves  → plots/
  7. Save best model artifact  → models/earnings_model.pkl

Betting accuracy definition
────────────────────────────
  Among all events the model predicts as "beat" (score ≥ threshold),
  what fraction actually beat?  This directly answers: "if I trade
  every high-confidence predicted beat, how often am I right?"
"""

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")   # headless; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR  = ROOT / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Modelling features (order matters for lgb.feature_name_)
NUMERIC_FEATURES = [
    "historical_beat_rate",
    "beat_streak",
    "days_since_last_beat",
    "estimate_revision_trend",
    "guidance_sentiment",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "roe",
    "debt_equity",
    "current_ratio",
    "margin_trend",
    "sentiment_score",
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(
    features_path: Path | None  = None,
    sentiment_path: Path | None = None,
) -> pd.DataFrame:

    features_path  = features_path  or DATA_DIR / "features.csv"
    sentiment_path = sentiment_path or DATA_DIR / "sentiment_scores.csv"

    df = pd.read_csv(features_path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")

    if sentiment_path.exists():
        sent = pd.read_csv(sentiment_path)[["ticker", "sentiment_score"]]
        # Drop pre-existing column so the merge doesn't create _x/_y duplicates
        df = df.drop(columns=["sentiment_score"], errors="ignore")
        df = df.merge(sent, on="ticker", how="left")
        logger.info("Merged sentiment scores for %d tickers", sent["ticker"].nunique())
    elif "sentiment_score" not in df.columns:
        logger.warning("sentiment_scores.csv not found — using sentiment_score = 0")
        df["sentiment_score"] = 0.0

    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)

    # Keep only labeled rows
    df = df[df["beat"].notna()].copy()
    df["beat"] = df["beat"].astype(int)

    logger.info("Dataset: %d labeled events  |  beat rate %.1f%%",
                len(df), df["beat"].mean() * 100)
    return df


# ─── Feature preparation ──────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) ready for modelling."""
    sector_cols = [c for c in df.columns if c.startswith("sector_")]
    all_features = NUMERIC_FEATURES + sector_cols
    available   = [c for c in all_features if c in df.columns]

    X = df[available].copy()
    y = df["beat"]

    logger.info("Feature matrix: %d rows × %d cols", *X.shape)
    return X, y, available


# ─── Temporal split ───────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, test_start: str = "2022-01-01"):
    cut = pd.Timestamp(test_start)
    train = df[df["earnings_date"] < cut].copy()
    test  = df[df["earnings_date"] >= cut].copy()
    logger.info("Train: %d rows  (%s → %s)",
                len(train), train["earnings_date"].min().date(), train["earnings_date"].max().date())
    logger.info("Test:  %d rows  (%s → %s)",
                len(test),  test["earnings_date"].min().date(),  test["earnings_date"].max().date())
    return train, test


# ─── Model training ───────────────────────────────────────────────────────────

def train_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
) -> lgb.LGBMClassifier:
    """Train LightGBM with early stopping on an internal validation set."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos if pos else 1.0
    logger.info("Class counts — beat: %d  miss: %d  scale_pos_weight: %.2f", pos, neg, spw)

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Logistic regression baseline with standard-scaling."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "model":     name,
        "n_test":    len(y_test),
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test,    y_pred, zero_division=0),
        "f1":        f1_score(y_test,        y_pred, zero_division=0),
    }

    # Betting accuracy at the default 0.50 threshold
    pred_beat = y_pred == 1
    metrics["betting_accuracy_50"] = float(y_test[pred_beat].mean()) if pred_beat.sum() else 0.0
    metrics["n_predicted_beat_50"] = int(pred_beat.sum())

    # High-confidence thresholds
    for thresh in (0.55, 0.60, 0.65, 0.70):
        mask = y_proba >= thresh
        key  = f"_{int(thresh*100)}"
        metrics[f"betting_accuracy{key}"] = float(y_test[mask].mean()) if mask.sum() else np.nan
        metrics[f"n_predicted_beat{key}"] = int(mask.sum())

    logger.info("\n%s", "=" * 55)
    logger.info("%-20s %s", "Model:", name)
    logger.info("%-20s %.4f", "AUC-ROC:",  metrics["auc_roc"])
    logger.info("%-20s %.4f", "Precision:", metrics["precision"])
    logger.info("%-20s %.4f", "Recall:",    metrics["recall"])
    logger.info("%-20s %.4f", "F1:",        metrics["f1"])
    logger.info("%-20s %.1f%% (n=%d)", "Betting acc @50%%:",
                metrics["betting_accuracy_50"] * 100, metrics["n_predicted_beat_50"])
    for thresh in (0.60, 0.65, 0.70):
        key = f"_{int(thresh*100)}"
        n   = metrics[f"n_predicted_beat{key}"]
        acc = metrics[f"betting_accuracy{key}"]
        if n > 0:
            logger.info("Betting acc @%d%%:    %.1f%% (n=%d)", int(thresh*100), acc*100, n)
    logger.info("\n%s", classification_report(y_test, y_pred))
    return metrics


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_feature_importance(model: lgb.LGBMClassifier, feature_names: list, top_n: int = 15):
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    # Shorten sector dummy names for display
    imp_df["label"] = imp_df["feature"].str.replace("sector_", "sector: ", regex=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("RdYlGn", top_n)[::-1]
    ax.barh(range(len(imp_df)), imp_df["importance"], color=palette)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (LightGBM split count)")
    ax.set_title(f"Top {top_n} Features — Earnings Surprise Prediction")
    plt.tight_layout()

    path = PLOTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot → %s", path)
    return imp_df


def plot_calibration(model, X_test: pd.DataFrame, y_test: pd.Series, label: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "bo-", label=label)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Actual Beats")
    ax.set_title("Calibration Curve — Earnings Beat Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "calibration_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved calibration curve → %s", path)


def plot_roc_curves(models_preds: list[tuple[str, np.ndarray]], y_test: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline (AUC=0.500)")
    colors = ["steelblue", "tomato", "seagreen"]
    for i, (name, y_proba) in enumerate(models_preds):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Earnings Surprise Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = PLOTS_DIR / "roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves → %s", path)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def run_training():
    df = load_data()

    if len(df) < 100:
        logger.error(
            "Too few labeled samples (%d) to train a meaningful model. "
            "Run data_pull.py first.", len(df)
        )
        return

    train_df, test_df = temporal_split(df)

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Train or test split is empty — check date range of your data.")
        return

    X_train, y_train, feat_names = prepare_features(train_df)
    X_test,  y_test,  _          = prepare_features(test_df)

    # Impute with training-set medians (avoids look-ahead leakage)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    # Hold out the most recent 15% of training events for early-stopping
    val_cut = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train.iloc[:val_cut], y_train.iloc[:val_cut]
    X_vl, y_vl = X_train.iloc[val_cut:], y_train.iloc[val_cut:]

    logger.info("\nTraining LightGBM classifier…")
    lgbm = train_lgbm(X_tr, y_tr, X_vl, y_vl)

    logger.info("\nTraining Logistic Regression baseline…")
    lr = train_logistic_regression(X_train, y_train)

    lgbm_metrics = evaluate_model(lgbm, X_test, y_test, "LightGBM")
    lr_metrics   = evaluate_model(lr,   X_test, y_test, "Logistic Regression")

    # Save evaluation table
    metrics_df = pd.DataFrame([lgbm_metrics, lr_metrics])
    metrics_path = MODELS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved evaluation metrics → %s", metrics_path)

    # Plots
    plot_feature_importance(lgbm, feat_names)
    plot_calibration(lgbm, X_test, y_test, "LightGBM")
    plot_roc_curves(
        [("LightGBM", lgbm.predict_proba(X_test)[:, 1]),
         ("Logistic Regression", lr.predict_proba(X_test)[:, 1])],
        y_test,
    )

    # Persist model artifact
    artifact = {
        "model":          lgbm,
        "feature_names":  feat_names,
        "train_medians":  train_medians.to_dict(),
        "train_end_date": str(train_df["earnings_date"].max().date()),
        "metrics":        lgbm_metrics,
    }
    model_path = MODELS_DIR / "earnings_model.pkl"
    joblib.dump(artifact, model_path)
    logger.info("Saved best model → %s", model_path)

    # Also persist medians as JSON for the Streamlit app
    with open(MODELS_DIR / "feature_medians.json", "w") as f:
        json.dump(train_medians.to_dict(), f, indent=2)

    return lgbm, feat_names, lgbm_metrics


if __name__ == "__main__":
    run_training()
