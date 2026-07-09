"""開發② 的 baseline 與第一個模型（Logistic Regression）。

流程：
    載入 games → 建特徵 → 依時間切分 train/test
    → 計算 baseline（主場必勝、多數類別）
    → 訓練 Logistic Regression → 對照多項指標

用法：
    python -m src.baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data_loader, features


def time_split(df: pd.DataFrame):
    """依賽季做時序切分：TEST_SEASON_START（含）以後為測試集。"""
    train = df[df["SEASON"] < config.TEST_SEASON_START].copy()
    test = df[df["SEASON"] >= config.TEST_SEASON_START].copy()
    return train, test


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        "LogLoss": log_loss(y_true, y_prob) if y_prob is not None else np.nan,
    }


def evaluate_baselines(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """兩個 baseline：主場必勝、訓練集多數類別。"""
    y_test = test[config.LABEL].values
    rows = {}

    # 主場必勝：一律預測主隊贏（=1）
    home_pred = np.ones_like(y_test)
    home_prob = np.full_like(y_test, train[config.LABEL].mean(), dtype=float)
    rows["Baseline: 主場必勝"] = _metrics(y_test, home_pred, home_prob)

    # 多數類別：預測訓練集出現最多的類別
    majority = int(round(train[config.LABEL].mean()))
    maj_pred = np.full_like(y_test, majority)
    maj_prob = np.full_like(y_test, train[config.LABEL].mean(), dtype=float)
    rows["Baseline: 多數類別"] = _metrics(y_test, maj_pred, maj_prob)

    return pd.DataFrame(rows).T


def train_logreg(train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str]):
    """訓練 Logistic Regression（含缺值填補與標準化）。"""
    X_train, y_train = train[feat_cols], train[config.LABEL]
    X_test, y_test = test[feat_cols], test[config.LABEL]

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_prob)
    return model, metrics


def run() -> pd.DataFrame:
    print("載入資料 ...")
    games = data_loader.load_games()

    print("建立特徵 ...")
    feats = features.build_features(games)

    # 丟棄早期沒有足夠歷史的比賽（滾動特徵為 NaN 的樣本，主要是每隊季初幾場）
    feat_cols = features.feature_columns(feats)
    key_cols = [c for c in feat_cols if c.startswith("diff_winrate")]
    before = len(feats)
    feats = feats.dropna(subset=key_cols).reset_index(drop=True)
    print(f"丟棄暖機期樣本：{before} → {len(feats)}")

    train, test = time_split(feats)
    print(f"時序切分：train={len(train)}（賽季 <{config.TEST_SEASON_START}）"
          f"／test={len(test)}（賽季 ≥{config.TEST_SEASON_START}）")

    results = evaluate_baselines(train, test)

    print("訓練 Logistic Regression ...")
    _, lr_metrics = train_logreg(train, test, feat_cols)
    results.loc["Logistic Regression"] = lr_metrics

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== 測試集表現 ===")
    print(results)

    config.ensure_dirs()
    out_path = config.REPORTS_DIR / "baseline_results.csv"
    results.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n結果已存至 {out_path}")

    # 是否打贏 baseline
    best_baseline_acc = results.loc[["Baseline: 主場必勝", "Baseline: 多數類別"], "Accuracy"].max()
    lr_acc = results.loc["Logistic Regression", "Accuracy"]
    verdict = "✅ 優於" if lr_acc > best_baseline_acc else "⚠️ 尚未超過"
    print(f"\nLogReg 準確率 {lr_acc:.4f} {verdict} 最佳 baseline {best_baseline_acc:.4f}")
    return results


if __name__ == "__main__":
    run()
