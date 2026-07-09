"""開發③ 訓練管線：Logistic Regression / Decision Tree / XGBoost。

流程：
    載入 games → 建特徵 → 丟棄暖機期 → 依時間切分 train/test
    → 訓練三個模型（XGBoost 另做 TimeSeriesSplit CV 超參數調校）
    → 在保留測試集上以多指標評估、選出 ROC-AUC 最佳的模型
    → 存最佳模型 + 特徵清單到 models/，存三模型比較表到 reports/

【防洩漏鐵律】
    - 時序切分沿用 baseline.time_split：train = 早賽季、
      test = SEASON >= config.TEST_SEASON_START 的最近賽季。絕不隨機切分。
    - XGBoost 超參數調校使用 sklearn.model_selection.TimeSeriesSplit
      （依時間切 fold），不使用會打亂時序的 KFold/StratifiedKFold。
      調參前資料已依 GAME_DATE_EST 排序（build_features 保證），
      直接交給 TimeSeriesSplit 即為時序 CV。

對外介面（供 evaluate.py / predict.py / tests 匯入，勿隨意改名）：
    BEST_MODEL_PATH, FEATURE_LIST_PATH
    prepare_data() -> (train_df, test_df, feature_cols)
    train_all(train_df, test_df, feature_cols) -> dict
    select_best(results) -> (best_name, best_entry)
    save_artifacts(best_entry, feature_cols)
    load_or_train() -> (model, feature_cols)

用法：
    python -m src.train
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config, data_loader, features
from .baseline import _metrics, time_split

# ---- 模組常數：對外路徑契約 ----
BEST_MODEL_PATH = config.MODELS_DIR / "best_model.joblib"
FEATURE_LIST_PATH = config.MODELS_DIR / "feature_columns.json"

# XGBoost 超參數搜尋空間（範圍刻意保守，避免調校耗時過久）
_XGB_PARAM_DIST = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0, 10.0],
}
_XGB_N_ITER = 25
_XGB_CV_SPLITS = 3


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """載入資料、建特徵、丟棄暖機期、依時序切分。

    回傳 (train_df, test_df, feature_cols)。
    """
    games = data_loader.load_games()
    feats = features.build_features(games)

    feat_cols = features.feature_columns(feats)
    # 與 baseline 相同作法：以 diff_winrate 開頭欄位當關鍵欄，丟棄暖機期（滾動特徵為 NaN 的樣本）
    key_cols = [c for c in feat_cols if c.startswith("diff_winrate")]
    feats = feats.dropna(subset=key_cols).reset_index(drop=True)

    train_df, test_df = time_split(feats)
    return train_df, test_df, feat_cols


def _train_logreg(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    X_train, y_train = train_df[feat_cols], train_df[config.LABEL]
    X_test, y_test = test_df[feat_cols], test_df[config.LABEL]

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_prob)
    return {"model": model, "metrics": metrics}


def _train_dtree(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    X_train, y_train = train_df[feat_cols], train_df[config.LABEL]
    X_test, y_test = test_df[feat_cols], test_df[config.LABEL]

    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            random_state=config.RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_prob)
    return {"model": model, "metrics": metrics}


def _train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str]) -> dict:
    # 資料已依 GAME_DATE_EST 排序（build_features 保證），符合 TimeSeriesSplit 的假設
    X_train, y_train = train_df[feat_cols], train_df[config.LABEL]
    X_test, y_test = test_df[feat_cols], test_df[config.LABEL]

    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    # 【時序 CV】超參數調校使用 TimeSeriesSplit（依時間切 fold），不可用隨機打亂的 CV
    tscv = TimeSeriesSplit(n_splits=_XGB_CV_SPLITS)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=_XGB_PARAM_DIST,
        n_iter=_XGB_N_ITER,
        scoring="roc_auc",
        cv=tscv,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, y_pred, y_prob)
    return {"model": model, "metrics": metrics, "best_params": search.best_params_}


def train_all(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """訓練三個模型並在 test 上評估。"""
    results: dict = {}

    print("訓練 Logistic Regression ...")
    results["Logistic Regression"] = _train_logreg(train_df, test_df, feature_cols)

    print("訓練 Decision Tree ...")
    results["Decision Tree"] = _train_dtree(train_df, test_df, feature_cols)

    print(f"訓練 XGBoost（TimeSeriesSplit x{_XGB_CV_SPLITS} 調校 {_XGB_N_ITER} 組參數）...")
    results["XGBoost"] = _train_xgboost(train_df, test_df, feature_cols)
    print(f"  最佳參數：{results['XGBoost']['best_params']}")

    return results


def select_best(results: dict) -> tuple[str, dict]:
    """依 ROC-AUC 選最佳模型，回傳 (best_name, best_entry)。"""
    best_name = max(results, key=lambda name: results[name]["metrics"]["ROC-AUC"])
    return best_name, results[best_name]


def save_artifacts(best_entry: dict, feature_cols: list[str]) -> None:
    """存最佳模型與特徵清單到 models/。"""
    config.ensure_dirs()
    joblib.dump(best_entry["model"], BEST_MODEL_PATH)
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"最佳模型已存至 {BEST_MODEL_PATH}")
    print(f"特徵清單已存至 {FEATURE_LIST_PATH}")


def load_or_train() -> tuple[object, list[str]]:
    """若已有存檔則直接載入；否則跑完整訓練流程並存檔。

    下游（predict.py / evaluate.py）依此取得模型；由於 models/ 為
    git-ignore，各 worktree 可能沒有現成模型檔，故此函式必須能自動訓練。
    """
    if BEST_MODEL_PATH.exists() and FEATURE_LIST_PATH.exists():
        model = joblib.load(BEST_MODEL_PATH)
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        return model, feature_cols

    train_df, test_df, feature_cols = prepare_data()
    results = train_all(train_df, test_df, feature_cols)
    _, best_entry = select_best(results)
    save_artifacts(best_entry, feature_cols)
    return best_entry["model"], feature_cols


def main() -> None:
    print("載入資料、建特徵、時序切分 ...")
    train_df, test_df, feature_cols = prepare_data()
    print(f"train={len(train_df)}（賽季 <{config.TEST_SEASON_START}）／"
          f"test={len(test_df)}（賽季 >={config.TEST_SEASON_START}）")

    results = train_all(train_df, test_df, feature_cols)

    comparison = pd.DataFrame(
        {name: entry["metrics"] for name, entry in results.items()}
    ).T

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== 三模型測試集表現比較 ===")
    print(comparison)

    config.ensure_dirs()
    out_path = config.REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n比較表已存至 {out_path}")

    best_name, best_entry = select_best(results)
    save_artifacts(best_entry, feature_cols)

    print(f"\n最佳模型（依 ROC-AUC）：{best_name}")
    print(f"  ROC-AUC = {best_entry['metrics']['ROC-AUC']:.4f}")
    print(f"  已存至 {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
