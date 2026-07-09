"""開發③ 評估報告：以保留測試集評估三模型，對最佳模型產生視覺化診斷圖。

流程：
    prepare_data() → train_all() → 印出多指標比較表並存
    reports/evaluation_metrics.csv → select_best() 取最佳模型 → 在測試集上
    產生四張圖（混淆矩陣／ROC 曲線／特徵重要度／機率校準）存到 reports/。

【模型無關的特徵重要度】
    最佳模型可能是：
      - sklearn Pipeline，結尾 estimator 為 LogisticRegression（用 coef_）
      - sklearn Pipeline，結尾 estimator 為 DecisionTreeClassifier（用 feature_importances_）
      - 未包在 Pipeline 裡的 XGBClassifier（用 feature_importances_）
    本模組一律先取出「結尾 estimator」，再依其擁有的屬性
    （feature_importances_ 優先，否則用 abs(coef_)）決定重要度來源，
    不寫死模型名稱或型別。

用法：
    PYTHONIOENCODING=utf-8 MPLBACKEND=Agg python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 無視窗後端：本模組僅存檔，絕不彈出視窗
import matplotlib.pyplot as plt
import seaborn as sns

# 圖表含中文標題／軸標籤，預設 DejaVu Sans 無中文字型會出現缺字方塊，
# 這裡改用 Windows 內建的 Microsoft JhengHei（繁中）；找不到則靜默退回預設字型。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline

from . import config
from .train import prepare_data, select_best, train_all

# 特徵重要度圖最多顯示幾名
_TOP_N_FEATURES = 20


# ---------------------------------------------------------------------------
# 共用小工具：從模型（可能是 Pipeline）取出「結尾 estimator」
# ---------------------------------------------------------------------------
def _final_estimator(model):
    """若為 Pipeline，回傳其最後一步 estimator；否則原樣回傳。"""
    if isinstance(model, Pipeline):
        return model[-1]
    return model


def _feature_importance_values(model) -> np.ndarray:
    """取得特徵重要度數值（一維陣列，與 feature_cols 順序一致）。

    優先使用 feature_importances_（樹模型／XGBoost）；
    否則若有 coef_（線性模型如 LogisticRegression）則取 abs(coef_)。
    """
    estimator = _final_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_).ravel()
    if hasattr(estimator, "coef_"):
        return np.abs(np.asarray(estimator.coef_)).ravel()
    raise ValueError(
        f"模型 {type(estimator).__name__} 既無 feature_importances_ 也無 coef_，"
        "無法產生特徵重要度圖。"
    )


# ---------------------------------------------------------------------------
# 四張圖
# ---------------------------------------------------------------------------
def plot_confusion_matrix(model, X_test, y_test, path) -> None:
    """畫混淆矩陣（含格內數字標註）並存檔。"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=["客隊勝 (0)", "主隊勝 (1)"],
        yticklabels=["客隊勝 (0)", "主隊勝 (1)"],
        ax=ax,
    )
    ax.set_title("混淆矩陣 (Confusion Matrix)")
    ax.set_xlabel("預測類別 (Predicted)")
    ax.set_ylabel("實際類別 (Actual)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(model, X_test, y_test, path) -> None:
    """畫 ROC 曲線（標註 AUC）並存檔。"""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC 曲線 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="隨機猜測")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("偽陽性率 (False Positive Rate)")
    ax.set_ylabel("真陽性率 (True Positive Rate)")
    ax.set_title("ROC 曲線 (ROC Curve)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, path, top_n: int = _TOP_N_FEATURES) -> None:
    """畫特徵重要度長條圖（前 top_n 名）並存檔。

    特徵重要度來源依模型型別自動判斷：
      - 有 feature_importances_（樹模型 / XGBoost）→ 直接使用
      - 有 coef_（線性模型如 LogisticRegression）→ 使用 abs(coef_)
    """
    importances = _feature_importance_values(model)
    if len(importances) != len(feature_cols):
        raise ValueError(
            f"特徵重要度數量 ({len(importances)}) 與 feature_cols 數量 "
            f"({len(feature_cols)}) 不一致。"
        )

    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_features))))
    sns.barplot(x=top_values, y=top_features, hue=top_features, ax=ax,
                palette="viridis", legend=False)
    ax.set_title(f"特徵重要度 Top {len(top_features)} (Feature Importance)")
    ax.set_xlabel("重要度 (importance / |coef|)")
    ax.set_ylabel("特徵 (Feature)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(model, X_test, y_test, path, n_bins: int = 10) -> None:
    """畫機率校準圖 (calibration curve) 並存檔。"""
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", color="steelblue", label="模型校準曲線")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="完美校準")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("預測機率 (Mean predicted probability)")
    ax.set_ylabel("實際發生比例 (Fraction of positives)")
    ax.set_title("機率校準圖 (Calibration Curve)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 統籌流程
# ---------------------------------------------------------------------------
def evaluate_report() -> dict:
    """跑完整評估流程：多指標比較表 + 最佳模型四張診斷圖。

    回傳 dict：{"results": results, "best_name": best_name,
                "best_entry": best_entry, "comparison": comparison}
    """
    config.ensure_dirs()

    print("載入資料、建特徵、時序切分 ...")
    train_df, test_df, feature_cols = prepare_data()
    print(f"train={len(train_df)}（賽季 <{config.TEST_SEASON_START}）／"
          f"test={len(test_df)}（賽季 >={config.TEST_SEASON_START}）")

    results = train_all(train_df, test_df, feature_cols)

    comparison = pd.DataFrame(
        {name: entry["metrics"] for name, entry in results.items()}
    ).T

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== 三模型測試集多指標比較 ===")
    print(comparison)

    metrics_path = config.REPORTS_DIR / "evaluation_metrics.csv"
    comparison.to_csv(metrics_path, encoding="utf-8-sig")
    print(f"\n多指標比較表已存至 {metrics_path}")

    best_name, best_entry = select_best(results)
    best_model = best_entry["model"]
    print(f"\n最佳模型（依 ROC-AUC）：{best_name}")
    print(f"  ROC-AUC = {best_entry['metrics']['ROC-AUC']:.4f}")

    X_test = test_df[feature_cols]
    y_test = test_df[config.LABEL]

    cm_path = config.REPORTS_DIR / "confusion_matrix.png"
    roc_path = config.REPORTS_DIR / "roc_curve.png"
    fi_path = config.REPORTS_DIR / "feature_importance.png"
    cal_path = config.REPORTS_DIR / "calibration_curve.png"

    print("\n產生視覺化診斷圖 ...")
    plot_confusion_matrix(best_model, X_test, y_test, cm_path)
    print(f"  混淆矩陣 -> {cm_path}")
    plot_roc(best_model, X_test, y_test, roc_path)
    print(f"  ROC 曲線 -> {roc_path}")
    plot_feature_importance(best_model, feature_cols, fi_path)
    print(f"  特徵重要度 -> {fi_path}")
    plot_calibration(best_model, X_test, y_test, cal_path)
    print(f"  機率校準圖 -> {cal_path}")

    return {
        "results": results,
        "best_name": best_name,
        "best_entry": best_entry,
        "comparison": comparison,
    }


def main() -> None:
    evaluate_report()


if __name__ == "__main__":
    main()
