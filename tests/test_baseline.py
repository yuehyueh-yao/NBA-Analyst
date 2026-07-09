"""baseline 對照測試：確保訓練出來的最佳模型明顯優於「亂猜」。

「主場必勝」是本任務最直觀、最強的 baseline（NBA 主場優勢是眾所皆知的
先驗），其準確率就等於測試集中主隊實際獲勝的比例。若模型連這個 baseline
都打不過，代表模型根本沒學到東西。
"""
from __future__ import annotations

import pytest

from src import config, train

# 主場必勝 baseline：預測「主隊必勝」的準確率 = 測試集中主隊實際獲勝比例。
HOME_WIN_BASELINE_ACCURACY = 0.5558


def test_home_win_baseline_matches_expected(prepared_data):
    """驗證測試集本身的主場勝率確實接近規格書載明的 baseline（0.5558）。

    這個測試同時保護了 prepared_data/prepare_data 的行為不被意外改動
    （例如時序切分範圍、暖機期丟棄規則變動）。
    """
    _, test_df, _ = prepared_data
    actual_home_win_rate = test_df[config.LABEL].mean()
    assert actual_home_win_rate == pytest.approx(HOME_WIN_BASELINE_ACCURACY, abs=0.01)


def test_best_model_beats_baseline(trained_results):
    """最佳模型（依 ROC-AUC 選出）在測試集上須明顯優於隨機/baseline。"""
    best_name, best_entry = train.select_best(trained_results)
    metrics = best_entry["metrics"]

    assert metrics["ROC-AUC"] > 0.60, (
        f"最佳模型（{best_name}）ROC-AUC={metrics['ROC-AUC']:.4f}，未明顯優於隨機（0.5）"
    )
    assert metrics["Accuracy"] > HOME_WIN_BASELINE_ACCURACY, (
        f"最佳模型（{best_name}）Accuracy={metrics['Accuracy']:.4f}，"
        f"未超過主場必勝 baseline（{HOME_WIN_BASELINE_ACCURACY}）"
    )


def test_all_models_beat_random_auc(trained_results):
    """三個模型的 ROC-AUC 都應優於隨機猜測（0.5），任何一個掉到 0.5 附近代表管線可能壞了。"""
    for name, entry in trained_results.items():
        auc = entry["metrics"]["ROC-AUC"]
        assert auc > 0.55, f"{name} 的 ROC-AUC={auc:.4f} 過低，接近隨機猜測"
