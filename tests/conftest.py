"""共用 pytest fixtures。

會訓練模型 / 建特徵的 fixture 一律用 session 層級快取，避免每個測試
函式各自重跑一次（build_features 約 0.5 秒、train_all 約十幾秒，
若每個測試都重跑會拖慢整體測試時間）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import data_loader, features, train


@pytest.fixture(scope="session")
def games() -> pd.DataFrame:
    """原始賽事資料（依日期排序），整個測試 session 只載入一次。"""
    return data_loader.load_games()


@pytest.fixture(scope="session")
def features_df(games: pd.DataFrame) -> pd.DataFrame:
    """由原始 games 建出的賽前特徵表，整個測試 session 只算一次。"""
    return features.build_features(games)


@pytest.fixture(scope="session")
def prepared_data():
    """時序切分後的 train/test 與特徵欄位清單，整個測試 session 只算一次。"""
    return train.prepare_data()


@pytest.fixture(scope="session")
def trained_results(prepared_data):
    """訓練三個模型並在測試集上評估，整個測試 session 只訓練一次。"""
    train_df, test_df, feat_cols = prepared_data
    return train.train_all(train_df, test_df, feat_cols)
