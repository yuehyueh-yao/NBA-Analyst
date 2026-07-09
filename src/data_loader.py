"""載入與清理原始賽事資料。

不論資料來自 Kaggle 或 nba_api，皆輸出同一份乾淨、依時間排序的 games 表，
供 EDA 與特徵工程使用。
"""
from __future__ import annotations

import pandas as pd

from . import config

# 產生賽前特徵所需的最小欄位集合
REQUIRED_COLUMNS = [
    "GAME_DATE_EST",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_ID",
    "SEASON",
    "PTS_home",
    "PTS_away",
    config.LABEL,
]


def load_teams() -> pd.DataFrame:
    """載入球隊對照表（TEAM_ID → 名稱）。"""
    if not config.TEAMS_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {config.TEAMS_CSV}；請先執行資料下載"
            f"（python -m src.download_nba_api）。"
        )
    return pd.read_csv(config.TEAMS_CSV)


def load_games(drop_incomplete: bool = True) -> pd.DataFrame:
    """載入賽事資料，清理並依比賽日期排序。

    Args:
        drop_incomplete: 是否丟棄關鍵欄位有缺值的比賽。
    """
    if not config.GAMES_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {config.GAMES_CSV}；請先執行資料下載"
            f"（python -m src.download_nba_api）。"
        )

    df = pd.read_csv(config.GAMES_CSV)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"games 資料缺少必要欄位：{missing}")

    # 日期轉型並排序（時序處理的基礎）
    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])

    if drop_incomplete:
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        dropped = before - len(df)
        if dropped:
            print(f"[data_loader] 丟棄 {dropped} 場缺值比賽（{before} → {len(df)}）")

    # 標籤轉為整數 0/1
    df[config.LABEL] = df[config.LABEL].astype(int)

    # 去重（同一 GAME_ID 只留一筆）並穩定排序
    if "GAME_ID" in df.columns:
        df = df.drop_duplicates(subset="GAME_ID")

    df = df.sort_values("GAME_DATE_EST", kind="mergesort").reset_index(drop=True)
    return df


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """回傳標籤（主隊是否獲勝）的比例分布。"""
    return df[config.LABEL].value_counts(normalize=True).sort_index()


if __name__ == "__main__":
    games = load_games()
    print(f"載入 {len(games)} 場比賽")
    print(f"賽季範圍：{games['SEASON'].min()}–{games['SEASON'].max()}")
    print(f"日期範圍：{games['GAME_DATE_EST'].min().date()} ~ "
          f"{games['GAME_DATE_EST'].max().date()}")
    print("主隊勝率分布：")
    print(label_distribution(games).rename({0: "客隊勝", 1: "主隊勝"}))
