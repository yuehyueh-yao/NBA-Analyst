"""從官方 stats.nba.com（nba_api）下載賽事資料，輸出與 Kaggle 相容的
data/raw/games.csv 與 data/raw/teams.csv（免帳號、免 token）。

用法：
    python -m src.download_nba_api --seasons 2018 2019 2020 2021 2022
    python -m src.download_nba_api                 # 預設近 6 個賽季

輸出欄位（games.csv）對齊 Kaggle NBA games dataset 的關鍵欄位，
確保 data_loader / features 不需分辨資料來源。
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from . import config


def _default_seasons() -> list[int]:
    # 固定清單，避免依賴當前年份；可用 --seasons 覆寫
    return [2017, 2018, 2019, 2020, 2021, 2022]


def fetch_teams() -> pd.DataFrame:
    """靜態球隊清單 → teams.csv。"""
    from nba_api.stats.static import teams as static_teams

    rows = static_teams.get_teams()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "TEAM_ID": df["id"],
            "ABBREVIATION": df["abbreviation"],
            "NICKNAME": df["nickname"],
            "CITY": df["city"],
            "FULL_NAME": df["full_name"],
        }
    )
    return out.sort_values("TEAM_ID").reset_index(drop=True)


def fetch_season(season: int, pause: float = 0.6) -> pd.DataFrame:
    """抓單一賽季例行賽，回傳「一場一列」的 games 資料。

    LeagueGameLog 每場比賽有兩列（主隊、客隊各一），以 MATCHUP 內
    'vs.'（主場）/ '@'（客場）區分，依 GAME_ID 配對還原成一場一列。
    """
    from nba_api.stats.endpoints import leaguegamelog

    season_str = f"{season}-{str(season + 1)[-2:]}"  # 2021 -> "2021-22"
    log = leaguegamelog.LeagueGameLog(
        season=season_str,
        season_type_all_star="Regular Season",
    )
    time.sleep(pause)  # 對官方 API 客氣一點，避免被限流
    df = log.get_data_frames()[0]

    keep = [
        "GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP", "WL",
        "PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB",
    ]
    df = df[keep].copy()

    is_home = df["MATCHUP"].str.contains("vs.", regex=False)
    home = df[is_home].copy()
    away = df[~is_home].copy()

    merged = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

    games = pd.DataFrame(
        {
            "GAME_ID": merged["GAME_ID"],
            "GAME_DATE_EST": merged["GAME_DATE_home"],
            "SEASON": season,
            "HOME_TEAM_ID": merged["TEAM_ID_home"],
            "VISITOR_TEAM_ID": merged["TEAM_ID_away"],
            "PTS_home": merged["PTS_home"],
            "PTS_away": merged["PTS_away"],
            "FG_PCT_home": merged["FG_PCT_home"],
            "FG_PCT_away": merged["FG_PCT_away"],
            "FT_PCT_home": merged["FT_PCT_home"],
            "FT_PCT_away": merged["FT_PCT_away"],
            "FG3_PCT_home": merged["FG3_PCT_home"],
            "FG3_PCT_away": merged["FG3_PCT_away"],
            "AST_home": merged["AST_home"],
            "AST_away": merged["AST_away"],
            "REB_home": merged["REB_home"],
            "REB_away": merged["REB_away"],
            "HOME_TEAM_WINS": (merged["WL_home"] == "W").astype(int),
        }
    )
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="下載 NBA 賽事資料（nba_api）")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=_default_seasons(),
        help="賽季起始年份清單，例如 --seasons 2018 2019 2020",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    print(f"下載球隊清單 ...")
    teams_df = fetch_teams()
    teams_df.to_csv(config.TEAMS_CSV, index=False, encoding="utf-8-sig")
    print(f"  → {config.TEAMS_CSV}（{len(teams_df)} 隊）")

    all_games = []
    for season in args.seasons:
        print(f"下載賽季 {season}-{str(season + 1)[-2:]} ...")
        g = fetch_season(season)
        print(f"  → {len(g)} 場")
        all_games.append(g)

    games_df = pd.concat(all_games, ignore_index=True)
    games_df = games_df.sort_values("GAME_DATE_EST").reset_index(drop=True)
    games_df.to_csv(config.GAMES_CSV, index=False, encoding="utf-8-sig")
    print(f"完成：{config.GAMES_CSV}（共 {len(games_df)} 場，{games_df['SEASON'].nunique()} 個賽季）")


if __name__ == "__main__":
    main()
