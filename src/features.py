"""賽前特徵工程（滾動統計 + 進階數據）。

【最重要的鐵律：防止資料洩漏】
任一場比賽的所有特徵，只能用「該場比賽日期之前」的比賽資料計算。
本模組所有滾動/累積統計都先在球隊時間序列上 `shift(1)`，確保當場結果
（得分、命中率、勝負…）永遠不會進入自己的特徵。

Elo 特徵於開發③ 於此檔案加入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# 進階單場數據：若原始資料含這些欄位，會一併做滾動（否則自動略過）
# key = 長表欄位名, value = (主隊欄位, 客隊欄位)
ADVANCED_STAT_MAP = {
    "fg_pct": ("FG_PCT_home", "FG_PCT_away"),
    "fg3_pct": ("FG3_PCT_home", "FG3_PCT_away"),
    "ft_pct": ("FT_PCT_home", "FT_PCT_away"),
    "ast": ("AST_home", "AST_away"),
    "reb": ("REB_home", "REB_away"),
}

# 開發② 實測：把單場命中率/助攻/籃板做成滾動特徵後，對線性 baseline 反而
# 掉了約 1% 準確率（雜訊 > 訊號，且彼此高度相關）。故預設關閉。
# 開發③ 若改用有內建正則化/特徵選擇的 XGBoost，可再開啟實驗。
# 註：淨得分差(pt_diff)與 EWMA 近況不受此開關影響，為預設保留的高訊號特徵。
USE_ADVANCED_BOXSCORE = False

# 長表中「非特徵」的欄位（含當場結果，嚴禁直接當特徵——會洩漏）
_BASE_LONG_COLS = {
    "GAME_ID", "GAME_DATE_EST", "SEASON", "team", "opponent", "is_home",
    "pts_for", "pts_against", "win", "pt_diff",
    *ADVANCED_STAT_MAP.keys(),
}


def build_team_game_view(games: pd.DataFrame) -> pd.DataFrame:
    """把「一場一列」展開成「一隊一列」的長表（每場兩列：主隊、客隊）。

    這樣就能用單一球隊的時間序列，方便計算各隊近期狀態。
    """

    def side(is_home: int) -> pd.DataFrame:
        home = is_home == 1
        d = pd.DataFrame(
            {
                "GAME_ID": games.get("GAME_ID", games.index),
                "GAME_DATE_EST": games["GAME_DATE_EST"],
                "SEASON": games["SEASON"],
                "team": games["HOME_TEAM_ID"] if home else games["VISITOR_TEAM_ID"],
                "opponent": games["VISITOR_TEAM_ID"] if home else games["HOME_TEAM_ID"],
                "is_home": is_home,
                "pts_for": games["PTS_home"] if home else games["PTS_away"],
                "pts_against": games["PTS_away"] if home else games["PTS_home"],
                "win": games[config.LABEL] if home else 1 - games[config.LABEL],
            }
        )
        # 進階單場數據（預設關閉，見 USE_ADVANCED_BOXSCORE 說明）
        if USE_ADVANCED_BOXSCORE:
            for name, (home_col, away_col) in ADVANCED_STAT_MAP.items():
                col = home_col if home else away_col
                if col in games.columns:
                    d[name] = games[col]
        return d

    long = pd.concat([side(1), side(0)], ignore_index=True)
    long["pt_diff"] = long["pts_for"] - long["pts_against"]  # 淨得分差（勝負幅度）
    # 依球隊 + 時間排序，穩定排序確保同日順序可重現
    long = long.sort_values(["team", "GAME_DATE_EST"], kind="mergesort").reset_index(drop=True)
    return long


def _roll_prior(long: pd.DataFrame, col: str, n: int) -> pd.Series:
    """球隊近 n 場某欄位的平均，只用「該場之前」的資料（先 shift(1)）。"""
    shifted = long.groupby("team", sort=False)[col].shift(1)
    return (
        shifted.groupby(long["team"], sort=False)
        .rolling(n, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )


def add_team_rolling_features(long: pd.DataFrame) -> pd.DataFrame:
    """在球隊長表上加入賽前滾動/累積特徵（全部 shift(1) 防洩漏）。"""
    long = long.copy()

    # ---- 近 N 場滾動 ----
    rolling_stats = ["win", "pts_for", "pts_against", "pt_diff"]
    rolling_stats += [s for s in ADVANCED_STAT_MAP if s in long.columns]
    for n in config.ROLL_WINDOWS:
        for stat in rolling_stats:
            # win 的滾動平均即近 N 場勝率，取個直覺的名字
            out_name = f"winrate_last{n}" if stat == "win" else f"{stat}_last{n}"
            long[out_name] = _roll_prior(long, stat, n)

    grp = long.groupby("team", sort=False)

    # ---- 休息天數與背靠背 ----
    long["rest_days"] = grp["GAME_DATE_EST"].diff().dt.days
    long["back_to_back"] = (long["rest_days"] == 1).astype("Int64")

    # ---- 近況：淨得分差的指數加權平均（近期權重更高）----
    long["form_ptdiff_ewma"] = (
        grp["pt_diff"]
        .apply(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # ---- 賽季至今累積勝率（依 team+season 分組，expanding 後 shift）----
    season_grp = long.groupby(["team", "SEASON"], sort=False)
    long["season_winrate"] = (
        season_grp["win"].apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # ---- 賽季至今主/客場勝率 ----
    long["venue_winrate"] = _venue_winrate(long)

    return long


def _venue_winrate(long: pd.DataFrame) -> pd.Series:
    """賽季至今、同場地（主或客）的勝率，只用先前比賽。"""
    result = pd.Series(index=long.index, dtype="float64")
    for _, idx in long.groupby(["team", "SEASON", "is_home"], sort=False).groups.items():
        sub = long.loc[idx].sort_values("GAME_DATE_EST", kind="mergesort")
        wr = sub["win"].shift(1).expanding().mean()
        result.loc[sub.index] = wr
    return result


def add_head_to_head(games: pd.DataFrame) -> pd.Series:
    """對戰歷史：主隊在兩隊先前交手中的勝率（只用先前交手）。

    以「排序後的兩隊 ID」為配對鍵，追蹤其中一支固定球隊(teamA)的歷史勝率，
    再依當場主隊是不是 teamA 換算成主隊視角。全程 shift(1) 防洩漏。
    """
    g = games.copy()
    g = g.sort_values("GAME_DATE_EST", kind="mergesort")
    a = np.minimum(g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"])
    b = np.maximum(g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"])
    g["_pair"] = list(zip(a, b))
    g["_teamA"] = a
    winner = np.where(g[config.LABEL] == 1, g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"])
    g["_teamA_win"] = (winner == g["_teamA"]).astype(int)

    # 每個配對，teamA 在先前交手的勝率
    teamA_prior = (
        g.groupby("_pair", sort=False)["_teamA_win"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    # 主隊即 teamA 時直接用；否則取 1 - 該值
    home_is_teamA = g["HOME_TEAM_ID"] == g["_teamA"]
    h2h_home = np.where(home_is_teamA, teamA_prior, 1 - teamA_prior)
    return pd.Series(h2h_home, index=g.index).reindex(games.index)


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """逐場計算標準 Elo 評分（賽前值）。

    【防洩漏】對每場比賽，先讀出兩隊「目前」的 Elo 當作該場賽前特徵
    （home_elo_pre / away_elo_pre / elo_prob_home），記錄完畢後才依該場
    實際結果更新兩隊 Elo，供之後的比賽使用。因此任何一場比賽的賽前 Elo
    只反映「該場之前」已結束的比賽；每隊在資料中第一次出現時，
    賽前 Elo 必為 config.ELO_INITIAL。

    跨賽季時，比賽前先將該隊的 Elo 往 config.ELO_INITIAL 回歸
    （regress = config.ELO_SEASON_REGRESS）。

    Returns:
        以 GAME_ID 為 index 的資料框，欄位：
        home_elo_pre, away_elo_pre, diff_elo, elo_prob_home。
    """
    g = games.sort_values("GAME_DATE_EST", kind="mergesort")
    game_ids = g["GAME_ID"].values if "GAME_ID" in g.columns else g.index.values

    ratings: dict = {}
    last_season: dict = {}

    n = len(g)
    home_pre = np.empty(n, dtype=float)
    away_pre = np.empty(n, dtype=float)
    prob_home = np.empty(n, dtype=float)

    home_ids = g["HOME_TEAM_ID"].values
    away_ids = g["VISITOR_TEAM_ID"].values
    seasons = g["SEASON"].values
    results = g[config.LABEL].values

    def _pre_rating(team_id, season) -> float:
        """取得某隊在本場「之前」的 Elo（含跨賽季回歸），並更新 last_season。"""
        if team_id not in ratings:
            ratings[team_id] = config.ELO_INITIAL
        elif last_season[team_id] != season:
            ratings[team_id] = config.ELO_INITIAL + config.ELO_SEASON_REGRESS * (
                ratings[team_id] - config.ELO_INITIAL
            )
        last_season[team_id] = season
        return ratings[team_id]

    for i in range(n):
        home_id, away_id, season = home_ids[i], away_ids[i], seasons[i]

        r_home = _pre_rating(home_id, season)
        r_away = _pre_rating(away_id, season)

        e_home = 1.0 / (1.0 + 10 ** ((r_away - (r_home + config.ELO_HOME_ADVANTAGE)) / 400.0))
        s_home = float(results[i])

        home_pre[i] = r_home
        away_pre[i] = r_away
        prob_home[i] = e_home

        # 記錄完賽前值之後，才依本場結果更新兩隊 Elo（供之後的比賽使用）
        ratings[home_id] = r_home + config.ELO_K * (s_home - e_home)
        ratings[away_id] = r_away + config.ELO_K * ((1 - s_home) - (1 - e_home))

    out = pd.DataFrame(
        {
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "elo_prob_home": prob_home,
        },
        index=game_ids,
    )
    out["diff_elo"] = out["home_elo_pre"] - out["away_elo_pre"]
    out.index.name = "GAME_ID"
    return out


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """組出最終賽前特徵表（一場一列）。

    回傳含主/客隊各自特徵、其差值、對戰歷史與標籤的資料框。
    """
    long = build_team_game_view(games)
    long = add_team_rolling_features(long)

    # 拆回主隊、客隊兩份，準備合併回一場一列
    home_side = long[long["is_home"] == 1].set_index("GAME_ID")
    away_side = long[long["is_home"] == 0].set_index("GAME_ID")

    # 工程特徵 = 長表中所有非「當場結果」的欄位
    engineered = [c for c in long.columns if c not in _BASE_LONG_COLS]

    base_cols = ["GAME_ID", "GAME_DATE_EST", "SEASON",
                 "HOME_TEAM_ID", "VISITOR_TEAM_ID", config.LABEL]
    out = games[[c for c in base_cols if c in games.columns]].copy()
    if "GAME_ID" not in out.columns:
        out["GAME_ID"] = games.index
    out = out.set_index("GAME_ID")

    for c in engineered:
        out[f"home_{c}"] = home_side[c]
        out[f"away_{c}"] = away_side[c]
        # 差值特徵（主隊 - 客隊），模型常更好學；二元旗標不取差
        if c != "back_to_back":
            out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]

    out["h2h_home_winrate"] = add_head_to_head(games).values

    # Elo 評分（賽前值，防洩漏見 compute_elo docstring）
    elo = compute_elo(games)
    for c in elo.columns:
        out[c] = elo[c]

    out = out.reset_index()
    out = out.sort_values("GAME_DATE_EST", kind="mergesort").reset_index(drop=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """回傳可餵給模型的特徵欄位（排除 ID、日期、標籤）。"""
    exclude = {"GAME_ID", "GAME_DATE_EST", "SEASON",
               "HOME_TEAM_ID", "VISITOR_TEAM_ID", config.LABEL}
    return [c for c in df.columns if c not in exclude]


def save_features(df: pd.DataFrame) -> None:
    config.ensure_dirs()
    df.to_csv(config.FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"特徵已存至 {config.FEATURES_CSV}（{len(df)} 列 × {df.shape[1]} 欄）")


if __name__ == "__main__":
    from . import data_loader

    games = data_loader.load_games()
    feats = build_features(games)
    save_features(feats)
    cols = feature_columns(feats)
    print(f"特徵欄位（{len(cols)} 個）：")
    for c in cols:
        na = feats[c].isna().mean()
        print(f"  {c:30s} 缺值 {na:5.1%}")
