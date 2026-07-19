"""推論介面：給定主客隊與可選日期，輸出主隊勝率、勝負判定與關鍵特徵。

供之後的 Streamlit app（開發③下一張工單）呼叫。

【設計原則：train/serve 一致 + 防洩漏】
本模組刻意不重新實作特徵計算邏輯（那會與訓練時的管線不一致，且容易漏掉
防洩漏細節），而是把「要預測的這場比賽」以一列合成資料
（sentinel GAME_ID、PTS 缺值、標籤佔位）附加到既有 games 表尾端，
再直接呼叫 features.build_features 產生賽前特徵：

    1. 讀入 games（依日期排序）。
    2. 組一列合成比賽：日期預設為資料中最後一場之後一天、賽季採最新賽季、
       主/客隊為使用者指定的兩隊，PTS/標籤皆為無意義佔位值。
    3. 把合成列 concat 到 games 尾端，餵給 build_features。
    4. 取出合成列對應的賽前特徵（elo/滾動/h2h 皆只用「該場之前」的比賽，
       shift(1) 已在 features.py 內完成），交給訓練好的模型 predict_proba。

因為 features.py 裡所有滾動/累積/Elo/h2h 特徵都是先 shift(1) 才計算，
合成列自己的佔位 PTS 與標籤永遠不會回頭影響「它自己那一列」的賽前特徵值
（它也是資料中的最後一場，之後沒有其他比賽會用到它，所以也不影響任何
其他列）。於是這裡取得的特徵向量，與訓練時 build_features 產生的特徵
在欄位定義與計算方式上完全一致。

對外介面（app 會用，勿隨意改名）：
    list_teams() -> pd.DataFrame
    resolve_team(name_or_id) -> int
    predict_matchup(home, away, date=None) -> dict

用法：
    PYTHONIOENCODING=utf-8 python -m src.predict
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from . import config, data_loader, features, train

# 合成比賽用的哨兵 GAME_ID（真實資料的 GAME_ID 皆為 8 位正整數，不會相撞）
_SYNTHETIC_GAME_ID = -1

# app 顯示用的關鍵特徵候選清單（只挑最有解釋力、最常見的幾個）
_KEY_FEATURE_CANDIDATES = [
    "elo_prob_home",
    "diff_elo",
    "diff_season_winrate",
    "diff_winrate_last10",
    "diff_winrate_last5",
    "h2h_home_winrate",
]


def list_teams() -> pd.DataFrame:
    """回傳可選球隊清單（含 TEAM_ID 與可顯示名稱），供下拉選單使用。"""
    return data_loader.load_teams().copy()


def resolve_team(name_or_id: Union[int, str]) -> int:
    """把 TEAM_ID（int）或球隊名稱/縮寫（str）解析成 TEAM_ID。

    接受 FULL_NAME（如 "Los Angeles Lakers"）、NICKNAME（如 "Lakers"）、
    ABBREVIATION（如 "LAL"）、CITY（如 "Los Angeles"），皆不分大小寫；
    也接受可轉為整數的字串（如 "1610612747"）。找不到時丟出 ValueError。
    """
    teams = list_teams()
    team_ids = set(teams["TEAM_ID"].tolist())

    if isinstance(name_or_id, (int, np.integer)) and not isinstance(name_or_id, bool):
        team_id = int(name_or_id)
        if team_id in team_ids:
            return team_id
        raise ValueError(f"找不到 TEAM_ID={team_id} 的球隊；可用球隊請見 list_teams()")

    name = str(name_or_id).strip()
    if name.isdigit():
        team_id = int(name)
        if team_id in team_ids:
            return team_id
        raise ValueError(f"找不到 TEAM_ID={team_id} 的球隊；可用球隊請見 list_teams()")

    name_lower = name.lower()
    for col in ("FULL_NAME", "NICKNAME", "ABBREVIATION", "CITY"):
        if col not in teams.columns:
            continue
        exact = teams[teams[col].astype(str).str.lower() == name_lower]
        if len(exact):
            return int(exact.iloc[0]["TEAM_ID"])

    # 寬鬆比對：名稱是否包含於 FULL_NAME（例如只打 "Lakers"）
    if "FULL_NAME" in teams.columns:
        contains = teams[teams["FULL_NAME"].astype(str).str.lower().str.contains(name_lower, na=False)]
        if len(contains):
            return int(contains.iloc[0]["TEAM_ID"])

    raise ValueError(
        f"無法辨識球隊：{name_or_id!r}。請用 TEAM_ID 或 FULL_NAME/NICKNAME/"
        f"ABBREVIATION/CITY（可呼叫 list_teams() 查看可用球隊）。"
    )


def _display_name(team_id: int) -> str:
    teams = list_teams()
    row = teams[teams["TEAM_ID"] == team_id]
    if len(row) and "FULL_NAME" in teams.columns:
        return str(row.iloc[0]["FULL_NAME"])
    return str(team_id)


def _build_synthetic_game(
    games: pd.DataFrame,
    home_id: int,
    away_id: int,
    date: Optional[Union[str, pd.Timestamp]],
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    """把待預測的比賽組成合成列，附加於 games 尾端；回傳 (合併後 games, 採用日期, 賽季)。

    【預設＝下一個新賽季開季】
    當 date 省略時（App 的預設路徑），把合成比賽視為「資料中最新賽季的下一個
    新賽季」開季戰：SEASON 取 max_season + 1、日期取該新賽季的代表開季日
    （10/21 前後）。因為 features.compute_elo 的跨賽季回歸只依 SEASON 整數
    是否改變觸發，這樣就會對兩隊套用一次跨季 Elo 回歸，反映「新賽季重新洗牌」
    的開季基準（而非沿用上季末的即時戰力）。

    當使用者明確傳入 date 時，維持舊行為：用該日期、賽季取 max_season，
    代表「在既有賽季脈絡下」的一場比賽（向後相容）。
    """
    max_season = int(games["SEASON"].max())
    if date is None:
        season = max_season + 1
        # 新賽季開季代表日；僅供顯示與時序排序，回歸與否只認 SEASON 值
        resolved_date = pd.Timestamp(year=season, month=10, day=21)
    else:
        season = max_season
        resolved_date = pd.Timestamp(date)

    synth = {c: np.nan for c in games.columns}
    synth.update(
        {
            "GAME_ID": _SYNTHETIC_GAME_ID,
            "GAME_DATE_EST": resolved_date,
            "SEASON": season,
            "HOME_TEAM_ID": home_id,
            "VISITOR_TEAM_ID": away_id,
            "PTS_home": np.nan,
            "PTS_away": np.nan,
            config.LABEL: 0,
        }
    )
    synth_row = pd.DataFrame([synth])[games.columns]

    all_games = pd.concat([games, synth_row], ignore_index=True)
    return all_games, resolved_date, season


def predict_matchup(
    home: Union[int, str],
    away: Union[int, str],
    date: Optional[Union[str, pd.Timestamp]] = None,
) -> dict:
    """預測主隊 home 對客隊 away 的勝負機率。

    Args:
        home: 主隊 TEAM_ID 或名稱/縮寫。
        away: 客隊 TEAM_ID 或名稱/縮寫。
        date: 比賽日期（可省略；預設為資料中最後一場之後一天）。

    Returns:
        dict，含 home_team/away_team（顯示名）、home_win_prob/away_win_prob、
        predicted_winner（"home"/"away"）、threshold、season_label（如
        "2026-27"，代表這是哪個賽季的開季預測）、key_features（賽前關鍵特徵
        與其數值的清單，供 app 做可解讀呈現）。
    """
    home_id = resolve_team(home)
    away_id = resolve_team(away)
    if home_id == away_id:
        raise ValueError("主隊與客隊不可相同")

    games = data_loader.load_games()
    all_games, resolved_date, season = _build_synthetic_game(games, home_id, away_id, date)

    feats = features.build_features(all_games)
    synth_row = feats[feats["GAME_ID"] == _SYNTHETIC_GAME_ID]
    if len(synth_row) != 1:
        raise RuntimeError("組合合成比賽特徵失敗，請確認 data/raw 資料完整")

    model, feat_cols = train.load_or_train()
    X = synth_row[feat_cols]
    home_win_prob = float(model.predict_proba(X)[:, 1][0])
    away_win_prob = 1.0 - home_win_prob

    threshold = 0.5
    predicted_winner = "home" if home_win_prob >= threshold else "away"

    key_features = []
    for col in _KEY_FEATURE_CANDIDATES:
        if col in synth_row.columns:
            value = synth_row.iloc[0][col]
            key_features.append(
                {"feature": col, "value": None if pd.isna(value) else float(value)}
            )

    return {
        "home_team": _display_name(home_id),
        "away_team": _display_name(away_id),
        "home_team_id": home_id,
        "away_team_id": away_id,
        "date": str(resolved_date.date()),
        "season_label": f"{season}-{str(season + 1)[-2:]}",
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "predicted_winner": predicted_winner,
        "threshold": threshold,
        "key_features": key_features,
    }


if __name__ == "__main__":
    teams = list_teams()

    # 冒煙測試：任選兩隊（優先嘗試 Lakers vs Celtics，找不到就退回資料前兩隊）
    try:
        home_id = resolve_team("Los Angeles Lakers")
        away_id = resolve_team("Boston Celtics")
    except ValueError:
        home_id, away_id = int(teams["TEAM_ID"].iloc[0]), int(teams["TEAM_ID"].iloc[1])

    result = predict_matchup(home_id, away_id)
    print(f"{result['home_team']}(主) vs {result['away_team']}(客)  日期={result['date']}")
    print(
        f"  主隊勝率 = {result['home_win_prob']:.3f}  "
        f"客隊勝率 = {result['away_win_prob']:.3f}  "
        f"預測勝方 = {result['predicted_winner']}"
    )
    print("  關鍵特徵：")
    for kf in result["key_features"]:
        print(f"    {kf['feature']:24s} = {kf['value']}")

    # 主客對調對照：驗證主場優勢方向合理
    swapped = predict_matchup(away_id, home_id)
    print(
        f"\n[主客對調] {swapped['home_team']}(主) vs {swapped['away_team']}(客)  "
        f"主隊勝率 = {swapped['home_win_prob']:.3f}"
    )

    # 名稱 vs TEAM_ID 一致性檢查
    by_name = predict_matchup("Los Angeles Lakers", "Boston Celtics") \
        if home_id == resolve_team("Los Angeles Lakers") else result
    by_id = predict_matchup(home_id, away_id)
    print(
        f"\n[名稱 vs ID 一致性] 名稱呼叫 = {by_name['home_win_prob']:.6f}  "
        f"ID 呼叫 = {by_id['home_win_prob']:.6f}"
    )
