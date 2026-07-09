"""防洩漏測試（本專案最重要的一類測試）。

核心主張：任一場比賽的賽前特徵，只能反映「該場比賽日期之前」已結束的
比賽，絕不可用到「該場自己」或「該場之後」的資訊。三個測試分別驗證：

1. 每支球隊第一次出現時，賽前 Elo 必為初始值（沒有任何歷史可用）。
2. 竄改某一場「自己的結果」不會改變那一場自己的特徵值（但會影響之後
   的比賽，用來證明竄改確實有效，不是測試寫錯而恰好沒差）。
3. 在資料尾端 append 一場合成的未來比賽，不會反過來改變任何既有比賽
   的特徵值（predict.py 正是靠這個性質才能安全地用同一套
   build_features 產生推論用特徵）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features


def _to_float_row(df: pd.DataFrame, game_id, cols: list[str]) -> np.ndarray:
    """取出某場比賽（GAME_ID）在指定欄位上的值，轉成 float ndarray（含 NaN）。"""
    row = df.loc[df["GAME_ID"] == game_id, cols]
    assert len(row) == 1, f"GAME_ID={game_id} 應唯一，實際找到 {len(row)} 列"
    return row.iloc[0].astype("float64").to_numpy()


def _first_game_ids_per_team(games: pd.DataFrame) -> set:
    """回傳每支球隊在資料中第一次出現的那場 GAME_ID 集合。"""
    long = pd.concat(
        [
            games[["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID"]].rename(
                columns={"HOME_TEAM_ID": "team"}
            ),
            games[["GAME_ID", "GAME_DATE_EST", "VISITOR_TEAM_ID"]].rename(
                columns={"VISITOR_TEAM_ID": "team"}
            ),
        ],
        ignore_index=True,
    )
    long = long.sort_values("GAME_DATE_EST", kind="mergesort")
    firsts = long.groupby("team", sort=False)["GAME_ID"].first()
    return set(firsts.tolist())


def test_elo_first_game_is_initial(features_df: pd.DataFrame):
    """每支球隊第一次出現的那場，賽前 Elo（依 is_home 取值）必為 ELO_INITIAL。"""
    feats = features_df.sort_values("GAME_DATE_EST", kind="mergesort")

    teams = pd.unique(
        pd.concat([feats["HOME_TEAM_ID"], feats["VISITOR_TEAM_ID"]], ignore_index=True)
    )
    assert len(teams) > 0

    checked = 0
    for team_id in teams:
        team_games = feats[
            (feats["HOME_TEAM_ID"] == team_id) | (feats["VISITOR_TEAM_ID"] == team_id)
        ]
        first = team_games.iloc[0]
        if first["HOME_TEAM_ID"] == team_id:
            elo_pre = first["home_elo_pre"]
        else:
            elo_pre = first["away_elo_pre"]
        assert elo_pre == pytest.approx(config.ELO_INITIAL), (
            f"球隊 {team_id} 第一場（GAME_ID={first['GAME_ID']}）賽前 Elo "
            f"應為 {config.ELO_INITIAL}，實際為 {elo_pre}"
        )
        checked += 1

    assert checked == len(teams)


def test_own_outcome_does_not_leak(games: pd.DataFrame):
    """竄改某一場自己的結果，不應改變那一場自己的特徵值（但應影響之後的比賽）。"""
    feats_before = features.build_features(games)
    feat_cols = features.feature_columns(feats_before)

    first_game_ids = _first_game_ids_per_team(games)

    # 任選一場「非某隊首場」的比賽：由資料中段開始找，確保挑到的比賽
    # 兩隊都已有歷史（首場的滾動/Elo 特徵本來就是 NaN/初始值，不適合拿來
    # 驗證「當場結果是否洩漏進當場特徵」這件事）。
    games_sorted = games.sort_values("GAME_DATE_EST", kind="mergesort").reset_index(drop=True)
    candidates = games_sorted[~games_sorted["GAME_ID"].isin(first_game_ids)]
    assert len(candidates) > 0
    target = candidates.iloc[len(candidates) // 2]
    target_id = target["GAME_ID"]
    target_date = target["GAME_DATE_EST"]
    home_id, away_id = target["HOME_TEAM_ID"], target["VISITOR_TEAM_ID"]

    # 竄改該場自己的結果：比分對調 + 勝負反轉
    games_corrupted = games.copy()
    mask = games_corrupted["GAME_ID"] == target_id
    assert mask.sum() == 1
    orig_pts_home = games_corrupted.loc[mask, "PTS_home"].iloc[0]
    orig_pts_away = games_corrupted.loc[mask, "PTS_away"].iloc[0]
    games_corrupted.loc[mask, "PTS_home"] = orig_pts_away
    games_corrupted.loc[mask, "PTS_away"] = orig_pts_home
    games_corrupted.loc[mask, config.LABEL] = 1 - games_corrupted.loc[mask, config.LABEL]

    feats_after = features.build_features(games_corrupted)

    # 斷言一：該場自己那一列的所有特徵值完全不變
    before_row = _to_float_row(feats_before, target_id, feat_cols)
    after_row = _to_float_row(feats_after, target_id, feat_cols)
    np.testing.assert_array_equal(
        before_row, after_row,
        err_msg=(
            f"竄改 GAME_ID={target_id} 自己的結果後，該場自己的特徵值不應改變"
            "（特徵只能用該場之前的資料計算）"
        ),
    )

    # 斷言二（sanity）：竄改確實有效——之後至少有一場（其中一隊參與的）
    # 比賽，特徵值應該改變，否則代表竄改沒有真的影響到下游計算。
    later_games = games_sorted[
        (games_sorted["GAME_DATE_EST"] > target_date)
        & (
            games_sorted["HOME_TEAM_ID"].isin([home_id, away_id])
            | games_sorted["VISITOR_TEAM_ID"].isin([home_id, away_id])
        )
    ]
    assert len(later_games) > 0, "測試資料不足以驗證竄改有向後傳播（找不到後續比賽）"

    changed = False
    for later_id in later_games["GAME_ID"]:
        b = _to_float_row(feats_before, later_id, feat_cols)
        a = _to_float_row(feats_after, later_id, feat_cols)
        if not np.array_equal(b, a, equal_nan=True):
            changed = True
            break
    assert changed, "竄改該場結果後，理應影響到之後至少一場比賽的特徵，但完全沒有改變"


def test_synthetic_append_is_non_invasive(games: pd.DataFrame):
    """在資料尾端 append 一場合成的未來比賽，不應改變任何既有比賽的特徵值。

    這正是 predict.py 推論管線賴以成立的性質：把待預測比賽接在尾端後
    重跑 build_features，得到的特徵必須跟訓練時的特徵計算方式完全一致，
    且不能反過來污染既有比賽的特徵。
    """
    feats_before = features.build_features(games)
    feat_cols = features.feature_columns(feats_before)

    games_sorted = games.sort_values("GAME_DATE_EST", kind="mergesort").reset_index(drop=True)
    last_row = games_sorted.iloc[-1]

    synth = {c: np.nan for c in games.columns}
    synth.update(
        {
            "GAME_ID": -1,
            "GAME_DATE_EST": last_row["GAME_DATE_EST"] + pd.Timedelta(days=1),
            "SEASON": int(games_sorted["SEASON"].max()),
            "HOME_TEAM_ID": last_row["HOME_TEAM_ID"],
            "VISITOR_TEAM_ID": last_row["VISITOR_TEAM_ID"],
            "PTS_home": np.nan,
            "PTS_away": np.nan,
            config.LABEL: 0,
        }
    )
    synth_row = pd.DataFrame([synth])[games.columns]
    games_with_synthetic = pd.concat([games, synth_row], ignore_index=True)

    feats_after = features.build_features(games_with_synthetic)

    # 合成比賽本身應該存在於輸出中
    assert (feats_after["GAME_ID"] == -1).sum() == 1

    # 所有原本真實比賽的特徵值必須完全不變
    real_ids = feats_before["GAME_ID"].tolist()
    before_idx = feats_before.set_index("GAME_ID").loc[real_ids, feat_cols]
    after_idx = feats_after.set_index("GAME_ID").loc[real_ids, feat_cols]

    pd.testing.assert_frame_equal(
        before_idx.astype("float64"),
        after_idx.astype("float64"),
        check_like=False,
    )
