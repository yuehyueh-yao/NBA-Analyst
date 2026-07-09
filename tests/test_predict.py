"""推論冒煙測試：predict_matchup / resolve_team 的基本行為與邊界情況。

不重新驗證特徵計算是否正確（那是 test_leakage.py 的事），只確保推論介面
本身可用、輸出型別/範圍合理、錯誤情況會正確丟出例外。

注意：predict_matchup 內部會呼叫 train.load_or_train()，第一次呼叫時
若 models/ 底下沒有現成的模型檔會觸發完整訓練（十幾秒），之後同一個
process 內（甚至同一份 models/ 快取檔）皆會重用，故本檔測試不刻意
共用 fixture 快取模型也不會太慢。
"""
from __future__ import annotations

import pytest

from src import predict


@pytest.fixture(scope="module")
def two_teams():
    """任選兩支存在的球隊（Lakers vs Celtics；找不到則退回資料前兩隊）。"""
    teams = predict.list_teams()
    try:
        home_id = predict.resolve_team("Los Angeles Lakers")
        away_id = predict.resolve_team("Boston Celtics")
    except ValueError:
        home_id = int(teams["TEAM_ID"].iloc[0])
        away_id = int(teams["TEAM_ID"].iloc[1])
    return home_id, away_id


def test_predict_matchup_home_win_prob_in_unit_interval(two_teams):
    home_id, away_id = two_teams
    result = predict.predict_matchup(home_id, away_id)

    assert "home_win_prob" in result
    assert "away_win_prob" in result
    assert 0.0 <= result["home_win_prob"] <= 1.0
    assert 0.0 <= result["away_win_prob"] <= 1.0
    assert result["home_win_prob"] == pytest.approx(1.0 - result["away_win_prob"])
    assert result["predicted_winner"] in ("home", "away")


def test_predict_matchup_swap_home_away_is_sane(two_teams):
    """主客對調兩次，機率都要合理落在 [0,1]，且兩者不應相等（主場優勢應造成差異）。"""
    home_id, away_id = two_teams

    forward = predict.predict_matchup(home_id, away_id)
    swapped = predict.predict_matchup(away_id, home_id)

    assert 0.0 <= forward["home_win_prob"] <= 1.0
    assert 0.0 <= swapped["home_win_prob"] <= 1.0
    assert forward["home_win_prob"] != pytest.approx(swapped["home_win_prob"])


def test_resolve_team_name_abbreviation_id_agree():
    teams = predict.list_teams()
    row = teams.iloc[0]
    team_id = int(row["TEAM_ID"])

    by_id = predict.resolve_team(team_id)
    by_id_str = predict.resolve_team(str(team_id))
    by_full_name = predict.resolve_team(row["FULL_NAME"])
    by_abbr = predict.resolve_team(row["ABBREVIATION"])

    assert by_id == by_id_str == by_full_name == by_abbr == team_id


def test_predict_matchup_same_team_raises_value_error(two_teams):
    home_id, _ = two_teams
    with pytest.raises(ValueError):
        predict.predict_matchup(home_id, home_id)


def test_resolve_team_unknown_name_raises_value_error():
    with pytest.raises(ValueError):
        predict.resolve_team("Definitely Not A Real NBA Team Name 12345")


def test_predict_matchup_unknown_team_raises_value_error(two_teams):
    home_id, _ = two_teams
    with pytest.raises(ValueError):
        predict.predict_matchup(home_id, "Definitely Not A Real NBA Team Name 12345")
