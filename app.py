"""NBA 賽前勝率預測 — Streamlit Demo。

給定主隊、客隊（與可選日期），呼叫既有推論介面 ``src.predict.predict_matchup``
取得主隊賽前勝率、勝負判定與關鍵特徵，並以互動介面呈現。

【Cloud 部署友善設計】
    - 模型載入/訓練透過 ``@st.cache_resource`` 包裝，確保整個 container
      生命週期只執行一次（Streamlit Community Cloud 上 models/ 不存在，
      首次會自動訓練，約需十幾秒；之後的互動皆重用快取，不會重跑）。
    - 球隊清單透過 ``@st.cache_data`` 快取，避免每次互動重新讀檔。
    - 所有重運算都放在被 ``st.cache_*`` 裝飾的函式內，於 app 執行期間
      （而非 import 期）才被呼叫。
    - 圖表全部使用 Streamlit 原生元件（st.progress / st.metric /
      st.bar_chart），不使用會嘗試開視窗的 matplotlib.pyplot.show。

用法：
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import predict, train

st.set_page_config(
    page_title="NBA 賽前勝率預測",
    page_icon="🏀",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 關鍵特徵的白話說明（顯示名稱 + 一句話解讀）
# ---------------------------------------------------------------------------
FEATURE_INFO: dict[str, tuple[str, str]] = {
    "elo_prob_home": (
        "Elo 期望勝率（主隊）",
        "根據雙方目前 Elo 等級分換算出的主隊賽前期望勝率；越接近 100% 代表系統認為主隊實力越明顯領先。",
    ),
    "diff_elo": (
        "Elo 分差（主－客）",
        "主隊與客隊目前 Elo 等級分之差；正值代表主隊的 Elo 評分較高、實力評等較強。",
    ),
    "diff_season_winrate": (
        "賽季勝率差（主－客）",
        "本賽季至目前為止，主隊與客隊戰績勝率之差；正值代表主隊本季戰績較佳。",
    ),
    "diff_winrate_last10": (
        "近10場勝率差（主－客）",
        "雙方最近 10 場比賽的勝率之差，反映中期狀態；正值代表主隊近況較好。",
    ),
    "diff_winrate_last5": (
        "近5場勝率差（主－客）",
        "雙方最近 5 場比賽的勝率之差，反映短期手感/狀態；正值代表主隊近況較好。",
    ),
    "h2h_home_winrate": (
        "對戰歷史勝率（主隊視角）",
        "兩隊過去交手紀錄中（只計入本場之前的交手），目前主隊一方獲勝的比例。",
    ),
}


# ---------------------------------------------------------------------------
# 快取：模型只在整個 container 生命週期載入/訓練一次
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached():
    """載入（或於必要時訓練）模型；以 st.cache_resource 確保只執行一次。"""
    return train.load_or_train()


# ---------------------------------------------------------------------------
# 快取：球隊清單（資料讀取，內容不變時不必重讀）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_teams_cached() -> pd.DataFrame:
    """讀取球隊對照表，供下拉選單使用；以 st.cache_data 快取。"""
    return predict.list_teams()


def main() -> None:
    st.title("🏀 NBA 賽前勝率預測 Demo")
    st.caption(
        "輸入主隊、客隊，模型會依「賽前可得」的特徵（Elo、近況、賽季戰績、"
        "對戰歷史等）預測主隊獲勝機率。資料來源：nba_api（官方 stats.nba.com）。"
    )

    # 首次載入需要訓練模型（models/ 在 Cloud 上不存在），給使用者提示。
    with st.spinner("首次載入：訓練模型中…（約十幾秒，之後的操作不會重跑）"):
        model, feature_cols = load_model_cached()

    teams = load_teams_cached()
    if teams.empty:
        st.error("找不到球隊資料，請確認 data/raw 資料完整。")
        return

    teams_sorted = teams.sort_values("FULL_NAME").reset_index(drop=True)
    team_ids = teams_sorted["TEAM_ID"].tolist()
    id_to_name = dict(zip(teams_sorted["TEAM_ID"], teams_sorted["FULL_NAME"]))

    default_home_idx = 0
    default_away_idx = 1 if len(team_ids) > 1 else 0

    col1, col2 = st.columns(2)
    with col1:
        home_id = st.selectbox(
            "主隊（Home）",
            options=team_ids,
            index=default_home_idx,
            format_func=lambda tid: id_to_name.get(tid, str(tid)),
        )
    with col2:
        away_id = st.selectbox(
            "客隊（Away）",
            options=team_ids,
            index=default_away_idx,
            format_func=lambda tid: id_to_name.get(tid, str(tid)),
        )

    predict_clicked = st.button("預測比賽結果", type="primary")

    if home_id == away_id:
        st.warning("主隊與客隊不可相同，請選擇兩支不同的球隊。")
        return

    if not predict_clicked:
        st.info("選好主客隊後，按下「預測比賽結果」開始預測。")
        return

    try:
        with st.spinner("預測中…"):
            result = predict.predict_matchup(home_id, away_id)
    except Exception as exc:  # noqa: BLE001 — 對使用者顯示友善錯誤訊息
        st.error(f"預測失敗：{exc}")
        return

    st.divider()
    st.subheader(f"{result['home_team']}（主）vs {result['away_team']}（客）")
    st.caption(f"預測日期：{result['date']}　｜　判定門檻：{result['threshold']:.0%}")

    home_prob = result["home_win_prob"]
    away_prob = result["away_win_prob"]

    m1, m2 = st.columns(2)
    m1.metric(f"{result['home_team']} 勝率", f"{home_prob:.1%}")
    m2.metric(f"{result['away_team']} 勝率", f"{away_prob:.1%}")

    st.progress(
        min(max(home_prob, 0.0), 1.0),
        text=f"主隊獲勝機率 {home_prob:.1%}（客隊 {away_prob:.1%}）",
    )

    winner_name = result["home_team"] if result["predicted_winner"] == "home" else result["away_team"]
    if result["predicted_winner"] == "home":
        st.success(f"預測勝方：**{winner_name}**（主場）")
    else:
        st.success(f"預測勝方：**{winner_name}**（客場）")

    st.markdown("#### 關鍵特徵（賽前可得，用來解讀這次預測）")
    key_features = result.get("key_features", [])
    if key_features:
        rows = []
        chart_rows = []
        for kf in key_features:
            feat_key = kf["feature"]
            value = kf["value"]
            label, explanation = FEATURE_INFO.get(feat_key, (feat_key, "（無額外說明）"))
            rows.append(
                {
                    "特徵": label,
                    "數值": "資料不足" if value is None else f"{value:.3f}",
                    "說明": explanation,
                }
            )
            if value is not None:
                chart_rows.append({"特徵": label, "數值": value})

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if chart_rows:
            chart_df = pd.DataFrame(chart_rows).set_index("特徵")
            st.bar_chart(chart_df)
    else:
        st.caption("此對戰暫無可顯示的關鍵特徵。")

    with st.expander("欄位定義與資料來源"):
        st.markdown(
            """
**資料來源**：nba_api（官方 stats.nba.com），涵蓋 6 個賽季的例行賽對戰紀錄。

**主要特徵欄位說明**（皆為「賽前可得」資料，計算時嚴格只使用比賽當日之前
已結束的比賽，杜絕資料洩漏）：

- **近5 / 近10場勝率**（`winrate_last5` / `winrate_last10`）：球隊最近 5、10 場
  比賽的勝率，反映短、中期狀態。
- **休息天數 / 背靠背**（`rest_days` / `back_to_back`）：距離上一場比賽的天數；
  若只休息 1 天（背靠背）通常較不利。
- **賽季勝率**（`season_winrate`）：本賽季至目前為止的累積勝率。
- **對戰歷史 h2h**（`h2h_home_winrate`）：兩隊過去交手紀錄中，主隊一方的勝率。
- **Elo 賽前分與期望勝率**（`home_elo_pre` / `away_elo_pre` / `elo_prob_home`）：
  逐場更新的 Elo 等級分（賽前值）及依此換算出的主隊期望勝率。

**防資料洩漏**：以上所有特徵在計算時都對球隊時間序列做了 `shift(1)`，
即每場比賽的特徵只反映「該場之前」已完成的比賽，不會使用到當場或未來的
比賽結果，確保訓練與預測階段的特徵計算方式完全一致（train/serve 一致）。
            """
        )


if __name__ == "__main__":
    main()
