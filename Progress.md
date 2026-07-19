# Progress — NBA 對戰勝率預測

專案進度日誌。每次有進展就在最上方新增一筆（日期 + 完成事項 + 下一步）。

---

## 里程碑總覽

| 里程碑 | 日期 | 狀態 |
|---|---|---|
| 首次諮詢：定方向 | 06/21 | ✅ 已完成 |
| 開發①：環境 + 方向討論 | 07/02 | ✅ 已完成 |
| 開發②：第一階段送審（資料/EDA/指標/滾動特徵/baseline） | 07/09 | ✅ 已完成 |
| 開發③：結案（Elo/建模調校/評估/Streamlit Demo） | 07/16 | ✅ 已完成（advisor-dispatch 派工開發） |

---

## 日誌

### 2026-07-19（資料更新至 2025-26 + 新賽季開季預測）
- ✅ 資料由 6 季（2018–2023）**更新為 8 季（2018–2025）**，共 **9509 場**，最後一場 2026-04-12
- ✅ `config.py`：`TEST_SEASON_START` 2021→**2024**（train=2018–2023 共 7043 場、test=2024–2025 共 2450 場）
- ✅ `predict.py`：預設預測改為 **2026-27 新賽季開季**（合成比賽 `SEASON=max+1`，觸發跨賽季 Elo 回歸＝真開季狀態）；回傳新增 `season_label`
- ✅ `app.py`：文案改為「2026-27 開季自選對戰組合預測」（不依賽程），顯示賽季標籤，涵蓋 8 季說明
- ✅ 重新訓練三模型：最佳 **Logistic Regression，ROC-AUC 0.7292 / Accuracy 0.6784**（測試集 2024–2025）
- ✅ 重生 `reports/`：evaluation_metrics.csv + 混淆矩陣/ROC/特徵重要度/校準圖
- ✅ 測試維持全綠（leakage / baseline / predict）
- 📌 設計文件：[`docs/superpowers/specs/2026-07-19-refresh-data-2025-newseason-prediction-design.md`](docs/superpowers/specs/2026-07-19-refresh-data-2025-newseason-prediction-design.md)

### 2026-07-09（開發③：結案，以 advisor-dispatch 派工開發）
- 🏗️ 主 session 當 advisor，把開發③ 拆成 6 張工單，派 sonnet subagent 在獨立 git worktree 平行實作、逐單 review、merge：
  - A `features.py` 加 **Elo**（防洩漏，雙 verifier 通過）：AUC 0.675→0.690
  - B `train.py`：LogReg/DecisionTree/XGBoost + **TimeSeriesSplit** 時序CV 調校；`load_or_train` 介面
  - C `evaluate.py`：多指標 + 混淆矩陣/ROC/特徵重要度/校準圖 → reports/
  - D `predict.py`：**合成比賽法**重用 build_features，train/serve 一致
  - E `tests/`：防洩漏 / baseline 對照 / 推論冒煙（pytest 12 條）
  - F `app.py`：Streamlit Demo（Cloud 部署友善，st.cache_resource）
- 🐛 **E 的防洩漏測試抓到既有真 bug**：`add_head_to_head` 因 `groupby.apply` 為分組順序、`np.where` 位置對齊而錯位，`h2h_home_winrate` 96% 列錯誤。經 systematic-debugging 確認根因（hand-verify GAME 22000002 truth=0.5 vs 現行0.75），一行 `reindex` 修正。修正後 **全量 pytest 12/12 通過**、LogReg AUC 0.6908/Acc 0.6385。
- 🚀 **部署準備**：精簡 requirements.txt + requirements-dev.txt + packages.txt(libgomp1) + .streamlit/config.toml + README 部署段；git init、raw CSV 納版控、.claude/ 排除；gh CLI 已裝並認證。
- 📦 已建 GitHub repo `NBA-Analyst`(public)，**尚未 push**（待最終同意）。
- ⏭️ 下一步：push 到 GitHub → Streamlit Community Cloud 部署 → 遠端驗證。

### 2026-07-09（開發②補強：進階特徵實驗）
- ➕ `features.py` 加入淨得分差(pt_diff)滾動、EWMA 近況、單場命中率/助攻/籃板滾動能力
- 🔬 對照實驗（同 LR/同切分）：原始 30 特徵 acc **0.6388**／+淨分差+EWMA(39) **0.6377**／全部含命中率(69) **0.6287**
- 💡 **關鍵發現**：單場命中率/助攻/籃板滾動是雜訊（掉 ~1%），已用 `USE_ADVANCED_BOXSCORE=False` 關閉；淨分差+EWMA 中性但概念佳，保留給開發③ XGBoost
- 📌 rolling box-score 的 AUC 天花板約 0.675，要再提升需靠**新訊號（Elo）**→ 開發③

### 2026-07-09（開發②）
- ✅ 建立專案骨架（data/、src/、notebooks/、models/、reports/、tests/）+ requirements.txt / README / .gitignore / config.py
- ✅ 安裝相依套件（含 xgboost、nba_api，Python 3.13 皆正常）
- ✅ 資料改用 **nba_api**（免帳號）下載 6 賽季（2018–2023）共 **7059 場**，轉成與 Kaggle 相容的 games.csv/teams.csv
- ✅ `src/data_loader.py`：清理 + 依日期排序；主場勝率 **55.97%**（符合 55–60% 預期）
- ✅ `src/features.py`：滾動特徵（近5/10場勝率/得失分、休息天數、背靠背、賽季/主客場勝率、對戰歷史），**全部 shift(1) 防洩漏**，共 36 欄
- ✅ `notebooks/01_eda.ipynb`：主場勝率、賽季分布、缺值、得分分布、特徵相關性（已執行含輸出）
- ✅ `src/baseline.py`：時序切分（<2021 訓練 / ≥2021 測試）；**baseline 55.58% → LogReg 63.88%（ROC-AUC 0.675）**，明顯打贏 baseline
- ⏭️ 下一步（開發③）：加入 Elo 特徵、XGBoost 調校、完整評估報告與視覺化、Streamlit Demo、防洩漏單元測試

### 2026-07-02（開發①）
- ✅ 完成需求釐清（brainstorming）：確定賽前預測、Kaggle 為主 + nba_api 補充、Streamlit 選兩隊出勝率、成功標準為明顯優於 baseline
- ✅ 技術路線定案：滾動統計 + Elo（方案 B）
- ✅ 完成 PRD 文件 `docs/superpowers/specs/2026-07-02-nba-win-prediction-design.md`
- ✅ 完成專案企劃書 docx
- ✅ 建立 TODO.md 與 Progress.md
- ⏭️ 下一步：建立專案骨架、下載 Kaggle 資料、開始 EDA

---

## 待決事項 / 風險
- git repo 尚未初始化（PRD 尚未 commit）
- 兩週時程偏緊：若開發② 落後，Elo 可延到開發③ 才加
- Kaggle 資料集賽季範圍需確認（影響樣本數與特徵計算）
