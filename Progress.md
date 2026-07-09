# Progress — NBA 對戰勝率預測

專案進度日誌。每次有進展就在最上方新增一筆（日期 + 完成事項 + 下一步）。

---

## 里程碑總覽

| 里程碑 | 日期 | 狀態 |
|---|---|---|
| 首次諮詢：定方向 | 06/21 | ✅ 已完成 |
| 開發①：環境 + 方向討論 | 07/02 | ✅ 已完成 |
| 開發②：第一階段送審（資料/EDA/指標/滾動特徵/baseline） | 07/09 | ✅ 已完成 |
| 開發③：結案（Elo/建模調校/評估/Streamlit Demo） | 07/16 | ⬜ 尚未開始 |

---

## 日誌

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
