# PRD：NBA 對戰勝率預測 — 結構型資料分類模型與 Streamlit 應用 Demo

- **文件版本**：v1.0
- **日期**：2026-07-02
- **專案執行者**：姚玥樂
- **諮詢講師**：Alex Huang
- **課程**：AI-10 結構型資料的分析案例
- **專案根目錄**：`D:\Eveyaoyao\nSchool - 人工智慧\NBA Project`

---

## 1. 產品概述

以歷史 NBA 賽事的結構型資料，建立一個**賽前勝率預測模型**：在比賽開打前，輸入對戰的兩支球隊，輸出主隊獲勝的機率與勝負判定。成果以 Streamlit 互動 Demo 呈現，並交付完整訓練/推論程式碼與模型效能評估報告。

本專案的核心價值在於完整走過資料科學標準流程（業務問題定義 → 資料處理 → 建模 → 評估 → 應用），並產出可量化、可解釋的預測工具。

---

## 2. 業務問題與目標

### 2.1 業務問題
在一場 NBA 比賽開打前，根據雙方球隊「賽前已知」的狀態資料，預測主隊是否會獲勝，並給出勝率機率，供球迷、分析與決策參考。

### 2.2 建模任務
- **任務類型**：二元分類（binary classification）
- **預測目標（標籤）**：`HOME_TEAM_WINS` ∈ {0, 1}（主隊是否獲勝）
- **輸出**：主隊獲勝機率（0–1）＋勝負判定（門檻 0.5，可調）

### 2.3 專案成功標準
1. 主力模型（XGBoost）的 **ROC-AUC 與準確率明顯優於 baseline**（「主場必勝」與「多數類別」）。
2. 完整報告多項指標：Accuracy、Precision、Recall、F1、ROC-AUC、Log-loss。
3. 交付可運作的 Streamlit Demo、完整訓練/推論程式碼、效能評估報告。

> 註：不設定硬性準確率門檻；重點在「打贏 baseline」與「指標解讀」。NBA 主場勝率歷史約 55–60%，賽前預測模型合理表現約落在 63–68% 準確率區間。

---

## 3. 核心設計原則：防止資料洩漏（Data Leakage）

這是本專案最重要的技術鐵律：

1. **只用賽前已知資訊**。嚴禁使用該場比賽的 box score（得分、命中率、籃板等結果數據）作為特徵——那等同於用結果預測結果。
2. **特徵逐場只用「該場比賽日期之前」的歷史資料**計算（滾動統計與 Elo 皆依時間順序更新）。
3. **訓練/測試依時間切分**，不可隨機切分：以較早賽季為訓練集、最近賽季為測試集，避免用未來資料預測過去。

---

## 4. 資料來源與欄位

### 4.1 資料來源
- **主要**：Kaggle — NBA games dataset（Nathan Lauga）
  https://www.kaggle.com/datasets/nathanlauga/nba-games
  （含 `games.csv`、`teams.csv`、`ranking.csv`、`games_details.csv` 等）
- **加分補充**：`nba_api`（官方 stats.nba.com）https://github.com/swar/nba_api
  用於補抓最新賽季或進階數據（時間允許時才做）。

### 4.2 原始關鍵欄位（來自 games.csv）
| 欄位 | 說明 |
|---|---|
| GAME_DATE_EST | 比賽日期（用於時序排序與切「之前」） |
| HOME_TEAM_ID / VISITOR_TEAM_ID | 主隊 / 客隊 ID |
| SEASON | 賽季 |
| HOME_TEAM_WINS | 標籤：主隊是否獲勝 |
| PTS_home / PTS_away 等 box score | **僅用於產生歷史滾動特徵，禁止作為當場特徵** |

---

## 5. 特徵設計（皆為賽前可得）

| 特徵群 | 內容 |
|---|---|
| 基本 | 主/客場（結構固定主隊為 home）、休息天數、是否背靠背（back-to-back） |
| 近期狀態 | 近 5 / 10 場勝率、平均得分、平均失分（分主隊、客隊） |
| 賽季累積 | 賽季至今勝率、主場勝率、客場勝率 |
| 對戰歷史 | 兩隊近期對戰（head-to-head）勝率 |
| Elo 評分 | 兩隊賽前 Elo、Elo 差值（每場賽後依結果更新，初始 1500） |
| 標籤 | HOME_TEAM_WINS（0/1） |

**Elo 更新規則**：標準 Elo，`R' = R + K * (S - E)`，其中 `E = 1 / (1 + 10^((R_opp - R)/400))`，K 值待實驗（常用 20），可加入主場優勢調整。Elo 逐場依比賽時間順序更新，確保任一場的賽前 Elo 只反映先前比賽。

---

## 6. 模型與評估

### 6.1 Baseline
- **主場必勝**：一律預測主隊贏，記錄其準確率。
- **多數類別**：預測訓練集多數類別。

### 6.2 候選模型
| 模型 | 角色 |
|---|---|
| Logistic Regression | 可解釋的線性基準 |
| Decision Tree | 非線性、易解釋對照 |
| XGBoost | 主力模型，預期表現最佳 |

### 6.3 驗證與調校
- 依賽季/時間做時序切分驗證（避免洩漏）。
- XGBoost 進行超參數調校（如 n_estimators、max_depth、learning_rate）。

### 6.4 評估指標
Accuracy、Precision、Recall、F1-score、ROC-AUC、Log-loss。

### 6.5 評估視覺化
混淆矩陣、ROC 曲線、特徵重要度（feature importance）、機率校準圖（calibration curve）。

---

## 7. 系統架構

```
NBA Project/
├─ data/
│  ├─ raw/          # Kaggle 原始 CSV
│  └─ processed/    # 特徵工程後的訓練資料
├─ notebooks/
│  ├─ 01_eda.ipynb          # 探索分析
│  └─ 02_modeling.ipynb     # 訓練/評估實驗
├─ src/
│  ├─ data_loader.py   # 載入 + 清理原始資料
│  ├─ features.py      # 賽前特徵工程（滾動統計 + Elo）
│  ├─ train.py         # 切分/訓練/交叉驗證/存模型
│  ├─ evaluate.py      # 指標計算 + 視覺化
│  └─ predict.py       # 推論介面（給 app 呼叫）
├─ models/            # 訓練好的模型 (.pkl / .json)
├─ reports/           # 評估報告 + 圖表
├─ app.py             # Streamlit Demo
├─ requirements.txt
└─ README.md
```

**關鍵設計**：`features.py` 的特徵函式由 `train.py` 與 `predict.py` 共用，確保訓練與推論特徵一致，避免 train/serve skew。

---

## 8. 資料流

```
原始 CSV
  → data_loader.py（清理、依日期排序）
  → features.py（逐場只用「該場之前」的資料算滾動特徵與 Elo）
  → 依時間切分 train / test
  → train.py（訓練 LogReg / DecisionTree / XGBoost + 調校）
  → evaluate.py（多指標 + 視覺化）→ 存最佳模型到 models/
                                          ↓
  Streamlit app.py：選主隊 / 客隊 → predict.py 組出賽前特徵 → 輸出主隊勝率
```

---

## 9. Streamlit Demo 規格

- **輸入**：下拉選單選主隊、客隊；可選比賽日期（預設用資料中最新狀態）。
- **輸出**：
  - 主隊獲勝機率 %（量表 / 進度條呈現）
  - 勝負判定（門檻 0.5）
  - Top 關鍵特徵貢獻說明（讓預測可解讀）
- **附加頁籤**：欄位定義與資料來源說明（對齊課程 Demo 驗收要求）。

---

## 10. 測試策略（輕量，聚焦防洩漏）

1. **特徵防洩漏單元測試**：驗證某場比賽的特徵不包含該場或未來的資料。
2. **baseline 對照測試**：訓練後模型準確率必須 > baseline。
3. **推論冒煙測試**：任選兩隊，`predict.py` 能正常輸出合理機率（0–1）。

---

## 11. 交付物（對齊課程完成驗收）

| 項目 | 對應檔案 |
|---|---|
| (a) 完整訓練與推論程式碼 | `src/`、`notebooks/`、`app.py` |
| (b) 模型效能評估報告 | `reports/`（含指標表與圖表） |
| (c) 應用情境 Demo 簡報（含欄位定義與來源） | Streamlit app + 簡報 |

---

## 12. 時程與里程碑（兩週）

| 里程碑 | 日期 | 內容 |
|---|---|---|
| 首次諮詢 | 06/21 | 定方向（已完成） |
| 開發① | 07/02 | 環境 + 方向討論（已完成） |
| 開發②（第一階段送審） | 07/09 | 資料載入、EDA、業務問題與指標定義、滾動特徵、baseline 模型 |
| 開發③（結案） | 07/16 | Elo 特徵、三模型訓練與調校、完整評估報告、Streamlit Demo |

---

## 13. 技術棧

Python、Jupyter Notebook、Pandas、NumPy、Scikit-learn、XGBoost、Matplotlib/Seaborn、Streamlit、（加分）nba_api。

---

## 14. 範圍界定（YAGNI）

**本次做**：賽前二元分類、滾動特徵 + Elo、三模型比較、多指標評估、Streamlit 選隊預測 Demo。

**本次不做**：即時比分/傷病資料串接、球員層級模型、比分差（point spread）迴歸、線上部署與帳號系統、下注策略回測。這些列為未來延伸。
