# NBA 對戰勝率預測

以歷史 NBA 賽事的結構型資料，建立**賽前勝率預測模型**：比賽開打前選定對戰兩隊，輸出主隊獲勝機率與勝負判定。成果以 Streamlit 互動 Demo 呈現。

> 課程：AI-10 結構型資料的分析案例 · 執行者：姚玥樂 · 講師：Alex Huang
> 完整設計見 [`docs/superpowers/specs/2026-07-02-nba-win-prediction-design.md`](docs/superpowers/specs/2026-07-02-nba-win-prediction-design.md)

---

## 核心設計原則：防止資料洩漏（Data Leakage）

1. **只用賽前已知資訊** — 嚴禁把該場比賽的 box score（得分、命中率等結果）當特徵。
2. **特徵逐場只用「該場比賽日期之前」的歷史資料**計算（滾動統計與 Elo 皆依時間順序更新）。
3. **訓練/測試依時間切分**，不可隨機切分（較早賽季訓練、最近賽季測試）。

---

## 專案結構

```
NBA Project/
├─ data/
│  ├─ raw/          # 原始 CSV（games.csv, teams.csv, ...）
│  └─ processed/    # 特徵工程後的訓練資料
├─ notebooks/
│  ├─ 01_eda.ipynb          # 探索分析
│  └─ 02_modeling.ipynb     # 訓練/評估實驗（開發③）
├─ src/
│  ├─ config.py        # 路徑與常數
│  ├─ data_loader.py   # 載入 + 清理原始資料
│  ├─ features.py      # 賽前特徵工程（滾動統計 + Elo）
│  ├─ baseline.py      # baseline 與第一個模型
│  ├─ train.py         # 切分/訓練/交叉驗證/存模型（開發③）
│  ├─ evaluate.py      # 指標計算 + 視覺化（開發③）
│  └─ predict.py       # 推論介面（開發③）
├─ models/            # 訓練好的模型
├─ reports/           # 評估報告 + 圖表
├─ tests/             # 防洩漏 / baseline 對照 / 冒煙測試
├─ app.py             # Streamlit Demo（開發③）
├─ requirements.txt
└─ README.md
```

---

## 環境安裝

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 取得資料

擇一：

**A. Kaggle（PRD 主要來源）** — 需 Kaggle 帳號與 API token
1. 至 https://www.kaggle.com/settings 產生 API token，下載 `kaggle.json`
2. 放到 `%USERPROFILE%\.kaggle\kaggle.json`
3. 執行：
   ```bash
   python -m src.download_kaggle
   ```
   （或手動下載 https://www.kaggle.com/datasets/nathanlauga/nba-games 解壓到 `data/raw/`）

**B. nba_api（免帳號）** — 直接從官方 stats.nba.com 抓
```bash
python -m src.download_nba_api --seasons 2018 2019 2020 2021 2022
```

兩者皆輸出符合欄位規格的 `data/raw/games.csv` 與 `data/raw/teams.csv`。

## 執行流程

```bash
# 1. 檢視資料與 EDA
jupyter notebook notebooks/01_eda.ipynb

# 2. 產生特徵 + baseline / 第一個模型（開發②）
python -m src.baseline

# 3. （開發③）完整訓練、評估、Demo
python -m src.train
streamlit run app.py
```

---

## 里程碑

| 里程碑 | 日期 | 狀態 |
|---|---|---|
| 開發①：環境 + 方向討論 | 07/02 | ✅ |
| 開發②：資料/EDA/指標/滾動特徵/baseline | 07/09 | ⏳ |
| 開發③：Elo/建模調校/評估/Streamlit Demo | 07/16 | ⬜ |
