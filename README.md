# NBA 對戰勝率預測

以歷史 NBA 賽事的結構型資料（2018–2025 共 8 個賽季例行賽），建立**賽前勝率預測模型**：比賽開打前選定對戰兩隊，輸出主隊獲勝機率與勝負判定。Streamlit Demo 以此預測 **2026-27 新賽季開季**的自選對戰組合（不依賽程，開季實力套用跨賽季 Elo 回歸）。

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
python -m src.download_nba_api --seasons 2018 2019 2020 2021 2022 2023 2024 2025
```

兩者皆輸出符合欄位規格的 `data/raw/games.csv` 與 `data/raw/teams.csv`。

## 執行流程

```bash
# 1. 檢視資料與 EDA
jupyter notebook notebooks/01_eda.ipynb

# 2. 產生特徵 + baseline / 第一個模型（開發②）
python -m src.baseline

# 3. （開發③）完整訓練、評估、Demo
python -m src.train          # 三模型訓練 + 調校，存最佳模型到 models/
python -m src.evaluate       # 多指標 + 混淆矩陣/ROC/特徵重要度/校準圖 → reports/
streamlit run app.py         # 互動 Demo：選兩隊 → 主隊勝率
```

> 注意：`models/` 為衍生產物、未進版控。第一次跑 `app.py` 會自動訓練並快取模型（約 10–15 秒），之後即從快取載入。

---

## 部署到 Streamlit Community Cloud（透過 GitHub）

本專案已備好雲端部署所需檔案：

| 檔案 | 用途 |
|---|---|
| `requirements.txt` | **App 執行時**精簡相依（streamlit/pandas/numpy/scikit-learn/xgboost）；Cloud 依此建置 |
| `requirements-dev.txt` | 開發/EDA/評估/資料下載的完整相依（Cloud 不需要） |
| `packages.txt` | 系統套件 `libgomp1`（xgboost 於 Debian 的執行相依） |
| `.streamlit/config.toml` | 佈景主題與伺服器設定 |
| `data/raw/games.csv` | 已納版控，Cloud 上直接可用；App 首次啟動自動訓練模型 |

**部署步驟**：
1. 把本 repo 推到 GitHub。
2. 到 https://share.streamlit.io → New app → 選此 repo、branch `main`、Main file path `app.py`。
3. Advanced settings 的 Python 版本建議選 **3.11 或 3.12**（相容性較穩）。
4. Deploy。首次啟動會安裝相依並自動訓練模型（十幾秒），完成後即可使用。

> 模型不進版控（避免 joblib/scikit-learn 版本不一致造成反序列化失敗）；改為在雲端**首次啟動時就地訓練**並以 `@st.cache_resource` 快取，較為穩健。

---

## 里程碑

| 里程碑 | 日期 | 狀態 |
|---|---|---|
| 開發①：環境 + 方向討論 | 07/02 | ✅ |
| 開發②：資料/EDA/指標/滾動特徵/baseline | 07/09 | ✅ |
| 開發③：Elo/建模調校/評估/Streamlit Demo | 07/16 | ⏳ 進行中 |
