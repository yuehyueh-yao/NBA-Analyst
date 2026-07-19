# TODO — NBA 對戰勝率預測

依 PRD（`docs/superpowers/specs/2026-07-02-nba-win-prediction-design.md`）拆解。
狀態標記：`[ ]` 未開始 · `[~]` 進行中 · `[x]` 完成

---

## 開發② — 第一階段送審（目標 07/09）

### 環境與專案骨架
- [x] 建立資料夾結構（data/、notebooks/、src/、models/、reports/、tests/）
- [x] 建立 `requirements.txt`（pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit）
- [x] 建立 `README.md`（專案說明、執行方式）

### 資料載入
- [x] 下載 NBA games 資料到 `data/raw/`（改用 nba_api，免帳號；6 賽季 7059 場）
- [x] `src/data_loader.py`：載入 games/teams、清理、依 GAME_DATE_EST 排序
- [x] 確認標籤欄位 HOME_TEAM_WINS 分布（主場勝率 55.97%）

### EDA
- [x] `notebooks/01_eda.ipynb`：主場勝率、賽季分布、缺值檢查、基本視覺化（已執行含輸出）

### 業務問題與指標定義
- [x] 文件化：業務問題、標籤、評估指標（見 PRD §2、§6.4；指標於 `src/baseline.py` 實作並輸出）

### 特徵工程（第一版：滾動統計）
- [x] `src/features.py`：近 5/10 場勝率、平均得分/失分、休息天數、背靠背
- [x] 賽季累積勝率、主/客場勝率、對戰歷史勝率
- [x] **防洩漏**：逐場只用「該場之前」的資料（全部 shift(1)）

### Baseline
- [x] baseline：主場必勝、多數類別，記錄準確率（55.58%）
- [x] 訓練第一個模型（Logistic Regression）並與 baseline 對照（63.88%，ROC-AUC 0.675）

### 送審
- [x] 整理第一階段成果，07/09 送審

---

## 開發③ — 結案（目標 07/16）

### Elo 特徵
- [x] `src/features.py` 加入 Elo：逐場更新、賽前 Elo、Elo 差值、Elo 期望勝率（AUC 0.675→0.690）
- [x] K 值/主場優勢/季間回歸（config），驗證提升表現

### 模型訓練與調校
- [x] `src/train.py`：時間切分 train/test、訓練 LogReg/DecisionTree/XGBoost
- [x] XGBoost 超參數調校、**時序交叉驗證（TimeSeriesSplit）**
- [x] 存最佳模型到 `models/`（`load_or_train` 介面，含首次自動訓練）

### 評估
- [x] `src/evaluate.py`：多指標表 + 混淆矩陣、ROC、特徵重要度、校準圖
- [x] 產出 `reports/` 評估報告

### 推論與 Demo
- [x] `src/predict.py`：共用 features（合成比賽法，train/serve 一致）→ 輸出勝率 + 關鍵特徵
- [x] `app.py`：Streamlit 選主/客隊 → 勝率% + 勝負 + 關鍵特徵 + 欄位定義/資料來源頁籤

### 測試
- [x] 特徵防洩漏單元測試（**抓到並修正 add_head_to_head h2h 錯位 bug**）
- [x] baseline 對照測試（模型 > baseline）
- [x] 推論冒煙測試（全量 pytest 12/12 通過）

### 部署（GitHub → Streamlit Community Cloud）
- [x] 部署 scaffolding：精簡 requirements.txt + requirements-dev.txt + packages.txt + .streamlit/config.toml + README 部署段
- [x] git init、raw CSV 納版控、gh CLI 安裝與認證、建立 repo `NBA-Analyst`
- [x] push 到 GitHub（含 8 季資料更新 + 預訓練模型，2026-07-19）
- [~] 在 Streamlit Cloud 部署並遠端驗證（GitHub 端就緒，待在 share.streamlit.io 建立 App）

### 交付
- [x] 完整程式碼、評估報告
- [ ] Demo 簡報（含欄位定義與來源）
- [ ] 07/16 結案送審

---

## 開發④ — 資料更新至 2025-26 + 新賽季開季預測（07/19）

設計：`docs/superpowers/specs/2026-07-19-refresh-data-2025-newseason-prediction-design.md`

- [x] 重抓 8 季（2018–2025）共 9509 場（新增 2024-25、2025-26）
- [x] `config.py`：`TEST_SEASON_START` 2021→2024
- [x] `predict.py`：預設改新賽季開季（`SEASON=max+1`、跨季 Elo 回歸）、回傳 `season_label`
- [x] `app.py`：文案改 2026-27 開季自選對戰組合預測、顯示賽季標籤、8 季說明
- [x] 重新訓練（最佳 LogReg，ROC-AUC 0.7292）
- [x] 重生評估報告（reports/）
- [x] pytest 全綠、predict 冒煙驗證跨季回歸生效
