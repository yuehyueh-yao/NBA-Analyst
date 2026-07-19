# 設計：更新資料至 2018–2025 賽季 + 新賽季（2026-27）開季組合預測

- 日期：2026-07-19
- 狀態：已與使用者確認，待實作

## 背景與目標

目前系統：
- 資料 `data/raw/games.csv` 涵蓋 2018–2023 共 6 個賽季例行賽。
- Streamlit App（`app.py` + `src/predict.py`）已支援「自選主客隊組合」的賽前勝率預測，
  但預設預測日期固定為「資料最後一場（2024-04-14）的隔天 2024-04-15」，且沿用同一賽季，
  並未觸發跨賽季 Elo 回歸。

目標：
1. 把訓練資料更新到最新已完成的 **2025-26 賽季**（season 值 2018–2025，共 8 季）。
2. 預測改為代表 **2026-27 新賽季開季**（該賽季尚未開打，無真實賽程）：
   不依賽程，而是由使用者自選主客隊組合，用「截至 2025-26 賽季」的資料計算開季勝率。
3. 開季實力採 **跨賽季 Elo 回歸**（真開季狀態），而非沿用上季末即時戰力。

## 可行性確認（已驗證）

- `nba_api` 1.11.4 已安裝，可連線 stats.nba.com。
- `LeagueGameLog` 可取得：2024-25（2024-10-22 ~ 2025-04-13）、2025-26（2025-10-21 ~ 2026-04-12），各 1230 場。
- 2026-27 賽季回傳空資料（尚未開打，約 10 月才開始）→ 只能以「假想對戰」形式預測。
- `features.py:compute_elo` 的跨季回歸只依 `SEASON` 整數是否改變觸發（`_pre_rating`，
  features.py:204-207）；把合成試合的 `SEASON` 設為 `max_season + 1` 即可自動套用回歸。
- `tests/test_predict.py` 不檢查預測日期，改預設行為不會弄壞測試。
- 舊 `games.csv` 已在 git 版控歷史中，不需另行手動備份。

## 改動範圍

### ① 資料更新
執行：
```
python -m src.download_nba_api --seasons 2018 2019 2020 2021 2022 2023 2024 2025
```
覆寫 `data/raw/games.csv`（6→8 季，約 9,840 場）與 `data/raw/teams.csv`。
欄位格式與現有一致（`download_nba_api.py` 已固定輸出欄位），下游 `data_loader`/`features` 不需改。

### ② `src/predict.py`：預設改為「新賽季開季」語意
- `_build_synthetic_game(games, home_id, away_id, date)`：
  - 當 `date is None`（App 的預設路徑）：
    - `new_season = int(games["SEASON"].max()) + 1`（= 2026）
    - 合成列 `SEASON = new_season` → 觸發跨賽季 Elo 回歸（真開季狀態）
    - `resolved_date` 設為新賽季開季代表日（`pd.Timestamp(year=new_season, month=10, day=21)`）；
      此日期僅供顯示與時序排序，回歸與否只認 `SEASON` 值。
  - 當使用者明確傳入 `date`：維持現有行為（用該日期、`SEASON = max_season`），確保向後相容。
  - 回傳時新增賽季標籤（例如 `season_label = f"{new_season}-{str(new_season+1)[-2:]}"` → `"2026-27"`）。
- `predict_matchup(...)` 回傳的 dict 新增 `season_label` 欄位，供 App 顯示。

### ③ `app.py`：文案與顯示更新
- 標題/caption 改為明確說明「以截至 2025-26 賽季的資料，預測 **2026-27 開季**的自選對戰（自選主客隊組合，不依賽程）」。
- 顯示 `result["season_label"]`（如「2026-27 賽季開季預測」）。
- expander 內「涵蓋 6 個賽季」→「涵蓋 8 個賽季（2018–2025）」。
- **不新增日期選擇器**：使用者只需選主客隊（YAGNI）。

### ④ `src/config.py`：時序切分點
- `TEST_SEASON_START`：2021 → **2024**。
  8 季下 train = 2018–2023（6 季）、test = 2024–2025（2 季），評估比例較合理。
  僅影響報告指標，不影響 App 預測輸出。

### ⑤ 重新訓練
- 刪除 `models/best_model.joblib` 與 `models/feature_columns.json`（快取模型），
  執行 `python -m src.train` 以新資料重訓三模型並存最佳模型。

### ⑥ 重生報告（建議）
- 執行 `python -m src.evaluate` 更新 `reports/` 的指標表與四張診斷圖（混淆矩陣／ROC／特徵重要度／校準）。

### ⑦ 文件
- README / Progress / TODO 中「6 個賽季」相關敘述更新為「8 個賽季（2018–2025）」，並補註新賽季開季預測模式。

## 驗證方式

1. 資料：讀 `games.csv`，確認 `SEASON` 為 2018–2025、最後一場為 2026-04-12、總場數約 9,840。
2. 預測：`python -m src.predict` 冒煙測試；確認回傳 `season_label == "2026-27"`、主客對調機率不同、機率落在 [0,1]。
3. 跨季回歸生效：確認開季預測的 Elo 已向 1500 回歸（與「沿用上季末」路徑數值不同）。
4. 測試：`pytest`（leakage / baseline / predict 全綠）。
5. App：`streamlit run app.py` 手動點選數組主客隊，畫面顯示 2026-27 開季文案與合理勝率。

## 非目標（YAGNI）

- 不接真實 2026-27 賽程表、不做逐日賽程預測。
- App 不加日期選擇器。
- 不改特徵工程邏輯、不新增特徵。
- 不改模型演算法或超參數搜尋空間。
