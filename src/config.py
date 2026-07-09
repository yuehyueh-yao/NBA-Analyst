"""路徑與常數設定。集中管理避免散落各處。"""
from __future__ import annotations

from pathlib import Path

# 專案根目錄（本檔在 src/ 之下）
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# 原始檔
GAMES_CSV = RAW_DIR / "games.csv"
TEAMS_CSV = RAW_DIR / "teams.csv"

# 特徵工程輸出
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# 標籤欄位
LABEL = "HOME_TEAM_WINS"

# 時序切分：此賽季（含）以後作為測試集，之前作為訓練集
TEST_SEASON_START = 2021

# 滾動窗口
ROLL_WINDOWS = (5, 10)

# Elo 參數
ELO_INITIAL = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 100.0  # 主場優勢（Elo 分數）

# 隨機種子
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """確保輸出資料夾存在。"""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
