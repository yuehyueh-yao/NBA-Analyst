"""產生 NBA 對戰勝率預測 — 結案報告 PPT。

用法（在專案根目錄執行）：
    python -m pip install -r requirements-dev.txt   # 首次執行需要 python-pptx
    python scripts/build_final_report_pptx.py

輸出：reports/姚玥樂_第二階段結案報告_NBA Analyst.pptx（16:9，21 頁）

【資料來源與數字口徑】
    本腳本文字中出現的所有數字，均取自 repo 既有事實來源，不憑空捏造：
      - reports/model_comparison.csv / reports/evaluation_metrics.csv （三模型六指標）
      - reports/baseline_results.csv （baseline 主場必勝 55.58% ← 6 季舊資料版本，
        僅用於對照說明「baseline 概念」；本報告採用的是 8 季新資料版本的
        test-set baseline 54.94%，由 src.train.prepare_data() 現場計算得出，
        詳見下方 _load_facts()）
      - models/feature_columns.json （43 個賽前特徵欄位清單）
      - data/raw/games.csv （總場數、賽季範圍、全期主場勝率）
      - src/config.py、src/features.py、src/train.py、src/predict.py 的
        docstring 與程式邏輯（Elo 參數、防洩漏設計、合成比賽法、模型切分方式）
      - Progress.md 開發日誌（進階 box-score 特徵實驗對照數字、防洩漏 bug 故事）
    四張診斷圖（混淆矩陣／ROC／特徵重要度／校準曲線）直接內嵌 reports/ 下的
    既有 PNG 檔，不重畫、不編造。

【重要】reports/*.png 在 .gitignore 中（衍生檔案，不進版控），若該路徑在
本機尚未存在，本腳本會呼叫 _ensure_diagnostic_plots() 用「既有」的
models/best_model.joblib（不重新訓練）現場產生四張圖，確保頁面一定有圖可嵌。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

# ---------------------------------------------------------------------------
# 視覺風格常數（與結案影片一致的運動數據風）
# ---------------------------------------------------------------------------
NAVY = "0B1F3A"          # 底色：深夜藍
ORANGE = "F97316"        # 強調色：籃球橘
TEXT_PRIMARY = "F4F6FB"  # 主文字
TEXT_SECONDARY = "B8C2D9"  # 次要文字
CARD_FILL_HEX = "FFFFFF"   # 卡片底：白色 + 低透明度 → 呈現半透明淺色填充
CARD_FILL_ALPHA = 8        # 百分比（8% 不透明）
CARD_FILL_ALPHA_STRONG = 14

FONT_CJK = "Microsoft JhengHei"   # 中文字型
FONT_NUM = "Arial"                # 數字/英文字型

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
MARGIN = 0.6

REPORTS_DIR = ROOT / "reports"
OUT_PATH = REPORTS_DIR / "姚玥樂_第二階段結案報告_NBA Analyst.pptx"
DEMO_SHOTS_DIR = REPORTS_DIR / "demo_screenshots"

REPORT_DATE = "2026-10-30"
GITHUB_URL = "https://github.com/yuehyueh-yao/NBA-Analyst"
DEMO_URL = "https://nba-analyst-ktgfbnvixska5ghseefnm8.streamlit.app/"

TOTAL_SLIDES = 21
_page_counter = {"n": 0}


# ---------------------------------------------------------------------------
# 事實查核：現場從 repo 讀取數字，不手動謄寫，避免謄寫誤差
# ---------------------------------------------------------------------------
def _load_facts() -> dict:
    """從 repo 既有資料/報表現場計算本 PPT 會用到的所有數字。

    刻意不重新訓練模型：train_df/test_df 只是「載入資料 + 建特徵 + 時序切分」，
    prepare_data() 本身不訓練；三模型的指標則直接讀現成的
    reports/model_comparison.csv（不重新訓練、不重新產生 CSV）。
    """
    import pandas as pd

    from src import config, data_loader  # noqa: E402
    from src.train import prepare_data  # noqa: E402

    games = data_loader.load_games()
    n_games = len(games)
    home_winrate_all = float(games[config.LABEL].mean())
    season_min, season_max = int(games["SEASON"].min()), int(games["SEASON"].max())
    date_min = str(games["GAME_DATE_EST"].min().date())
    date_max = str(games["GAME_DATE_EST"].max().date())

    train_df, test_df, feat_cols = prepare_data()
    n_train, n_test = len(train_df), len(test_df)
    baseline_test_acc = float(test_df[config.LABEL].mean())

    comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv", index_col=0)

    with open(ROOT / "models" / "feature_columns.json", "r", encoding="utf-8") as f:
        import json
        feature_columns = json.load(f)

    return {
        "n_games": n_games,
        "home_winrate_all": home_winrate_all,
        "season_min": season_min,
        "season_max": season_max,
        "date_min": date_min,
        "date_max": date_max,
        "n_train": n_train,
        "n_test": n_test,
        "n_warmup_dropped": n_games - (n_train + n_test),
        "baseline_test_acc": baseline_test_acc,
        "comparison": comparison,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
    }


def _ensure_diagnostic_plots(facts: dict) -> None:
    """確保四張診斷圖存在；若不存在則用「既有」預訓練模型現場產生（不重新訓練）。

    reports/*.png 屬 .gitignore 衍生檔，各 worktree 可能沒有現成圖檔。
    這裡直接呼叫 src.evaluate 的畫圖函式（沒有改動 evaluate.py 本身），
    模型來源是 src.train.load_or_train()（若 models/best_model.joblib
    已存在則直接載入，不重新訓練）。
    """
    names = ["confusion_matrix.png", "roc_curve.png", "feature_importance.png",
             "calibration_curve.png"]
    if all((REPORTS_DIR / n).exists() for n in names):
        return

    from src import config  # noqa: E402
    from src.train import load_or_train, prepare_data  # noqa: E402
    from src.evaluate import (  # noqa: E402
        plot_confusion_matrix, plot_roc, plot_feature_importance, plot_calibration,
    )

    model, feature_cols = load_or_train()
    train_df, test_df, feat_cols = prepare_data()
    X_test = test_df[feat_cols]
    y_test = test_df[config.LABEL]

    plot_confusion_matrix(model, X_test, y_test, REPORTS_DIR / "confusion_matrix.png")
    plot_roc(model, X_test, y_test, REPORTS_DIR / "roc_curve.png")
    plot_feature_importance(model, feat_cols, REPORTS_DIR / "feature_importance.png")
    plot_calibration(model, X_test, y_test, REPORTS_DIR / "calibration_curve.png")


# ---------------------------------------------------------------------------
# 低階樣式輔助函式
# ---------------------------------------------------------------------------
def _rgb(hex_str: str) -> RGBColor:
    return RGBColor.from_string(hex_str)


def set_fill_alpha(shape, hex_color: str, alpha_pct: float) -> None:
    """設定形狀為純色填滿，並疊加透明度（alpha_pct：0-100，數值越小越透明）。"""
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(hex_color)
    spPr = shape.fill._xPr
    solidFill = spPr.find(qn("a:solidFill"))
    srgbClr = solidFill.find(qn("a:srgbClr"))
    alpha_el = srgbClr.makeelement(qn("a:alpha"), {"val": str(int(alpha_pct * 1000))})
    srgbClr.append(alpha_el)


def style_run(run, size: float, color: str, bold: bool = False, italic: bool = False,
              font: str = FONT_CJK) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _rgb(color)
    run.font.name = font
    # 明確設定東亞字型槽位，確保中文一律用微軟正黑體渲染（不受預設拉丁字型影響）
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def new_textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, word_wrap=True):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = word_wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return box, tf


def no_shadow(shape) -> None:
    shape.shadow.inherit = False


def add_card(slide, x, y, w, h, alpha=CARD_FILL_ALPHA, border=True) -> None:
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    set_fill_alpha(card, CARD_FILL_HEX, alpha)
    if border:
        card.line.color.rgb = _rgb(ORANGE)
        card.line.width = Pt(1)
    else:
        card.line.fill.background()
    no_shadow(card)
    card.text_frame.clear()
    return card


def add_hairline(slide, x, y, w, color=ORANGE, weight=1.0) -> None:
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = _rgb(color)
    ln.line.width = Pt(weight)
    no_shadow(ln)


# ---------------------------------------------------------------------------
# 共用區塊：背景、頁首、頁尾、圓環裝飾
# ---------------------------------------------------------------------------
def new_slide(prs) -> "Slide":
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb(NAVY)
    _add_corner_ring(slide)
    return slide


def _add_corner_ring(slide) -> None:
    """右上角一枚細橘圓環，作為全片一致的視覺母題（呼應籃球意象）。"""
    ring = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(SLIDE_W_IN - 0.75), Inches(0.32), Inches(0.34), Inches(0.34)
    )
    ring.fill.background()
    ring.line.color.rgb = _rgb(ORANGE)
    ring.line.width = Pt(1.5)
    no_shadow(ring)


def add_footer(slide, section_zh: str) -> None:
    _page_counter["n"] += 1
    n = _page_counter["n"]
    _, tf = new_textbox(slide, MARGIN, SLIDE_H_IN - 0.42, 7.0, 0.3, anchor=MSO_ANCHOR.MIDDLE)
    r = tf.paragraphs[0].add_run()
    r.text = "NBA 對戰勝率預測 · 結案報告  ·  " + section_zh
    style_run(r, 9, TEXT_SECONDARY)

    _, tf2 = new_textbox(slide, SLIDE_W_IN - MARGIN - 1.2, SLIDE_H_IN - 0.42, 1.2, 0.3,
                          anchor=MSO_ANCHOR.MIDDLE)
    tf2.paragraphs[0].alignment = PP_ALIGN.RIGHT
    r2 = tf2.paragraphs[0].add_run()
    r2.text = f"{n:02d} / {TOTAL_SLIDES}"
    style_run(r2, 9, TEXT_SECONDARY, font=FONT_NUM)


def add_title_block(slide, eyebrow: str, title: str, y: float = 0.55) -> float:
    """加入「小標籤 + 大標題」，回傳標題區塊底部 y 座標（供下方內容接續排版）。"""
    _, tf_e = new_textbox(slide, MARGIN, y, 10.0, 0.32)
    r = tf_e.paragraphs[0].add_run()
    r.text = eyebrow
    style_run(r, 13, ORANGE, bold=True, font=FONT_NUM)

    _, tf_t = new_textbox(slide, MARGIN, y + 0.32, SLIDE_W_IN - 2 * MARGIN, 0.85)
    r2 = tf_t.paragraphs[0].add_run()
    r2.text = title
    style_run(r2, 30, TEXT_PRIMARY, bold=True)
    return y + 0.32 + 0.85


def add_card_title(tf, text: str, size: float = 16, color: str = ORANGE) -> None:
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    style_run(r, size, color, bold=True)


def add_bullets(tf, items, size: float = 13, color: str = TEXT_PRIMARY, space_after: float = 6,
                 bullet_color: str = ORANGE, first_paragraph_is_title: bool = True) -> None:
    """在既有 text_frame 加入項目符號清單（每項可為 str 或 (主文, 次文) tuple）。"""
    for item in items:
        p = tf.add_paragraph()
        p.space_after = Pt(space_after)
        pPr = p._pPr if p._pPr is not None else p.get_or_add_pPr()
        buChar = pPr.makeelement(qn("a:buChar"), {"char": "▸"})
        buClr = pPr.makeelement(qn("a:buClr"))
        srgb = buClr.makeelement(qn("a:srgbClr"), {"val": bullet_color})
        buClr.append(srgb)
        pPr.append(buClr)
        pPr.append(buChar)
        pPr.set("marL", "182880")
        pPr.set("indent", "-182880")
        if isinstance(item, tuple):
            main, sub = item
            r = p.add_run()
            r.text = main
            style_run(r, size, color)
            if sub:
                r2 = p.add_run()
                r2.text = "  " + sub
                style_run(r2, size - 1.5, TEXT_SECONDARY)
        else:
            r = p.add_run()
            r.text = item
            style_run(r, size, color)


def add_image_fit(slide, path, box_x, box_y, box_w, box_h) -> None:
    """在指定方框內置中嵌入圖片，維持長寬比不變形。"""
    im = Image.open(path)
    iw, ih = im.size
    box_ratio = box_w / box_h
    img_ratio = iw / ih
    if img_ratio > box_ratio:
        w = box_w
        h = w / img_ratio
    else:
        h = box_h
        w = h * img_ratio
    x = box_x + (box_w - w) / 2
    y = box_y + (box_h - h) / 2
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    no_shadow(pic)
    return pic


def fmt_cell(cell, text, size, color, bold=False, fill_hex=None, fill_alpha=None, align=PP_ALIGN.CENTER,
             font=None):
    cell.margin_left = Inches(0.08)
    cell.margin_right = Inches(0.08)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill_hex is not None:
        if fill_alpha is not None:
            set_fill_alpha(cell, fill_hex, fill_alpha)
        else:
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb(fill_hex)
    else:
        cell.fill.background()
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    if font is None:
        font = FONT_NUM if any(ch.isdigit() for ch in text) else FONT_CJK
    style_run(r, size, color, bold=bold, font=font)


def add_text_table(slide, x, y, w, header, rows, col_ratio, row_h=0.42, size=11.5,
                   header_size=12.5, row_colors=None):
    """深色底上的文字表格：橘色表頭 + 白色低透明度資料列；回傳表格高度。"""
    n_rows = len(rows) + 1
    gframe = slide.shapes.add_table(n_rows, len(header), Inches(x), Inches(y),
                                    Inches(w), Inches(row_h * n_rows))
    table = gframe.table
    for i, ratio in enumerate(col_ratio):
        table.columns[i].width = Inches(w * ratio)
    for r in range(n_rows):
        table.rows[r].height = Inches(row_h)
    for c, name in enumerate(header):
        fmt_cell(table.cell(0, c), name, header_size, NAVY, bold=True, fill_hex=ORANGE,
                 align=PP_ALIGN.LEFT, font=FONT_CJK)
    for r_i, row in enumerate(rows, start=1):
        for c_i, text in enumerate(row):
            color = (row_colors or {}).get((r_i - 1, c_i), TEXT_PRIMARY)
            fmt_cell(table.cell(r_i, c_i), text, size, color, bold=(c_i == 0),
                     fill_hex=CARD_FILL_HEX, fill_alpha=6, align=PP_ALIGN.LEFT, font=FONT_CJK)
    return row_h * n_rows


def add_stat(slide, x, y, w, value: str, label: str, value_size=34, label_size=11,
             value_color=ORANGE, align=PP_ALIGN.LEFT) -> None:
    _, tf = new_textbox(slide, x, y, w, 0.6)
    tf.paragraphs[0].alignment = align
    r = tf.paragraphs[0].add_run()
    r.text = value
    style_run(r, value_size, value_color, bold=True, font=FONT_NUM)

    _, tf2 = new_textbox(slide, x, y + 0.58, w, 0.4)
    tf2.paragraphs[0].alignment = align
    r2 = tf2.paragraphs[0].add_run()
    r2.text = label
    style_run(r2, label_size, TEXT_SECONDARY)


# ---------------------------------------------------------------------------
# 各頁內容
# ---------------------------------------------------------------------------
def build_cover(prs, facts) -> None:
    slide = new_slide(prs)

    accent = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.8), Inches(-2.0), Inches(6), Inches(6))
    set_fill_alpha(accent, ORANGE, 10)
    accent.line.fill.background()
    no_shadow(accent)

    _, tf_tag = new_textbox(slide, MARGIN, 1.9, 8, 0.4)
    r = tf_tag.paragraphs[0].add_run()
    r.text = "課程 AI-10 · 結構型資料的分析案例  結案報告"
    style_run(r, 15, ORANGE, bold=True)

    _, tf_title = new_textbox(slide, MARGIN, 2.35, 11.5, 1.9)
    r1 = tf_title.paragraphs[0].add_run()
    r1.text = "NBA 對戰勝率預測"
    style_run(r1, 52, TEXT_PRIMARY, bold=True)
    p2 = tf_title.add_paragraph()
    p2.space_before = Pt(6)
    r2 = p2.add_run()
    r2.text = "結構型資料建模：從賽前特徵到 2026-27 開季勝率 Demo"
    style_run(r2, 19, TEXT_SECONDARY)

    meta_y = 5.55
    metas = [
        ("執行者", "姚玥樂"),
        ("講師", "Alex Huang"),
        ("日期", REPORT_DATE),
        ("交付", "GitHub · Streamlit Cloud"),
    ]
    col_w = (SLIDE_W_IN - 2 * MARGIN) / 4
    for i, (k, v) in enumerate(metas):
        x = MARGIN + i * col_w
        add_hairline(slide, x, meta_y, col_w - 0.35, color=ORANGE, weight=1.0)
        _, tf_k = new_textbox(slide, x, meta_y + 0.12, col_w - 0.35, 0.3)
        rk = tf_k.paragraphs[0].add_run()
        rk.text = k
        style_run(rk, 11, TEXT_SECONDARY)
        _, tf_v = new_textbox(slide, x, meta_y + 0.42, col_w - 0.35, 0.4)
        rv = tf_v.paragraphs[0].add_run()
        rv.text = v
        style_run(rv, 15, TEXT_PRIMARY, bold=True)

    add_footer(slide, "封面")


def build_agenda(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "AGENDA", "大綱")

    groups = [
        ("01", "問題定義與資料", "目標、成功標準／baseline、資料來源、欄位定義與 EDA"),
        ("02", "特徵工程", "43 個賽前特徵、防洩漏三鐵律、Elo 設計與實驗取捨"),
        ("03", "建模與評估", "三模型調校、六指標總表、四張診斷圖"),
        ("04", "工程品質與推論", "pytest 防洩漏測試、抓到的真 bug、合成比賽推論法"),
        ("05", "Demo、成果與未來", "Streamlit Demo 實際畫面、提案目標 → 實際成果、下一步規劃"),
    ]
    y = 1.85
    h = 0.92
    gap = 0.14
    for i, (num, title, desc) in enumerate(groups):
        yy = y + i * (h + gap)
        card = add_card(slide, MARGIN, yy, SLIDE_W_IN - 2 * MARGIN, h, alpha=CARD_FILL_ALPHA)
        _, tf_n = new_textbox(slide, MARGIN + 0.25, yy, 1.0, h, anchor=MSO_ANCHOR.MIDDLE)
        rn = tf_n.paragraphs[0].add_run()
        rn.text = num
        style_run(rn, 26, ORANGE, bold=True, font=FONT_NUM)

        _, tf_c = new_textbox(slide, MARGIN + 1.4, yy, SLIDE_W_IN - 2 * MARGIN - 1.7, h,
                               anchor=MSO_ANCHOR.MIDDLE)
        p = tf_c.paragraphs[0]
        r1 = p.add_run()
        r1.text = title + "   "
        style_run(r1, 17, TEXT_PRIMARY, bold=True)
        r2 = p.add_run()
        r2.text = desc
        style_run(r2, 12.5, TEXT_SECONDARY)

    add_footer(slide, "大綱")


def build_goal(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "01 · 專案目標", "賽前二元分類：預測主場是否獲勝")

    y = 1.9
    add_card(slide, MARGIN, y, 5.9, 4.3)
    _, tf = new_textbox(slide, MARGIN + 0.35, y + 0.3, 5.2, 3.7)
    add_card_title(tf, "問題定義")
    add_bullets(tf, [
        ("輸入", "比賽開打前已知的主／客隊對戰組合"),
        ("輸出", "主隊獲勝機率（0–1）＋ 勝負判定（閾值 0.5）"),
        ("標籤", "HOME_TEAM_WINS（主隊贏=1／輸=0）"),
        ("範圍", "例行賽賽前預測，不使用當場任何結果資訊"),
    ], size=14)

    add_card(slide, MARGIN + 6.15, y, 5.78, 4.3)
    _, tf2 = new_textbox(slide, MARGIN + 6.5, y + 0.3, 5.1, 3.7)
    add_card_title(tf2, "為什麼是這個問題")
    add_bullets(tf2, [
        "結構型資料（每場比賽一列、欄位固定），適合示範完整的特徵工程 → 建模 → 評估 → 部署流程",
        "主場優勢是已知的強先驗，天然形成一個「不難打贏、但也不容易大幅超越」的挑戰性 baseline",
        "可延伸做成互動 Demo：選兩隊即得到可解讀的勝率預測",
    ], size=14)

    add_footer(slide, "專案目標")


def build_baseline(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "01 · 成功標準", "要打敗的門檻：主場必勝 baseline")

    baseline_pct = facts["baseline_test_acc"] * 100
    best = facts["comparison"].sort_values("ROC-AUC", ascending=False).iloc[0]
    best_name = facts["comparison"]["ROC-AUC"].idxmax()
    best_acc = best["Accuracy"] * 100
    delta = best_acc - baseline_pct

    y = 1.9
    add_card(slide, MARGIN, y, 3.7, 4.3)
    add_stat(slide, MARGIN + 0.35, y + 0.5, 3.1, f"{baseline_pct:.2f}%", "測試集 baseline（主場必勝）",
             value_size=40, value_color=TEXT_SECONDARY)
    _, tfb = new_textbox(slide, MARGIN + 0.35, y + 1.9, 3.1, 2.0)
    add_bullets(tfb, [
        "test 2024–2025 賽季（2,450 場）",
        "策略：永遠預測主隊贏",
        "ROC-AUC = 0.5（無鑑別力）",
    ], size=12.5, first_paragraph_is_title=False)

    add_card(slide, MARGIN + 3.95, y, 3.7, 4.3, alpha=CARD_FILL_ALPHA_STRONG)
    add_stat(slide, MARGIN + 3.95 + 0.35, y + 0.5, 3.1, f"{best_acc:.2f}%", f"最佳模型（{best_name}）",
             value_size=40, value_color=ORANGE)
    _, tfc = new_textbox(slide, MARGIN + 3.95 + 0.35, y + 1.9, 3.1, 2.0)
    add_bullets(tfc, [
        f"ROC-AUC = {best['ROC-AUC']:.4f}",
        f"較 baseline 提升 {delta:+.2f} pp",
        "同一測試集、同一時序切分，公平對照",
    ], size=12.5, first_paragraph_is_title=False)

    add_card(slide, MARGIN + 7.9, y, 4.03, 4.3)
    _, tf3 = new_textbox(slide, MARGIN + 7.9 + 0.3, y + 0.3, 3.5, 3.7)
    add_card_title(tf3, "成功標準")
    add_bullets(tf3, [
        "Accuracy／ROC-AUC 明顯高於主場必勝 baseline",
        "機率輸出需合理校準，可直接用於 Demo 呈現勝率",
        "全程防洩漏：特徵只用賽前資訊，時序切分train/test",
    ], size=13)

    add_footer(slide, "成功標準與 baseline")


def build_data(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "01 · 資料", "資料來源與規模")

    y = 1.9
    stats = [
        (f"{facts['n_games']:,}", "總場數（例行賽）"),
        (f"{facts['season_max'] - facts['season_min'] + 1}", "涵蓋賽季數"),
        (f"{facts['home_winrate_all']*100:.2f}%", "全期主場勝率"),
        (f"{facts['n_features']}", "賽前特徵欄位"),
    ]
    col_w = (SLIDE_W_IN - 2 * MARGIN - 3 * 0.25) / 4
    for i, (val, label) in enumerate(stats):
        x = MARGIN + i * (col_w + 0.25)
        add_card(slide, x, y, col_w, 1.5)
        add_stat(slide, x + 0.25, y + 0.22, col_w - 0.5, val, label, value_size=30)

    y2 = y + 1.8
    add_card(slide, MARGIN, y2, 6.9, 2.4)
    _, tf = new_textbox(slide, MARGIN + 0.3, y2 + 0.25, 6.3, 2.0)
    add_card_title(tf, "來源與涵蓋範圍")
    add_bullets(tf, [
        "nba_api（免帳號）直接呼叫 stats.nba.com 官方例行賽資料",
        f"賽季 {facts['season_min']}–{facts['season_max']}（共 8 季），"
        f"最後一場 {facts['date_max']}",
        "已納入版控：data/raw/games.csv、teams.csv",
    ], size=13)

    add_card(slide, MARGIN + 7.2, y2, 4.73, 2.4)
    _, tf2 = new_textbox(slide, MARGIN + 7.2 + 0.3, y2 + 0.25, 4.15, 2.0)
    add_card_title(tf2, "主要欄位")
    add_bullets(tf2, [
        "GAME_ID／GAME_DATE_EST／SEASON",
        "HOME_TEAM_ID／VISITOR_TEAM_ID",
        "PTS_home／PTS_away",
        "HOME_TEAM_WINS（標籤）",
    ], size=12.5)

    add_footer(slide, "資料來源與規模")


def build_field_defs(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "01 · 欄位定義", "欄位定義與資料來源")

    src_games = "games.csv"
    rows = [
        ("GAME_ID", "比賽唯一編號", src_games),
        ("GAME_DATE_EST", "比賽日期（美東時間）；時序排序與「賽前」切點的依據", src_games),
        ("SEASON", "賽季起始年（例：2025 = 2025-26 賽季）", src_games),
        ("HOME_TEAM_ID／VISITOR_TEAM_ID", "主隊／客隊的球隊 ID", src_games),
        ("PTS_home／PTS_away", "主隊／客隊得分；只用來計算「過去比賽」的滾動統計與 Elo，不當作當場特徵",
         src_games),
        ("FG_PCT／FT_PCT／FG3_PCT／AST／REB", "投籃／罰球／三分命中率、助攻、籃板（進階實驗用，預設關閉）",
         src_games),
        ("HOME_TEAM_WINS", "預測標籤：主隊贏 = 1、輸 = 0", "games.csv（由主隊勝負 WL 轉換）"),
        ("TEAM_ID／ABBREVIATION／FULL_NAME", "球隊 ID、縮寫、全名（Demo 下拉選單使用）",
         "teams.csv"),
        (f"賽前特徵（{facts['n_features']} 欄）",
         "例：diff_elo（Elo 差）、elo_prob_home（Elo 期望勝率）、home_winrate_last10（近 10 場勝率）",
         "src/features.py 由上列欄位計算（shift(1) 只用賽前資料）"),
    ]
    w = SLIDE_W_IN - 2 * MARGIN
    h = add_text_table(slide, MARGIN, 1.8, w, ["欄位名稱", "代表什麼", "資料來源"], rows,
                       [0.27, 0.48, 0.25], row_h=0.44, size=11)

    _, tf = new_textbox(slide, MARGIN, 1.8 + h + 0.15, w, 0.35)
    r = tf.paragraphs[0].add_run()
    r.text = ("資料來源：games.csv ← nba_api LeagueGameLog（stats.nba.com 官方例行賽）；teams.csv ← nba_api static teams。"
              "完整 43 個特徵清單見 models/feature_columns.json")
    style_run(r, 11, TEXT_SECONDARY, italic=True)

    add_footer(slide, "欄位定義與資料來源")


def build_demo_screens(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "05 · Demo 畫面", "Demo 實際操作畫面")

    y = 1.85
    box_h = 4.55
    col_w = (SLIDE_W_IN - 2 * MARGIN - 0.3) / 2
    shots = [
        ("demo_predict.png", "① 選 Boston Celtics（主）vs Los Angeles Lakers（客）→ 按「預測比賽結果」→ 雙方勝率與預測勝方"),
        ("demo_key_features.png", "② 關鍵特徵表：Elo 期望勝率、Elo 差、近況與對戰歷史，解讀這次預測"),
    ]
    for i, (fname, caption) in enumerate(shots):
        x = MARGIN + i * (col_w + 0.3)
        add_card(slide, x, y, col_w, box_h)
        add_image_fit(slide, DEMO_SHOTS_DIR / fname, x + 0.15, y + 0.15, col_w - 0.3, box_h - 0.3)
        _, tf = new_textbox(slide, x, y + box_h + 0.1, col_w, 0.5)
        r = tf.paragraphs[0].add_run()
        r.text = caption
        style_run(r, 11.5, TEXT_SECONDARY)

    add_footer(slide, "Demo 實際畫面")


def build_proposal_vs_result(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "05 · 提案對照", "提案目標 → 實際成果")

    comp = facts["comparison"]
    best_name = comp["ROC-AUC"].idxmax()
    best_acc = comp.loc[best_name, "Accuracy"] * 100
    best_auc = comp.loc[best_name, "ROC-AUC"]
    baseline = facts["baseline_test_acc"] * 100

    done, adjusted, delayed, bonus = "✔ 完成", "△ 調整", "△ 延後", "＋ 加分"
    rows = [
        ("業務問題", "賽前預測主隊是否獲勝（二元分類）＋勝率", "HOME_TEAM_WINS 二元分類，輸出勝率與勝負判定", done),
        ("資料來源", "Kaggle NBA games 為主、nba_api 補充",
         f"全面改用 nba_api（免帳號、可更新到最新）：{facts['season_min']}–{facts['season_max']} "
         f"共 8 季 {facts['n_games']:,} 場", adjusted),
        ("特徵工程", "近期戰績、得失分、休息天數、賽季勝率、進階數據",
         f"{facts['n_features']} 個賽前特徵＋Elo；進階 box-score 經實驗為雜訊、預設關閉", done),
        ("Baseline", "主場必勝、多數類別", f"測試集主場必勝 baseline {baseline:.2f}%", done),
        ("模型與調參", "LogReg／Decision Tree／XGBoost＋交叉驗證",
         "三模型完成；XGBoost 以 TimeSeriesSplit 時序 CV 調參", done),
        ("評估", "Accuracy／Precision／Recall／F1／ROC-AUC／Log-loss", "六指標總表＋四張診斷圖", done),
        ("成功標準", "ROC-AUC 與準確率明顯優於基線",
         f"{best_name}：Accuracy {best_acc:.2f}%（{best_acc - baseline:+.2f} pp）、ROC-AUC {best_auc:.4f}", done),
        ("交付物", "訓練／推論程式碼、評估報告、Streamlit Demo",
         "GitHub 公開 repo、reports/、Streamlit Cloud 線上 Demo", done),
        ("額外成果", "（提案未列）", "pytest 12 條測試、資料更新至 2025-26、2026-27 開季預測", bonus),
        ("時程", "07/16 繳交結案報告", "程式 07/09–07/19 完成；結案報告 10/30 送審", delayed),
    ]
    status_color = {done: "4ADE80", adjusted: ORANGE, delayed: ORANGE, bonus: "60A5FA"}
    colors = {(i, 3): status_color[row[3]] for i, row in enumerate(rows)}
    w = SLIDE_W_IN - 2 * MARGIN
    add_text_table(slide, MARGIN, 1.75, w, ["項目", "第一階段提案", "實際成果", "狀態"], rows,
                   [0.12, 0.33, 0.45, 0.10], row_h=0.44, size=10.5, row_colors=colors)

    add_footer(slide, "提案目標 → 實際成果")


def build_eda(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "01 · EDA", "探索性分析重點")

    y = 1.9
    add_card(slide, MARGIN, y, 3.9, 4.3)
    add_stat(slide, MARGIN + 0.3, y + 0.35, 3.3, f"{facts['home_winrate_all']*100:.2f}%",
             "全期主場勝率（9,509 場）", value_size=34)
    _, tf = new_textbox(slide, MARGIN + 0.3, y + 1.7, 3.3, 2.3)
    add_bullets(tf, [
        "主場優勢明顯，符合 NBA 歷史經驗（一般落在 55–60%）",
        "是本專案 baseline 的直接來源",
    ], size=12.5, first_paragraph_is_title=False)

    add_card(slide, MARGIN + 4.1, y, 3.9, 4.3)
    add_stat(slide, MARGIN + 4.1 + 0.3, y + 0.35, 3.3,
             f"{facts['season_max'] - facts['season_min'] + 1} 季", "賽季分布均勻", value_size=34)
    _, tf2 = new_textbox(slide, MARGIN + 4.1 + 0.3, y + 1.7, 3.3, 2.3)
    add_bullets(tf2, [
        f"{facts['season_min']}–{facts['season_max']}，每季約 1,180–1,230 場",
        "時序涵蓋長，足以支撐時序切分 train/test",
    ], size=12.5, first_paragraph_is_title=False)

    add_card(slide, MARGIN + 8.2, y, 3.73, 4.3)
    add_stat(slide, MARGIN + 8.2 + 0.3, y + 0.35, 3.1,
             f"{facts['n_warmup_dropped']} 場", "缺值 / 暖機期處理", value_size=34,
             value_color=TEXT_SECONDARY)
    _, tf3 = new_textbox(slide, MARGIN + 8.2 + 0.3, y + 1.7, 3.1, 2.3)
    add_bullets(tf3, [
        "各隊資料首現時，滾動特徵為 NaN（尚無歷史可用）",
        "以關鍵欄位 dropna 丟棄暖機期樣本後，"
        f"train={facts['n_train']:,} / test={facts['n_test']:,}",
    ], size=12.5, first_paragraph_is_title=False)

    add_footer(slide, "EDA 重點")


def build_features_overview(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "02 · 特徵工程", f"{facts['n_features']} 個賽前特徵總覽")

    rows = [
        ("滾動統計（近 5／10 場）", "24", "勝率／得分／失分／淨得分差，主隊、客隊、差值三種視角"),
        ("EWMA 近況", "3", "淨得分差的指數加權平均，近期表現權重更高"),
        ("休息與背靠背", "5", "休息天數、是否背靠背（rest_days／back_to_back）"),
        ("賽季與主客場勝率", "6", "賽季至今累積勝率、同場地（主／客）勝率"),
        ("對戰歷史", "1", "兩隊交手歷史：主隊視角的勝率（h2h）"),
        ("Elo 評分", "4", "賽前 Elo、期望勝率、Elo 差值（開發③新增）"),
    ]
    y = 1.9
    row_h = 0.62
    x0 = MARGIN
    w_total = SLIDE_W_IN - 2 * MARGIN
    col_widths = [0.55 * w_total, 0.13 * w_total, 0.32 * w_total]

    header = ["特徵分類", "欄位數", "說明"]
    hx = x0
    for htext, cw in zip(header, col_widths):
        _, tf = new_textbox(slide, hx, y, cw - 0.15, 0.35)
        r = tf.paragraphs[0].add_run()
        r.text = htext
        style_run(r, 12, ORANGE, bold=True)
        hx += cw
    add_hairline(slide, x0, y + 0.36, w_total)

    yy = y + 0.5
    for cat, n, desc in rows:
        card = add_card(slide, x0, yy, w_total, row_h, alpha=CARD_FILL_ALPHA, border=False)
        card.line.color.rgb = _rgb(ORANGE)
        card.line.width = Pt(0.5)
        hx = x0
        vals = [cat, n, desc]
        sizes = [13.5, 13.5, 12.5]
        colors = [TEXT_PRIMARY, ORANGE, TEXT_SECONDARY]
        for val, cw, sz, col in zip(vals, col_widths, sizes, colors):
            _, tf = new_textbox(slide, hx + 0.2, yy, cw - 0.3, row_h, anchor=MSO_ANCHOR.MIDDLE)
            r = tf.paragraphs[0].add_run()
            r.text = val
            style_run(r, sz, col, bold=(cw == col_widths[1]), font=(FONT_NUM if cw == col_widths[1] else FONT_CJK))
            hx += cw
        yy += row_h + 0.1

    total_check = sum(int(r[1]) for r in rows)
    _, tf_note = new_textbox(slide, x0, yy + 0.05, w_total, 0.35)
    r = tf_note.paragraphs[0].add_run()
    r.text = f"六大類合計 {total_check} 欄，對齊 models/feature_columns.json 的 {facts['n_features']} 個特徵"
    style_run(r, 11, TEXT_SECONDARY, italic=True)

    add_footer(slide, "特徵工程總覽")


def build_leakage(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "02 · 防洩漏設計", "防止資料洩漏的三鐵律")

    rules = [
        ("1", "只用賽前已知資訊", "嚴禁把該場比賽的 box score（得分、命中率等結果）當特徵；"
         "特徵只能反映「比賽開打前」已知的狀態。"),
        ("2", "逐場 shift(1) 才計算", "所有滾動統計／累積勝率／Elo，都在球隊時間序列上先 "
         "shift(1)，確保當場結果永遠不會進入自己的特徵。"),
        ("3", "時序切分 train/test", "訓練用較早賽季、測試用最近賽季，絕不隨機切分，"
         "貼近「用過去預測未來」的真實情境。"),
    ]
    y = 1.9
    col_w = (SLIDE_W_IN - 2 * MARGIN - 2 * 0.3) / 3
    for i, (num, title, desc) in enumerate(rules):
        x = MARGIN + i * (col_w + 0.3)
        add_card(slide, x, y, col_w, 3.9, alpha=CARD_FILL_ALPHA_STRONG)
        _, tf_n = new_textbox(slide, x + 0.3, y + 0.25, col_w - 0.6, 0.7)
        r = tf_n.paragraphs[0].add_run()
        r.text = num
        style_run(r, 36, ORANGE, bold=True, font=FONT_NUM)
        _, tf_t = new_textbox(slide, x + 0.3, y + 1.0, col_w - 0.6, 0.7)
        r2 = tf_t.paragraphs[0].add_run()
        r2.text = title
        style_run(r2, 16, TEXT_PRIMARY, bold=True)
        _, tf_d = new_textbox(slide, x + 0.3, y + 1.65, col_w - 0.6, 2.1)
        r3 = tf_d.paragraphs[0].add_run()
        r3.text = desc
        style_run(r3, 12.5, TEXT_SECONDARY)
        tf_d.word_wrap = True

    add_footer(slide, "防洩漏三鐵律")


def build_elo(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "02 · Elo 特徵", "Elo 特徵設計")

    y = 1.9
    add_card(slide, MARGIN, y, 6.2, 4.3)
    _, tf = new_textbox(slide, MARGIN + 0.3, y + 0.25, 5.6, 3.9)
    add_card_title(tf, "更新公式（概念）")
    p1 = tf.add_paragraph()
    p1.space_after = Pt(4)
    r1 = p1.add_run()
    r1.text = "期望勝率　E_home = 1 / (1 + 10^(-(R_home+HFA-R_away)/400))"
    style_run(r1, 12.5, TEXT_PRIMARY, font="Consolas")
    p2 = tf.add_paragraph()
    p2.space_after = Pt(4)
    r2 = p2.add_run()
    r2.text = "賽後更新　R_home ← R_home + K × (S_home − E_home)"
    style_run(r2, 12.5, TEXT_PRIMARY, font="Consolas")
    p3 = tf.add_paragraph()
    p3.space_before = Pt(10)
    r3 = p3.add_run()
    r3.text = ("賽前先讀出兩隊「目前」Elo 當特徵，記錄完才依實際結果更新——"
               "確保賽前 Elo 只反映該場之前的比賽（防洩漏）。")
    style_run(r3, 12.5, TEXT_SECONDARY)

    add_card(slide, MARGIN + 6.5, y, 5.23, 4.3)
    _, tf2 = new_textbox(slide, MARGIN + 6.5 + 0.3, y + 0.25, 4.6, 3.9)
    add_card_title(tf2, "參數與貢獻")
    add_bullets(tf2, [
        ("初始值 1500", ""),
        ("K = 20", "每場結果的更新幅度"),
        ("主場優勢 +100", "計入期望勝率的等效分差"),
        ("跨季回歸 0.75", "新賽季開打前，Elo 向 1500 回歸，"
         "保留上季偏移量的 75%"),
    ], size=13)
    p_c = tf2.add_paragraph()
    p_c.space_before = Pt(10)
    rc = p_c.add_run()
    rc.text = "貢獻：加入 Elo 後 ROC-AUC 0.675 → 0.690"
    style_run(rc, 13.5, ORANGE, bold=True)
    p_c2 = tf2.add_paragraph()
    rc2 = p_c2.add_run()
    rc2.text = "（開發③加入 Elo 時的消融對照，6 季資料期間；"
    style_run(rc2, 10.5, TEXT_SECONDARY, italic=True)
    p_c3 = tf2.add_paragraph()
    rc3 = p_c3.add_run()
    rc3.text = "與本報告 8 季／43 特徵最終結果 AUC 0.7292 分屬不同版本）"
    style_run(rc3, 10.5, TEXT_SECONDARY, italic=True)

    add_footer(slide, "Elo 特徵設計")


def build_feature_experiment(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "02 · 實驗與取捨", "特徵實驗：進階 box-score 滾動是雜訊")

    y = 1.9
    add_card(slide, MARGIN, y, 7.0, 4.3)
    _, tf = new_textbox(slide, MARGIN + 0.3, y + 0.25, 6.4, 3.9)
    add_card_title(tf, "對照實驗（同 LogReg、同時序切分）")
    add_bullets(tf, [
        ("原始 30 個特徵", "Accuracy 63.88%"),
        ("+ 淨得分差 + EWMA（39 欄）", "Accuracy 63.77%（中性）"),
        ("+ 單場命中率／助攻／籃板滾動（69 欄）", "Accuracy 62.87%（下降）"),
    ], size=13.5)
    p = tf.add_paragraph()
    p.space_before = Pt(12)
    r = p.add_run()
    r.text = "結論：單場命中率／助攻／籃板滾動彼此高度相關、訊號 < 雜訊，"
    style_run(r, 12.5, TEXT_SECONDARY)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "加入後反而掉約 1% 準確率。"
    style_run(r2, 12.5, TEXT_SECONDARY)

    add_card(slide, MARGIN + 7.3, y, 4.43, 4.3, alpha=CARD_FILL_ALPHA_STRONG)
    _, tf2 = new_textbox(slide, MARGIN + 7.3 + 0.3, y + 0.3, 3.85, 3.7)
    add_card_title(tf2, "取捨決策", color=ORANGE)
    add_bullets(tf2, [
        "USE_ADVANCED_BOXSCORE = False（src/features.py，預設關閉）",
        "淨得分差(pt_diff) + EWMA 近況：訊號中性但概念佳，保留供 "
        "XGBoost 等有正則化能力的模型使用",
    ], size=12.5)

    add_footer(slide, "特徵實驗與取捨")


def build_modeling(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "03 · 建模", "建模與調校")

    y = 1.9
    cards = [
        ("Logistic Regression", [
            "Pipeline：中位數補值 → 標準化 → LogisticRegression",
            "max_iter=1000，作為可解釋的線性基準模型",
        ]),
        ("Decision Tree", [
            "max_depth=5、min_samples_leaf=20",
            "刻意限制深度／葉節點樣本數，避免過擬合",
        ]),
        ("XGBoost", [
            "RandomizedSearchCV：25 組參數 × TimeSeriesSplit(3-fold)",
            "搜尋空間涵蓋 n_estimators／max_depth／learning_rate／"
            "subsample／colsample_bytree／reg_lambda",
        ]),
    ]
    col_w = (SLIDE_W_IN - 2 * MARGIN - 2 * 0.3) / 3
    for i, (name, bullets) in enumerate(cards):
        x = MARGIN + i * (col_w + 0.3)
        add_card(slide, x, y, col_w, 2.85)
        _, tf = new_textbox(slide, x + 0.28, y + 0.25, col_w - 0.56, 2.4)
        add_card_title(tf, name, size=15)
        add_bullets(tf, bullets, size=12)

    y2 = y + 3.1
    add_card(slide, MARGIN, y2, SLIDE_W_IN - 2 * MARGIN, 1.15)
    _, tf3 = new_textbox(slide, MARGIN + 0.3, y2, SLIDE_W_IN - 2 * MARGIN - 0.6, 1.15,
                          anchor=MSO_ANCHOR.MIDDLE)
    p = tf3.paragraphs[0]
    r1 = p.add_run()
    r1.text = "時序切分（防洩漏鐵律）　"
    style_run(r1, 14, ORANGE, bold=True)
    r2 = p.add_run()
    r2.text = (f"train = {facts['season_min']}–2023（{facts['n_train']:,} 場） ／ "
               f"test = 2024–{facts['season_max']}（{facts['n_test']:,} 場）"
               "　絕不隨機切分，XGBoost 調參亦用 TimeSeriesSplit 而非會打亂時序的 KFold")
    style_run(r2, 13, TEXT_PRIMARY)

    add_footer(slide, "建模與調校")


def build_eval_table(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "03 · 評估結果", "三模型 × 六指標總表")

    comp = facts["comparison"]
    best_name = comp["ROC-AUC"].idxmax()

    cols = ["模型", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "LogLoss"]
    y = 2.05
    x = MARGIN
    w = SLIDE_W_IN - 2 * MARGIN
    n_rows = len(comp) + 1
    row_h = 0.62
    table_h = row_h * n_rows

    gframe = slide.shapes.add_table(n_rows, len(cols), Inches(x), Inches(y), Inches(w), Inches(table_h))
    table = gframe.table
    col_ratio = [0.24] + [0.1267] * 6
    for i, ratio in enumerate(col_ratio):
        table.columns[i].width = Inches(w * ratio)
    for r in range(n_rows):
        table.rows[r].height = Inches(row_h)

    for c, name in enumerate(cols):
        fmt_cell(table.cell(0, c), name, 13, "0B1F3A", bold=True, fill_hex=ORANGE,
                  align=(PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER))

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "LogLoss"]
    for r_i, model_name in enumerate(comp.index, start=1):
        row = comp.loc[model_name]
        is_best = (model_name == best_name)
        # 最佳模型列用橘色低透明度暈染，與一般列的白色低透明度區隔開，
        # 讓「最佳模型」在深色底上有明顯可辨識的視覺重量（非只靠文字顏色/粗體）。
        fill_hex = ORANGE if is_best else CARD_FILL_HEX
        fill_alpha = 16 if is_best else 6
        name_color = ORANGE if is_best else TEXT_PRIMARY
        label = model_name + ("  ★ 最佳" if is_best else "")
        fmt_cell(table.cell(r_i, 0), label, 13, name_color, bold=is_best,
                  fill_hex=fill_hex, fill_alpha=fill_alpha, align=PP_ALIGN.LEFT)
        for c_i, mc in enumerate(metric_cols, start=1):
            val = row[mc]
            if mc in ("Accuracy", "Precision", "Recall", "F1"):
                text = f"{val*100:.2f}%"
            else:
                text = f"{val:.4f}"
            fmt_cell(table.cell(r_i, c_i), text, 13, TEXT_PRIMARY, bold=is_best,
                      fill_hex=fill_hex, fill_alpha=fill_alpha)

    _, tf_note = new_textbox(slide, MARGIN, y + table_h + 0.25, w, 0.4)
    r = tf_note.paragraphs[0].add_run()
    r.text = "測試集：2024–2025 賽季，2,450 場；依 ROC-AUC 選出最佳模型 → Logistic Regression"
    style_run(r, 12, TEXT_SECONDARY, italic=True)

    add_footer(slide, "評估結果總表")


def build_diag_1(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "03 · 診斷圖", "混淆矩陣與 ROC 曲線")

    y = 1.95
    box_h = 4.15
    box_w = 5.75
    add_card(slide, MARGIN, y, box_w, box_h)
    add_image_fit(slide, REPORTS_DIR / "confusion_matrix.png",
                   MARGIN + 0.15, y + 0.15, box_w - 0.3, box_h - 0.3)

    add_card(slide, MARGIN + box_w + 0.3, y, box_w, box_h)
    add_image_fit(slide, REPORTS_DIR / "roc_curve.png",
                   MARGIN + box_w + 0.3 + 0.15, y + 0.15, box_w - 0.3, box_h - 0.3)

    _, tf = new_textbox(slide, MARGIN, y + box_h + 0.15, SLIDE_W_IN - 2 * MARGIN, 0.35)
    r = tf.paragraphs[0].add_run()
    r.text = "測試集 2,450 場（2024–2025 賽季）· 最佳模型 Logistic Regression · ROC-AUC = 0.7292"
    style_run(r, 12, TEXT_SECONDARY, italic=True)

    add_footer(slide, "診斷圖：混淆矩陣 / ROC")


def build_diag_2(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "03 · 診斷圖", "特徵重要度與機率校準")

    y = 1.95
    box_h = 4.15
    left_w = 5.9
    right_w = 5.6
    add_card(slide, MARGIN, y, left_w, box_h)
    add_image_fit(slide, REPORTS_DIR / "feature_importance.png",
                   MARGIN + 0.15, y + 0.15, left_w - 0.3, box_h - 0.3)

    add_card(slide, MARGIN + left_w + 0.3, y, right_w, box_h)
    add_image_fit(slide, REPORTS_DIR / "calibration_curve.png",
                   MARGIN + left_w + 0.3 + 0.15, y + 0.15, right_w - 0.3, box_h - 0.3)

    _, tf = new_textbox(slide, MARGIN, y + box_h + 0.15, SLIDE_W_IN - 2 * MARGIN, 0.35)
    r = tf.paragraphs[0].add_run()
    r.text = "diff_elo／home_elo_pre／away_elo_pre 高居重要度前三 → 驗證 Elo 特徵的貢獻；" \
             "校準曲線貼近對角線，機率輸出可直接用於 Demo 呈現勝率"
    style_run(r, 12, TEXT_SECONDARY, italic=True)

    add_footer(slide, "診斷圖：特徵重要度 / 校準")


def build_engineering(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "04 · 工程品質", "pytest 12 條測試，抓到一個真 bug")

    y = 1.9
    add_card(slide, MARGIN, y, 3.9, 4.3)
    add_stat(slide, MARGIN + 0.3, y + 0.4, 3.3, "12 / 12", "pytest 全綠", value_size=38)
    _, tf = new_textbox(slide, MARGIN + 0.3, y + 1.75, 3.3, 2.3)
    add_bullets(tf, [
        "3 條防洩漏測試（Elo 初值／竄改結果不回流／合成比賽不污染既有資料）",
        "3 條 baseline 對照測試",
        "6 條推論冒煙測試",
    ], size=12, first_paragraph_is_title=False)

    add_card(slide, MARGIN + 4.1, y, 7.83, 4.3, alpha=CARD_FILL_ALPHA_STRONG)
    _, tf2 = new_textbox(slide, MARGIN + 4.1 + 0.35, y + 0.28, 7.1, 3.8)
    add_card_title(tf2, "防洩漏測試抓到的真 bug", color=ORANGE)
    add_bullets(tf2, [
        ("問題：", "add_head_to_head 用 groupby.apply 依配對分組計算，"
         "結果內部順序已依組別重排；後續 np.where 卻用「位置」對齊、"
         "沒有 reindex 回原始列序"),
        ("後果：", "h2h_home_winrate 欄位 96% 的值被錯套到別的比賽列上"),
        ("如何抓到：", "以 GAME 22000002 手算驗證，正確值應為 0.5，"
         "程式當時算出 0.75 —— systematic-debugging 確認為真 bug（不是測試寫錯）"),
        ("修法：", "一行 .reindex(g.index)，把依組重排的結果對齊回原始列序"),
    ], size=12.5)
    p = tf2.add_paragraph()
    p.space_before = Pt(10)
    r = p.add_run()
    r.text = "修正後：全量 pytest 12/12 通過"
    style_run(r, 13, ORANGE, bold=True)

    add_footer(slide, "工程品質")


def build_inference(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "04 · 推論設計", "合成比賽法：train / serve 共用同一條管線")

    steps = [
        ("1", "組合成列", "待預測比賽附加一列合成資料：GAME_ID=-1、PTS 缺值、標籤佔位"),
        ("2", "接到 games 尾端", "concat 到既有 games 表最後一列，不修改任何既有資料"),
        ("3", "呼叫 build_features", "與訓練時完全相同的特徵管線，所有滾動/Elo/h2h 皆已 shift(1)"),
        ("4", "取出該列特徵 → predict", "合成列的佔位值不會回頭影響自己或任何既有比賽的特徵"),
    ]
    y = 1.95
    col_w = (SLIDE_W_IN - 2 * MARGIN - 3 * 0.55) / 4
    for i, (num, title, desc) in enumerate(steps):
        x = MARGIN + i * (col_w + 0.55)
        add_card(slide, x, y, col_w, 2.9)
        _, tf_n = new_textbox(slide, x + 0.22, y + 0.2, col_w - 0.44, 0.55)
        r = tf_n.paragraphs[0].add_run()
        r.text = num
        style_run(r, 26, ORANGE, bold=True, font=FONT_NUM)
        _, tf_t = new_textbox(slide, x + 0.22, y + 0.78, col_w - 0.44, 0.5)
        r2 = tf_t.paragraphs[0].add_run()
        r2.text = title
        style_run(r2, 13.5, TEXT_PRIMARY, bold=True)
        _, tf_d = new_textbox(slide, x + 0.22, y + 1.32, col_w - 0.44, 1.45)
        r3 = tf_d.paragraphs[0].add_run()
        r3.text = desc
        style_run(r3, 11, TEXT_SECONDARY)
        if i < len(steps) - 1:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.CHEVRON, Inches(x + col_w + 0.06), Inches(y + 2.9 / 2 - 0.16),
                Inches(0.42), Inches(0.32),
            )
            set_fill_alpha(arrow, ORANGE, 55)
            arrow.line.fill.background()
            no_shadow(arrow)

    y2 = y + 3.15
    add_card(slide, MARGIN, y2, SLIDE_W_IN - 2 * MARGIN, 1.05)
    _, tf3 = new_textbox(slide, MARGIN + 0.3, y2, SLIDE_W_IN - 2 * MARGIN - 0.6, 1.05,
                          anchor=MSO_ANCHOR.MIDDLE)
    r = tf3.paragraphs[0].add_run()
    r.text = ("為什麼安全：features.py 所有特徵皆先 shift(1) 才計算，合成列是資料最後一場、"
              "之後沒有其他比賽會用到它 → 不會污染自己也不會污染任何既有列。"
              "train/serve 用同一份 build_features 程式碼，避免管線不一致。")
    style_run(r, 12.5, TEXT_PRIMARY)
    tf3.word_wrap = True

    add_footer(slide, "推論設計")


def build_demo_deploy(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "05 · Demo 與部署", "Streamlit Demo：選兩隊 → 2026-27 開季勝率")

    y = 1.9
    add_card(slide, MARGIN, y, 6.9, 4.3)
    _, tf = new_textbox(slide, MARGIN + 0.3, y + 0.3, 6.3, 3.7)
    add_card_title(tf, "Demo 互動流程")
    add_bullets(tf, [
        "使用者選定主隊／客隊",
        "輸出：主隊勝率（%）、勝負判定、關鍵特徵"
        "（Elo 期望勝率、Elo 差值、賽季勝率差、近況勝率差、對戰歷史等）",
        "預設不依實際賽程，模擬「2026-27 新賽季開季」：SEASON 取"
        "最新賽季+1，觸發跨賽季 Elo 回歸，反映開季重新洗牌的基準戰力",
        "頁籤附欄位定義與資料來源說明，方便非技術聽眾理解",
    ], size=13)

    add_card(slide, MARGIN + 7.2, y, 4.73, 4.3, alpha=CARD_FILL_ALPHA_STRONG)
    _, tf2 = new_textbox(slide, MARGIN + 7.2 + 0.3, y + 0.3, 4.15, 3.7)
    add_card_title(tf2, "交付與部署", color=ORANGE)
    add_bullets(tf2, [
        "程式碼（GitHub，public）：" + GITHUB_URL,
        "線上 Demo（Streamlit Community Cloud）：" + DEMO_URL,
        "requirements.txt（App 執行）／requirements-dev.txt（開發）分離",
        "預訓練模型 models/best_model.joblib 納入版控，"
        "Cloud 端免現場訓練、秒開",
    ], size=13)

    add_footer(slide, "Demo 與部署")


def build_future_conclusion(prs, facts) -> None:
    slide = new_slide(prs)
    add_title_block(slide, "05 · 結論", "未來方向與結論")

    stages = [
        ("STAGE 1", "每日自動更新重訓", "每日抓取新賽果，排程重跑特徵管線與訓練，\n讓 Elo／滾動狀態與模型持續反映最新戰況"),
        ("STAGE 2", "賽季中任一日期預測", "推廣合成比賽法，支援指定賽季中任一日期的\n預測，不侷限於開季基準"),
        ("STAGE 3", "傷兵／輪休／疲勞特徵", "納入傷兵名單、輪休(load management)、\n背靠背與連續客場的行程疲勞訊號"),
    ]
    y = 1.9
    col_w = (SLIDE_W_IN - 2 * MARGIN - 2 * 0.3) / 3
    for i, (tag, title, desc) in enumerate(stages):
        x = MARGIN + i * (col_w + 0.3)
        add_card(slide, x, y, col_w, 2.75)
        _, tf_g = new_textbox(slide, x + 0.25, y + 0.22, col_w - 0.5, 0.32)
        r = tf_g.paragraphs[0].add_run()
        r.text = tag
        style_run(r, 11, ORANGE, bold=True, font=FONT_NUM)
        _, tf_t = new_textbox(slide, x + 0.25, y + 0.58, col_w - 0.5, 0.55)
        r2 = tf_t.paragraphs[0].add_run()
        r2.text = title
        style_run(r2, 15, TEXT_PRIMARY, bold=True)
        _, tf_d = new_textbox(slide, x + 0.25, y + 1.15, col_w - 0.5, 1.45)
        r3 = tf_d.paragraphs[0].add_run()
        r3.text = desc
        style_run(r3, 11.5, TEXT_SECONDARY)

    y2 = y + 2.98
    comp = facts["comparison"]
    best_name = comp["ROC-AUC"].idxmax()
    best_acc = comp.loc[best_name, "Accuracy"] * 100
    best_auc = comp.loc[best_name, "ROC-AUC"]
    delta = best_acc - facts["baseline_test_acc"] * 100
    add_card(slide, MARGIN, y2, SLIDE_W_IN - 2 * MARGIN, 1.3, alpha=CARD_FILL_ALPHA_STRONG)
    _, tf3 = new_textbox(slide, MARGIN + 0.35, y2, SLIDE_W_IN - 2 * MARGIN - 0.7, 1.3,
                          anchor=MSO_ANCHOR.MIDDLE)
    p = tf3.paragraphs[0]
    r1 = p.add_run()
    r1.text = "結論　"
    style_run(r1, 14, ORANGE, bold=True)
    r2 = p.add_run()
    r2.text = (f"{best_name} 以 Accuracy {best_acc:.2f}%（優於 baseline {delta:+.2f} pp）、"
               f"ROC-AUC {best_auc:.4f} 勝出；43 個賽前特徵 + Elo + 嚴謹防洩漏設計"
               "（3 條測試、時序切分、shift(1)）、pytest 12/12 全綠，"
               "並以合成比賽法完成 train/serve 一致的推論與 Demo。")
    style_run(r2, 13, TEXT_PRIMARY)
    tf3.word_wrap = True

    add_footer(slide, "未來方向與結論")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("讀取 repo 既有事實（資料、特徵、評估報表），不憑空捏造數字 ...")
    facts = _load_facts()
    _ensure_diagnostic_plots(facts)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)

    builders = [
        build_cover,
        build_agenda,
        build_goal,
        build_baseline,
        build_data,
        build_field_defs,
        build_eda,
        build_features_overview,
        build_leakage,
        build_elo,
        build_feature_experiment,
        build_modeling,
        build_eval_table,
        build_diag_1,
        build_diag_2,
        build_engineering,
        build_inference,
        build_demo_deploy,
        build_demo_screens,
        build_proposal_vs_result,
        build_future_conclusion,
    ]
    for fn in builders:
        fn(prs, facts)

    prs.save(str(OUT_PATH))
    print(f"已產生 {OUT_PATH}（{len(builders)} 頁）")


if __name__ == "__main__":
    main()
