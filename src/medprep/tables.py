"""medprep.tables — 論文にそのまま載る Table 1 / Table 2 を作る。

**なぜ 2 枚に分けるか**

`table_one()` は「全体の記述」と「群間比較」を 1 枚に詰めていた。
読む側の目的が違うのに 1 枚だと、どちらも読みにくい。論文でも普通は分ける。

    Table 1 … 全症例の背景（N と分布だけ。★p 値は載せない★）
    Table 2 … 群間比較（全体 + 群ごと + p 値）

群分けの指定が無ければ **Table 1 だけ**を作る。

**書式**（gtsummary に合わせてある）

    連続変数    平均値 ± 標準偏差 [最小値, 最大値]
    離散変数    n (%)
    p 値        右端の 1 列だけ。★検定手法は表の中に書かず、脚注に回す★

検定手法を行ごとに書くと表が横に伸びて読めなくなる。しかし
**どの検定を使ったかを書かない表は査読に通らない**ので、脚注に必ず出す。

**日本語版と英語版を同じ数値から作る。**
人が訳し直すと、そこで誤訳と写し間違いが入る。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from . import i18n
from .clean import _norm, build_alias_map, load_dict
from .describe import compare_groups, group_levels
from .schema import BINARY, DATETIME, GROUP, HIGH_CARDINALITY, ID, NOMINAL, NUMERIC, ORDINAL
from .textfmt import frame_text
from .timing import detect_timing

CONTINUOUS, CATEGORICAL = "continuous", "categorical"

_LABEL_JA = {"characteristic": "特性", "overall": "全体", "p": "p値"}
_LABEL_EN = {"characteristic": "Characteristic", "overall": "Overall", "p": "p-value"}


# ============================================================== 数値の書き方
def _digits(v: pd.Series) -> int:
    """小数何桁で出すか。**元の値より細かくしない。**"""
    x = pd.to_numeric(v, errors="coerce").dropna()
    if x.empty:
        return 1
    if (x == x.round(0)).all():
        return 0
    m = float(x.abs().median())
    if m < 1:
        return 2
    return 1


def _num(x: float, d: int) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:,.{d}f}"


def _pct(k: int, n: int) -> str:
    """割合。**10% 未満は有効数字 2 桁にする。**0% と 0.4% を同じに見せない。"""
    if not n:
        return "—"
    p = 100.0 * k / n
    if p == 0 or p == 100:
        return f"{p:.0f}%"
    return f"{p:.1f}%" if p >= 10 else f"{p:.2g}%"


def _p_str(p: float) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "—"
    if p < 0.001:
        return "<0.001"
    if p > 0.999:
        return ">0.999"
    return f"{p:.3f}"


def _cont_cell(x: pd.Series, d: int) -> str:
    """★平均値 ± 標準偏差 [最小値, 最大値]★"""
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return "—"
    return (f"{_num(float(v.mean()), d)} ± {_num(float(v.std(ddof=1)), d)} "
            f"[{_num(float(v.min()), d)}, {_num(float(v.max()), d)}]")


def _cat_cell(x: pd.Series, level) -> str:
    v = x.dropna()
    return f"{int((v == level).sum())} ({_pct(int((v == level).sum()), len(v))})"


# ============================================================== 列の見出し
class _Labeller:
    """列名 → 表示名（日本語・英語）。単位と採血時点を添える。"""

    def __init__(self, dic: dict | None = None):
        self.dic = dic or load_dict()
        self.amap = build_alias_map(self.dic)

    def key(self, col: str):
        t, base = detect_timing(col)
        return self.amap.get(_norm(base)) or self.amap.get(_norm(col)), t

    def unit(self, col: str) -> str:
        k, _ = self.key(col)
        spec = self.dic["items"].get(k) if k else None
        u = (spec or {}).get("unit") or ""
        return "" if u in ("index", "") else str(u)

    def ja(self, col: str) -> str:
        u = self.unit(col)
        return f"{col} ({u})" if u and f"({u})" not in col and u not in col else str(col)

    def en(self, col: str) -> str:
        k, t = self.key(col)
        name = i18n.column_en(col, k, t)
        u = i18n.unit_en(self.unit(col))
        # ★英語の表に日本語の単位（歳・月・時間）を残さない。★
        return f"{name} ({u})" if u and f"({u})" not in name else name


# ============================================================== 表そのもの
@dataclass
class GTTable:
    """1 枚の表。日本語版と英語版を同じ数値から持つ。"""
    kind: str = "table1"
    number: int = 1
    title_ja: str = ""
    title_en: str = ""
    frame_ja: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame_en: pd.DataFrame = field(default_factory=pd.DataFrame)
    footnote_ja: str = ""
    footnote_en: str = ""
    notes: list = field(default_factory=list)

    def frame(self, lang: str = "ja") -> pd.DataFrame:
        return self.frame_ja if lang == "ja" else self.frame_en

    def title(self, lang: str = "ja") -> str:
        return self.title_ja if lang == "ja" else self.title_en

    def footnote(self, lang: str = "ja") -> str:
        return self.footnote_ja if lang == "ja" else self.footnote_en

    def text(self, lang: str = "ja") -> str:
        return "\n".join([self.title(lang),
                          frame_text(self.frame(lang)),
                          self.footnote(lang)])

    def to_html(self, lang: str = "ja") -> str:
        """**Word にそのまま貼れる HTML。** 配置は中詰め。"""
        f = self.frame(lang)
        head = "".join(f"<th style='{_TH}'>{_esc(c)}</th>" for c in f.columns)
        body = "".join(
            "<tr>" + "".join(f"<td style='{_TD}'>{_esc(v)}</td>" for v in row) + "</tr>"
            for row in f.astype(str).itertuples(index=False, name=None))
        return (f"<div style='font-family:Yu Gothic,Meiryo,sans-serif;font-size:11pt'>"
                f"<p style='{_CAP}'>{_esc(self.title(lang))}</p>"
                f"<table style='{_TABLE}'><thead><tr>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table>"
                f"<p style='{_FN}'>{_esc(self.footnote(lang))}</p></div>")


_TABLE = "border-collapse:collapse;margin:0 auto;"
_TH = ("border-top:1.5pt solid #000;border-bottom:1pt solid #000;"
       "padding:4px 10px;text-align:center;font-weight:bold;")
_TD = "border:none;padding:3px 10px;text-align:center;"
_CAP = "text-align:center;font-weight:bold;margin:6px 0;"
_FN = "text-align:left;font-size:9pt;margin:4px 0 14px 0;"


def _esc(v) -> str:
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


@dataclass
class GTSummary:
    """Table 1（と、群分けがあれば Table 2）。書き出しもここが持つ。"""
    table1: GTTable | None = None
    table2: GTTable | None = None
    group: str | None = None
    notes: list = field(default_factory=list)

    def tables(self) -> list:
        return [t for t in (self.table1, self.table2) if t is not None]

    def report(self, lang: str = "ja") -> str:
        return "\n\n".join(t.text(lang) for t in self.tables())

    def to_html(self, lang: str = "ja") -> str:
        return "\n".join(t.to_html(lang) for t in self.tables())

    # ---------------------------------------------------------- Excel
    def to_excel(self, path) -> str:
        """★sheet1 = 日本語版、sheet2 = 英語版。★ 配置は中詰め。"""
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, Side

        wb = Workbook()
        thick, thin = Side(style="medium"), Side(style="thin")
        centre = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for i, (lang, sheet) in enumerate((("ja", "日本語"), ("en", "English"))):
            ws = wb.active if i == 0 else wb.create_sheet()
            ws.title = sheet
            r = 1
            for t in self.tables():
                f = t.frame(lang)
                ws.cell(r, 1, t.title(lang)).font = Font(bold=True, size=11)
                ws.cell(r, 1).alignment = centre
                ws.merge_cells(start_row=r, start_column=1,
                               end_row=r, end_column=max(1, f.shape[1]))
                r += 1
                head = r
                for j, c in enumerate(f.columns, start=1):
                    cell = ws.cell(r, j, str(c))
                    cell.font = Font(bold=True)
                    cell.alignment = centre
                    cell.border = Border(top=thick, bottom=thin)
                r += 1
                for row in f.astype(str).itertuples(index=False, name=None):
                    for j, v in enumerate(row, start=1):
                        cell = ws.cell(r, j, v)
                        cell.alignment = centre
                    r += 1
                for j in range(1, f.shape[1] + 1):
                    ws.cell(r - 1, j).border = Border(bottom=thick)
                ws.cell(r, 1, t.footnote(lang)).alignment = Alignment(
                    horizontal="left", vertical="top", wrap_text=True)
                ws.merge_cells(start_row=r, start_column=1,
                               end_row=r, end_column=max(1, f.shape[1]))
                r += 3
                ws.column_dimensions["A"].width = 46
                for j in range(2, f.shape[1] + 1):
                    ws.column_dimensions[chr(64 + j)].width = 24
                ws.row_dimensions[head].height = 30
        wb.save(str(path))
        return str(path)

    # ---------------------------------------------------------- Word
    def to_docx(self, path, *, landscape_from: int = 4) -> str:
        """★日本語版と英語版をページを分けて載せる。★ 表は中詰め。

        群が増えると Table 2 は横に伸びる。縦置きのままだとセルの中で
        「57.3 ± 10.6 [30.0, 88.3]」が 3 行に折り返して読めなくなるので、
        列数が `landscape_from` を超える表だけ**横置きのページ**に置く。
        """
        from docx import Document
        from docx.enum.section import WD_ORIENT, WD_SECTION
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt

        doc = Document()
        first = True
        for lang in ("ja", "en"):
            for t in self.tables():
                f = t.frame(lang)
                wide = f.shape[1] > landscape_from
                sec = doc.sections[0] if first else doc.add_section(WD_SECTION.NEW_PAGE)
                first = False
                _orient(sec, WD_ORIENT.LANDSCAPE if wide else WD_ORIENT.PORTRAIT)
                size = Pt(8.5) if wide else Pt(10)

                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(t.title(lang))
                run.bold = True
                run.font.size = Pt(11)

                tbl = doc.add_table(rows=1, cols=f.shape[1])
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                tbl.autofit = False
                avail = (sec.page_width - sec.left_margin - sec.right_margin)
                w0 = int(avail * (0.30 if wide else 0.52))
                rest = int((avail - w0) / max(1, f.shape[1] - 1))
                widths = [w0] + [rest] * (f.shape[1] - 1)

                for j, c in enumerate(f.columns):
                    cell = tbl.rows[0].cells[j]
                    cell.width = widths[j]
                    p_ = cell.paragraphs[0]
                    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p_.add_run(str(c))
                    r.bold = True
                    r.font.size = size
                for row in f.astype(str).itertuples(index=False, name=None):
                    cells = tbl.add_row().cells
                    for j, v in enumerate(row):
                        cells[j].width = widths[j]
                        p_ = cells[j].paragraphs[0]
                        p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        r = p_.add_run(v)
                        r.font.size = size
                _rule_only(tbl)
                _no_wrap_numbers(tbl)
                _repeat_header(tbl)

                fn = doc.add_paragraph()
                fn.alignment = WD_ALIGN_PARAGRAPH.LEFT
                r = fn.add_run(t.footnote(lang))
                r.font.size = Pt(8.5)
        doc.save(str(path))
        return str(path)


def _orient(section, orient) -> None:
    """節の向きを変える。★width と height も入れ替える（向きだけでは変わらない）★"""
    from docx.enum.section import WD_ORIENT
    from docx.shared import Cm

    w, h = section.page_width, section.page_height
    section.orientation = orient
    if ((orient == WD_ORIENT.LANDSCAPE and w < h)
            or (orient == WD_ORIENT.PORTRAIT and w > h)):
        section.page_width, section.page_height = h, w
    section.left_margin = section.right_margin = Cm(1.6)


def _repeat_header(tbl) -> None:
    """見出し行をページごとに繰り返す。長い表が次ページで見出しを失わないように。"""
    from docx.oxml.ns import qn
    from docx.oxml.parser import OxmlElement

    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _no_wrap_numbers(tbl) -> None:
    """数値セルの中で折り返させない。`57.3 ± 10.6 [30.0, 88.3]` は 1 行で読む。"""
    for row in tbl.rows:
        for cell in list(row.cells)[1:]:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.text = r.text.replace(" ", "\u00a0")


def _rule_only(tbl) -> None:
    """罫線を **上端・見出し下・下端の 3 本だけ**にする（学術誌の体裁）。"""
    from docx.oxml.ns import qn
    from docx.oxml.parser import OxmlElement

    n_rows = len(tbl.rows)
    for i, row in enumerate(tbl.rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            borders = OxmlElement("w:tcBorders")
            for edge, size in (("top", 12 if i == 0 else 0),
                               ("bottom", 6 if i == 0 else (12 if i == n_rows - 1 else 0)),
                               ("left", 0), ("right", 0)):
                e = OxmlElement(f"w:{edge}")
                e.set(qn("w:val"), "single" if size else "nil")
                if size:
                    e.set(qn("w:sz"), str(size))
                    e.set(qn("w:color"), "000000")
                borders.append(e)
            tcPr.append(borders)


# ============================================================== 組み立て
_SKIP_EXACT = ("除外推奨", "除外推奨_理由")
#: 掃除の過程で付けた内部フラグ。**人が読む表には出さない。**
_SKIP_PATTERNS = ("__censored_low", "__censored_high", "__unit_converted")
_SKIP_PREFIX = ("欠損あり_",)


def _is_internal(col: str) -> bool:
    c = str(col)
    return (c in _SKIP_EXACT or c.startswith(_SKIP_PREFIX)
            or any(pat in c for pat in _SKIP_PATTERNS))


def _ordered_levels(s: pd.Series, schema, col: str) -> list:
    """水準の並び。**「1 に相当する水準」を先に置く。**

    `性別 — 女 / 男` では、どちらが主語か分からない。schema が二値の対応表
    （`value_map`）を持っているときは 1 側を先にする。
    """
    lv = group_levels(s)
    spec = getattr(schema, "columns", {}).get(col) if schema is not None else None
    vmap = getattr(spec, "value_map", None)
    if vmap and len(lv) == 2:
        norm = {_norm(k): v for k, v in vmap.items()}
        pos = [x for x in lv if norm.get(_norm(x)) == 1]
        neg = [x for x in lv if norm.get(_norm(x)) == 0]
        if len(pos) == 1 and len(neg) == 1:
            return pos + neg
    return lv


def _kind_of(s: pd.Series, role: str | None, max_levels: int) -> str:
    if role in (NOMINAL, GROUP, BINARY, ORDINAL):
        return CATEGORICAL
    if role == NUMERIC:
        return CONTINUOUS
    if pd.api.types.is_numeric_dtype(s) and s.dropna().nunique() > max_levels:
        return CONTINUOUS
    return CATEGORICAL


def _pick_columns(df, schema, group, max_levels) -> list:
    if schema is not None:
        cols = [c for c in schema.kept() if c in df.columns]
        cols = [c for c in cols
                if schema.columns[c].role not in (ID, DATETIME, HIGH_CARDINALITY)]
    else:
        cols = [c for c in df.columns
                if not pd.api.types.is_datetime64_any_dtype(df[c])]
    return [c for c in cols if c != group and not _is_internal(c)]


def gt_tables(
    df: pd.DataFrame,
    schema=None,
    *,
    group: str | None = None,
    columns: list | None = None,
    dic: dict | None = None,
    max_levels: int = 10,
    alpha: float = 0.05,
) -> GTSummary:
    """Table 1（全症例の背景）と、`group` があれば Table 2（群間比較）を作る。

    `group` が None なら **Table 1 だけ**。
    検定は `compare_groups()` が選ぶ（正規性・群数・期待度数で自動）。
    使われた検定の名前は**表ではなく脚注**に出る。
    """
    lab = _Labeller(dic)
    cols = columns or _pick_columns(df, schema, group, max_levels)
    out = GTSummary(group=group)

    # ---------------------------------------------------------- Table 1
    n = len(df)
    rows_ja, rows_en = [], []
    missing = []
    for c in cols:
        s = df[c]
        role = schema.columns[c].role if (schema and c in schema.columns) else None
        kind = _kind_of(s, role, max_levels)
        nmiss = int(s.isna().sum())
        if nmiss:
            missing.append((lab.ja(c), lab.en(c), nmiss))
        if kind == CONTINUOUS:
            d = _digits(s)
            rows_ja.append([lab.ja(c), _cont_cell(s, d)])
            rows_en.append([lab.en(c), _cont_cell(s, d)])
            continue
        levels = _ordered_levels(s, schema, c)
        if not levels:
            continue
        if len(levels) > max_levels:
            rows_ja.append([lab.ja(c), f"水準が {len(levels)} 個（多すぎるため省略）"])
            rows_en.append([lab.en(c), f"{len(levels)} levels (omitted)"])
            continue
        cells = [f"{_cat_cell(s, lv)}" for lv in levels]
        if len(levels) == 2:
            rows_ja.append([f"{lab.ja(c)} — {levels[0]} / {levels[1]}",
                            " / ".join(cells)])
            rows_en.append([f"{lab.en(c)} — {i18n.level_en(levels[0])} / "
                            f"{i18n.level_en(levels[1])}", " / ".join(cells)])
        else:
            rows_ja.append([lab.ja(c),
                            "; ".join(f"{lv} {x}" for lv, x in zip(levels, cells))])
            rows_en.append([lab.en(c), "; ".join(
                f"{i18n.level_en(lv)} {x}" for lv, x in zip(levels, cells))])

    h_ja = [_LABEL_JA["characteristic"], f"全体 (N = {n}) ¹"]
    h_en = [_LABEL_EN["characteristic"], f"Overall (N = {n}) ¹"]
    t1 = GTTable(
        kind="table1", number=1,
        title_ja=f"表1. 全症例の背景（N = {n}）",
        title_en=f"Table 1. Baseline characteristics of all patients (N = {n}).",
        frame_ja=pd.DataFrame(rows_ja, columns=h_ja),
        frame_en=pd.DataFrame(rows_en, columns=h_en),
        footnote_ja="¹ 平均値 ± 標準偏差 [最小値, 最大値]；n (%)。"
                    + _missing_note(missing, "ja"),
        footnote_en="¹ Mean ± SD [min, max]; n (%)." + _missing_note(missing, "en"),
    )
    out.table1 = t1
    if not group or group not in df.columns:
        if group:
            out.notes.append(f"群分け列 '{group}' がデータに無いため Table 2 は作らなかった")
        return out

    # ---------------------------------------------------------- Table 2
    g = df[group]
    levels = group_levels(g)
    if len(levels) < 2:
        out.notes.append(f"'{group}' の水準が {len(levels)} 個しかないため Table 2 は作らなかった")
        return out

    cmp_cols = [c for c in cols if c != group]
    res = compare_groups(df, group, cmp_cols, schema=schema, dic=dic, posthoc=False)
    pmap = dict(zip(cmp_cols, res.comparisons))
    sizes = {lv: int((g == lv).sum()) for lv in levels}

    rows_ja, rows_en, tests = [], [], []
    for c in cmp_cols:
        s, cm = df[c], pmap.get(c)
        role = schema.columns[c].role if (schema and c in schema.columns) else None
        kind = _kind_of(s, role, max_levels)
        pv = _p_str(getattr(cm, "p", float("nan")))
        if cm is not None and getattr(cm, "test", ""):
            tests.append(cm.test)
        if kind == CONTINUOUS:
            d = _digits(s)
            rows_ja.append([lab.ja(c), _cont_cell(s, d)]
                           + [_cont_cell(s[g == lv], d) for lv in levels] + [pv])
            rows_en.append([lab.en(c), _cont_cell(s, d)]
                           + [_cont_cell(s[g == lv], d) for lv in levels] + [pv])
            continue
        lv_all = _ordered_levels(s, schema, c)
        if not lv_all or len(lv_all) > max_levels:
            continue
        if len(lv_all) == 2:
            pos = lv_all[0]        # _ordered_levels が 1 側を先に置いている
            rows_ja.append([f"{lab.ja(c)} — {pos}", _cat_cell(s, pos)]
                           + [_cat_cell(s[g == lv], pos) for lv in levels] + [pv])
            rows_en.append([f"{lab.en(c)} — {i18n.level_en(pos)}", _cat_cell(s, pos)]
                           + [_cat_cell(s[g == lv], pos) for lv in levels] + [pv])
        else:
            blank = ["—"] * (len(levels) + 1)
            rows_ja.append([lab.ja(c), *blank, pv])
            rows_en.append([lab.en(c), *blank, pv])
            for x in lv_all:
                rows_ja.append([f"　{x}", _cat_cell(s, x)]
                               + [_cat_cell(s[g == lv], x) for lv in levels] + [""])
                rows_en.append([f"　{i18n.level_en(x)}", _cat_cell(s, x)]
                               + [_cat_cell(s[g == lv], x) for lv in levels] + [""])

    h_ja = ([_LABEL_JA["characteristic"], f"全体 (N = {n}) ¹"]
            + [f"{lv} (N = {sizes[lv]})" for lv in levels] + [f"{_LABEL_JA['p']} ²"])
    h_en = ([_LABEL_EN["characteristic"], f"Overall (N = {n}) ¹"]
            + [f"{i18n.level_en(lv)} (N = {sizes[lv]})" for lv in levels]
            + [f"{_LABEL_EN['p']} ²"])
    uniq = list(dict.fromkeys(tests))
    t2 = GTTable(
        kind="table2", number=2,
        title_ja=f"表2. {group} 別の背景の比較（N = {n}）",
        title_en=(f"Table 2. Comparison of characteristics by "
                  f"{i18n.column_en(group, *lab.key(group))} (N = {n})."),
        frame_ja=pd.DataFrame(rows_ja, columns=h_ja),
        frame_en=pd.DataFrame(rows_en, columns=h_en),
        footnote_ja=("¹ 平均値 ± 標準偏差 [最小値, 最大値]；n (%)。 ² "
                     + "；".join(uniq) + "。"
                     + f" p < {alpha} を有意とした。"
                     + "★多項目の p 値を同時に見ているので、個々の p だけで有意と述べない★"),
        footnote_en=("¹ Mean ± SD [min, max]; n (%). ² "
                     + "; ".join(i18n.TEST_EN.get(t, t) for t in uniq) + "."
                     + f" Significance level {alpha}."),
        notes=list(res.notes),
    )
    out.table2 = t2
    return out


def _missing_note(missing, lang) -> str:
    if not missing:
        return ""
    top = sorted(missing, key=lambda x: -x[2])[:8]
    if lang == "ja":
        body = "、".join(f"{ja} {k}" for ja, _en, k in top)
        return f" 欠測あり: {body}" + ("、ほか" if len(missing) > len(top) else "") + "。"
    body = "; ".join(f"{en} {k}" for _ja, en, k in top)
    return f" Missing: {body}" + ("; others" if len(missing) > len(top) else "") + "."
