"""medprep.viz — 図。**1 つの色体系で、目で見て読める形に出す。**

色の決め方
----------
色は最後に決める。まず「そのデータの仕事は何か」で図の形を決め、
仕事に応じて色の役割を割り当てる。

    identity（どれがどれか）  → カテゴリ配色。**決まった順に使い、循環させない**
    magnitude（大きさ）       → 単一色相の明→暗（順次配色）
    polarity（符号のある量）  → 2 色相 ＋ **灰色の中点**（発散配色）
    state（良し悪し）         → 状態色（凡例ではなくラベルと併記する）

相関係数は符号を持つので**発散配色**、η² や Cramér's V は 0〜1 の大きさなので
**順次配色**。ここを取り違えて虹色を当てると、大きさの順序が色から読めなくなる。

配色は検証済みである（色覚特性のある読者で隣り合う色が区別できること、
明度帯、彩度の下限、背景とのコントラスト）。**目分量で決めていない。**
`PALETTE` の値を変える場合は、同じ検証を通してから変えること。

散布図など**すべての対が同時に見える図では 3 色まで**にする。
4 色目で黄と橙が並び、色覚特性のある読者には区別できなくなる。
それ以上は「その他」にまとめるか、図を分ける。

コントラストが 3:1 に満たない色（aqua・yellow・magenta）を使うときは、
**凡例だけに頼らず、直接ラベルか表を添える**。`report.py` は図の隣に必ず表を置く。

この図は静止画である
--------------------
matplotlib の PNG を HTML に埋めるので、ホバーも切り替えも無い。
その代わりに **図と同じ内容の表を必ず併記する**（レポート側の責務）。
配色は明るい背景の 1 種類だけを用意する（印刷と添付を前提にしているため）。
"""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .describe import eta_squared
from .schema import BINARY, GROUP, NOMINAL, NUMERIC, ORDINAL, Schema, cramers_v


# ================================================================== 配色
@dataclass(frozen=True)
class Palette:
    """検証済みの配色。**順番に意味がある。**"""

    # identity — 決まった順に使う。循環させない（9 色目を作らない）
    categorical: tuple = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                          "#e87ba4", "#008300", "#4a3aa7", "#e34948")
    # すべての対が同時に見える図（散布図・PCA）で使ってよい数
    scatter_max: int = 3
    # magnitude — 単一色相の明→暗
    sequential: tuple = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                         "#256abf", "#184f95", "#0d366b")
    # polarity — 暖色と寒色の 2 極、中点は灰色（虹色にしない）
    diverging_low: str = "#2a78d6"
    diverging_mid: str = "#f0efec"
    diverging_high: str = "#d03b3b"
    # state — 系列色には使わない
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"
    # 図の地と線
    surface: str = "#fcfcfb"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"

    def series(self, i: int) -> str:
        """i 番目の系列の色。**循環させず、8 を超えたら呼び出し側がまとめる。**"""
        if i >= len(self.categorical):
            raise ValueError(
                f"系列が {i + 1} 本目になった。カテゴリ配色は 8 色までで、"
                f"9 色目を作ると色覚特性のある読者には区別できない。"
                f"『その他』にまとめるか、図を分けること")
        return self.categorical[i]

    def colors_for(self, levels, *, all_pairs: bool = False) -> dict:
        """水準 → 色。`all_pairs=True`（散布図など）では 3 色までに抑える。"""
        cap = self.scatter_max if all_pairs else len(self.categorical)
        out = {}
        for i, lv in enumerate(levels):
            out[lv] = self.categorical[i] if i < cap else self.muted
        return out


PALETTE = Palette()


def sequential_cmap(name: str = "medprep_seq"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(PALETTE.sequential))


def diverging_cmap(name: str = "medprep_div"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        name, [PALETTE.diverging_low, PALETTE.diverging_mid, PALETTE.diverging_high])


# ================================================================== 下ごしらえ
def _fontja():
    with contextlib.suppress(ImportError):
        import matplotlib_fontja  # noqa: F401


def _new(figsize=(8, 5), ax=None):
    _fontja()
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE.surface)
    return fig, ax


def _style(ax, *, grid_axis="y"):
    """軸と格子を控えめにする。データより目立たせない。"""
    ax.set_facecolor(PALETTE.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE.axis)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=PALETTE.muted, labelsize=9, length=3)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(PALETTE.ink_secondary)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE.grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(PALETTE.ink_secondary)
    ax.yaxis.label.set_color(PALETTE.ink_secondary)
    ax.title.set_color(PALETTE.ink)
    return ax


# 日本語フォント（IPAexGothic）に無い文字。**そのまま描くと豆腐（□）になる。**
#   例外は出ない。気づけない壊れ方なので、描く前に置き換える。
#   µ は MICRO SIGN(U+00B5)、μ は GREEK SMALL MU(U+03BC)。辞書の単位に前者が混じる。
GLYPH_FIX = {"≥": "≧", "≤": "≦", "✗": "×", "✕": "×", "µ": "μ", "≈": "≒"}


def safe_text(s: str) -> str:
    """フォントに無い文字を、同じ意味の持っている文字に置き換える。"""
    out = str(s)
    for a, b in GLYPH_FIX.items():
        out = out.replace(a, b)
    return out


def _fix_glyphs(fig):
    """図の中のすべての文字を安全な文字に直す。

    各関数の末尾（`_save`）で一度だけ通す。呼び出し側が書き忘れても効く。
    """
    from matplotlib.text import Text
    for obj in fig.findobj(Text):
        s = obj.get_text()
        if s:
            fixed = safe_text(s)
            if fixed != s:
                obj.set_text(fixed)
    return fig


def missing_glyphs(fig) -> set:
    """図の中で、いま使っているフォントに無い文字を返す。**テスト用。**"""
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.text import Text
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return set()
    import matplotlib.pyplot as plt
    f = TTFont(findfont(FontProperties(family=plt.rcParams["font.family"])), fontNumber=0)
    have = set()
    for tbl in f["cmap"].tables:
        have |= set(tbl.cmap.keys())
    used = set()
    for obj in fig.findobj(Text):
        used |= set(obj.get_text() or "")
    return {c for c in used if c.strip() and ord(c) not in have}


def _save(fig, save):
    _fix_glyphs(fig)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=PALETTE.surface)
    return fig


def _numeric_columns(df, schema, columns) -> list:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    if schema is not None:
        return [c for c in schema.kept()
                if c in df.columns and schema.columns[c].role == NUMERIC]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_columns(df, schema, columns) -> list:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    if schema is not None:
        return [c for c in schema.kept() if c in df.columns
                and schema.columns[c].role in (BINARY, NOMINAL, ORDINAL, GROUP)]
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) <= 20]


def _fold_levels(s: pd.Series, cap: int):
    """水準が多すぎるとき、上位 cap-1 を残して残りを『その他』にまとめる。

    ★9 色目を作らない。★ 循環させた色は、色覚特性のある読者には区別できない。
    """
    levels = list(s.value_counts().index)
    if len(levels) <= cap:
        return s.astype(str), [str(x) for x in levels], False
    keep = [str(x) for x in levels[:cap - 1]]
    out = s.astype(str).where(s.astype(str).isin(keep), "その他")
    return out, [*keep, "その他"], True


# ================================================================== 欠損
def fit_tick_labels(ax, *, axis: str = "x", fig_width_in: float | None = None) -> float:
    """★目盛りラベルが重ならない字の大きさと向きを決める。★

    列が 40 本あって名前が「β2マイクログロブリン(β2MG)」のように長いと、
    既定の大きさ・45 度では**文字どうしが重なって 1 文字も読めなくなる**。
    図としては描けているので例外は出ない。人が見て初めて分かる壊れ方である。

    縦書き（90 度）にすると、横に要る幅は**字 1 つぶん**で済む。
    1 列あたりの幅（インチ）から、その幅に収まる大きさを計算して当てる。
    日本語は全角なので、字の幅は大きさ（pt）とほぼ同じとみなす。
    """
    fig = ax.get_figure()
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    n = len(labels)
    if not n:
        return 0.0
    width_in = fig_width_in if fig_width_in else fig.get_size_inches()[0]
    slot_pt = (width_in / n) * 72.0                 # 1 列あたりの幅（pt）
    size = float(min(13.0, max(4.5, slot_pt * 0.85)))
    longest = max((len(t.get_text()) for t in labels), default=0)
    rot = 90 if (n > 10 or longest > 8) else 45
    for t in labels:
        t.set_rotation(rot)
        t.set_fontsize(size)
        t.set_ha("center" if rot == 90 else "left")
        t.set_va("bottom")
    return size


def missing_map(df: pd.DataFrame, schema: Schema | None = None, *,
                columns: list | None = None, kind: str = "matrix",
                save=None, figsize=None):
    """欠損の地図。どの症例でどの列が同時に欠けるかを見る。

    ★図の幅と字の大きさは列数から決める。★ 既定のまま 40 列を描くと
    列名が重なって読めない（例外は出ないので、人が見るまで気づかない）。
    """
    _fontja()
    import matplotlib.pyplot as plt
    import missingno as msno
    cols = columns or ([c for c in schema.kept() if c in df.columns] if schema
                       else list(df.columns))
    fn = {"matrix": msno.matrix, "bar": msno.bar,
          "heatmap": msno.heatmap, "dendrogram": msno.dendrogram}.get(kind)
    if fn is None:
        raise ValueError("kind は matrix / bar / heatmap / dendrogram のいずれか")
    if figsize is None:
        n = max(1, len(cols))
        longest = max((len(str(c)) for c in cols), default=8)
        # 1 列あたり 0.34 インチ。名前が長いほど上の余白も要る。
        figsize = (min(26.0, max(10.0, 0.34 * n)),
                   min(11.0, 5.0 + 0.11 * min(longest, 24)))
    kw = {"color": _hex_to_rgb(PALETTE.categorical[0])} if kind in ("matrix", "bar") else {}
    ax = fn(df[cols], figsize=figsize, **kw)
    fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
    fig.patch.set_facecolor(PALETTE.surface)
    if kind in ("matrix", "bar"):
        fit_tick_labels(ax, fig_width_in=figsize[0])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fig.tight_layout()
    return _save(fig, save)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ================================================================== 分布
def distributions(df: pd.DataFrame, schema: Schema | None = None, *,
                  columns: list | None = None, by: str | None = None,
                  ncols: int = 3, max_columns: int = 12,
                  save=None, figsize=None):
    """連続変数の分布。ヒストグラムと箱ひげを縦に並べる。

    `by` を渡すと群別に重ねる。**水準は 8 つまで**で、それを超えたら
    上位をまとめる（色を循環させない）。
    """
    cols = _numeric_columns(df, schema, columns)[:max_columns]
    if not cols:
        raise ValueError("連続変数が無い")
    _fontja()
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(cols) / ncols)
    figsize = figsize or (4.2 * ncols, 3.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor(PALETTE.surface)

    if by and by in df.columns:
        g, levels, folded = _fold_levels(df[by], len(PALETTE.categorical))
        cmap = PALETTE.colors_for(levels)
    else:
        g, levels, folded, cmap = None, [], False, {}

    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        v = pd.to_numeric(df[c], errors="coerce")
        if g is None:
            ax.hist(v.dropna(), bins=30, color=PALETTE.categorical[0],
                    edgecolor=PALETTE.surface, linewidth=0.6)
        else:
            for lv in levels:
                sub = v[g == lv].dropna()
                if len(sub) < 2:
                    continue
                ax.hist(sub, bins=25, color=cmap[lv], alpha=0.55,
                        label=f"{lv} (n={len(sub)})", edgecolor=PALETTE.surface,
                        linewidth=0.5)
        med = v.median()
        if pd.notna(med):
            ax.axvline(med, color=PALETTE.ink_secondary, lw=1.2, ls="--")
            ax.annotate(f"中央値 {med:.3g}", xy=(med, ax.get_ylim()[1]),
                        xytext=(3, -10), textcoords="offset points",
                        fontsize=8, color=PALETTE.ink_secondary)
        ax.set_title(c, fontsize=10)
        ax.set_ylabel("症例数")
        _style(ax)
        if g is not None and i == 0:
            ax.legend(fontsize=8, frameon=False)

    for j in range(len(cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if folded:
        fig.text(0.01, 0.005, f"※ '{by}' の水準が多いため、上位以外は『その他』にまとめた",
                 fontsize=8, color=PALETTE.muted)
    fig.tight_layout()
    return _save(fig, save)


def category_bars(df: pd.DataFrame, schema: Schema | None = None, *,
                  columns: list | None = None, by: str | None = None,
                  ncols: int = 3, max_columns: int = 9, save=None, figsize=None):
    """カテゴリ変数の内訳。水準ごとの症例数を横棒で出す。"""
    cols = _categorical_columns(df, schema, columns)
    cols = [c for c in cols if c != by][:max_columns]
    if not cols:
        raise ValueError("カテゴリ変数が無い")
    _fontja()
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(cols) / ncols)
    figsize = figsize or (4.4 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor(PALETTE.surface)
    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        vc = df[c].astype(str).value_counts().head(12).iloc[::-1]
        # ★名義カテゴリに値の濃淡を当てない。★ 1 系列は 1 色。
        ax.barh(range(len(vc)), vc.to_numpy(), color=PALETTE.categorical[0], height=0.68)
        ax.set_yticks(range(len(vc)))
        ax.set_yticklabels(vc.index, fontsize=9)
        for y, v in enumerate(vc.to_numpy()):
            ax.annotate(f"{v}（{v / max(len(df), 1):.0%}）", xy=(v, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=PALETTE.ink_secondary)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("症例数")
        ax.set_xlim(0, float(vc.max()) * 1.35)
        _style(ax, grid_axis="x")
    for j in range(len(cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== 相関・関連
def correlation_heatmap(df: pd.DataFrame, schema: Schema | None = None, *,
                        columns: list | None = None, method: str = "spearman",
                        cluster: bool = True, annotate: bool | None = None,
                        save=None, figsize=None):
    """連続変数どうしの相関。**符号があるので発散配色、中点は灰色。**

    既定は Spearman。医学データは歪んだ分布が多く、Pearson は外れ値に弱い。
    """
    cols = _numeric_columns(df, schema, columns)
    cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    if len(cols) < 2:
        raise ValueError("相関を出すには連続変数が 2 つ以上必要")
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)
    if cluster and len(cols) > 2:
        corr = corr.loc[_cluster_order(corr), _cluster_order(corr)]
    return _matrix_plot(corr, title=f"相関（{method}、クラスタ順）",
                        cmap=diverging_cmap(), vmin=-1, vmax=1,
                        annotate=annotate, save=save, figsize=figsize,
                        cbar_label="相関係数")


def association_heatmap(df: pd.DataFrame, schema: Schema | None = None, *,
                        columns: list | None = None, save=None, figsize=None,
                        annotate: bool | None = None):
    """型の違う列どうしの関連の強さ（0〜1）。**大きさなので順次配色。**

        数値 × 数値     |Spearman の ρ|
        数値 × カテゴリ  相関比 η²
        カテゴリ × カテゴリ Cramér's V

    相関行列と違って符号が無い。発散配色を当てると中点の灰色が
    「関連が弱い」ではなく「負の関連」に見えてしまう。
    """
    num = _numeric_columns(df, schema, columns)
    cat = [c for c in _categorical_columns(df, schema, None)
           if (columns is None or c in columns) and df[c].nunique(dropna=True) <= 20]
    cols = num + cat
    if len(cols) < 2:
        raise ValueError("関連を出すには列が 2 つ以上必要")
    m = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        m.loc[a, a] = 1.0
        for b in cols[i + 1:]:
            v = _association(df, a, b, num)
            m.loc[a, b] = m.loc[b, a] = v
    return _matrix_plot(m, title="関連の強さ（|ρ| / η² / Cramér's V）",
                        cmap=sequential_cmap(), vmin=0, vmax=1,
                        annotate=annotate, save=save, figsize=figsize,
                        cbar_label="関連の強さ（0〜1）")


def _association(df, a, b, numeric_cols) -> float:
    an, bn = a in numeric_cols, b in numeric_cols
    try:
        if an and bn:
            r = (pd.to_numeric(df[a], errors="coerce")
                 .corr(pd.to_numeric(df[b], errors="coerce"), method="spearman"))
            return abs(float(r)) if pd.notna(r) else np.nan
        if an != bn:
            num_col, cat_col = (a, b) if an else (b, a)
            v = pd.to_numeric(df[num_col], errors="coerce")
            groups = [v[df[cat_col].astype(str) == lv].dropna().to_numpy()
                      for lv in df[cat_col].dropna().astype(str).unique()]
            groups = [g for g in groups if len(g) > 1]
            return eta_squared(groups) if len(groups) >= 2 else np.nan
        v = cramers_v(df[a].astype(str), df[b].astype(str))
        return np.nan if v is None else float(v)
    except Exception:                                              # noqa: BLE001
        return np.nan


def _cluster_order(corr: pd.DataFrame) -> list:
    """似た動きをする列を隣どうしにする。塊が見えるようにするため。"""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        d = 1 - corr.abs().fillna(0).to_numpy()
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2
        z = linkage(squareform(d, checks=False), method="average")
        return [corr.index[i] for i in leaves_list(z)]
    except Exception:                                              # noqa: BLE001
        return list(corr.index)


def _matrix_plot(m: pd.DataFrame, *, title, cmap, vmin, vmax, annotate,
                 save, figsize, cbar_label):
    _fontja()
    import matplotlib.pyplot as plt
    n = len(m)
    if annotate is None:
        annotate = n <= 14
    figsize = figsize or (max(6.0, 0.55 * n + 3), max(5.0, 0.55 * n + 2))
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE.surface)
    im = ax.imshow(m.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(m.index, fontsize=9)
    ax.set_title(title, fontsize=11, color=PALETTE.ink)
    # セルの境目を地の色で 1px 空ける（隣り合う面を分ける）
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=PALETTE.surface, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=PALETTE.muted, length=0)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(PALETTE.ink_secondary)
    if annotate:
        for i in range(n):
            for j in range(n):
                v = m.iat[i, j]
                if pd.isna(v):
                    continue
                strong = abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.32
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if strong else PALETTE.ink)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label, color=PALETTE.ink_secondary, fontsize=9)
    cb.ax.tick_params(colors=PALETTE.muted, labelsize=8)
    cb.outline.set_edgecolor(PALETTE.axis)
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== 目的変数
def outcome_relations(df: pd.DataFrame, outcome: str, schema: Schema | None = None, *,
                      task: str | None = None, columns: list | None = None,
                      ncols: int = 3, max_columns: int = 9, save=None, figsize=None):
    """目的変数と各説明変数の関係。

    回帰なら散布図＋移動中央値、分類ならクラス別の分布を重ねる。
    """
    if outcome not in df.columns:
        raise ValueError(f"目的変数 '{outcome}' がデータに無い")
    y = df[outcome]
    task = task or ("classification" if y.nunique(dropna=True) <= 10 else "regression")
    cols = [c for c in _numeric_columns(df, schema, columns) if c != outcome][:max_columns]
    if not cols:
        raise ValueError("説明変数（連続）が無い")
    _fontja()
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(cols) / ncols)
    figsize = figsize or (4.2 * ncols, 3.1 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor(PALETTE.surface)

    if task.startswith("class"):
        gy, levels, folded = _fold_levels(y, PALETTE.scatter_max)
        cmap = PALETTE.colors_for(levels, all_pairs=True)
    else:
        gy = levels = None
        folded = False

    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        v = pd.to_numeric(df[c], errors="coerce")
        if task.startswith("class"):
            for lv in levels:
                sub = v[gy == lv].dropna()
                if len(sub) < 2:
                    continue
                ax.hist(sub, bins=22, alpha=0.55, color=cmap[lv],
                        label=f"{outcome}={lv} (n={len(sub)})",
                        edgecolor=PALETTE.surface, linewidth=0.5)
            ax.set_ylabel("症例数")
            if i == 0:
                ax.legend(fontsize=8, frameon=False)
        else:
            yy = pd.to_numeric(y, errors="coerce")
            ok = v.notna() & yy.notna()
            ax.scatter(v[ok], yy[ok], s=14, alpha=0.5,
                       color=PALETTE.categorical[0], linewidths=0)
            band = _running_median(v[ok], yy[ok])
            if band is not None:
                ax.plot(band[0], band[1], color=PALETTE.categorical[1], lw=2.0,
                        label="移動中央値")
                if i == 0:
                    ax.legend(fontsize=8, frameon=False)
            ax.set_ylabel(outcome)
        ax.set_xlabel(c)
        ax.set_title(c, fontsize=10)
        _style(ax)
    for j in range(len(cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if folded:
        fig.text(0.01, 0.005, "※ クラスが多いため、上位以外は『その他』にまとめた",
                 fontsize=8, color=PALETTE.muted)
    fig.tight_layout()
    return _save(fig, save)


def _running_median(x, y, bins: int = 12):
    x = pd.Series(x).to_numpy(dtype=float)
    y = pd.Series(y).to_numpy(dtype=float)
    if len(x) < bins * 3:
        return None
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return None
    cx, cy = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 3:
            cx.append(float(np.median(x[m])))
            cy.append(float(np.median(y[m])))
    return (cx, cy) if len(cx) >= 3 else None


def univariate_auc(df: pd.DataFrame, outcome: str, schema: Schema | None = None, *,
                   columns: list | None = None, save=None, figsize=None):
    """説明変数 1 本ずつの AUC。**1 系列なので 1 色。**

    0.5 が「まったく予測できない」。棒の長さではなく 0.5 からの距離を見る。
    """
    from sklearn.metrics import roc_auc_score
    y = pd.to_numeric(df[outcome], errors="coerce")
    if y.nunique(dropna=True) != 2:
        raise ValueError(f"'{outcome}' が 2 値でないため AUC を出せない")
    cols = [c for c in _numeric_columns(df, schema, columns) if c != outcome]
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v.notna() & y.notna()
        if ok.sum() < 20 or y[ok].nunique() != 2:
            continue
        try:
            a = float(roc_auc_score(y[ok], v[ok]))
        except ValueError:
            continue
        rows.append({"変数": c, "AUC": a, "0.5からの距離": abs(a - 0.5)})
    t = pd.DataFrame(rows).sort_values("0.5からの距離", ascending=True)
    if not len(t):
        raise ValueError("AUC を出せる変数が無い")

    fig, ax = _new(figsize or (7.5, max(3.0, 0.38 * len(t) + 1.4)))
    ax.barh(range(len(t)), t["AUC"] - 0.5, left=0.5, height=0.66,
            color=PALETTE.categorical[0])
    ax.axvline(0.5, color=PALETTE.axis, lw=1.2)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["変数"], fontsize=9)
    for i, a in enumerate(t["AUC"].to_numpy()):
        ax.annotate(f"{a:.3f}", xy=(a, i), xytext=(6 if a >= 0.5 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if a >= 0.5 else "right",
                    fontsize=8, color=PALETTE.ink_secondary)
    ax.set_xlabel("単変量 AUC（0.5 = 予測できない）")
    ax.set_title(f"'{outcome}' に対する単変量 AUC", fontsize=11)
    ax.set_xlim(min(0.35, float(t["AUC"].min()) - 0.08),
                max(0.65, float(t["AUC"].max()) + 0.08))
    _style(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== SMD
def smd_forest(balance: pd.DataFrame, *, threshold: float = 0.1,
               label_col: str = "列", smd_col: str = "SMD",
               title: str = "train と test の標準化差（SMD）",
               save=None, figsize=None):
    """SMD のフォレストプロット。**群間バランスは p 値ではなくこれで見る。**

    |SMD| < 0.1 が目安。閾値の線を引き、超えたものだけ状態色にして、
    **色だけに頼らずラベルも付ける**。
    """
    t = balance.dropna(subset=[smd_col]).copy()
    if not len(t):
        raise ValueError("SMD の値が無い")
    t["_abs"] = t[smd_col].astype(float).abs()
    t = t.sort_values("_abs")
    fig, ax = _new(figsize or (7.5, max(3.0, 0.36 * len(t) + 1.4)))
    over = t["_abs"] >= threshold
    ax.axvspan(-threshold, threshold, color=PALETTE.grid, alpha=0.55, zorder=0)
    ax.axvline(0, color=PALETTE.axis, lw=1.2, zorder=1)
    ax.scatter(t.loc[~over, smd_col], np.flatnonzero(~over.to_numpy()),
               s=46, color=PALETTE.categorical[0], zorder=3,
               edgecolors=PALETTE.surface, linewidths=1.2, label="目安の内側")
    if over.any():
        ax.scatter(t.loc[over, smd_col], np.flatnonzero(over.to_numpy()),
                   s=54, color=PALETTE.critical, marker="D", zorder=3,
                   edgecolors=PALETTE.surface, linewidths=1.2,
                   label=f"|SMD| ≧ {threshold}（偏り）")
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t[label_col], fontsize=9)
    for i, v in enumerate(t[smd_col].astype(float).to_numpy()):
        ax.annotate(f"{v:+.3f}", xy=(v, i), xytext=(9 if v >= 0 else -9, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8,
                    color=PALETTE.ink_secondary)
    ax.set_xlabel("標準化差（SMD）")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    lim = max(0.25, float(t["_abs"].max()) * 1.45)
    ax.set_xlim(-lim, lim)
    _style(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== 管理目標
def achievement_bars(achievement: pd.DataFrame, *, save=None, figsize=None,
                     title: str = "管理目標の達成率"):
    """管理目標の達成率。境界値の行は除いて、群ごとに並べる。"""
    t = achievement[~achievement["群"].astype(str).str.contains("境界")].copy()
    if not len(t):
        raise ValueError("達成率の行が無い")
    t["率"] = t["達成率"].astype(str).str.rstrip("%").replace("—", np.nan).astype(float)
    items = list(dict.fromkeys(t["項目"]))
    groups = [g for g in dict.fromkeys(t["群"]) if g != "全体"]
    fig, ax = _new(figsize or (max(7.0, 1.5 * len(items) + 2), 4.4))
    width = 0.8 / max(len(groups) + 1, 1)
    x = np.arange(len(items))
    series = ["全体", *groups][:len(PALETTE.categorical)]
    for k, g in enumerate(series):
        vals = [float(t[(t["項目"] == it) & (t["群"] == g)]["率"].mean()) for it in items]
        ax.bar(x + k * width - 0.4 + width / 2, vals, width * 0.9,
               color=PALETTE.categorical[k], label=str(g),
               edgecolor=PALETTE.surface, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(items, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("達成率（%）")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=min(len(series), 5))
    _style(ax)
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== 次元削減
def pca_scatter(df: pd.DataFrame, schema: Schema | None = None, *,
                columns: list | None = None, by: str | None = None,
                save=None, figsize=(11, 4.6)):
    """主成分分析。群の分離・施設差・外れ値の塊を目で見る。

    **散布図はすべての対が同時に見えるので 3 色まで。** 超える分はまとめる。
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cols = _numeric_columns(df, schema, columns)
    cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    if len(cols) < 2:
        raise ValueError("PCA には連続変数が 2 つ以上必要")
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    p = PCA(n_components=min(len(cols), 10)).fit(x)
    z = p.transform(x)

    _fontja()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(PALETTE.surface)

    ax = axes[0]
    if by and by in df.columns:
        g, levels, folded = _fold_levels(df[by], PALETTE.scatter_max)
        cmap = PALETTE.colors_for(levels, all_pairs=True)
        for lv in levels:
            m = (g == lv).to_numpy()
            ax.scatter(z[m, 0], z[m, 1], s=16, alpha=0.65, color=cmap[lv],
                       label=f"{lv} (n={int(m.sum())})", linewidths=0)
        ax.legend(fontsize=8, frameon=False)
    else:
        folded = False
        ax.scatter(z[:, 0], z[:, 1], s=16, alpha=0.6,
                   color=PALETTE.categorical[0], linewidths=0)
    ax.set_xlabel(f"第1主成分（{p.explained_variance_ratio_[0]:.1%}）")
    ax.set_ylabel(f"第2主成分（{p.explained_variance_ratio_[1]:.1%}）")
    ax.set_title("主成分得点", fontsize=11)
    _style(ax, grid_axis="both")

    ax2 = axes[1]
    k = len(p.explained_variance_ratio_)
    ax2.bar(range(1, k + 1), p.explained_variance_ratio_ * 100,
            color=PALETTE.categorical[0], edgecolor=PALETTE.surface, linewidth=1.0)
    ax2.plot(range(1, k + 1), np.cumsum(p.explained_variance_ratio_) * 100,
             color=PALETTE.categorical[1], marker="o", ms=5, lw=2.0, label="累積")
    ax2.set_xlabel("主成分")
    ax2.set_ylabel("寄与率（%）")
    ax2.set_title("寄与率", fontsize=11)
    ax2.legend(fontsize=8, frameon=False)
    _style(ax2)
    if folded:
        fig.text(0.01, 0.005, f"※ 散布図は 3 色までのため、'{by}' の上位以外は『その他』",
                 fontsize=8, color=PALETTE.muted)
    fig.tight_layout()
    return _save(fig, save)


# ================================================================== 図の束
@dataclass
class FigureSet:
    """図とその説明をひとまとめにする。レポートがこれを並べる。"""
    figures: list = field(default_factory=list)      # (見出し, 説明, Figure)
    skipped: list = field(default_factory=list)      # (見出し, 描けなかった理由)

    def add(self, title: str, caption: str, maker):
        try:
            fig = maker()
        except Exception as e:                                     # noqa: BLE001
            self.skipped.append((title, f"{type(e).__name__}: {e}"))
            return None
        self.figures.append((title, caption, fig))
        return fig

    def __len__(self):
        return len(self.figures)

    def save_all(self, directory, *, dpi: int = 300, prefix: str = "") -> list:
        """全部の図を PNG で保存し、書いたファイルの一覧を返す。

        ファイル名は見出しから作る。**ファイル名に使えない文字は `_` にする**
        （Windows では `:` `?` `*` がファイル名に使えず、保存が例外で止まる）。
        """
        import os
        os.makedirs(directory, exist_ok=True)
        written = []
        for i, (title, _caption, fig) in enumerate(self.figures, start=1):
            stem = re.sub(r'[\\/:*?"<>|\s]+', "_", str(title)).strip("_") or f"figure{i}"
            path = os.path.join(directory, f"{prefix}{i:02d}_{stem}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
            written.append(path)
        return written


def overview(df: pd.DataFrame, schema: Schema | None = None, *,
             outcome: str | None = None, task: str | None = None,
             by: str | None = None, achievement: pd.DataFrame | None = None,
             balance: pd.DataFrame | None = None) -> FigureSet:
    """データを一通り見る図をまとめて作る。描けないものは理由を残して飛ばす。"""
    fs = FigureSet()
    outcome = outcome or (schema.target["name"] if (schema and schema.target) else None)
    task = task or (schema.target.get("task") if (schema and schema.target) else None)
    by = by or (schema.by_role(GROUP)[0] if (schema and schema.by_role(GROUP)) else None)

    fs.add("欠損の地図", "どの症例でどの列が同時に欠けるかを見る。"
           "縦の筋が揃っていれば、同じ検査がまとめて行われなかったということ。",
           lambda: missing_map(df, schema))
    fs.add("連続変数の分布",
           "中央値の破線と分布の形を見る。二つ山なら別の集団が混ざっている疑いがある"
           "（施設差、測定法の変更、単位の混在）。",
           lambda: distributions(df, schema, by=by))
    fs.add("カテゴリ変数の内訳", "症例数の少ない水準は検定が不安定になる。",
           lambda: category_bars(df, schema, by=by))
    fs.add("相関", "符号のある量なので発散配色。中点の灰色が『関連なし』。"
           "似た動きの列が隣に来るよう並べ替えてある。",
           lambda: correlation_heatmap(df, schema))
    fs.add("関連の強さ", "型の違う列どうしも比べられる 0〜1 の指標。"
           "こちらは符号が無いので順次配色（濃いほど強い）。",
           lambda: association_heatmap(df, schema))
    fs.add("主成分", "群の分離・施設差・外れ値の塊を目で見る。",
           lambda: pca_scatter(df, schema, by=by))
    if outcome:
        fs.add("目的変数との関係", "説明変数ごとに、目的変数との関係を見る。",
               lambda: outcome_relations(df, outcome, schema, task=task))
        if pd.to_numeric(df[outcome], errors="coerce").nunique(dropna=True) == 2:
            fs.add("単変量 AUC", "0.5 が『まったく予測できない』。"
                   "**これは探索であって、ここで変数を選ぶと選択後推論の問題が起きる。**",
                   lambda: univariate_auc(df, outcome, schema))
    if achievement is not None and len(achievement):
        fs.add("管理目標の達成率", "境界値ちょうどの扱いは表のほうで確かめること。",
               lambda: achievement_bars(achievement))
    if balance is not None and len(balance):
        fs.add("train と test のバランス", "群間バランスは p 値ではなく SMD で見る。"
               "|SMD| < 0.1 が目安。",
               lambda: smd_forest(balance))
    return fs


def to_base64(fig, *, dpi: int = 130) -> str:
    """図を base64 の PNG にする。HTML 1 枚に埋め込むため。"""
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE.surface)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def close_all():
    """作った図をまとめて閉じる。レポートを書き出したあとに呼ぶ。"""
    import matplotlib.pyplot as plt
    plt.close("all")
