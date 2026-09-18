"""medprep.describe — 記述統計・Table 1・群間比較・管理目標の達成率。

医学論文の Table 1 をそのまま出す。ただし「出す」だけでは足りない。
**どの検定をなぜ選んだか**を表に残す。そこが説明できなければ査読に耐えない。

このモジュールが引き受けている判断
----------------------------------
1. **平均(SD) で書くか、中央値[Q1,Q3] で書くか**（正規性の判定）
2. **どの検定を使うか**（連続/カテゴリ、群数、正規性、対応の有無）
3. **多重比較をどう扱うか**（BH の q 値を併記し、3群以上は事後比較まで出す）
4. **効果量を必ず併記する**（p 値は例数で動く。効果量は動かない）

正規性の判定について
--------------------
Shapiro-Wilk を n=600 に当てると、実用上どうでもよい歪みでも p<0.05 になる。
検定だけで決めると、ほぼすべての項目が「非正規」になり、
中央値[Q1,Q3] ばかりの読みにくい Table 1 ができあがる。

ここで決めたいのは「厳密に正規分布か」ではなく
**「平均と中央値のどちらがこの分布を忠実に代表するか」** である。
したがって medprep は、**検定が棄却し、かつ歪み（歪度・尖度）が実際に大きい**
ときにだけ非正規とする。判定の根拠は必ず表に残す。

Table 1 の p 値について
-----------------------
ベースライン表の p 値は、群分けが無作為割付なら**定義上意味が無く**、
観察研究なら**例数が増えるだけで小さくなる**。
医学統計の標準的な助言は「p 値ではなく標準化差（SMD）で群間バランスを見る」である。
`table_one()` はこの注意書きを結果に必ず添える。消したければ `notes` を読まないこと。
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .clean import build_alias_map, load_dict
from .schema import (
    BINARY,
    GROUP,
    NOMINAL,
    NUMERIC,
    ORDINAL,
    OUTCOME,
    Schema,
    _norm,
    cramers_v,
)
from .targets import describe_target, in_target
from .textfmt import frame_text
from .timing import detect_timing

CONTINUOUS, CATEGORICAL = "continuous", "categorical"
_TIMING_LABEL = {"pre": "透析前", "post": "透析後", "unknown": ""}


# ================================================================== 正規性
@dataclass
class Normality:
    """正規性の判定。**根拠を持ち歩く。**"""
    is_normal: bool
    test: str
    p: float
    skew: float
    kurtosis: float
    n: int
    reason: str


def normality(x, alpha: float = 0.05, skew_limit: float = 0.5,
              kurt_limit: float = 1.0) -> Normality:
    """平均(SD) で要約してよいかを判定する。

    n ≤ 5000 は Shapiro-Wilk、それより大きければ Anderson-Darling を使う。
    ただし**検定の棄却だけでは非正規としない**（大標本では必ず棄却されるため）。
    歪度 |skew| ≥ 0.5 か 過剰尖度 ≥ 1.0 を伴うときに非正規とする。
    """
    v = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 3 or np.nanstd(v) == 0:
        return Normality(True, "—", float("nan"), 0.0, 0.0, n,
                         f"n={n} または分散 0 のため判定できない。平均(SD) で表示する")
    sk = float(stats.skew(v, bias=False))
    ku = float(stats.kurtosis(v, fisher=True, bias=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n <= 5000:
            test, p = "Shapiro-Wilk", float(stats.shapiro(v).pvalue)
        else:
            r = stats.anderson(v, dist="norm")
            # Anderson-Darling は p 値を返さないので、有意水準 5% の臨界値と比べる
            crit = float(r.critical_values[list(r.significance_level).index(5.0)])
            test, p = "Anderson-Darling", (0.01 if r.statistic > crit else 0.20)

    rejected = p < alpha
    shaped = abs(sk) >= skew_limit or ku >= kurt_limit
    if n < 20:
        # 小標本では検定の検出力が無い。形だけで決める。
        is_normal = not shaped
        reason = (f"n={n} と小さく検定の検出力が無い。歪度 {sk:.2f}・過剰尖度 {ku:.2f} で判断"
                  f"（{'非正規' if shaped else '正規とみなす'}）")
    elif rejected and shaped:
        is_normal = False
        # 実際に効いた基準だけを書く。「歪度も尖度も大きい」と書いてしまうと、
        # 尖度が小さい場合に読み手が数字と文言の食い違いに気づく。
        which = []
        if abs(sk) >= skew_limit:
            which.append(f"歪度 {sk:.2f}（|skew| ≥ {skew_limit}）")
        if ku >= kurt_limit:
            which.append(f"過剰尖度 {ku:.2f}（≥ {kurt_limit}）")
        reason = f"{test} p={p:.3g} < {alpha} で棄却され、" + "・".join(which) + " も大きい"
    elif rejected:
        is_normal = True
        reason = (f"{test} は p={p:.3g} で棄却するが、歪度 {sk:.2f}・過剰尖度 {ku:.2f} は小さい。"
                  f"大標本では小さな歪みでも棄却されるため、平均(SD) で表示する")
    else:
        is_normal = True
        reason = f"{test} p={p:.3g} ≥ {alpha}。歪度 {sk:.2f}"
    return Normality(is_normal, test, p, sk, ku, n, reason)


# ================================================================== 効果量
def hedges_g(a, b) -> float:
    """2 群の標準化平均差（小標本補正つき）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    if sp2 <= 0:
        return 0.0
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def cliffs_delta(a, b) -> float:
    """順位に基づく効果量（−1〜1）。非正規データの 2 群比較に使う。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # U 統計量から出す（全対比較を作らないので大標本でも軽い）
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def eta_squared(groups) -> float:
    """一元配置分散分析の η²（群間平方和 / 全平方和）。"""
    vals = [np.asarray(g, float) for g in groups]
    vals = [v[~np.isnan(v)] for v in vals]
    allv = np.concatenate(vals) if vals else np.array([])
    if len(allv) < 2:
        return float("nan")
    gm = allv.mean()
    ss_b = sum(len(v) * (v.mean() - gm) ** 2 for v in vals if len(v))
    ss_t = ((allv - gm) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else 0.0


def epsilon_squared(h: float, n: int, k: int) -> float:
    """Kruskal-Wallis の ε²。H 統計量から出す。

    群間差が偶然より小さいと (H − k + 1) が負になる。効果量として負の値を出すと
    「小さい」ではなく「逆向き」と読まれるので 0 に丸める。
    """
    if n <= k:
        return float("nan")
    return float(min(max((h - k + 1) / (n - k), 0.0), 1.0))


# ================================================================== 標準化差
def smd_continuous(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float("nan") if s == 0 else float((a.mean() - b.mean()) / s)


def smd_categorical(a: pd.Series, b: pd.Series) -> float:
    """多水準にも使える標準化差（Yang & Dalton 2012）。

    2 水準なら (p1−p2)/√((p1(1−p1)+p2(1−p2))/2) に一致する。
    """
    levels = sorted(set(a.dropna().astype(str)) | set(b.dropna().astype(str)))
    if len(levels) < 2:
        return float("nan")
    p1 = np.array([(a.astype(str) == lv).mean() for lv in levels])[:-1]
    p2 = np.array([(b.astype(str) == lv).mean() for lv in levels])[:-1]
    k = len(p1)
    if k == 0:
        return float("nan")

    def cov(p):
        s = -np.outer(p, p)
        s[np.diag_indices(k)] = p * (1 - p)
        return s

    s = (cov(p1) + cov(p2)) / 2
    d = (p1 - p2).reshape(-1, 1)
    try:
        v = float(d.T @ np.linalg.pinv(s) @ d)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.sqrt(max(v, 0.0)))


def group_levels(g: pd.Series) -> list:
    """群の水準を、並べられるなら昇順で返す（型が混ざっていれば出現順のまま）。"""
    levels = list(pd.unique(g.dropna()))
    with contextlib.suppress(TypeError):
        levels = sorted(levels)
    return levels


def max_pairwise_smd(series: pd.Series, groups: pd.Series, kind: str) -> float:
    """3 群以上のとき、対ごとの SMD の最大値を返す（tableone と同じ規約）。"""
    levels = group_levels(groups)
    best = float("nan")
    for i, gi in enumerate(levels):
        for gj in levels[i + 1:]:
            a, b = series[groups == gi], series[groups == gj]
            v = (smd_continuous(a, b) if kind == CONTINUOUS
                 else smd_categorical(a, b))
            if pd.notna(v) and (pd.isna(best) or abs(v) > abs(best)):
                best = v
    return best


# ================================================================== 事後比較
# Tukey と Dunn の結果を 1 つの表に積めるよう、列をそろえておく
_POSTHOC_COLUMNS = ["群1", "群2", "検定", "統計量", "統計量の意味", "p", "p補正", "補正法"]


def dunn_test(groups: dict, correction: str = "holm") -> pd.DataFrame:
    """Kruskal-Wallis の事後比較（Dunn 1964、同順位補正つき）。

    scipy に実装が無いので自前で持つ。Holm 補正を既定にする。
    """
    names = list(groups)
    vals = {k: pd.Series(v).dropna().to_numpy(dtype=float) for k, v in groups.items()}
    allv = np.concatenate([vals[k] for k in names])
    n = len(allv)
    ranks = stats.rankdata(allv)
    pos, mean_rank, sizes = 0, {}, {}
    for k in names:
        m = len(vals[k])
        mean_rank[k] = ranks[pos:pos + m].mean() if m else np.nan
        sizes[k] = m
        pos += m
    # 同順位補正
    _, counts = np.unique(allv, return_counts=True)
    ties = float(((counts ** 3) - counts).sum())
    sigma2 = n * (n + 1) / 12 - ties / (12 * (n - 1)) if n > 1 else np.nan

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sizes[a] == 0 or sizes[b] == 0 or not np.isfinite(sigma2) or sigma2 <= 0:
                continue
            se = np.sqrt(sigma2 * (1 / sizes[a] + 1 / sizes[b]))
            z = (mean_rank[a] - mean_rank[b]) / se
            rows.append({"群1": a, "群2": b, "検定": "Dunn",
                         "統計量": float(z), "統計量の意味": "z",
                         "p": float(2 * stats.norm.sf(abs(z)))})
    out = pd.DataFrame(rows, columns=_POSTHOC_COLUMNS)
    if len(out):
        out["p補正"] = _adjust(out["p"].to_numpy(), correction)
        out["補正法"] = correction.upper()
    return out


def tukey_hsd(groups: dict) -> pd.DataFrame:
    names = list(groups)
    arrs = [pd.Series(groups[k]).dropna().to_numpy(dtype=float) for k in names]
    if any(len(a) < 2 for a in arrs):
        return pd.DataFrame()
    r = stats.tukey_hsd(*arrs)
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            rows.append({"群1": a, "群2": b, "検定": "Tukey HSD",
                         "統計量": float(r.statistic[i, j]), "統計量の意味": "平均差",
                         # Tukey の p 値はもともと族全体で補正されている。
                         # 補正前の p は存在しないので空にしておく。
                         "p": float("nan"), "p補正": float(r.pvalue[i, j]),
                         "補正法": "Tukey"})
    return pd.DataFrame(rows, columns=_POSTHOC_COLUMNS)


def _adjust(p, method: str = "bh"):
    """多重比較の補正。statsmodels があればそれを使い、無ければ自前で計算する。"""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return out
    key = {"bh": "fdr_bh", "fdr": "fdr_bh", "holm": "holm",
           "bonferroni": "bonferroni"}.get(method, method)
    try:
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method=key)[1]
    except Exception:                                              # noqa: BLE001
        v = p[ok]
        order = np.argsort(v)
        m = len(v)
        if key == "bonferroni":
            adj = np.minimum(v * m, 1.0)
        else:
            ranked = v[order]
            adj_sorted = np.minimum.accumulate(
                (ranked * m / np.arange(1, m + 1))[::-1])[::-1]
            adj = np.empty(m)
            adj[order] = np.minimum(adj_sorted, 1.0)
        out[ok] = adj
    return out


# ================================================================== 群間比較
@dataclass
class Comparison:
    """1 変数ぶんの群間比較。**なぜその検定を選んだかを持つ。**"""
    name: str
    kind: str
    test: str
    reason: str
    statistic: float = float("nan")
    p: float = float("nan")
    q: float = float("nan")
    effect: float = float("nan")
    effect_name: str = ""
    smd: float = float("nan")
    n: int = 0
    n_missing: int = 0
    summaries: dict = field(default_factory=dict)      # 群名 -> 表示用の文字列
    normality: Normality | None = None
    posthoc: pd.DataFrame | None = None
    note: str = ""


@dataclass
class ComparisonResult:
    comparisons: list = field(default_factory=list)
    by: str = ""
    groups: list = field(default_factory=list)
    sizes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    correction: str = "bh"

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for c in self.comparisons:
            row = {"項目": c.name, "尺度": "連続" if c.kind == CONTINUOUS else "カテゴリ"}
            for g in self.groups:
                row[str(g)] = c.summaries.get(g, "")
            row.update({
                "欠測": c.n_missing, "検定": c.test,
                "統計量": _r(c.statistic), "p": _p(c.p),
                f"q({self.correction.upper()})": _p(c.q),
                "効果量": (f"{c.effect_name} = {_r(c.effect)}"
                          if (c.effect_name and _r(c.effect)) else ""),
                "SMD": _r(c.smd), "判定の根拠": c.reason,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def posthoc_frame(self) -> pd.DataFrame:
        out = []
        for c in self.comparisons:
            if c.posthoc is not None and len(c.posthoc):
                d = c.posthoc.copy()
                d.insert(0, "項目", c.name)
                out.append(d)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def report(self) -> str:
        head = (f"群間比較  {self.by} = "
                + "、".join(f"{g}（n={self.sizes.get(g, 0)}）" for g in self.groups))
        lines = [head, frame_text(self.to_frame())]
        ph = self.posthoc_frame()
        if len(ph):
            lines += ["\n事後比較:", frame_text(ph)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)


def _r(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{digits}f}"


def _p(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _kind_of(s: pd.Series, role: str | None) -> str:
    if role in (NUMERIC,):
        return CONTINUOUS
    if role in (BINARY, NOMINAL, ORDINAL, GROUP):
        return CATEGORICAL
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
        return CONTINUOUS
    return CATEGORICAL


def _fmt_continuous(v: pd.Series, normal: bool) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if len(v) == 0:
        return "—"
    if normal:
        return f"{v.mean():.1f} ({v.std(ddof=1):.1f})"
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    return f"{v.median():.1f} [{q1:.1f}, {q3:.1f}]"


def _fmt_categorical(v: pd.Series, levels: list) -> str:
    v = v.dropna().astype(str)
    n = len(v)
    if n == 0:
        return "—"
    parts = [f"{lv}: {int((v == lv).sum())} ({(v == lv).mean():.1%})" for lv in levels]
    return " / ".join(parts)


def _label_of(col: str, dic: dict, amap: dict) -> str:
    """列名に単位と採血時点を添えた表示名を作る。

    ★透析前 BUN と透析後 BUN が同じ表に並ぶので、時点を書かないと読み違える。★
    """
    timing, base = detect_timing(col)
    key = amap.get(_norm(base)) or amap.get(_norm(col))
    spec = dic["items"].get(key) if key else None
    label = col
    if spec:
        unit = spec.get("unit")
        if unit and unit not in ("index", ""):
            label = f"{col} [{unit}]"
    tl = _TIMING_LABEL.get(timing, "")
    return f"{label}（{tl}）" if tl else label


def compare_groups(
    df: pd.DataFrame,
    by: str,
    columns: list | None = None,
    *,
    schema: Schema | None = None,
    dic: dict | None = None,
    paired_by: str | None = None,
    alpha: float = 0.05,
    correction: str = "bh",
    posthoc: bool = True,
    min_group: int = 2,
) -> ComparisonResult:
    """群間比較。検定を自動で選び、**選んだ理由と効果量を必ず添える。**

    | 状況 | 検定 |
    |---|---|
    | 連続・2群・正規 | Welch の t 検定 |
    | 連続・2群・非正規 | Mann-Whitney U |
    | 連続・3群以上・正規 | 一元配置分散分析 →（事後）Tukey HSD |
    | 連続・3群以上・非正規 | Kruskal-Wallis →（事後）Dunn（Holm 補正） |
    | カテゴリ | χ²（期待度数 < 5 が 2 割超なら Fisher 正確検定に自動切替） |
    | 対応あり（`paired_by`） | 対応のある t / Wilcoxon 符号付順位 |
    """
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    if by not in df.columns:
        raise ValueError(f"群分け列 '{by}' がデータに無い")

    g = df[by]
    levels = group_levels(g)
    sizes = {lv: int((g == lv).sum()) for lv in levels}
    res = ComparisonResult(by=by, groups=levels, sizes=sizes, correction=correction)

    if len(levels) < 2:
        res.notes.append(f"'{by}' の水準が {len(levels)} 個しかないため比較できない")
        return res
    small = [lv for lv, n in sizes.items() if n < 5]
    if small:
        res.notes.append(f"例数が 5 未満の群がある（{small}）。検定は不安定になる")

    if columns is None:
        if schema is not None:
            columns = [c for c in schema.kept()
                       if c != by and schema.columns[c].role in
                       (NUMERIC, BINARY, ORDINAL, NOMINAL, GROUP, OUTCOME)]
        else:
            columns = [c for c in df.columns if c != by]
    columns = [c for c in columns if c in df.columns and c != by]

    for col in columns:
        s = df[col]
        role = schema.columns[col].role if (schema and col in schema.columns) else None
        kind = _kind_of(s, role)
        cmp_ = (_compare_continuous(s, g, levels, paired_by, df, alpha, posthoc, correction)
                if kind == CONTINUOUS
                else _compare_categorical(s, g, levels, paired_by, df))
        cmp_.name = _label_of(col, dic, amap)
        cmp_.n_missing = int(s.isna().sum())
        res.comparisons.append(cmp_)

    qs = _adjust([c.p for c in res.comparisons], correction)
    for c, q in zip(res.comparisons, qs):
        c.q = float(q)
    res.notes.append(
        f"p 値は {len(res.comparisons)} 項目ぶん出しているので、{correction.upper()} で補正した "
        f"q 値を併記した。個々の p だけを見て有意と述べない")
    return res


def _compare_continuous(s, g, levels, paired_by, df, alpha, posthoc, correction) -> Comparison:
    v = pd.to_numeric(s, errors="coerce")
    groups = {lv: v[g == lv].dropna() for lv in levels}
    nrm = normality(v)
    summaries = {lv: _fmt_continuous(groups[lv], nrm.is_normal) for lv in levels}
    n = int(sum(len(x) for x in groups.values()))
    c = Comparison(name=str(s.name), kind=CONTINUOUS, test="—", reason="", n=n,
                   summaries=summaries, normality=nrm)

    usable = [lv for lv in levels if len(groups[lv]) >= 2]
    if len(usable) < 2:
        c.test, c.reason = "—", "有効な例数が 2 未満の群があるため検定しない"
        return c

    if paired_by and len(levels) == 2:
        a, b = _paired_vectors(df, v, g, levels, paired_by)
        if len(a) >= 2:
            if nrm.is_normal:
                r = stats.ttest_rel(a, b)
                c.test, c.reason = "対応のある t 検定", f"対応あり・正規。{nrm.reason}"
            else:
                r = stats.wilcoxon(a, b)
                c.test, c.reason = "Wilcoxon 符号付順位検定", f"対応あり・非正規。{nrm.reason}"
            c.statistic, c.p = float(r.statistic), float(r.pvalue)
            c.effect, c.effect_name = hedges_g(a, b), "Hedges' g"
            c.smd = smd_continuous(a, b)
            c.note = f"対応のある {len(a)} 組で比較した（'{paired_by}' で対応づけ）"
            return c
        c.note = f"'{paired_by}' で対応がつく組が無いため、対応なしとして扱った"

    if len(levels) == 2:
        a, b = groups[levels[0]], groups[levels[1]]
        if nrm.is_normal:
            r = stats.ttest_ind(a, b, equal_var=False)
            c.test = "Welch の t 検定"
            c.reason = f"2群・正規。{nrm.reason}。分散の等質性を仮定しない Welch を既定にする"
            c.effect, c.effect_name = hedges_g(a, b), "Hedges' g"
        else:
            r = stats.mannwhitneyu(a, b, alternative="two-sided")
            c.test = "Mann-Whitney U 検定"
            c.reason = f"2群・非正規。{nrm.reason}"
            c.effect, c.effect_name = cliffs_delta(a, b), "Cliff's δ"
        c.statistic, c.p = float(r.statistic), float(r.pvalue)
        c.smd = smd_continuous(a, b)
        return c

    arrs = [groups[lv] for lv in usable]
    if nrm.is_normal:
        r = stats.f_oneway(*arrs)
        c.test, c.reason = "一元配置分散分析", f"3群以上・正規。{nrm.reason}"
        c.effect, c.effect_name = eta_squared(arrs), "η²"
        if posthoc:
            c.posthoc = tukey_hsd({lv: groups[lv] for lv in usable})
    else:
        r = stats.kruskal(*arrs)
        c.test, c.reason = "Kruskal-Wallis 検定", f"3群以上・非正規。{nrm.reason}"
        c.effect = epsilon_squared(float(r.statistic), int(sum(len(a) for a in arrs)),
                                   len(arrs))
        c.effect_name = "ε²"
        if posthoc:
            c.posthoc = dunn_test({lv: groups[lv] for lv in usable}, "holm")
    c.statistic, c.p = float(r.statistic), float(r.pvalue)
    c.smd = max_pairwise_smd(v, g, CONTINUOUS)
    c.note = "SMD は対ごとの最大値"
    return c


def _compare_categorical(s, g, levels, paired_by, df) -> Comparison:
    v = s.astype("object").where(s.notna())
    cat_levels = sorted({str(x) for x in v.dropna().unique()})
    summaries = {lv: _fmt_categorical(v[g == lv], cat_levels) for lv in levels}
    c = Comparison(name=str(s.name), kind=CATEGORICAL, test="—", reason="",
                   n=int(v.notna().sum()), summaries=summaries)

    t = pd.crosstab(v.astype(str), g)
    t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        c.test, c.reason = "—", "水準が 1 つしかないため検定しない"
        return c

    chi2, p, _dof, exp = stats.chi2_contingency(t)
    small_ratio = float((exp < 5).mean())
    if small_ratio > 0.2:
        try:
            r = stats.fisher_exact(t.to_numpy())
            c.test = "Fisher 正確検定"
            c.reason = (f"期待度数 < 5 のセルが {small_ratio:.0%}（2 割超）のため "
                        f"χ² から切り替えた")
            c.p = float(r.pvalue if hasattr(r, "pvalue") else r[1])
            c.statistic = float("nan")
        except (ValueError, TypeError) as e:                       # noqa: BLE001
            c.test = "χ² 検定"
            c.reason = (f"期待度数 < 5 のセルが {small_ratio:.0%} あり Fisher が望ましいが、"
                        f"この scipy では {t.shape[0]}×{t.shape[1]} 表の正確検定ができない（{e}）。"
                        f"χ² の p 値は過小評価になりうる")
            c.statistic, c.p = float(chi2), float(p)
    else:
        c.test = "χ² 検定"
        c.reason = f"期待度数 < 5 のセルは {small_ratio:.0%}（2 割以下）"
        c.statistic, c.p = float(chi2), float(p)

    # ★`cramers_v(...) or nan` と書いてはいけない。★
    #   関連がまったく無いとき Cramér's V は 0.0 を返し、0.0 は偽と評価されるので
    #   「算出できなかった」ことにされてしまう。None かどうかで判定する。
    cv = cramers_v(v.astype(str), g.astype(str))
    c.effect = float("nan") if cv is None else float(cv)
    c.effect_name = "Cramér's V"
    c.smd = (smd_categorical(v[g == levels[0]], v[g == levels[1]]) if len(levels) == 2
             else max_pairwise_smd(v, g, CATEGORICAL))
    if len(levels) > 2:
        c.note = "SMD は対ごとの最大値"
    return c


def _paired_vectors(df, v, g, levels, paired_by):
    """`paired_by` で 2 群を対応づけ、両群に値がある対だけを返す。"""
    if paired_by not in df.columns:
        return np.array([]), np.array([])
    w = pd.DataFrame({"id": df[paired_by], "g": g, "v": v}).dropna()
    a = w[w["g"] == levels[0]].drop_duplicates("id").set_index("id")["v"]
    b = w[w["g"] == levels[1]].drop_duplicates("id").set_index("id")["v"]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)


# ================================================================== Table 1
_P_VALUE_NOTE = (
    "【Table 1 の p 値について】無作為割付なら、ベースラインの群間差は偶然によるものと"
    "分かっているので p 値に意味は無い。観察研究なら、例数が増えるだけで p は小さくなる。"
    "群間バランスは p ではなく SMD（目安 |SMD| < 0.1 でバランスが取れているとみなす）で"
    "評価すること。"
)


@dataclass
class TableOneResult:
    table: pd.DataFrame
    groups: list = field(default_factory=list)
    sizes: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    comparison: ComparisonResult | None = None
    groupby: str | None = None

    def to_frame(self) -> pd.DataFrame:
        return self.table

    def __str__(self) -> str:
        return self.report()

    def report(self) -> str:
        head = "Table 1"
        if self.groupby:
            head += f"（{self.groupby} で層別）"
        lines = [head, frame_text(self.table)]
        for n in self.notes:
            lines.append(f"\n[注記] {n}")
        return "\n".join(lines)

    def to_excel(self, path):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            self.table.to_excel(w, sheet_name="Table1", index=False)
            if self.comparison is not None:
                self.comparison.to_frame().to_excel(w, sheet_name="群間比較", index=False)
                ph = self.comparison.posthoc_frame()
                if len(ph):
                    ph.to_excel(w, sheet_name="事後比較", index=False)
            pd.DataFrame({"注記": self.notes}).to_excel(w, sheet_name="注記", index=False)
        return path

    def to_html(self, path=None) -> str:
        style = ("<style>body{font-family:sans-serif;font-size:14px}"
                 "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px 8px}"
                 "th{background:#f2f2f2}.note{color:#555;font-size:12px;margin-top:1em}</style>")
        notes = "".join(f"<p class='note'>{n}</p>" for n in self.notes)
        html = (f"<html><head><meta charset='utf-8'>{style}</head><body>"
                f"<h2>Table 1</h2>{self.table.to_html(index=False, escape=False)}{notes}"
                f"</body></html>")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html


def table_one(
    df: pd.DataFrame,
    schema: Schema | None = None,
    *,
    groupby: str | None = None,
    columns: list | None = None,
    pval: bool = True,
    smd: bool = True,
    overall: bool = True,
    dic: dict | None = None,
    correction: str = "bh",
    paired_by: str | None = None,
    max_levels: int = 20,
) -> TableOneResult:
    """医学論文の Table 1 を作る。

    連続変数は正規性の判定に従って 平均(SD) か 中央値[Q1,Q3] で書き分け、
    カテゴリ変数は水準ごとに n(%) を出す。`groupby` を渡すと群間比較を併記する。
    """
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    if schema is None:
        schema = Schema.infer(df, group=groupby, dic=dic)
    if columns is None:
        columns = [c for c in schema.kept()
                   if c != groupby and schema.columns[c].role in
                   (NUMERIC, BINARY, ORDINAL, NOMINAL, GROUP, OUTCOME)]
    columns = [c for c in columns if c in df.columns and c != groupby]

    comp = None
    if groupby:
        g = df[groupby]
        levels = group_levels(g)
        sizes = {lv: int((g == lv).sum()) for lv in levels}
        comp = compare_groups(df, groupby, columns, schema=schema, dic=dic,
                              correction=correction, paired_by=paired_by)
        cmap = {c.name: c for c in comp.comparisons}
    else:
        levels, sizes, cmap = [], {}, {}

    rows = []
    notes_extra: list = []
    head = {"項目": "n"}
    if overall:
        head["全体"] = str(len(df))
    for lv in levels:
        head[str(lv)] = str(sizes[lv])
    if groupby and pval:
        head["検定"] = ""
        head["p"] = ""
        head[f"q({correction.upper()})"] = ""
    if groupby and smd:
        head["SMD"] = ""
    head["欠測"] = ""
    rows.append(head)

    for col in columns:
        s = df[col]
        role = schema.columns[col].role if col in schema.columns else None
        kind = _kind_of(s, role)
        label = _label_of(col, dic, amap)
        c = cmap.get(label)
        nrm = c.normality if (c and c.normality) else normality(s) if kind == CONTINUOUS else None

        if kind == CONTINUOUS:
            suffix = "平均 (SD)" if (nrm and nrm.is_normal) else "中央値 [Q1, Q3]"
            row = {"項目": f"{label}, {suffix}"}
            if overall:
                row["全体"] = _fmt_continuous(s, bool(nrm and nrm.is_normal))
            for lv in levels:
                row[str(lv)] = _fmt_continuous(s[df[groupby] == lv],
                                               bool(nrm and nrm.is_normal))
            _attach(row, c, groupby, pval, smd, correction)
            row["欠測"] = int(s.isna().sum())
            rows.append(row)
        else:
            cat_levels = sorted({str(x) for x in s.dropna().unique()})
            if len(cat_levels) > max_levels:
                # ★水準ごとに 1 行ずつ出してはならない。★
                #   自由記載やカルテ番号のような列がカテゴリとみなされると、
                #   Table 1 が数百行になり、検定（χ²）も意味を持たない。
                #   1 行にまとめ、何が起きたかを言う。
                row = {"項目": f"{label}（{len(cat_levels)} 水準。多すぎるため内訳は省いた）",
                       **_blank(overall, levels, groupby, pval, smd, correction)}
                row["欠測"] = int(s.isna().sum())
                rows.append(row)
                notes_extra.append(
                    f"★'{col}' は {len(cat_levels)} 水準あり、Table 1 の内訳を省いた。★ "
                    f"自由記載や識別子であれば schema で除くこと"
                    f"（max_levels= で閾値を変えられる）")
                continue
            rows.append({"項目": f"{label}, n (%)", **_blank(overall, levels, groupby,
                                                             pval, smd, correction)})
            # 検定の結果は変数の見出し行に載せる（水準ごとに出すと読みにくい）
            _attach(rows[-1], c, groupby, pval, smd, correction)
            rows[-1]["欠測"] = int(s.isna().sum())
            for lv_name in cat_levels:
                r = {"項目": f"　{lv_name}"}
                if overall:
                    v = s.dropna().astype(str)
                    r["全体"] = f"{int((v == lv_name).sum())} ({(v == lv_name).mean():.1%})"
                for lv in levels:
                    v = s[df[groupby] == lv].dropna().astype(str)
                    r[str(lv)] = ("—" if len(v) == 0 else
                                  f"{int((v == lv_name).sum())} ({(v == lv_name).mean():.1%})")
                r.update(_blank(overall, levels, groupby, pval, smd, correction,
                                only_stats=True))
                r["欠測"] = ""
                rows.append(r)

    table = pd.DataFrame(rows).fillna("")
    notes = []
    if groupby and pval:
        notes.append(_P_VALUE_NOTE)
        notes += comp.notes if comp else []
    notes.append("連続変数は正規性の判定に従って 平均(SD) と 中央値[Q1,Q3] を書き分けている。"
                 "判定の根拠は compare_groups() の『判定の根拠』列に残してある。")
    notes += notes_extra
    return TableOneResult(table=table, groups=levels, sizes=sizes, notes=notes,
                          comparison=comp, groupby=groupby)


def _blank(overall, levels, groupby, pval, smd, correction, only_stats=False):
    d = {}
    if not only_stats:
        if overall:
            d["全体"] = ""
        for lv in levels:
            d[str(lv)] = ""
    if groupby and pval:
        d["検定"] = ""
        d["p"] = ""
        d[f"q({correction.upper()})"] = ""
    if groupby and smd:
        d["SMD"] = ""
    return d


def _attach(row, c, groupby, pval, smd, correction):
    if not groupby or c is None:
        return
    if pval:
        row["検定"] = c.test
        row["p"] = _p(c.p)
        row[f"q({correction.upper()})"] = _p(c.q)
    if smd:
        row["SMD"] = _r(c.smd)


# ================================================================== 管理目標
def target_achievement(df: pd.DataFrame, *, by: str | None = None,
                       dic: dict | None = None, colmap: dict | None = None) -> pd.DataFrame:
    """辞書に管理目標がある項目について、達成率を出す。

    列の対応（辞書キー → 列名）は別名表から自動で作る。
    `colmap` を渡せば明示的に指定できる。

    開区間と閉区間を取り違えないこと（`medprep.targets`）。
    P 5.5・補正Ca 9.5・Hb 12.0・iPTH 240 はいずれも**目標範囲に含まれない**ため、
    境界値ちょうどの症例数を必ず併記する。
    """
    dic = dic or load_dict()
    if colmap is None:
        amap = build_alias_map(dic)
        colmap = {}
        for c in df.columns:
            timing, base = detect_timing(c)
            k = amap.get(_norm(base)) or amap.get(_norm(c))
            if k and "target" in dic["items"].get(k, {}) and k not in colmap:
                colmap[k] = c
    from .targets import achievement
    return achievement(df, dic, colmap, by=by)


def target_summary(df: pd.DataFrame, *, dic: dict | None = None) -> pd.DataFrame:
    """データにある項目の管理目標を一覧にする（達成率ではなく目標そのもの）。"""
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    rows = []
    for c in df.columns:
        _t, base = detect_timing(c)
        k = amap.get(_norm(base)) or amap.get(_norm(c))
        spec = dic["items"].get(k) if k else None
        if not spec or "target" not in spec:
            continue
        t = spec["target"]
        v = in_target(df[c], t)
        rows.append({"列": c, "項目": spec["name_ja"], "単位": spec.get("unit", ""),
                     "管理目標": describe_target(t), "評価可能例数": int(v.notna().sum()),
                     "出典": t.get("source", "")})
    return pd.DataFrame(rows)
