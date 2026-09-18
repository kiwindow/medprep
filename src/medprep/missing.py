"""medprep.missing — 欠損の分析。**補完はしない。**

補完は「データから統計量を学習する」処理なので、train でしか fit できない。
したがって補完は `pipeline.py` の `ColumnTransformer` の中にしか置かない。
このモジュールは**欠損を見て、考えるための材料を出すところまで**を担う。

医学データの欠損について
------------------------
1. **「測っていない」こと自体が情報である。**
   重症だから測った／軽症だから測らなかった、という選択が必ず入る。
   したがって欠損指示子（`add_indicator`）を既定で残す。
   指示子を黙って捨てるのは、予後情報を捨てることに等しい。

2. **目的変数の欠損は補完してはならない。**
   補完した目的変数で学習した結果は解釈できない。該当症例を除外し、
   **除外した数を必ず報告する**。`drop_missing_outcome()` がこれを行う。

3. **欠損が群に偏っていれば MCAR ではない。**
   全体の中央値で埋めると、群間差が人工的に作られる。
   `mcar_signals()` が「欠損の有無」と他の列の関連を総当たりで検定し、
   有意な組合せを列挙する（Little の MCAR 検定は仮定が強いので採らない）。

4. **形式Cの生存時間では、空欄が意味を持つ。**
   「イベント発生日が空欄」＝イベントが起きなかった、である。
   欠損率の対象から外す（`quality.audit` も同じ扱いをする）。
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import pandas as pd
from scipy import stats

from .describe import _adjust, _p, group_levels
from .schema import DATETIME, NUMERIC, Schema


# ================================================================== 報告
@dataclass
class MissingReport:
    columns: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0
    n_complete: int = 0
    notes: list = field(default_factory=list)

    @property
    def complete_rate(self) -> float:
        return 0.0 if self.n_rows == 0 else self.n_complete / self.n_rows

    def report(self) -> str:
        lines = [
            f"欠損の分析  {self.n_rows} 行中、欠損がまったく無い行 {self.n_complete} "
            f"（{self.complete_rate:.1%}）",
            "\n列ごと:", self.columns.to_string(index=False),
        ]
        if len(self.patterns):
            lines += ["\n欠損パターン（上位）:", self.patterns.to_string(index=False)]
        if len(self.signals):
            lines += ["\n★欠損が他の列と関連している（MCAR ではない）:",
                      self.signals.to_string(index=False)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    def show(self):
        print(self.report())

    def plot(self, df: pd.DataFrame, kind: str = "matrix", *, save: str | None = None,
             figsize=(10, 5)):
        """missingno による可視化。matrix / bar / heatmap / dendrogram。"""
        with contextlib.suppress(ImportError):
            import matplotlib_fontja  # noqa: F401
        import matplotlib.pyplot as plt
        import missingno as msno
        fn = {"matrix": msno.matrix, "bar": msno.bar,
              "heatmap": msno.heatmap, "dendrogram": msno.dendrogram}.get(kind)
        if fn is None:
            raise ValueError("kind は matrix / bar / heatmap / dendrogram のいずれか")
        cols = [c for c in self.columns["列"] if c in df.columns]
        ax = fn(df[cols], figsize=figsize)
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
        # missingno の図は tight_layout と相性が悪い軸を持つ。警告を出させない。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig


# ================================================================== 分析
def analyze(
    df: pd.DataFrame,
    schema: Schema | None = None,
    *,
    columns: list | None = None,
    group: str | None = None,
    outcome: str | None = None,
    top_patterns: int = 20,
    signal_alpha: float = 0.001,
) -> MissingReport:
    """欠損を列ごと・症例ごと・パターンごとに見る。**データは変更しない。**"""
    cols = _target_columns(df, schema, columns)
    rep = MissingReport(n_rows=len(df))
    if not cols:
        rep.notes.append("対象の列が無い")
        return rep

    sub = df[cols]
    na = sub.isna()
    rep.n_complete = int((~na.any(axis=1)).sum())

    rows = []
    for c in cols:
        n_na = int(na[c].sum())
        rows.append({
            "列": c, "欠損": n_na, "欠損率": f"{n_na / max(len(df), 1):.1%}",
            "非欠損": int(len(df) - n_na),
            "型": ("数値" if pd.api.types.is_numeric_dtype(df[c]) else "カテゴリ"),
            "水準/範囲": _range_text(df[c]),
        })
    rep.columns = pd.DataFrame(rows).sort_values("欠損", ascending=False)

    rowmiss = na.sum(axis=1)
    rep.rows = (pd.DataFrame({"欠損している列数": rowmiss.value_counts().sort_index().index,
                              "症例数": rowmiss.value_counts().sort_index().to_numpy()})
                .assign(割合=lambda d: (d["症例数"] / max(len(df), 1)).map("{:.1%}".format)))

    # --- 欠損パターン（どの列が同時に欠けるか）
    if na.any().any():
        key = na.apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
        vc = key.value_counts().head(top_patterns)
        rep.patterns = pd.DataFrame({
            "パターン": vc.index,
            "症例数": vc.to_numpy(),
            "割合": [f"{v / max(len(df), 1):.1%}" for v in vc.to_numpy()],
            "欠損している列": ["、".join(c for c, f in zip(cols, k) if f == "1") or "（欠損なし）"
                        for k in vc.index],
        })

    rep.signals = mcar_signals(df, cols, alpha=signal_alpha, group=group)

    # --- 注記
    high = rep.columns[rep.columns["欠損"] / max(len(df), 1) >= 0.5]
    if len(high):
        rep.notes.append(
            f"欠損率 50% 以上の列がある（{list(high['列'])}）。"
            f"半分以上を補完で埋めた列は、実質的に補完アルゴリズムの出力である。"
            f"列ごと外すか、『測ったかどうか』の 2 値として扱うことを検討すること")
    if len(rep.signals):
        rep.notes.append(
            "欠損が他の列と関連している（MCAR ではない）。全体の中央値で埋めると"
            "群間差が人工的に作られる。多重代入（MICE）か、群別の補完を検討すること")
    rep.notes.append("**補完はこのモジュールでは行わない。** "
                     "補完は統計量を学習する処理なので、train でしか fit できない。"
                     "pipeline.build_preprocessor() の中で行う")
    rep.notes.append("欠損指示子（add_indicator）は既定で残す。"
                     "『測っていない』こと自体が医学では情報である")

    if outcome and outcome in df.columns:
        n = int(df[outcome].isna().sum())
        if n:
            rep.notes.append(
                f"★目的変数 '{outcome}' が {n} 例で欠損している。★ "
                f"目的変数は補完してはならない。drop_missing_outcome() で除外すること")
    return rep


def mcar_signals(df: pd.DataFrame, columns: list | None = None, *,
                 alpha: float = 0.001, group: str | None = None,
                 min_rate: float = 0.02) -> pd.DataFrame:
    """「欠損の有無」が他の列と関連しているかを総当たりで検定する。

    欠損が完全にランダム（MCAR）なら、欠損フラグはどの列とも関連しないはずである。
    関連があれば MAR 以上であり、単純補完は分布を歪める。

    Little の MCAR 検定は多変量正規を仮定するため採らない。
    ここでの目的は「検定に通ったかどうか」ではなく、
    **どの列の欠損がどの列と結びついているかを名指しすること**である。
    """
    cols = columns or list(df.columns)
    tested = []
    for miss_col in cols:
        m = df[miss_col].isna()
        if not (min_rate < m.mean() < 1 - min_rate):
            continue
        for other in cols:
            if other == miss_col:
                continue
            s = df[other]
            if s.isna().all():
                continue
            try:
                if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10:
                    a = pd.to_numeric(s[m], errors="coerce").dropna()
                    b = pd.to_numeric(s[~m], errors="coerce").dropna()
                    if len(a) < 5 or len(b) < 5:
                        continue
                    r = stats.mannwhitneyu(a, b, alternative="two-sided")
                    stat, p, test = float(r.statistic), float(r.pvalue), "Mann-Whitney U"
                    detail = f"欠損あり 中央値 {a.median():.3g} / 欠損なし {b.median():.3g}"
                else:
                    if s.nunique(dropna=True) > 20:
                        continue
                    t = pd.crosstab(s.astype(str), m)
                    t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
                    if t.shape[0] < 2 or t.shape[1] < 2:
                        continue
                    chi2, p, _dof, _e = stats.chi2_contingency(t)
                    stat, test = float(chi2), "χ²"
                    rate = m.groupby(s.astype(str)).mean().sort_values(ascending=False)
                    detail = "、".join(f"{k}={v:.0%}" for k, v in rate.head(4).items())
            except Exception:                                      # noqa: BLE001
                continue
            tested.append({"欠損した列": miss_col, "関連する列": other, "検定": test,
                           "統計量": round(stat, 3), "p": p, "内訳": detail})
    out = pd.DataFrame(tested)
    if not len(out):
        return out
    out["q(BH)"] = _adjust(out["p"].to_numpy(), "bh")
    out = out[out["q(BH)"] < alpha].copy()
    if len(out):
        out["p"] = out["p"].map(_p)
        out["q(BH)"] = out["q(BH)"].map(_p)
        if group:
            out = pd.concat([out[out["関連する列"] == group], out[out["関連する列"] != group]])
    return out.reset_index(drop=True)


def drop_missing_outcome(df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, dict]:
    """目的変数が欠損している症例を除く。**補完しない。**

    Returns
    -------
    (残ったデータ, {"除外": n, "残り": n, "理由": ...})
    """
    if outcome not in df.columns:
        raise ValueError(f"目的変数 '{outcome}' がデータに無い")
    miss = df[outcome].isna()
    info = {
        "除外": int(miss.sum()), "残り": int((~miss).sum()),
        "除外率": f"{miss.mean():.1%}",
        "理由": f"目的変数 '{outcome}' が欠損。**目的変数は補完してはならない**"
                f"（補完した目的変数で学習した結果は解釈できない）",
    }
    return df[~miss].copy(), info


def missingness_by_group(df: pd.DataFrame, group: str,
                         columns: list | None = None) -> pd.DataFrame:
    """群ごとの欠損率。施設差の把握に使う。"""
    if group not in df.columns:
        raise ValueError(f"群分け列 '{group}' がデータに無い")
    cols = [c for c in (columns or df.columns) if c != group and c in df.columns]
    levels = group_levels(df[group])
    rows = []
    for c in cols:
        row = {"列": c, "全体": f"{df[c].isna().mean():.1%}"}
        for lv in levels:
            row[str(lv)] = f"{df.loc[df[group] == lv, c].isna().mean():.1%}"
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================== 補助
def _target_columns(df, schema, columns) -> list:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    if schema is None:
        return list(df.columns)
    sv = schema.survival or {}
    # 形式C の日付列は「空欄が意味を持つ」ので欠損率の対象から外す
    skip = {sv.get("event_date"), sv.get("censor_date")} - {None} if sv.get("form") == "C" else set()
    return [c for c in schema.kept() if c in df.columns and c not in skip]


def _range_text(s: pd.Series) -> str:
    v = s.dropna()
    if len(v) == 0:
        return "—"
    if pd.api.types.is_numeric_dtype(v):
        return f"{float(v.min()):.3g} – {float(v.max()):.3g}"
    if pd.api.types.is_datetime64_any_dtype(v):
        return f"{v.min()} – {v.max()}"
    n = v.nunique()
    return f"{n} 水準"


def is_datetime_role(schema: Schema, col: str) -> bool:
    return col in schema.columns and schema.columns[col].role == DATETIME


def numeric_columns(schema: Schema) -> list:
    return [c for c, s in schema.columns.items() if s.role == NUMERIC and s.action == "keep"]
