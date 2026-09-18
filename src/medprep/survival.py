"""medprep.survival — Kaplan-Meier・log-rank・Cox 回帰・比例ハザードの検定。

`survival_input` が「日付から (duration, event) を作る」ところまでを担い、
ここはその先の解析を担う。

このモジュールが見張っていること
--------------------------------
1. **欠測による症例の脱落を黙って起こさない。**
   Cox 回帰は共変量に欠測があると、その症例を**黙って捨てる**。
   共変量を 1 本足しただけで n が 578 から 401 に減り、
   それに気づかないまま「ハザード比が変わった」と解釈するのは、
   実データの解析で最も多い事故のひとつである。
   medprep は fit の前に**何例・何イベントが落ちるか**を数えて報告する。

2. **EPV（イベント数 ÷ 共変量数）。**
   10 を下回る多変量 Cox は過学習が強く疑われる。警告を出す。

3. **比例ハザード仮定。**
   Schoenfeld 残差で検定し、違反があれば**どうすればよいか**を言う
   （層別化するか、時間依存項を入れるか、RMST に切り替えるか）。
   「p < 0.05 でした」だけでは何の役にも立たない。

4. **生存期間中央値に到達しない群。**
   打ち切りが多ければ中央値は出ない。medprep は NaN ではなく **NR**（not reached）
   と表示し、「追跡期間が足りない」という事実として扱う。

やらないこと
------------
**逐次変数選択（stepwise）を既定にしない。**
AIC や p 値による逐次選択は、選ばれた変数の p 値・信頼区間・ハザード比を
すべて楽観方向に歪める（選択後推論の問題）。既定は
「単変量で p < 0.10 の変数を多変量に入れる」であり、これも選択であることを
`notes` に明示する。臨床的に重要な変数は p 値によらず `covariates=` で明示的に入れる。
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.statistics import (
    logrank_test,
    multivariate_logrank_test,
    pairwise_logrank_test,
    proportional_hazard_test,
)
from lifelines.utils import median_survival_times, restricted_mean_survival_time

from .describe import _adjust, _p, _r, group_levels
from .schema import BINARY, GROUP, NOMINAL, NUMERIC, ORDINAL, Schema
from .survival_input import SurvivalFrame

NR = "NR"           # not reached（生存期間中央値に到達しなかった）
EPV_MIN = 10


def _fontja():
    """日本語フォントを一度だけ有効にする。図に豆腐（□）を出さない。"""
    with contextlib.suppress(ImportError):
        import matplotlib_fontja  # noqa: F401


# ================================================================== KM
@dataclass
class KMResult:
    table: pd.DataFrame
    fitters: dict = field(default_factory=dict)
    by: str | None = None
    unit: str = "days"
    notes: list = field(default_factory=list)
    figure: object = None

    def report(self) -> str:
        lines = [f"Kaplan-Meier 推定（単位: {self.unit}）", self.table.to_string(index=False)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)


# ================================================================== log-rank
@dataclass
class LogRankResult:
    test: str
    statistic: float
    p: float
    df: int
    by: str | None = None
    pairwise: pd.DataFrame | None = None
    trend: dict | None = None
    correction: str = "holm"
    notes: list = field(default_factory=list)

    def report(self) -> str:
        lines = [f"{self.test}: χ² = {self.statistic:.3f}（df={self.df}）, p = {_p(self.p)}"]
        if self.trend:
            t = self.trend
            lines.append(f"傾向検定（log-rank trend、スコア {t['scores']}）: "
                         f"χ² = {t['statistic']:.3f}（df=1）, p = {_p(t['p'])}")
        if self.pairwise is not None and len(self.pairwise):
            lines += [f"対比較（{self.correction.upper()} 補正）:",
                      self.pairwise.to_string(index=False)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)


# ================================================================== Cox
@dataclass
class CoxResult:
    model: CoxPHFitter | None = None
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    univariate: pd.DataFrame = field(default_factory=pd.DataFrame)
    ph: pd.DataFrame = field(default_factory=pd.DataFrame)
    covariates: list = field(default_factory=list)
    reference_levels: dict = field(default_factory=dict)
    strata: list = field(default_factory=list)
    fit_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    n: int = 0
    n_events: int = 0
    n_dropped: int = 0
    events_dropped: int = 0
    c_index: float = float("nan")
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    failed: str = ""

    @property
    def epv(self) -> float:
        k = len(self.covariates)
        return float("inf") if k == 0 else self.n_events / k

    @property
    def ph_violations(self) -> list:
        if self.ph is None or not len(self.ph):
            return []
        return [i for i, p in zip(self.ph.index, self.ph["p"]) if p < 0.05]

    def report(self) -> str:
        if self.failed:
            return f"Cox 回帰は実行できなかった: {self.failed}"
        lines = [
            f"Cox 比例ハザード回帰  n = {self.n}、イベント = {self.n_events}、"
            f"共変量 = {len(self.covariates)}、EPV = {self.epv:.1f}"
            + ("  ★EPV < 10★" if self.epv < EPV_MIN else ""),
        ]
        if self.n_dropped:
            lines.append(f"欠測により除外: {self.n_dropped} 例"
                         f"（うちイベント {self.events_dropped} 件）")
        if len(self.univariate):
            lines += ["\n単変量スクリーニング:", self.univariate.to_string(index=False)]
        lines += ["\n多変量:", self.summary.to_string()]
        lines.append(f"\nC-index = {self.c_index:.3f}")
        if len(self.ph):
            lines += ["\n比例ハザード仮定の検定（Schoenfeld 残差）:",
                      self.ph.to_string()]
        for w in self.warnings:
            lines.append(f"[警告] {w}")
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)


# ================================================================== 本体
class Survival:
    """生存時間解析の入口。

        sf = mp.build_survival(df, id_col=..., start_date=..., event_date=..., censor_date=...)
        s  = mp.Survival.from_survival_frame(sf)
        s.km(by="施設")
        s.logrank(by="施設")
        s.cox(covariates="auto")
    """

    def __init__(self, data: pd.DataFrame, time: str = "duration", event: str = "event",
                 *, unit: str = "days", schema: Schema | None = None):
        for c in (time, event):
            if c not in data.columns:
                raise ValueError(f"列 '{c}' がデータに無い")
        self.data = data.copy()
        self.time, self.event, self.unit = time, event, unit
        self.schema = schema
        self.notes: list = []

        t = pd.to_numeric(self.data[time], errors="coerce")
        e = pd.to_numeric(self.data[event], errors="coerce")
        bad_e = e.notna() & ~e.isin([0, 1])
        if bad_e.any():
            raise ValueError(f"イベント列 '{event}' に 0/1 以外の値が {int(bad_e.sum())} 件ある")
        bad_t = t.notna() & (t <= 0)
        if bad_t.any():
            self.notes.append(f"観察期間が 0 以下の症例が {int(bad_t.sum())} 件ある。"
                              f"build_survival の zero_duration= で扱いを決めること")
        drop = t.isna() | e.isna()
        if drop.any():
            self.notes.append(f"観察期間またはイベントが欠測の {int(drop.sum())} 例を除いた")
            self.data = self.data[~drop]
        self.data[time] = pd.to_numeric(self.data[time], errors="coerce")
        self.data[event] = pd.to_numeric(self.data[event], errors="coerce").astype(int)

    # -------------------------------------------------------------- 入口
    @classmethod
    def from_survival_frame(cls, sf: SurvivalFrame, schema: Schema | None = None) -> Survival:
        return cls(sf.data, time=sf.duration_col, event=sf.event_col,
                   unit=sf.unit, schema=schema)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def n_events(self) -> int:
        return int(self.data[self.event].sum())

    def summary(self) -> pd.DataFrame:
        d = self.data[self.time]
        km = KaplanMeierFitter().fit(d, self.data[self.event])
        med, lo, hi = _median_ci(km)
        return pd.DataFrame({
            "項目": ["解析対象例数", "イベント数", "打ち切り数", "イベント率",
                     f"生存期間中央値 [{self.unit}]", f"観察期間中央値 [{self.unit}]",
                     f"観察期間 最小–最大 [{self.unit}]"],
            "値": [self.n, self.n_events, self.n - self.n_events,
                   f"{0.0 if self.n == 0 else self.n_events / self.n:.1%}",
                   _fmt_median(med, lo, hi),
                   f"{_reverse_km_median(self.data[self.time], self.data[self.event]):.1f}",
                   f"{d.min():.1f} – {d.max():.1f}"],
        })

    # -------------------------------------------------------------- KM
    def km(self, by: str | None = None, *, ci="auto", at_risk: bool = True,
           at_risk_rows=("At risk",), ax=None, title: str | None = None,
           save: str | None = None, figsize=None, censors: bool = True) -> KMResult:
        """Kaplan-Meier 曲線と、群ごとの生存期間中央値（95%CI）。

        at-risk 表は既定で「At risk」の行だけを出す。lifelines の既定は
        At risk / Censored / Events の 3 行で、4 群なら 12 行になり、
        **曲線が細い帯に潰れて読めなくなる。**
        """
        _fontja()
        import matplotlib.pyplot as plt

        d, e = self.data[self.time], self.data[self.event]
        groups = {"全体": slice(None)} if by is None else \
            {lv: (self.data[by] == lv) for lv in group_levels(self.data[by])}

        # 信頼区間の帯は、3 群以上だと重なって曇りガラスになる。
        # 既定では 2 群までで出す（ci=True / False で明示的に上書きできる）。
        if ci == "auto":
            ci = len(groups) <= 2
        # at-risk 表は群の数だけ縦に伸びる。図の高さを群数に合わせて広げる。
        if figsize is None:
            extra = (len(groups) * len(at_risk_rows) * 0.28) if at_risk else 0.0
            figsize = (8, 4.6 + extra)
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        rows, fitters, not_reached = [], {}, []
        for name, mask in groups.items():
            dd = d if mask is slice(None) else d[mask]
            ee = e if mask is slice(None) else e[mask]
            if len(dd) == 0:
                continue
            k = KaplanMeierFitter(label=f"{name} (n={len(dd)})").fit(dd, ee)
            fitters[name] = k
            k.plot_survival_function(ax=ax, ci_show=ci, show_censors=censors,
                                      censor_styles={"marker": "|", "ms": 5, "alpha": 0.6})
            med, lo, hi = _median_ci(k)
            if not np.isfinite(med):
                not_reached.append(name)
            rows.append({
                "群": name, "n": len(dd), "イベント": int(ee.sum()),
                "打ち切り": int(len(dd) - ee.sum()),
                f"生存期間中央値 [{self.unit}]": _fmt_median(med, lo, hi),
                f"追跡期間中央値 [{self.unit}]": f"{_reverse_km_median(dd, ee):.1f}",
                "1年生存率": _at(k, 365.25 / _unit_days(self.unit)),
                "3年生存率": _at(k, 3 * 365.25 / _unit_days(self.unit)),
            })

        ax.set_xlabel(f"観察期間（{_unit_ja(self.unit)}）")
        ax.set_ylabel("生存率")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(title or ("Kaplan-Meier 曲線" if by is None else f"{by} 別の生存曲線"))
        if at_risk and len(fitters):
            from lifelines.plotting import add_at_risk_counts
            add_at_risk_counts(*fitters.values(), ax=ax,
                               rows_to_show=list(at_risk_rows))
        if fig is not None:
            fig.tight_layout()
        if save:
            (fig or ax.figure).savefig(save, dpi=150, bbox_inches="tight")

        res = KMResult(table=pd.DataFrame(rows), fitters=fitters, by=by,
                       unit=self.unit, figure=fig or ax.figure)
        # ★中央値そのものが出なかった群だけを挙げる。★
        #   信頼区間の上限が NR なのは普通にあることで、別の話である。
        nr = not_reached
        if nr:
            res.notes.append(
                f"生存期間中央値に到達しなかった群がある（{nr}）。NR と表示している。"
                f"これは『生存が良い』というより『追跡期間が足りない』という事実である")
        if not ci and len(groups) > 2:
            res.notes.append(f"群が {len(groups)} つあるため信頼区間の帯は描いていない"
                             f"（重なって読めなくなる）。ci=True で描ける")
        res.notes.append("追跡期間中央値は reverse Kaplan-Meier 法（イベントと打ち切りを"
                         "入れ替えて推定）で出している。生存期間中央値と混同しないこと")
        return res

    # -------------------------------------------------------------- log-rank
    def logrank(self, by: str, *, pairwise: bool = True, correction: str = "holm",
                trend: bool | None = None, order: list | None = None) -> LogRankResult:
        """群間の生存曲線の差を検定する。

        2 群なら log-rank、3 群以上なら多群 log-rank。
        3 群以上では対比較を Holm 補正つきで併記し、
        群に順序があるなら傾向検定（log-rank trend）も出す。
        """
        if by not in self.data.columns:
            raise ValueError(f"群分け列 '{by}' がデータに無い")
        g = self.data[by]
        levels = order if order is not None else group_levels(g)
        levels = [lv for lv in levels if (g == lv).sum() > 0]
        if len(levels) < 2:
            raise ValueError(f"'{by}' の水準が {len(levels)} 個しかない")

        d, e = self.data[self.time], self.data[self.event]
        small = [lv for lv in levels if int((g == lv).sum()) < 10]
        notes = []
        if small:
            notes.append(f"例数 10 未満の群がある（{small}）。log-rank の近似は不安定になる")
        zero_ev = [lv for lv in levels if int(e[g == lv].sum()) == 0]
        if zero_ev:
            notes.append(f"イベントが 1 件も無い群がある（{zero_ev}）。"
                         f"その群を含む比較は解釈できない")

        if len(levels) == 2:
            a, b = (g == levels[0]), (g == levels[1])
            r = logrank_test(d[a], d[b], e[a], e[b])
            res = LogRankResult("log-rank 検定", float(r.test_statistic), float(r.p_value), 1,
                                by=by, notes=notes)
        else:
            r = multivariate_logrank_test(d, g, e)
            res = LogRankResult(f"多群 log-rank 検定（{len(levels)} 群）",
                                float(r.test_statistic), float(r.p_value),
                                len(levels) - 1, by=by, correction=correction, notes=notes)
            if pairwise:
                pr = pairwise_logrank_test(d, g, e)
                rows = []
                for (a, b) in _pairs(pr):
                    rows.append({"群1": a, "群2": b,
                                 "χ²": float(pr.summary.loc[(a, b), "test_statistic"]),
                                 "p": float(pr.summary.loc[(a, b), "p"])})
                pw = pd.DataFrame(rows)
                if len(pw):
                    pw["p補正"] = _adjust(pw["p"].to_numpy(), correction)
                    pw["補正法"] = correction.upper()
                res.pairwise = pw
                notes.append(f"対比較は {len(pw)} 通りある。補正せずに並べると、"
                             f"どれか 1 つが偶然 p<0.05 になる確率が跳ね上がる")

        want_trend = trend if trend is not None else (len(levels) >= 3)
        if want_trend:
            scores = list(range(len(levels)))
            st, pt = logrank_trend(d, g, e, levels=levels, scores=scores)
            res.trend = {"statistic": st, "p": pt,
                         "scores": dict(zip(map(str, levels), scores))}
            notes.append("傾向検定は群に順序があるときにだけ意味を持つ（病期・stage など）。"
                         "順序の無い群（施設名など）に当てはめないこと。"
                         "順序は order= で明示できる")
        return res

    # -------------------------------------------------------------- RMST
    def rmst(self, by: str | None = None, t: float | None = None) -> pd.DataFrame:
        """制限付き平均生存時間（RMST）。

        **比例ハザードが成り立たないときの逃げ道。**
        ハザード比は「差が時間によらず一定」を前提にするが、RMST は
        「時点 t までの平均生存時間」なので前提が要らず、単位も直感的（年・日）である。
        """
        tau = t if t is not None else float(self.data[self.time].max())
        d, e = self.data[self.time], self.data[self.event]
        groups = {"全体": slice(None)} if by is None else \
            {lv: (self.data[by] == lv) for lv in group_levels(self.data[by])}
        rows = []
        for name, mask in groups.items():
            dd = d if mask is slice(None) else d[mask]
            ee = e if mask is slice(None) else e[mask]
            if len(dd) == 0:
                continue
            k = KaplanMeierFitter().fit(dd, ee)
            v = restricted_mean_survival_time(k, t=tau)
            rows.append({"群": name, "n": len(dd), f"τ [{self.unit}]": round(tau, 2),
                         f"RMST [{self.unit}]": round(float(v), 3)})
        out = pd.DataFrame(rows)
        if by is not None and len(out) == 2:
            out.attrs["差"] = float(out.iloc[0, -1] - out.iloc[1, -1])
        return out

    # -------------------------------------------------------------- Cox
    def cox(self, covariates="auto", *, strata=None, screen_p: float = 0.10,
            penalizer: float = 0.0, check_ph: bool = True) -> CoxResult:
        """Cox 比例ハザード回帰。

        Parameters
        ----------
        covariates : "auto" か 列名のリスト
            "auto" なら schema の説明変数を単変量でスクリーニングし、
            p < `screen_p` のものを多変量に入れる。**逐次選択は使わない。**
        strata : 層別化する列（比例ハザードが成り立たない変数の逃げ道）
        """
        res = CoxResult(strata=[strata] if isinstance(strata, str) else list(strata or []))
        cands = self._candidates(covariates)
        if not cands:
            res.failed = "共変量が 1 つも無い（schema を確認するか covariates= で指定すること）"
            return res

        design, refs, dropped_cols, notes = self._design(cands)
        res.reference_levels = refs
        res.notes += notes
        if dropped_cols:
            res.warnings.append(f"使えない共変量を外した: {dropped_cols}")
        if design.empty or not len(design.columns):
            res.failed = "使える共変量が残らなかった"
            return res

        # --- 単変量スクリーニング
        uni_rows, keep = [], []
        for c in design.columns:
            sub = pd.concat([self.data[[self.time, self.event]], design[c]], axis=1).dropna()
            if sub[c].nunique() < 2 or sub[self.event].sum() < 2:
                uni_rows.append({"変数": c, "n": len(sub), "HR": np.nan, "95%CI下限": np.nan,
                                 "95%CI上限": np.nan, "p": np.nan, "判定": "算出不可"})
                continue
            try:
                m = CoxPHFitter(penalizer=penalizer).fit(sub, self.time, self.event)
                r = m.summary.loc[c]
                p = float(r["p"])
                uni_rows.append({"変数": c, "n": len(sub), "HR": float(r["exp(coef)"]),
                                 "95%CI下限": float(r["exp(coef) lower 95%"]),
                                 "95%CI上限": float(r["exp(coef) upper 95%"]),
                                 "p": p,
                                 "判定": "多変量へ" if p < screen_p else f"p ≥ {screen_p}"})
                if p < screen_p:
                    keep.append(c)
            except (ConvergenceError, ValueError) as e:              # noqa: BLE001
                uni_rows.append({"変数": c, "n": len(sub), "HR": np.nan, "95%CI下限": np.nan,
                                 "95%CI上限": np.nan, "p": np.nan,
                                 "判定": f"収束せず（{type(e).__name__}）"})
        res.univariate = pd.DataFrame(uni_rows)

        if covariates == "auto":
            use = keep or list(design.columns)
            if not keep:
                res.warnings.append(
                    f"単変量で p < {screen_p} の変数が 1 つも無かったため、"
                    f"全変数で多変量を組んだ。結果は探索的なものとして扱うこと")
            res.notes.append(
                f"共変量は「単変量で p < {screen_p}」で選んだ。"
                f"**逐次選択（stepwise）は使っていない**。逐次選択は選ばれた変数の "
                f"p 値・信頼区間・ハザード比をすべて楽観方向に歪める。"
                f"臨床的に重要な変数は p 値によらず covariates= で明示的に入れること")
        else:
            use = list(design.columns)
        res.covariates = use

        # --- 欠測による脱落を数える（★黙って減らさない★）
        fit_df = pd.concat([self.data[[self.time, self.event] + res.strata], design[use]],
                           axis=1)
        fit_df = fit_df.loc[:, ~fit_df.columns.duplicated()]
        before_n, before_e = len(fit_df), int(fit_df[self.event].sum())
        fit_df = fit_df.dropna()
        res.n, res.n_events = len(fit_df), int(fit_df[self.event].sum())
        res.n_dropped = before_n - res.n
        res.events_dropped = before_e - res.n_events
        if res.n_dropped:
            worst = (self.data[use].isna().sum().sort_values(ascending=False)
                     if all(c in self.data.columns for c in use)
                     else design[use].isna().sum().sort_values(ascending=False))
            top = "、".join(f"{k}: {int(v)}" for k, v in worst.head(3).items() if v > 0)
            res.warnings.append(
                f"共変量の欠測により {res.n_dropped} 例（イベント {res.events_dropped} 件）が"
                f"解析から落ちた（{before_n} → {res.n}）。欠測の多い列: {top}。"
                f"Cox は完全ケースだけで推定するので、**共変量を足すと n が黙って減る**。"
                f"補完するか、その変数を外すかを決めること")
        if res.n_events == 0:
            res.failed = "欠測を除いたあとにイベントが 1 件も残らなかった"
            return res
        if res.epv < EPV_MIN:
            res.warnings.append(
                f"EPV = {res.epv:.1f} < {EPV_MIN}（イベント {res.n_events} 件 / "
                f"共変量 {len(use)} 本）。過学習が強く疑われる。共変量を減らすか、"
                f"罰則つき Cox（penalizer=）を検討すること")

        # --- 多変量
        #   lifelines は完全分離や共線性を ConvergenceWarning で知らせる。
        #   ★警告は標準エラーに流れて消える。★ 結果に取り込んで必ず読ませる。
        try:
            model = CoxPHFitter(penalizer=penalizer, strata=res.strata or None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(fit_df, self.time, self.event)
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    res.warnings.append("lifelines からの収束警告: "
                                        + _first_sentence(str(w.message)))
        except ConvergenceError as e:
            res.failed = (f"収束しなかった（{e}）。共変量どうしが強く相関しているか、"
                          f"ある水準でイベントが 0 件（完全分離）の可能性がある。"
                          f"audit の多重共線性の所見と、群ごとのイベント数を確認すること")
            return res
        res.model = model
        res.fit_frame = fit_df
        s = model.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%",
                           "coef", "se(coef)", "p"]].copy()
        s.columns = ["HR", "95%CI下限", "95%CI上限", "係数", "標準誤差", "p"]
        res.summary = s.round(4)
        res.c_index = float(model.concordance_index_)

        wide = s[(s["95%CI上限"] > 100) | (s["95%CI下限"] < 0.01)]
        if len(wide):
            res.warnings.append(
                f"信頼区間が極端に広い変数がある（{list(wide.index)}）。"
                f"完全分離か、水準あたりのイベントが少なすぎる兆候である")

        if check_ph:
            res.ph = self.check_ph(res)
            if res.ph_violations:
                res.warnings.append(
                    f"比例ハザード仮定に違反している変数がある（{res.ph_violations}）。"
                    f"ハザード比が『時間によらず一定』でないので、1 つの HR で要約できない。"
                    f"(1) その変数で層別する cox(strata='...')、"
                    f"(2) 時間依存項を入れる、(3) RMST に切り替える rmst(by='...')、"
                    f"のいずれかを選ぶこと")
        return res

    def check_ph(self, cox: CoxResult, time_transform: str = "rank") -> pd.DataFrame:
        """Schoenfeld 残差による比例ハザード仮定の検定。

        検定には **fit に使ったそのままの行列** を渡す。作り直すと、
        ダミーの基準水準や欠測による脱落がずれて別のモデルを検定してしまう。
        """
        if cox.model is None or not len(cox.fit_frame):
            return pd.DataFrame()
        try:
            r = proportional_hazard_test(cox.model, cox.fit_frame,
                                         time_transform=time_transform)
        except Exception as e:                                      # noqa: BLE001
            return pd.DataFrame({"エラー": [str(e)]})
        out = r.summary[["test_statistic", "p"]].copy()
        out.columns = ["χ²", "p"]
        out["判定"] = ["★違反（p<0.05）" if p < 0.05 else "違反なし" for p in out["p"]]
        return out.round(4)

    # -------------------------------------------------------------- 図
    def forest(self, cox: CoxResult, *, ax=None, save: str | None = None,
               title: str = "多変量 Cox 回帰（ハザード比と 95%CI）", figsize=(7.5, None)):
        """ハザード比のフォレストプロット（対数軸）。"""
        if cox.model is None or not len(cox.summary):
            raise ValueError("Cox が実行できていないため描けない")
        _fontja()
        import matplotlib.pyplot as plt

        s = cox.summary.iloc[::-1]
        h = figsize[1] or max(2.4, 0.45 * len(s) + 1.4)
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(figsize[0], h))
        y = np.arange(len(s))
        ax.errorbar(s["HR"], y,
                    xerr=[s["HR"] - s["95%CI下限"], s["95%CI上限"] - s["HR"]],
                    fmt="o", color="#1f3b57", ecolor="#7f9db9", capsize=3, lw=1.6)
        ax.axvline(1.0, color="#999", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s.index)
        ax.set_xscale("log")
        # ★対数軸の既定の目盛（4×10⁻¹）は読めない。ハザード比は素の数で見せる。★
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0), numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(1, 10) * 0.1, numticks=24))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
        ax.set_xlabel("ハザード比（対数軸）")
        ax.set_title(title)
        for i, (_k, r) in enumerate(s.iterrows()):
            ax.text(1.02, i, f"  {r['HR']:.2f} [{r['95%CI下限']:.2f}–{r['95%CI上限']:.2f}]"
                             f"  {_p_label(r['p'])}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_xlim(left=max(min(s["95%CI下限"].min() * 0.8, 0.9), 1e-3))
        if fig is not None:
            fig.tight_layout()
        if save:
            (fig or ax.figure).savefig(save, dpi=150, bbox_inches="tight")
        return fig or ax.figure

    def schoenfeld_plot(self, cox: CoxResult, *, save: str | None = None, figsize=(10, None)):
        """Schoenfeld 残差の図。**違反があると言われたら、まず形を見る。**"""
        if cox.model is None:
            raise ValueError("Cox が実行できていない")
        _fontja()
        import matplotlib.pyplot as plt
        df = cox.fit_frame
        r = proportional_hazard_test(cox.model, df, time_transform="rank")
        k = len(cox.covariates)
        cols = min(3, k)
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0],
                                                      figsize[1] or 3.0 * rows),
                                 squeeze=False)
        for i, c in enumerate(cox.covariates):
            ax = axes[i // cols][i % cols]
            try:
                cox.model.compute_residuals(df, kind="scaled_schoenfeld")[c].pipe(
                    lambda s, ax=ax: ax.scatter(np.arange(len(s)), s.to_numpy(), s=8,
                                                color="#1f3b57", alpha=0.6))
                ax.axhline(0, color="#999", ls="--", lw=1)
            except Exception as e:                                  # noqa: BLE001
                ax.text(0.5, 0.5, f"描けない: {e}", ha="center", va="center")
            p = float(r.summary.loc[c, "p"]) if c in r.summary.index else float("nan")
            ax.set_title(f"{c}  {_p_label(p)}" + ("  ★違反" if p < 0.05 else ""), fontsize=10)
            ax.set_xlabel("イベント時刻の順位")
            ax.set_ylabel("Schoenfeld 残差")
        for j in range(k, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    # -------------------------------------------------------------- 補助
    def _candidates(self, covariates) -> list:
        if covariates != "auto":
            miss = [c for c in covariates if c not in self.data.columns]
            if miss:
                raise ValueError(f"共変量がデータに無い: {miss}")
            return list(covariates)
        if self.schema is not None:
            return [c for c in self.schema.kept()
                    if c not in (self.time, self.event)
                    and self.schema.columns[c].role in
                    (NUMERIC, BINARY, ORDINAL, NOMINAL, GROUP)]
        return [c for c in self.data.columns
                if c not in (self.time, self.event)
                and (pd.api.types.is_numeric_dtype(self.data[c])
                     or self.data[c].nunique(dropna=True) <= 10)]

    def _design(self, cands: list):
        """共変量を数値の設計行列にする。

        カテゴリはダミー化し、**どの水準を基準にしたかを必ず記録する。**
        基準が分からなければハザード比は読めない。
        """
        cols, refs, dropped, notes = {}, {}, [], []
        for c in cands:
            if c not in self.data.columns:
                dropped.append(f"{c}（データに無い）")
                continue
            s = self.data[c]
            nu = s.nunique(dropna=True)
            if nu < 2:
                dropped.append(f"{c}（値が 1 種類）")
                continue
            if pd.api.types.is_numeric_dtype(s):
                if float(pd.to_numeric(s, errors="coerce").std(ddof=0) or 0) == 0:
                    dropped.append(f"{c}（分散 0）")
                    continue
                cols[c] = pd.to_numeric(s, errors="coerce")
                continue
            if nu > 10:
                dropped.append(f"{c}（水準 {nu} 個は多すぎる）")
                continue
            levels = group_levels(s)
            ref = max(levels, key=lambda lv: (s == lv).sum())     # 最頻水準を基準にする
            refs[c] = str(ref)
            for lv in levels:
                if lv == ref:
                    continue
                name = f"{c}={lv}"
                cols[name] = (s.astype(str) == str(lv)).astype(float).where(s.notna())
            notes.append(f"'{c}' をダミー化した（基準 = '{ref}'。"
                         f"ハザード比は基準水準との比である）")
        return pd.DataFrame(cols, index=self.data.index), refs, dropped, notes

    # -------------------------------------------------------------- 報告
    def report(self, by: str | None = None, covariates="auto") -> str:
        lines = ["生存時間解析", self.summary().to_string(index=False)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        if by:
            lines += ["", self.km(by=by).report(), "", self.logrank(by=by).report()]
        lines += ["", self.cox(covariates=covariates).report()]
        return "\n".join(lines)


# ================================================================== 傾向検定
def logrank_trend(durations, groups, events, *, levels=None, scores=None):
    """log-rank trend test（順序のある群での傾向検定）。

    lifelines に無いので自前で持つ。群にスコア s_j を与え、

        U = Σ_j s_j (O_j − E_j),
        V = Σ_t d(n−d)/(n−1) [ Σ_j s_j² n_j/n − (Σ_j s_j n_j/n)² ],
        χ² = U²/V（自由度 1）

    2 群でスコアを (0, 1) にすると、通常の log-rank 検定と一致する
    （`tests/test_survival.py` がこれを回帰テストとして固定している）。
    """
    d = pd.to_numeric(pd.Series(durations), errors="coerce").to_numpy(float)
    e = pd.to_numeric(pd.Series(events), errors="coerce").to_numpy(float)
    g = pd.Series(groups).to_numpy()
    ok = ~(np.isnan(d) | np.isnan(e))
    d, e, g = d[ok], e[ok], g[ok]
    levels = list(levels) if levels is not None else group_levels(pd.Series(g))
    scores = np.asarray(scores if scores is not None else range(len(levels)), dtype=float)
    idx = {lv: i for i, lv in enumerate(levels)}
    gi = np.array([idx.get(x, -1) for x in g])
    keep = gi >= 0
    d, e, gi = d[keep], e[keep], gi[keep]

    u = 0.0
    v = 0.0
    for t in np.unique(d[e == 1]):
        at_risk = d >= t
        n = at_risk.sum()
        if n <= 1:
            continue
        died = at_risk & (d == t) & (e == 1)
        dd = died.sum()
        nj = np.array([(at_risk & (gi == j)).sum() for j in range(len(levels))], dtype=float)
        dj = np.array([(died & (gi == j)).sum() for j in range(len(levels))], dtype=float)
        p = nj / n
        u += float((scores * (dj - dd * p)).sum())
        v += float(dd * (n - dd) / (n - 1) * ((scores ** 2 * p).sum() - ((scores * p).sum()) ** 2))
    if v <= 0:
        return float("nan"), float("nan")
    from scipy import stats as _st
    chi2 = u * u / v
    return float(chi2), float(_st.chi2.sf(chi2, 1))


# ================================================================== 小道具
def _median_ci(km: KaplanMeierFitter):
    med = float(km.median_survival_time_)
    try:
        ci = median_survival_times(km.confidence_interval_)
        lo, hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
    except Exception:                                               # noqa: BLE001
        lo = hi = float("nan")
    return med, lo, hi


def _fmt_median(med, lo, hi) -> str:
    if not np.isfinite(med):
        return NR
    s = f"{med:.1f}"
    if np.isfinite(lo) or np.isfinite(hi):
        s += f" [{_r(lo, 1) or NR}–{_r(hi, 1) or NR}]"
    return s


def _at(km: KaplanMeierFitter, t: float) -> str:
    """時点 t での生存率。観察が届いていなければ '—'。"""
    if not np.isfinite(t) or t <= 0 or t > km.timeline.max():
        return "—"
    return f"{float(km.predict(t)):.1%}"


def _reverse_km_median(d, e) -> float:
    """追跡期間中央値（reverse Kaplan-Meier）。イベントと打ち切りを入れ替えて推定する。"""
    k = KaplanMeierFitter().fit(d, 1 - pd.Series(e).to_numpy())
    m = float(k.median_survival_time_)
    return m if np.isfinite(m) else float(pd.Series(d).median())


def _p_label(p) -> str:
    """'p=<0.001' という読みにくい書き方を避ける。"""
    s = _p(p)
    if not s:
        return "p=—"
    return f"p{s}" if s.startswith("<") else f"p={s}"


def _first_sentence(msg: str, limit: int = 300) -> str:
    """lifelines の長い警告文から、原因を述べている先頭だけを取り出す。"""
    s = " ".join(str(msg).split())
    for sep in (". For example", ". See ", ". This could", "  >>>"):
        s = s.split(sep)[0]
    return s[:limit] + ("…" if len(s) > limit else "")


def _unit_days(unit: str) -> float:
    return {"days": 1.0, "weeks": 7.0, "months": 30.4375, "years": 365.25}.get(unit, 1.0)


def _unit_ja(unit: str) -> str:
    return {"days": "日", "weeks": "週", "months": "月", "years": "年"}.get(unit, unit)


def _pairs(pr):
    """pairwise_logrank_test の結果から (群1, 群2) の組を取り出す。"""
    seen = set()
    for a, b in pr.summary.index:
        if (b, a) in seen:
            continue
        seen.add((a, b))
        yield a, b
