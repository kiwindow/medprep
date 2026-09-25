"""medprep.horizon — 生存時間から「τ 以内にイベントが起きたか」の目的変数を作る。

目的変数を指定せず、観察開始日・イベント発生日・打ち切り日の 3 つだけを
指定したとき、`autoprep` は**イベントの有無**を目的変数にする。

★ただし「観察期間中にイベントが起きたか」をそのまま 0/1 にしてはならない。★
0.1 年で打ち切られた人も、5 年間イベントの無かった人も、同じ「0」になる。
観察期間の短い人が多い群ほどリスクが低く見える。**打ち切りを扱えることが
生存時間解析の存在理由**であり、そこを捨てた 0/1 は査読で必ず指摘される。

そこで、区切りの期間 τ を決めて、症例を 3 つに分ける。

| 区分 | 目的変数 |
|---|---|
| τ までにイベントが起きた | 1 |
| τ まで観察され、その間にイベントが無かった | 0 |
| τ より前に打ち切られた | **判定できない**（目的変数なし。3〜6 から除く） |

τ を長くすると 1 は増えるが、判定できない症例も増える。**τ はデータごとに決める。**

決め方
------
1. 候補: 観察期間の分位点（5%〜95%）を、観察が 2 年以上に及ぶなら 0.5 年刻み、
   そうでなければ 3 か月刻みに丸めたもの
2. 上限: 追跡期間の中央値（reverse Kaplan-Meier）。これより長い τ では
   打ち切り症例の偏りが大きくなる
3. 条件: 判定できない症例が全体の `max_undetermined`（既定 20%）以下
4. 選び方: 条件を満たす候補のうち、**0 と 1 の少ないほうの例数が最も多いもの**
   （同じなら判定できる例数が多いもの → τ が短いもの）

候補ごとの表と、選んだ理由を必ず残す。`tau=` で人が上書きできる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .textfmt import frame_text


@dataclass
class HorizonResult:
    tau: float
    unit: str
    name: str                              # 作った目的変数の列名
    outcome: pd.Series                     # 1 / 0 / NaN（判定できない）
    table: pd.DataFrame                    # 候補ごとの 0・1・判定できない の数
    reason: str = ""
    upper: float = float("nan")            # reverse KM による追跡期間中央値
    warnings: list = field(default_factory=list)

    @property
    def n1(self) -> int:
        return int((self.outcome == 1).sum())

    @property
    def n0(self) -> int:
        return int((self.outcome == 0).sum())

    @property
    def n_undetermined(self) -> int:
        return int(self.outcome.isna().sum())

    def report(self) -> str:
        lines = [f"目的変数 '{self.name}' を作った（τ = {tau_label(self.tau, self.unit)}）",
                 f"  1（τ までにイベント）: {self.n1} 例 / 0（τ まで観察してイベントなし）: "
                 f"{self.n0} 例 / 判定できない（τ より前に打ち切り）: {self.n_undetermined} 例",
                 f"  選んだ理由: {self.reason}",
                 "\nτ の候補:", frame_text(self.table)]
        for w in self.warnings:
            lines.append(f"[警告] {w}")
        return "\n".join(lines)


def tau_label(tau: float, unit: str = "years") -> str:
    if unit == "years":
        if tau < 1:
            return f"{round(tau * 12):g}か月"
        return f"{tau:g}年"
    return {"days": f"{tau:g}日", "weeks": f"{tau:g}週", "months": f"{tau:g}か月"}.get(
        unit, f"{tau:g}{unit}")


def binary_at(duration, event, tau: float) -> pd.Series:
    """τ 以内のイベントを 1、τ まで観察してイベントなしを 0、判定できないものを NaN にする。"""
    d = pd.to_numeric(pd.Series(duration), errors="coerce")
    e = pd.to_numeric(pd.Series(event), errors="coerce")
    out = pd.Series(np.nan, index=d.index, dtype=float)
    out[d >= tau] = 0.0                     # τ より後のイベントも「τ までは無かった」
    out[(e == 1) & (d <= tau)] = 1.0        # τ ちょうどのイベントは 1
    out[d.isna() | e.isna()] = np.nan
    return out


def _grid(d: pd.Series, unit: str) -> float:
    if unit == "years":
        return 0.5 if d.max() >= 2 else 0.25
    if unit == "months":
        return 6.0 if d.max() >= 24 else 3.0
    if unit == "days":
        return 182.0 if d.max() >= 730 else 91.0
    return float(max(d.max() / 20, 1e-6))


def choose_horizon(duration, event, *, unit: str = "years", tau: float | None = None,
                   max_undetermined: float = 0.20, name: str | None = None) -> HorizonResult:
    """τ を選んで「τ 以内のイベント」の 0/1 を作る。"""
    d = pd.to_numeric(pd.Series(duration), errors="coerce")
    e = pd.to_numeric(pd.Series(event), errors="coerce")
    ok = d.notna() & e.notna()
    if ok.sum() == 0:
        raise ValueError("観察期間とイベントが揃った症例が無い")
    dd, ee = d[ok], e[ok]
    n = int(ok.sum())
    warns = []

    # --- 上限：reverse Kaplan-Meier による追跡期間中央値
    try:
        from .survival import _reverse_km_median
        upper = float(_reverse_km_median(dd.to_numpy(), ee.to_numpy()))
    except Exception:                                                # noqa: BLE001
        upper = float(dd.median())

    step = _grid(dd, unit)
    qs = dd.quantile(np.linspace(0.05, 0.95, 19)).to_numpy()
    cands = sorted({round(float(np.round(q / step) * step), 6) for q in qs} - {0.0})
    cands = [c for c in cands if c > 0]

    rows = []
    for c in cands:
        y = binary_at(dd, ee, c)
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        nu = int(y.isna().sum())
        rows.append({"τ": tau_label(c, unit), "_tau": c, "1（イベント）": n1, "0（なし）": n0,
                     "判定できない": nu, "判定できない割合": round(nu / n, 3),
                     "少ないほうの例数": min(n0, n1),
                     "上限以内": "○" if c <= upper + 1e-9 else "×（追跡期間中央値を超える）",
                     "条件を満たす": "○" if (nu / n <= max_undetermined and c <= upper + 1e-9)
                     else "×"})
    table = pd.DataFrame(rows)

    if tau is not None:
        chosen = float(tau)
        reason = "人が tau= で指定した"
    else:
        good = table[table["条件を満たす"] == "○"] if len(table) else table
        if len(good):
            best = good.assign(_det=good["1（イベント）"] + good["0（なし）"]).sort_values(
                ["少ないほうの例数", "_det", "_tau"], ascending=[False, False, True]).iloc[0]
            chosen = float(best["_tau"])
            reason = (f"判定できない症例が {max_undetermined:.0%} 以下で、追跡期間中央値"
                      f"（reverse KM で {upper:.2f}）以内の候補のうち、0 と 1 の少ないほうの"
                      f"例数（{int(best['少ないほうの例数'])} 例）が最も多い")
        elif len(table):
            best = table.sort_values(["判定できない割合", "_tau"]).iloc[0]
            chosen = float(best["_tau"])
            reason = (f"★条件（判定できない ≤ {max_undetermined:.0%}・追跡期間中央値以内）を"
                      f"満たす候補が無かった。判定できない割合が最も小さいものを選んだ★")
            warns.append(reason)
        else:
            chosen = float(dd.median())
            reason = "候補が作れなかったので観察期間の中央値にした"
            warns.append(reason)

    y = binary_at(d, e, chosen)
    label = tau_label(chosen, unit)
    nm = name or f"イベント_{label}以内"
    if len(table):
        table = table.assign(選んだ=np.where(np.isclose(table["_tau"], chosen), "★", ""))
        table = table.drop(columns="_tau")
    nu = int(y[ok].isna().sum())
    if nu / n > max_undetermined:
        warns.append(f"判定できない症例が {nu} 例（{nu / n:.0%}）ある。"
                     f"τ より前に打ち切られた症例で、3〜6 のデータからは除かれる")
    return HorizonResult(tau=chosen, unit=unit, name=nm, outcome=y, table=table,
                         reason=reason, upper=upper, warnings=warns)
