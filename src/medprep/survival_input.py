"""medprep.survival_input — 生存時間データの入力形式を吸収する。

生存時間解析は (観察期間, イベントの有無) の 2 列を要求するが、
現場のデータ入力では観察期間を人が計算するのが手間であり、計算間違いも起きる。
そこで medprep は次の 3 形式をすべて受け付け、内部で (duration, event) に正規化する。

  形式A  duration + event                      従来型（2列）
  形式B  start_date + end_date + event         終了日とイベント有無（3列）
  形式C  id + start_date + event_date + censor_date   ← 日付だけで完結（4列）

形式C の規則
------------
  * event_date  が埋まっている  → event = 1、終了日 = event_date
  * censor_date が埋まっている  → event = 0、終了日 = censor_date
  * 両方埋まっている            → 矛盾。既定では除外して報告する（黙って一方を採らない）
  * 両方空欄                    → 終了日が無い。除外して報告する
  * id 列は解析に一切使わない。除外症例の報告にのみ用いる（仮名加工IDを想定）

日付は medprep.dates が受け付けるあらゆる表記（2013/3/3、{2013, 3, 3}、2013.3.3、
2013,3,3、2013-3-3 14:32:11、和暦、Excelシリアル値、全角）を解釈し、時刻は切り捨てる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .dates import parse_date_series

_UNITS = {"days": 1.0, "weeks": 7.0, "months": 30.4375, "years": 365.25}


@dataclass
class SurvivalFrame:
    """正規化された生存時間データと、除外された症例の記録。"""
    data: pd.DataFrame                    # duration, event と持ち込んだ共変量
    duration_col: str = "duration"
    event_col: str = "event"
    unit: str = "days"
    form: str = ""
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def n_events(self) -> int:
        return int(self.data[self.event_col].sum()) if len(self.data) else 0

    def epv(self, n_covariates: int) -> float:
        """Events Per Variable。10 を下回る多変量 Cox は過学習が強く疑われる。"""
        return float("inf") if n_covariates == 0 else self.n_events / n_covariates

    def summary(self) -> pd.DataFrame:
        d = self.data[self.duration_col]
        return pd.DataFrame({
            "項目": ["解析対象例数", "イベント数", "打ち切り数", "イベント率",
                     f"観察期間 中央値 [{self.unit}]", f"観察期間 最小–最大 [{self.unit}]",
                     "除外例数"],
            "値": [self.n, self.n_events, self.n - self.n_events,
                   f"{0.0 if self.n == 0 else self.n_events / self.n:.1%}",
                   f"{d.median():.1f}" if self.n else "-",
                   f"{d.min():.1f} – {d.max():.1f}" if self.n else "-",
                   len(self.excluded)],
        })

    def report(self) -> str:
        lines = [f"生存時間データ（入力形式: {self.form}、単位: {self.unit}）",
                 self.summary().to_string(index=False)]
        if len(self.excluded):
            lines.append("\n除外された症例（理由別）:")
            lines.append(self.excluded["理由"].value_counts().to_string())
        for w in self.warnings:
            lines.append(f"[警告] {w}")
        for nt in self.notes:
            lines.append(f"[注記] {nt}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<SurvivalFrame n={self.n} events={self.n_events} "
                f"excluded={len(self.excluded)} unit={self.unit}>")


def _to_dates(df, col, order, out_notes, out_warn):
    r = parse_date_series(df[col], order=order, name=col)
    out_notes.extend(r.notes)
    if r.n_failed:
        head = "、".join(f"{repr(v)}" for _, v in r.failures[:5])
        out_warn.append(f"列 '{col}': {r.n_failed} 件の日付を解釈できなかった（例: {head}）。")
    return r.values


def build_survival(
    df: pd.DataFrame,
    *,
    # --- 形式C（4列）---
    id_col: str | None = None,
    start_date: str | None = None,
    event_date: str | None = None,
    censor_date: str | None = None,
    # --- 形式B（3列）---
    end_date: str | None = None,
    # --- 形式A（2列）---
    duration: str | None = None,
    event: str | None = None,
    # --- 共通設定 ---
    covariates: list | None = None,
    unit: str = "days",
    order: str = "auto",
    both_dates: str = "exclude",       # 'exclude' | 'event' | 'censor' | 'raise'
    zero_duration: str = "half",       # 'half' | 'drop' | 'keep' | 'raise'
    inclusive: bool = False,           # True で両端入れ（(end-start).days + 1）
    max_years: float = 60.0,
) -> SurvivalFrame:
    """3 つの入力形式のいずれからでも (duration, event) を作る。

    Returns
    -------
    SurvivalFrame
        `.data` は lifelines にそのまま渡せる DataFrame。
        `.excluded` は除外された症例（ID・理由・元の値）。**必ず確認すること。**
    """
    if unit not in _UNITS:
        raise ValueError(f"unit は {list(_UNITS)} のいずれか。")

    notes: list = []
    warns: list = []
    work = pd.DataFrame(index=df.index)
    reasons = pd.Series("", index=df.index, dtype=object)

    # ------------------------------------------------ 形式の判定
    if start_date is not None and (event_date is not None or censor_date is not None):
        form = "C: id + 開始日 + イベント日 + 打ち切り日（4列）"
        sd = _to_dates(df, start_date, order, notes, warns)
        ed = (_to_dates(df, event_date, order, notes, warns)
              if event_date else pd.Series(pd.NaT, index=df.index))
        cd = (_to_dates(df, censor_date, order, notes, warns)
              if censor_date else pd.Series(pd.NaT, index=df.index))

        has_e, has_c = ed.notna(), cd.notna()

        # 矛盾: 両方に日付がある
        both = has_e & has_c
        if both.any():
            if both_dates == "raise":
                raise ValueError(f"{int(both.sum())} 件でイベント日と打ち切り日の両方が入力されている。")
            if both_dates == "exclude":
                reasons[both & (reasons == "")] = "イベント日と打ち切り日の両方が入力されている"
            elif both_dates == "event":
                warns.append(f"{int(both.sum())} 件でイベント日と打ち切り日が両方あり、イベント日を採用した。")
            elif both_dates == "censor":
                warns.append(f"{int(both.sum())} 件でイベント日と打ち切り日が両方あり、打ち切り日を採用した。")

        # 終了日とイベントを決める
        if both_dates == "censor":
            evt = (has_e & ~has_c).astype(float)
            end = cd.where(has_c, ed)
        else:                                  # 'event' / 'exclude' / 'raise'
            evt = has_e.astype(float)
            end = ed.where(has_e, cd)

        neither = ~has_e & ~has_c
        reasons[neither & (reasons == "")] = "イベント日・打ち切り日ともに空欄（観察終了日が不明）"
        work["_start"], work["_end"], work["event"] = sd, end, evt

    elif start_date is not None and end_date is not None:
        form = "B: 開始日 + 終了日 + イベント（3列）"
        sd = _to_dates(df, start_date, order, notes, warns)
        ed2 = _to_dates(df, end_date, order, notes, warns)
        if event is None:
            raise ValueError("形式B では event 列の指定が必要。")
        work["_start"], work["_end"] = sd, ed2
        work["event"] = pd.to_numeric(df[event], errors="coerce")

    elif duration is not None and event is not None:
        form = "A: 観察期間 + イベント（2列）"
        work["duration"] = pd.to_numeric(df[duration], errors="coerce")
        work["event"] = pd.to_numeric(df[event], errors="coerce")
        work["_start"] = pd.NaT
        work["_end"] = pd.NaT
    else:
        raise ValueError(
            "入力形式を特定できない。次のいずれかを指定すること:\n"
            "  形式A: duration=..., event=...\n"
            "  形式B: start_date=..., end_date=..., event=...\n"
            "  形式C: start_date=..., event_date=..., censor_date=...")

    # ------------------------------------------------ 期間の算出
    if form.startswith(("B", "C")):
        miss_start = work["_start"].isna()
        reasons[miss_start & (reasons == "")] = "観察開始日が空欄または解釈不能"
        miss_end = work["_end"].isna() & ~miss_start
        reasons[miss_end & (reasons == "")] = "観察終了日が空欄または解釈不能"

        days = (work["_end"] - work["_start"]).dt.days.astype("float")
        if inclusive:
            days = days + 1
            notes.append("両端入れ（開始日・終了日の両方を含む）で日数を数えた。")
        work["duration"] = days / _UNITS[unit]

        rev = days < 0
        reasons[rev & (reasons == "")] = "観察終了日が観察開始日より前（日付の逆転）"

        today = pd.Timestamp.today().normalize()
        fut = (work["_start"] > today) | (work["_end"] > today)
        reasons[fut.fillna(False) & (reasons == "")] = "未来の日付が入力されている"

        too_long = days > max_years * 365.25
        reasons[too_long.fillna(False) & (reasons == "")] = f"観察期間が {max_years:.0f} 年を超える"
    else:
        neg = work["duration"] < 0
        reasons[neg.fillna(False) & (reasons == "")] = "観察期間が負"

    # ------------------------------------------------ イベント列の検査
    bad_evt = ~work["event"].isin([0, 1]) & work["event"].notna()
    reasons[bad_evt & (reasons == "")] = "イベント列が 0/1 以外"
    reasons[work["event"].isna() & (reasons == "")] = "イベント列が空欄"

    # ------------------------------------------------ 期間 0 の扱い
    zero = (work["duration"] == 0) & (reasons == "")
    if zero.any():
        n0 = int(zero.sum())
        if zero_duration == "drop":
            reasons[zero] = "観察期間が 0（同日イベント）"
        elif zero_duration == "half":
            half = 0.5 / _UNITS[unit]
            work.loc[zero, "duration"] = half
            warns.append(f"観察期間 0 の {n0} 件を 0.5 日として扱った"
                         f"（lifelines は duration>0 を要求する。zero_duration= で変更可）。")
        elif zero_duration == "raise":
            raise ValueError(f"観察期間が 0 の症例が {n0} 件ある。")
        else:
            warns.append(f"観察期間 0 の {n0} 件をそのまま残した（Cox が失敗する可能性がある）。")

    # ------------------------------------------------ 除外と出力
    keep = reasons == ""
    exc_cols = {}
    if id_col is not None and id_col in df.columns:
        exc_cols["ID"] = df.loc[~keep, id_col]
    exc_cols["理由"] = reasons[~keep]
    for c, lbl in ((start_date, "観察開始日"), (event_date, "イベント発生日"),
                   (censor_date, "打ち切り日"), (end_date, "観察終了日"),
                   (duration, "観察期間"), (event, "イベント")):
        if c is not None and c in df.columns:
            exc_cols[f"入力値:{lbl}"] = df.loc[~keep, c]
    excluded = pd.DataFrame(exc_cols)

    out = pd.DataFrame({"duration": work.loc[keep, "duration"].astype(float),
                        "event": work.loc[keep, "event"].astype(int)})
    if form.startswith(("B", "C")):
        out["_start"] = work.loc[keep, "_start"]
        out["_end"] = work.loc[keep, "_end"]
    if id_col is not None and id_col in df.columns:
        out.insert(0, id_col, df.loc[keep, id_col])
    for c in (covariates or []):
        if c in df.columns:
            out[c] = df.loc[keep, c]

    sf = SurvivalFrame(data=out, unit=unit, form=form, excluded=excluded,
                       warnings=warns, notes=notes)

    # 解析可能性の確認
    if sf.n_events == 0:
        sf.warnings.append("イベントが 1 件も無い。生存時間解析はできない。")
    elif sf.n_events == sf.n:
        sf.warnings.append("全例でイベントが発生している（打ち切りが無い）。入力を確認すること。")
    if 0 < sf.n_events < 10:
        sf.warnings.append(f"イベント数が {sf.n_events} 件と少ない。多変量 Cox は避けること。")
    if len(excluded):
        sf.warnings.append(f"{len(excluded)} 例を除外した。`.excluded` を必ず確認すること。")
    return sf
