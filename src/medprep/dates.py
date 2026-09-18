"""medprep.dates — 多様な日付表記を日付（時刻なし）に正規化する。

医学データの日付列は表記が揃っていないことが常態である。
同じ列に「2013/3/3」「2013.3.3」「{2013, 3, 3}」「H25.3.3」「41336」（Excelシリアル値）
「2013-3-3 14:32:11」が混在しうる。ここではそれを列単位で解決する。

方針
----
1. 1セルずつ独立に解析するのではなく、**列全体を見て曖昧さを解く**。
   日/月の順序（3/4/2013 が 3月4日か 4月3日か）は 1 セルでは決まらないが、
   列に 13 以上の値が 1 つでもあればその位置が「日」だと確定する。
2. 時刻は捨てる（年月日以下を自動削除）。
3. 解析できなかったセルは NaT にし、**元の文字列を必ず記録して報告する**。
   黙って欠損にしない。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------- 和暦
# 改元日。元年 = 1年。
_ERAS = {
    "明治": (1868, 9, 8), "M": (1868, 9, 8),
    "大正": (1912, 7, 30), "T": (1912, 7, 30),
    "昭和": (1926, 12, 25), "S": (1926, 12, 25),
    "平成": (1989, 1, 8), "H": (1989, 1, 8),
    "令和": (2019, 5, 1), "R": (2019, 5, 1),
}
_ERA_RE = re.compile(
    r"^(明治|大正|昭和|平成|令和|[MTSHR])\s*(元|\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D*$"
)

# Excel シリアル値の起点（1900 日付システム。1900年をうるう年とする既知のバグを含む）
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
# シリアル値とみなす範囲: 1970-01-01 (25569) 〜 2100-01-01 (73051)
_SERIAL_MIN, _SERIAL_MAX = 25569, 73051

# 3 つの数値に分解する。区切りは / - . , 年月日 空白 { } ( ) など何でもよい。
_TRIPLE_RE = re.compile(r"^\D*(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})(?:\D.*)?$")
# 8 桁連結（20130303）
_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
# 年月のみ（2013/3、2013年3月）— 日を 1 とみなすか欠損にするかは policy
_YM_RE = re.compile(r"^\D*(\d{4})\D+(\d{1,2})\D*$")

_NULLS = {
    "", "-", "ー", "―", "‐", ".", "・", "なし", "無", "無し", "未実施", "未測定",
    "不明", "na", "n/a", "nan", "none", "null", "nat", "#n/a", "#value!", "9999",
    "99999999", "0", "00000000",
}


def _normalize_text(x) -> str:
    """全角→半角、記号ゆれの吸収、前後空白の除去。"""
    if x is None:
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)          # ２０１３／３／３ → 2013/3/3
    s = s.replace("　", " ").strip()
    s = s.strip("{}[]()<>\"'　 ")                  # {2013, 3, 3} の括弧を外す
    return s.strip()


def _from_wareki(s: str):
    m = _ERA_RE.match(s)
    if not m:
        return None
    era, yy, mm, dd = m.groups()
    base = _ERAS.get(era)
    if base is None:
        return None
    yy = 1 if yy == "元" else int(yy)
    year = base[0] + yy - 1
    try:
        return pd.Timestamp(year=year, month=int(mm), day=int(dd))
    except ValueError:
        return None


def _from_serial(s: str):
    """Excel のシリアル値。整数部だけを使い、小数部（時刻）は捨てる。"""
    try:
        v = float(s)
    except ValueError:
        return None
    if not (_SERIAL_MIN <= v <= _SERIAL_MAX):
        return None
    return _EXCEL_EPOCH + pd.Timedelta(days=int(v))


@dataclass
class DateParseResult:
    """列 1 本の解析結果。"""
    values: pd.Series                      # datetime64[ns]（時刻は 00:00:00）
    order: str                             # 'ymd' | 'dmy' | 'mdy' | 'mixed' | 'none'
    n_total: int = 0
    n_parsed: int = 0
    n_null: int = 0                        # もともと空欄
    n_failed: int = 0
    failures: list = field(default_factory=list)   # (index, 元の値)
    notes: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        d = self.n_total - self.n_null
        return 1.0 if d == 0 else self.n_parsed / d

    def __repr__(self):
        return (f"<DateParseResult order={self.order} "
                f"parsed={self.n_parsed}/{self.n_total - self.n_null} "
                f"null={self.n_null} failed={self.n_failed}>")


def _split_triple(s: str):
    """文字列から (a, b, c) の 3 整数を取り出す。取り出せなければ None。"""
    m = _YYYYMMDD_RE.match(s)
    if m:
        return tuple(int(g) for g in m.groups()), "ymd"
    m = _TRIPLE_RE.match(s)
    if m:
        return tuple(int(g) for g in m.groups()), None
    return None


def _infer_order(triples) -> str:
    """列全体から日月年の並びを推定する。

    - 先頭が 4 桁 → ymd（日本の医療データはほぼこれ）
    - 先頭に 13 以上がある → その位置は day
    - 決め手がなければ 'ymd' を返さず 'ambiguous' を返して呼び出し側に委ねる
    """
    if not triples:
        return "none"
    a = [t[0] for t in triples]
    b = [t[1] for t in triples]
    c = [t[2] for t in triples]
    if all(v >= 1000 for v in a):
        return "ymd"
    if all(v >= 1000 for v in c):
        # 末尾が年。先頭が 13 以上あれば日、そうでなければ曖昧
        if any(v > 12 for v in a):
            return "dmy"
        if any(v > 12 for v in b):
            return "mdy"
        return "ambiguous_dmy_mdy"
    if any(v >= 1000 for v in a):
        return "ymd"          # 2 桁年と 4 桁年が混在 → 年が先とみなす
    return "ambiguous"


def _build(order: str, t) -> pd.Timestamp | None:
    a, b, c = t
    try:
        if order == "ymd":
            y, m, d = a, b, c
        elif order == "dmy":
            d, m, y = a, b, c
        elif order == "mdy":
            m, d, y = a, b, c
        else:
            return None
        if y < 100:                      # 2 桁年: 69 以下は 20xx、70 以上は 19xx
            y = 2000 + y if y <= 69 else 1900 + y
        return pd.Timestamp(year=y, month=m, day=d)
    except (ValueError, OverflowError):
        return None


def parse_date_series(
    s: pd.Series,
    order: str = "auto",
    allow_year_month: bool = False,
    name: str | None = None,
) -> DateParseResult:
    """日付らしき列を datetime64 に正規化する（時刻は落とす）。

    Parameters
    ----------
    order : 'auto' | 'ymd' | 'dmy' | 'mdy'
        'auto' は列全体から推定する。推定できない（日month両方 12 以下しかない）場合は
        'ymd' を仮定したうえで notes に警告を残す。
    allow_year_month : bool
        '2013/3' のような年月のみの値を、その月の 1 日として受け入れるか。
        既定は False（受け入れず失敗として報告する。医学データでは日の欠落が
        観察期間の計算誤差に直結するため、黙って 1 日にしない）。
    """
    name = name or (s.name if s.name is not None else "date")
    n = len(s)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    res = DateParseResult(values=out, order="none", n_total=n)
    notes = res.notes

    # すでに datetime 型（Excel/pandas が解釈済み）— 時刻を落として終わり
    if pd.api.types.is_datetime64_any_dtype(s):
        res.values = s.dt.normalize()
        res.order = "ymd"
        res.n_null = int(s.isna().sum())
        res.n_parsed = n - res.n_null
        notes.append("既に datetime 型。時刻を切り捨てた。")
        return res

    raw = s.map(_normalize_text)
    is_null = raw.str.lower().isin(_NULLS) | raw.eq("")
    res.n_null = int(is_null.sum())

    # 1 パス目: 和暦・シリアル値・8 桁連結を先に処理し、残りを triple に分解
    direct = {}
    triples = {}
    for i, v in raw.items():
        if is_null[i]:
            continue
        w = _from_wareki(v)
        if w is not None:
            direct[i] = w
            continue
        t = _split_triple(v)
        if t is not None:
            trip, fixed = t
            if fixed == "ymd":
                d = _build("ymd", trip)
                if d is not None:
                    direct[i] = d
                    continue
            triples[i] = trip
            continue
        sv = _from_serial(v)
        if sv is not None:
            direct[i] = sv
            continue
        if allow_year_month:
            m = _YM_RE.match(v)
            if m:
                d = _build("ymd", (int(m.group(1)), int(m.group(2)), 1))
                if d is not None:
                    direct[i] = d
                    notes.append(f"行 {i}: 年月のみ '{v}' を月初として解釈した。")
                    continue

    # 2 パス目: 並びを推定して triple を組み立てる
    if order == "auto":
        inferred = _infer_order(list(triples.values()))
        if inferred.startswith("ambiguous"):
            # 推測して黙って進まない。列を丸ごと未解析のまま返し、指定を促す。
            notes.append(
                f"列 '{name}': 日と月の順序が確定できない（年が末尾で、日・月とも 12 以下しか"
                f"現れない）。誤って解釈すると観察期間が最大 11 か月ずれるため、自動では"
                f"解釈しない。order='dmy'（日/月/年）または order='mdy'（月/日/年）を"
                f"明示して再実行すること。"
            )
            res.order = inferred
            res.n_parsed = len(direct)
            for i, d in direct.items():
                out.at[i] = d
            failed_idx = [i for i in s.index if not is_null[i] and pd.isna(out.at[i])]
            res.n_failed = len(failed_idx)
            res.failures = [(i, s.at[i]) for i in failed_idx[:50]]
            res.values = out
            return res
        use = inferred if inferred != "none" else "ymd"
    else:
        use = order

    for i, t in triples.items():
        d = _build(use, t)
        if d is not None:
            direct[i] = d

    for i, d in direct.items():
        out.at[i] = d

    parsed = out.notna()
    res.values = out
    res.order = use if triples else ("ymd" if direct else "none")
    res.n_parsed = int(parsed.sum())
    failed_idx = [i for i in s.index if not is_null[i] and pd.isna(out.at[i])]
    res.n_failed = len(failed_idx)
    res.failures = [(i, s.at[i]) for i in failed_idx[:50]]
    if res.n_failed:
        notes.append(f"列 '{name}': {res.n_failed} 件を日付として解釈できなかった（先頭50件を failures に保持）。")
    return res


def parse_date_frame(df: pd.DataFrame, cols, order="auto", **kw) -> dict:
    """複数の日付列をまとめて解析する。戻り値は {列名: DateParseResult}。"""
    return {c: parse_date_series(df[c], order=order, name=c, **kw) for c in cols}
