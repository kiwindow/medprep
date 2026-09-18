"""medprep.schema — 列の役割を推定し、その判断を人が直せる形で書き出す。

このモジュールが medprep の中核である。`autoprep()` は 1 行で最後まで走るが、
そこで下した判断はすべてここで `Schema` に集約され、`schema.yaml` として
**理由つきで**書き出される。人がそれを直して再実行できる。

    1 回目: 全自動で走らせる → schema.yaml を読む
    2 回目: schema.yaml を数行直して再実行 → 差分を見る

医学研究では「なぜこの列を落としたのか」「なぜこの症例が除かれたのか」を
後から説明できることが成果物の価値を決める。したがって **`reason` の無い判断を
作らない**。推定できなかったものは `unknown` にして、そう書く。

役割
----
    id                 症例識別子。解析には使わない
    outcome            目的変数
    time / event       生存時間の観察期間とイベント
    group              群分け・層別に使う名義変数
    datetime           日付
    numeric            連続値
    binary             2 値
    ordinal            順序尺度
    nominal            名義尺度
    high_cardinality   水準が多すぎる文字列（既定は drop）
    text               自由記載
    constant           定数・準定数（既定は drop）
    duplicate_of       他列と実質同じ（既定は drop）
    unknown            推定できなかった。**人に返す**
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import yaml

from .clean import build_alias_map, load_dict
from .dates import parse_date_series
from .timing import UNKNOWN as TIMING_UNKNOWN
from .timing import detect_timing

# ------------------------------------------------------------------ 役割
ID = "id"
OUTCOME = "outcome"
TIME = "time"
EVENT = "event"
GROUP = "group"
DATETIME = "datetime"
NUMERIC = "numeric"
BINARY = "binary"
ORDINAL = "ordinal"
NOMINAL = "nominal"
HIGH_CARDINALITY = "high_cardinality"
TEXT = "text"
CONSTANT = "constant"
DUPLICATE = "duplicate_of"
UNKNOWN = "unknown"

# 既定で解析から外す役割
_DROP_BY_DEFAULT = {ID, HIGH_CARDINALITY, TEXT, CONSTANT, DUPLICATE}

# 列名が ID を示唆する語
_ID_NAME = re.compile(
    r"(^|[_\-\s])(id|ID|Id)($|[_\-\s])|患者番号|カルテ番号|カルテno|症例番号|整理番号|"
    r"仮名id|匿名id|被験者番号|登録番号|受付番号|^no$|^no\.|通し番号",
    re.IGNORECASE)

# 順序尺度の辞書。値の集合が部分集合として一致すれば ordinal とみなす。
ORDINAL_SCALES: dict[str, list] = {
    "CKD病期": ["G1", "G2", "G3a", "G3b", "G4", "G5", "G5D"],
    "NYHA": ["I", "II", "III", "IV"],
    "ECOG": [0, 1, 2, 3, 4],
    "重症度": ["軽症", "中等症", "重症", "最重症"],
    "程度": ["なし", "軽度", "中等度", "高度"],
    "TNM_T": ["T0", "T1", "T2", "T3", "T4"],
    "TNM_N": ["N0", "N1", "N2", "N3"],
    "TNM_M": ["M0", "M1"],
    "CKD_MBD_stage": ["stage1", "stage2", "stage3", "stage4"],
    "GNRIリスク": ["リスクなし", "軽度リスク", "中等度リスク", "高度リスク"],
    "3段階": ["低", "中", "高"],
    "5段階": ["1", "2", "3", "4", "5"],
}

# 二値の対応表（男/女 のような列を 0/1 にする。**どちらを 1 にしたかを必ず残す**）
BINARY_MAPS: list[dict] = [
    {"男": 1, "女": 0},
    {"男性": 1, "女性": 0},
    {"M": 1, "F": 0},
    {"Male": 1, "Female": 0},
    {"あり": 1, "なし": 0},
    {"有": 1, "無": 0},
    {"陽性": 1, "陰性": 0},
    {"はい": 1, "いいえ": 0},
    {"Yes": 1, "No": 0},
    {"Y": 1, "N": 0},
    {"True": 1, "False": 0},
    {"生存": 0, "死亡": 1},
    {"1": 1, "0": 0},
]


def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip().lower()


# ================================================================== 列
@dataclass
class ColumnSpec:
    """1 列についての判断。**`reason` を必ず持つ。**"""
    name: str
    role: str
    reason: str
    action: str = "keep"                  # keep | drop
    dict_key: str | None = None           # 医学領域辞書のキー
    timing: str = TIMING_UNKNOWN          # 採血時点（透析前/後/不詳）
    unit: str | None = None
    n_unique: int | None = None
    missing_rate: float | None = None
    levels: list | None = None            # 名義・二値の水準
    order: list | None = None             # 順序尺度の並び
    value_map: dict | None = None         # 二値の対応（どちらを 1 にしたか）
    duplicate_of: str | None = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if v is not None and v != [] and k != "name"}
        return d


# ================================================================== 推定
def _unique_ratio(s: pd.Series) -> float:
    n = s.notna().sum()
    return 0.0 if n == 0 else s.nunique(dropna=True) / n


def _top_share(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) == 0:
        return 1.0
    return v.value_counts(normalize=True).iloc[0]


# 数値として読める文字列: 先頭が数値で、続くのは短い単位だけ。
#   ★「数字以外を全部落として数値化できるか」で判定してはならない。★
#   それをやると '特記なし_12'（自由記載）が 12 になり、備考欄が数値列にされる。
#   数値は必ず先頭にあること、後ろに続くのは数字を含まない短い単位であることを要求する。
#     受け入れる: '12', '-0.5', '1,200', '<0.1', '3.2 mg/dL', '5回', '1.2e3'
#     受け入れない: '特記なし_12', 'P0001', 'A院', '2013/3/3'
_NUMERIC_TOKEN = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*[^\d]{0,12}$")


def _numeric_convertible(s: pd.Series) -> float:
    """カンマ・比較演算子・全角を落としたときに数値として読める割合。"""
    v = s.dropna()
    if len(v) == 0:
        return 0.0
    if pd.api.types.is_numeric_dtype(v):
        return 1.0
    cleaned = (v.astype(str)
                .map(lambda x: unicodedata.normalize("NFKC", x))
                .str.replace(",", "", regex=False)
                .str.replace(r"[<>≦≧]|以上|以下|未満", "", regex=True)
                .str.strip())
    return float(cleaned.str.match(_NUMERIC_TOKEN).mean())


def _match_ordinal(s: pd.Series, colname: str = "") -> tuple[str, list] | None:
    """順序尺度の辞書に一致するか。

    ★数値列は、列名が尺度名を示さないかぎり順序尺度とみなさない。★
    施設コード {1,2,3,4} は ECOG {0,1,2,3,4} の部分集合になってしまう。
    小さい整数の集合から順序尺度かどうかは決まらないので、推測しない。
    順序で扱いたければ人が schema.yaml で role を ordinal にする。
    """
    vals = {_norm(v) for v in s.dropna().unique()}
    if not vals or len(vals) > 12 or len(vals) < 3:
        return None
    numeric_col = pd.api.types.is_numeric_dtype(s)
    cname = _norm(colname)
    for name, order in ORDINAL_SCALES.items():
        normed = [_norm(o) for o in order]
        if not vals <= set(normed):
            continue
        if numeric_col and _norm(name) not in cname:
            continue          # 数値列は列名の裏づけが無ければ採らない
        return name, [o for o, n in zip(order, normed) if n in vals]
    return None


def _match_binary_map(s: pd.Series) -> dict | None:
    vals = {_norm(v) for v in s.dropna().unique()}
    if len(vals) != 2:
        return None
    for m in BINARY_MAPS:
        if vals == {_norm(k) for k in m}:
            return m
    return None


def _is_datetime_like(s: pd.Series) -> tuple[bool, float, str]:
    """dates.py に解釈させて、成功率と推定した並びを返す。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True, 1.0, "ymd"
    if pd.api.types.is_numeric_dtype(s):
        # Excel シリアル値だけの列を日付と誤認しないよう、数値列は列名で判断する
        return False, 0.0, ""
    try:
        r = parse_date_series(s)
    except Exception:
        return False, 0.0, ""
    return r.success_rate >= 0.9 and r.n_parsed > 0, r.success_rate, r.order


def infer_column(
    s: pd.Series,
    name: str,
    *,
    dict_key: str | None = None,
    spec: dict | None = None,
    n_rows: int | None = None,
) -> ColumnSpec:
    """1 列の役割を推定する。判断の根拠を必ず `reason` に書く。"""
    n_rows = n_rows or len(s)
    nn = int(s.notna().sum())
    nu = int(s.nunique(dropna=True))
    miss = float(s.isna().mean())
    timing, _base = detect_timing(name)
    unit = (spec or {}).get("unit")

    def mk(role, reason, **kw):
        return ColumnSpec(
            name=name, role=role, reason=reason,
            action="drop" if role in _DROP_BY_DEFAULT else "keep",
            dict_key=dict_key, timing=timing, unit=unit,
            n_unique=nu, missing_rate=round(miss, 4), **kw)

    # --- 全欠損
    if nn == 0:
        return mk(CONSTANT, "全行が欠損している")

    # --- 定数・準定数
    share = _top_share(s)
    if nu == 1:
        return mk(CONSTANT, f"値が 1 種類しかない（{s.dropna().iloc[0]!r}）")
    if share >= 0.99:
        return mk(CONSTANT, f"最頻値が {share:.1%} を占める（準定数）")

    # --- ID（列名が示す場合。これは日付判定より強い）
    if _ID_NAME.search(str(name)):
        return mk(ID, f"列名が識別子を示す（ユニーク率 {_unique_ratio(s):.2f}）")

    # --- 日付
    #     ★ユニーク率による ID 判定より前に行う★
    #     日付列はほぼ全行で異なるので、順序を逆にすると観察開始日が ID にされる。
    is_dt, rate, order = _is_datetime_like(s)
    if is_dt:
        return mk(DATETIME, f"{rate:.0%} の値を日付として解釈できた（並び: {order}）")

    # --- ほぼ全行で異なる列
    #     ★ユニーク率だけで「識別子」と決めない。★
    #     備考のような自由記載もユニーク率 1.00 になる。識別子か自由記載かは
    #     ユニーク率からは決まらないので、id_col の指定か列名の裏づけが無ければ
    #     high_cardinality（特徴量として使えない列）として落とすにとどめる。
    #     「識別子ではないか」の問いかけは audit が行う。
    #
    #     ★ただし数値として読める列には適用しない。★
    #     年齢・Cr・BNP のような連続値は、症例数が少なければユニーク率が 1.00 になる。
    #     ユニーク率で落としていたら、いちばん大事な説明変数から先に消える。
    #     数値の識別子は列名（_ID_NAME）か id_col の指定で捕まえる。
    ur = _unique_ratio(s)
    conv = _numeric_convertible(s)
    if ur > 0.9 and nn >= 10 and conv < 0.95:
        return mk(HIGH_CARDINALITY,
                  f"ユニーク率 {ur:.2f} > 0.90（ほぼ全行で異なる）。"
                  f"識別子なら id_col で指定すること")

    # --- 二値
    if nu == 2:
        bm = _match_binary_map(s)
        levels = sorted(map(str, s.dropna().unique()))
        if bm:
            return mk(BINARY, f"2 値で、対応表に一致（{bm}）", levels=levels, value_map=bm)
        return mk(BINARY, f"値が 2 種類（{levels}）。0/1 の割り当ては人が決めること",
                  levels=levels)

    # --- 順序尺度
    om = _match_ordinal(s, name)
    if om:
        scale, order_vals = om
        return mk(ORDINAL, f"順序尺度 '{scale}' の水準に一致", order=order_vals,
                  levels=sorted(map(str, s.dropna().unique())))

    # --- 数値
    if conv >= 0.95:
        if pd.api.types.is_numeric_dtype(s):
            reason = "数値型"
        else:
            reason = f"前処理（記号・単位の除去）で {conv:.0%} が数値化できる"
        return mk(NUMERIC, reason)

    # --- 文字列
    avg_len = s.dropna().astype(str).str.len().mean()
    if avg_len > 20 and ur > 0.5:
        return mk(TEXT, f"平均 {avg_len:.0f} 文字・ユニーク率 {ur:.2f}（自由記載とみなす）")

    limit = max(20, int(math.sqrt(max(n_rows, 1))))
    if nu <= limit:
        return mk(NOMINAL, f"水準 {nu} ≤ {limit}（= max(20, √n)）",
                  levels=sorted(map(str, s.dropna().unique()))[:50])

    return mk(HIGH_CARDINALITY,
              f"水準 {nu} > {limit}（= max(20, √n)）。既定では解析に入れない")


# ================================================================== Schema
@dataclass
class Schema:
    """列ごとの判断・目的変数・前処理方針をまとめたもの。YAML に往復できる。"""
    columns: dict = field(default_factory=dict)          # name -> ColumnSpec
    target: dict | None = None
    survival: dict | None = None
    policy: dict = field(default_factory=dict)
    source: str = "medprep.schema.infer"
    notes: list = field(default_factory=list)

    DEFAULT_POLICY = {
        "missing": {"numeric": "median", "categorical": "most_frequent", "add_indicator": True},
        "outlier": {"method": "iqr", "fold": 1.5, "action": "winsorize"},
        "encode": {"nominal": "onehot", "min_frequency": 0.01, "ordinal": "ordinal"},
        "scale": {"method": "standard"},
        "detection_limit": {"policy": "half", "keep_flag": True},
    }

    # ------------------------------------------------------------ 構築
    @classmethod
    def infer(
        cls,
        df: pd.DataFrame,
        *,
        outcome: str | None = None,
        task: str | None = None,
        group: str | None = None,
        survival: tuple[str, str] | None = None,
        survival_dates: tuple[str, str, str] | None = None,
        id_col: str | None = None,
        dic: dict | None = None,
        duplicate_threshold: float = 0.999,
    ) -> Schema:
        dic = dic or load_dict()
        amap = build_alias_map(dic)
        items = dic["items"]

        cols: dict[str, ColumnSpec] = {}
        for c in df.columns:
            key = amap.get(_norm(c).replace(" ", ""))
            cols[c] = infer_column(df[c], c, dict_key=key,
                                   spec=items.get(key) if key else None,
                                   n_rows=len(df))

        sch = cls(columns=cols, policy=dict(cls.DEFAULT_POLICY))

        # --- 人が指定した役割で上書きする（推定より指定が強い）
        if id_col and id_col in cols:
            cols[id_col].role = ID
            cols[id_col].action = "drop"
            cols[id_col].reason = "人が id_col として指定した"
        if group and group in cols:
            cols[group].role = GROUP
            cols[group].action = "keep"
            cols[group].reason = "人が group として指定した"
        if outcome and outcome in cols:
            prev = cols[outcome].role
            cols[outcome].role = OUTCOME
            cols[outcome].action = "keep"
            cols[outcome].reason = f"人が outcome として指定した（推定は {prev}）"
            sch.target = {"name": outcome, "task": task, "inferred_role": prev}
        if survival:
            t, e = survival
            for c, role, label in ((t, TIME, "観察期間"), (e, EVENT, "イベント")):
                if c in cols:
                    cols[c].role = role
                    cols[c].action = "keep"
                    cols[c].reason = f"人が生存時間の{label}として指定した"
            sch.survival = {"form": "A", "time": t, "event": e}
        elif survival_dates:
            # 形式C: 観察開始日 / イベント発生日 / 打ち切り日 の 3 つの日付から
            # (duration, event) を導く（medprep.survival_input）。
            start, ev, cens = survival_dates
            for c, label in ((start, "観察開始日"), (ev, "イベント発生日"), (cens, "打ち切り日")):
                if c in cols:
                    cols[c].role = DATETIME
                    cols[c].action = "keep"
                    cols[c].reason = f"人が生存時間の{label}として指定した（形式C）"
            sch.survival = {"form": "C", "start_date": start,
                            "event_date": ev, "censor_date": cens}

        # --- 重複列（実質同じ情報を持つ列）
        sch._mark_duplicates(df, threshold=duplicate_threshold)
        return sch

    def _mark_duplicates(self, df: pd.DataFrame, threshold: float = 0.999) -> None:
        """数値は相関、カテゴリは Cramér's V で、実質同じ列を見つける。

        施設 と 施設コード のような対は、両方入れると多重共線性を作るだけでなく、
        片方を落としたつもりで情報が残る事故のもとになる。
        """
        keep = [c for c, sp in self.columns.items()
                if sp.action == "keep" and sp.role in (NUMERIC, BINARY, ORDINAL, NOMINAL, GROUP)]
        num = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
        # ★型をまたいで比べる★ 施設（文字列）と 施設コード（整数）は完全に対応するのに、
        #   数値どうし・カテゴリどうしだけ見ていると気づけない。
        #   水準が少ない列は dtype を問わずカテゴリとして扱って Cramér's V を取る。
        low_card = [c for c in keep if 1 < df[c].nunique(dropna=True) <= 50]

        def _prefer(a: str, b: str) -> tuple[str, str]:
            """どちらを残すか。辞書に載っている列 → 読める（非数値）列 → 先に来る列。"""
            ka, kb = self.columns[a].dict_key, self.columns[b].dict_key
            if bool(ka) != bool(kb):
                return (a, b) if ka else (b, a)
            na = pd.api.types.is_numeric_dtype(df[a])
            nb = pd.api.types.is_numeric_dtype(df[b])
            if na != nb:
                return (b, a) if na else (a, b)
            return (a, b)

        def _mark(keep_col: str, drop_col: str, reason: str) -> None:
            if self.columns[drop_col].action != "keep":
                return
            self.columns[drop_col].role = DUPLICATE
            self.columns[drop_col].action = "drop"
            self.columns[drop_col].duplicate_of = keep_col
            self.columns[drop_col].reason = reason

        if len(num) >= 2:
            corr = df[num].corr(numeric_only=True).abs()
            for i, a in enumerate(num):
                for b in num[i + 1:]:
                    r = corr.loc[a, b]
                    if pd.notna(r) and r >= threshold:
                        k, d = _prefer(a, b)
                        _mark(k, d, f"'{k}' と相関 |r| = {r:.4f} ≥ {threshold}")

        for i, a in enumerate(low_card):
            for b in low_card[i + 1:]:
                if self.columns[a].action != "keep" or self.columns[b].action != "keep":
                    continue
                v = cramers_v(df[a], df[b])
                if v is not None and v >= threshold:
                    k, d = _prefer(a, b)
                    _mark(k, d, f"'{k}' と Cramér's V = {v:.4f} ≥ {threshold}（実質同じ情報）")

    # ------------------------------------------------------------ 参照
    def by_role(self, *roles: str) -> list:
        return [c for c, s in self.columns.items() if s.role in roles]

    def kept(self) -> list:
        return [c for c, s in self.columns.items() if s.action == "keep"]

    def dropped(self) -> list:
        return [c for c, s in self.columns.items() if s.action == "drop"]

    def unknown(self) -> list:
        return [c for c, s in self.columns.items() if s.role == UNKNOWN]

    def features(self) -> list:
        """説明変数として使う列（目的変数・ID・生存時間の 2 列を除く）。"""
        excl = {OUTCOME, ID, TIME, EVENT}
        return [c for c, s in self.columns.items() if s.action == "keep" and s.role not in excl]

    # ------------------------------------------------------------ 報告
    def to_frame(self) -> pd.DataFrame:
        rows = []
        for c, s in self.columns.items():
            rows.append({
                "列名": c, "役割": s.role, "扱い": s.action,
                "辞書": s.dict_key or "—", "採血時点": s.timing,
                "水準数": s.n_unique, "欠損率": s.missing_rate,
                "判断の根拠": s.reason,
            })
        return pd.DataFrame(rows)

    def report(self) -> str:
        f = self.to_frame()
        lines = [f"列 {len(self.columns)} 本（残す {len(self.kept())} / 落とす {len(self.dropped())}）",
                 f.to_string(index=False)]
        if self.target:
            lines.append(f"\n目的変数: {self.target['name']}（task={self.target.get('task')}）")
        if self.survival:
            sv = self.survival
            if sv.get("form") == "C":
                lines.append(f"生存時間（形式C・日付4列）: 開始={sv['start_date']} / "
                             f"イベント日={sv['event_date']} / 打ち切り日={sv['censor_date']}")
            else:
                lines.append(f"生存時間（形式A）: 観察期間={sv.get('time')} / "
                             f"イベント={sv.get('event')}")
        if self.unknown():
            lines.append(f"\n★役割を推定できなかった列: {self.unknown()}")
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 往復
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "survival": self.survival,
            "policy": self.policy,
            "columns": {c: s.to_dict() for c, s in self.columns.items()},
            "notes": self.notes,
        }

    def to_yaml(self, path=None) -> str:
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False,
                              default_flow_style=False, width=100)
        header = (
            "# medprep schema — 自動推定の結果。**人が直して再実行できる。**\n"
            "#\n"
            "#   role   : 列の役割。drop したくなければ action を keep にする\n"
            "#   action : keep / drop\n"
            "#   reason : なぜそう判断したか（自動生成。直した場合は書き換えること）\n"
            "#\n"
            "# このファイルと元データと medprep の版があれば、前処理を完全に再現できる。\n"
            "# 論文の Methods に「前処理は medprep、設定は補足資料の schema.yaml のとおり」\n"
            "# と書ける状態を保つこと。\n\n")
        text = header + text
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_yaml(cls, path_or_text: str) -> Schema:
        try:
            with open(path_or_text, encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except OSError:
            d = yaml.safe_load(path_or_text)
        cols = {}
        for name, spec in (d.get("columns") or {}).items():
            cols[name] = ColumnSpec(name=name, **spec)
        return cls(columns=cols, target=d.get("target"), survival=d.get("survival"),
                   policy=d.get("policy") or {}, source=d.get("source", "schema.yaml"),
                   notes=d.get("notes") or [])

    def diff(self, other: Schema) -> pd.DataFrame:
        """2 つの schema の違いを表にする。**手で直した箇所を確かめるために使う。**"""
        rows = []
        for c in sorted(set(self.columns) | set(other.columns)):
            a, b = self.columns.get(c), other.columns.get(c)
            if a is None:
                rows.append({"列名": c, "項目": "存在", "変更前": "—", "変更後": "あり"})
                continue
            if b is None:
                rows.append({"列名": c, "項目": "存在", "変更前": "あり", "変更後": "—"})
                continue
            for fld in ("role", "action", "timing", "dict_key"):
                x, y = getattr(a, fld), getattr(b, fld)
                if x != y:
                    rows.append({"列名": c, "項目": fld, "変更前": x, "変更後": y})
        return pd.DataFrame(rows)


# ================================================================== 補助
def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """2 つのカテゴリ列の関連の強さ（0〜1）。バイアス補正つき。"""
    t = pd.crosstab(a, b)
    if t.size == 0 or t.shape[0] < 2 or t.shape[1] < 2:
        return None
    chi2 = _chi2(t.to_numpy(dtype=float))
    n = t.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = t.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return None if denom <= 0 else float(np.sqrt(phi2corr / denom))


def _chi2(obs: np.ndarray) -> float:
    total = obs.sum()
    if total == 0:
        return 0.0
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)
    return float(t.sum())
