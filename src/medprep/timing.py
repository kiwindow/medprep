"""medprep.timing — 血液透析データの採血時点（透析前／透析後／不詳）を扱う。

血液透析患者の検査値には必ず「透析前の値か、透析後の値か」という形容詞がつく。
そのいずれでもない one point の値はほとんど存在しない。体重も同様である。
したがって medprep は、**すべての測定値を (項目, 採血時点) の組として扱う**。

採用する時点は3つ。
    pre     : 透析前
    post    : 透析後
    unknown : 不詳（どちらか判明しない）

【設計上の中核原則】
    unknown の値から透析指標（spKt/V・nPCR・%CGR・クリアスペース率・URR）を
    **計算しない**。これらはすべて前後の組を要求する式であり、
    どちらか分からない値を当てはめれば結果は意味を失う。
    計算できない場合は NaN を返し、「どの時点の値が欠けていたか」を報告する。

    また、同一項目の pre と post を 1 つの列に混ぜたまま Table 1 や Cox に
    投入すると、群間差が「病態の差」ではなく「採血時点の差」になる。
    audit がこれを検出する。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from .textfmt import frame_text

PRE, POST, UNKNOWN = "pre", "post", "unknown"

_LABEL = {PRE: "透析前", POST: "透析後", UNKNOWN: "不詳"}

# 列名から時点を読み取るパターン（NFKC 正規化・小文字化のあとに照合）
_PRE_PATTERNS = [
    r"透析前", r"透前", r"hd前", r"透析開始前", r"開始前", r"前値", r"週初め",
    r"(?:^|[_\-\(\（\[\s])pre(?:$|[_\-\)\）\]\s])", r"^pre", r"pre$",
    r"^前[^回週年月日]", r"[_\-\(\（]前[\)\）]?$",
]
_POST_PATTERNS = [
    r"透析後", r"透後", r"hd後", r"透析終了後", r"終了後", r"後値",
    r"(?:^|[_\-\(\（\[\s])post(?:$|[_\-\)\）\]\s])", r"^post", r"post$",
    r"^後[^日週年月]", r"[_\-\(\（]後[\)\）]?$",
]

# 長持ち形式の時点列に入りうる値
_VALUE_MAP = {
    PRE: {"透析前", "透前", "hd前", "前", "pre", "開始前", "透析開始前", "before", "1"},
    POST: {"透析後", "透後", "hd後", "後", "post", "終了後", "透析終了後", "after", "2"},
}

# 時点を必ず持つべき項目（辞書キー）。unknown なら警告する。
TIMING_REQUIRED = {
    "BUN", "Cr", "K", "P", "Ca", "Na", "Cl", "Alb", "Hb", "Ht", "weight",
    "SBP", "DBP", "b2MG", "B2MG", "TP", "Mg", "UA", "GLU",
}


def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip().lower()


_BRACKETS = "()（）[]［］{}｛｝〔〕<>＜＞"


def _strip_brackets(s: str) -> str:
    """時点の語を抜いたあとに残る括弧・区切り記号を落とす。"""
    s = re.sub(r"[（(\[｛{〔]\s*[）)\]｝}〕]", "", s)      # 空になった括弧対を削除
    prev = None
    while prev != s:
        prev = s
        s = s.strip(" 　_-–—・/\\|:：")
        # 開き括弧だけ、閉じ括弧だけが端に残っている場合に落とす
        if s and s[0] in "）)］]｝}〕＞>":
            s = s[1:]
        if s and s[-1] in "（(［[｛{〔＜<":
            s = s[:-1]
        if s and s[-1] in "）)］]｝}〕" and not any(c in s for c in "（(［[｛{〔"):
            s = s[:-1]
        if s and s[0] in "（(［[｛{〔" and not any(c in s for c in "）)］]｝}〕"):
            s = s[1:]
    return s.strip()


def detect_timing(colname: str) -> tuple[str, str]:
    """列名から (時点, 時点を取り除いた基底名) を返す。

    >>> detect_timing("透析前BUN")        -> ('pre',  'BUN')
    >>> detect_timing("BUN_post")         -> ('post', 'BUN')
    >>> detect_timing("Cr（透析後）")       -> ('post', 'Cr')
    >>> detect_timing("アルブミン(Alb)")    -> ('unknown', 'アルブミン(Alb)')
    """
    raw = str(colname)
    s = _norm(raw)
    for pats, tag in ((_PRE_PATTERNS, PRE), (_POST_PATTERNS, POST)):
        for p in pats:
            m = re.search(p, s)
            if m:
                # 元の文字列から、一致した範囲に対応する部分を落とす
                base = (raw[:m.start()] + raw[m.end():]) if len(raw) == len(s) else raw
                base = _strip_brackets(base)
                return tag, (base or raw)
    return UNKNOWN, raw


def detect_timing_value(v) -> str:
    s = _norm(v)
    for tag, vals in _VALUE_MAP.items():
        if s in vals:
            return tag
    return UNKNOWN


@dataclass
class TimingSchema:
    """列 → (基底項目名, 時点) の対応表。schema.yaml に保存される。"""
    columns: dict = field(default_factory=dict)        # col -> (base, timing)
    default_timing: str | None = None                  # 明示的に指定されたときのみ
    source: str = "列名からの自動推定"
    notes: list = field(default_factory=list)
    alias_map: dict = field(default_factory=dict)      # 正規化した別名 -> 辞書キー
    keys: dict = field(default_factory=dict)           # col -> 辞書キー

    # ------------------------------------------------------------ 構築
    @classmethod
    def infer(cls, df: pd.DataFrame, default_timing: str | None = None,
              alias_map: dict | None = None) -> TimingSchema:
        """列名から推定する。

        `default_timing` は「このデータは全部透析前値である」と
        **人が明示したとき** にだけ渡す。自動では決して仮定しない。
        `alias_map` に ranges_ja.yaml の別名表を渡すと、
        「透析前体重」→ 辞書キー `weight` のように基底項目を同定できる。
        """
        alias_map = alias_map or {}
        cols, keys = {}, {}
        for c in df.columns:
            t, base = detect_timing(c)
            if t == UNKNOWN and default_timing is not None:
                t = default_timing
            cols[c] = (base, t)
            k = alias_map.get(_norm(base)) or alias_map.get(_norm(c))
            if k:
                keys[c] = k
        ts = cls(columns=cols, default_timing=default_timing,
                 alias_map=alias_map, keys=keys)
        if default_timing:
            ts.notes.append(
                f"時点が読み取れない列に、人の指定により既定 '{_LABEL[default_timing]}' を適用した。")
        return ts

    @classmethod
    def from_long(cls, df: pd.DataFrame, timing_col: str) -> TimingSchema:
        """`採血時点` のような時点列を持つ縦持ちデータから作る。"""
        vals = df[timing_col].map(detect_timing_value)
        ts = cls(columns={}, source=f"時点列 '{timing_col}'")
        n_unknown = int((vals == UNKNOWN).sum())
        if n_unknown:
            ts.notes.append(f"時点列 '{timing_col}' の {n_unknown} 行が不詳。")
        ts.row_timing = vals
        return ts

    # ------------------------------------------------------------ 参照
    def get(self, base: str, timing: str) -> str | None:
        """基底項目名（または辞書キー）と時点から列名を引く。無ければ None。"""
        b = _norm(base)
        key = self.alias_map.get(b, base)
        for col, (bb, tt) in self.columns.items():
            if tt != timing:
                continue
            if self.keys.get(col) == key:
                return col
            if _norm(bb) == b or _norm(col) == b:
                return col
        return None

    def timing_of(self, col: str) -> str:
        return self.columns.get(col, (col, UNKNOWN))[1]

    def bases(self) -> set:
        return {b for b, _ in self.columns.values()}

    # ------------------------------------------------------------ 報告
    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(c, b, _LABEL[t]) for c, (b, t) in self.columns.items()],
            columns=["列名", "基底項目", "採血時点"])

    def unknown_columns(self, only_required=True, dict_keys=None) -> list:
        """時点が不詳の列。`only_required` なら時点が必須の項目に絞る。"""
        out = []
        for c, (_base, t) in self.columns.items():
            if t != UNKNOWN:
                continue
            if only_required:
                key = (dict_keys or {}).get(c)
                if key is None or key not in TIMING_REQUIRED:
                    continue
            out.append(c)
        return out

    def pairs(self) -> pd.DataFrame:
        """項目ごとに pre/post が揃っているかの一覧。"""
        rows = []
        for b in sorted(self.bases(), key=str):
            pre = self.get(b, PRE)
            post = self.get(b, POST)
            unk = self.get(b, UNKNOWN)
            rows.append((b, pre or "—", post or "—", unk or "—",
                         "揃っている" if (pre and post) else
                         ("片方のみ" if (pre or post) else "時点不詳のみ")))
        return pd.DataFrame(rows, columns=["項目", "透析前", "透析後", "不詳", "状態"])

    def report(self) -> str:
        p = self.pairs()
        lines = [f"採血時点の判定（{self.source}）", frame_text(p)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    def to_yaml_dict(self) -> dict:
        return {"source": self.source, "default_timing": self.default_timing,
                "columns": {c: {"base": b, "timing": t} for c, (b, t) in self.columns.items()}}


# ====================================================================
# 透析指標の一括算出（必要な時点が揃っているものだけ計算する）
# ====================================================================
# 指標 -> 必要な (基底項目, 時点) の組
REQUIREMENTS = {
    "URR":          [("BUN", PRE), ("BUN", POST)],
    "spKt/V":       [("BUN", PRE), ("BUN", POST), ("weight", PRE), ("weight", POST), ("Td", None)],
    "nPCR":         [("BUN", PRE), ("BUN", POST), ("weight", PRE), ("weight", POST), ("Td", None)],
    "%CGR":         [("BUN", PRE), ("BUN", POST), ("Cr", PRE), ("Cr", POST),
                     ("weight", PRE), ("weight", POST), ("Td", None), ("age", None), ("sex", None)],
    "クリアスペース率A/V(推算)": [("BUN", PRE), ("BUN", POST), ("weight", PRE), ("weight", POST),
                          ("V_post", None)],
    "推定塩分量":      [("weight_gain", None), ("Na", PRE)],
    "TAC-BUN(簡便式)": [("BUN", POST), ("BUN_next_pre", None)],
    "TAC-BUN(時間加重)": [("BUN", PRE), ("BUN", POST), ("BUN_next_pre", None),
                       ("Td", None), ("interdialytic_hours", None)],
    "補正Ca":        [("Ca", None), ("Alb", None)],
    "iCa(mg/dL)":   [("Ca", None), ("Alb", None)],
    "TSAT":         [("Fe", None), ("TIBC", None)],
    "GNRI":         [("Alb", None), ("weight", None), ("height", None)],
    "BMI":          [("weight", None), ("height", None)],
}


def check_requirements(ts: TimingSchema, available: dict) -> pd.DataFrame:
    """各指標について、必要な (項目, 時点) が揃っているかを判定する。

    Parameters
    ----------
    available : dict
        時点を問わない項目（Td, age, sex, V_post 等）の列名を渡す。
        例: {"Td": "透析時間", "age": "年齢", "sex": "性別"}

    Returns
    -------
    DataFrame  指標 / 算出可否 / 不足している入力
    """
    rows = []
    for name, reqs in REQUIREMENTS.items():
        missing = []
        for base, t in reqs:
            if t is None:
                if base in available and available[base] is not None:
                    continue
                col = ts.get(base, PRE) or ts.get(base, POST) or ts.get(base, UNKNOWN)
                if col is None:
                    missing.append(base)
            else:
                if ts.get(base, t) is None:
                    # 不詳の列があるなら「時点不詳のため使えない」と明示する
                    if ts.get(base, UNKNOWN) is not None:
                        missing.append(f"{base}（{_LABEL[t]}。列はあるが採血時点が不詳）")
                    else:
                        missing.append(f"{base}（{_LABEL[t]}）")
        rows.append((name, "算出可" if not missing else "算出不可",
                     "、".join(missing) if missing else "—"))
    return pd.DataFrame(rows, columns=["指標", "判定", "不足している入力"])
