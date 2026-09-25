"""medprep.quality — データ品質監査。**このまま解析してよいか**を先に問う。

`schema.py` が「この列は何か」を決めるのに対し、`audit.py` は
「このデータで出した結果を信じてよいか」を問う。

前処理でいちばん危ないのは、壊れたデータが**例外を出さずに通ってしまう**ことである。
Hb 0 g/dL は clean が捕まえる。だが次のようなものは、どの列も単独では正常に見える。

    * 同じ患者が 2 行ある（Cox も logistic も独立を仮定している）
    * 透析前と透析後の列が入れ替わっている（BUN が透析で上がっている）
    * 白血球分画の合計が 140%
    * 群 D が n=8 しかないのに Table 1 で検定している
    * 2020 年 4 月を挟んで ALP の分布に段差がある（測定法変更であって病態ではない）
    * 「転帰」列が目的変数とほぼ同じ（リーク）
    * 欠測が施設に強く偏っている（単純補完が群間差を作る）

これらは**型の決まった壊れ方**である。型が分かっているものは機械に見張らせる。

モジュール名について
--------------------
公開する関数の名前は `audit()` である。モジュール名を `audit.py` にすると
`import medprep.audit` がパッケージ属性 `medprep.audit`（関数）を
モジュールで上書きしてしまい、`mp.audit(df)` が突然呼べなくなる。
名前がふたつの意味を持つ状態を作らないため、モジュールは `quality` とした。

方針
----
1. **audit は直さない。報告する。** 何をどう直すかは人が決める。
   自動修正は「気づかないうちに直っていた」を生み、それは壊れ方として最悪である。
2. **所見には必ず「どうすればよいか」を書く。** 警告だけ出して放り出さない。
3. **error は「このまま解析すると誤った結論になる」ものに限る。**
   error を乱発すると読まれなくなる。

使い方
------
    import medprep as mp

    sch = mp.Schema.infer(df, id_col="仮名ID", group="施設",
                          survival_dates=("観察開始年月日", "event発生年月日", "観察打ち切り年月日"))
    rep = mp.audit(df, sch, id_col="仮名ID", group="施設", date_col="観察開始年月日")
    rep.show()
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from .clean import build_alias_map, load_dict
from .dates import parse_date_series
from .schema import (
    BINARY,
    DATETIME,
    DUPLICATE,
    GROUP,
    HIGH_CARDINALITY,
    NOMINAL,
    NUMERIC,
    ORDINAL,
    Schema,
    cramers_v,
)
from .timing import POST, PRE, TimingSchema

ERROR, WARN, INFO = "error", "warn", "info"
_ORDER = {ERROR: 0, WARN: 1, INFO: 2}
_MARK = {ERROR: "✗", WARN: "!", INFO: "·"}

# 透析の前後で値が動く向き。**逆になっていれば列の取り違えを疑う。**
#   減少: 透析で除去される溶質と、除水による体重
#   増加: 血液濃縮で見かけ上あがるもの（Hb・Ht・TP・Alb）
# 血圧は個人差が大きく向きが定まらないので入れない。
PRE_POST_DIRECTION: dict[str, str] = {
    "BUN": "decrease", "Cr": "decrease", "K": "decrease", "P": "decrease",
    "UA": "decrease", "B2MG": "decrease", "Mg": "decrease", "weight": "decrease",
    "Hb": "increase", "Ht": "increase", "TP": "increase", "Alb": "increase",
}


def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").lower()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ================================================================== 所見
@dataclass
class Finding:
    """1 件の所見。**`action`（どうすればよいか）を必ず持つ。**"""
    severity: str
    category: str
    message: str
    action: str
    columns: list = field(default_factory=list)
    n: int | None = None
    examples: list = field(default_factory=list)      # 該当する症例（最大 20）

    def __str__(self) -> str:
        head = f"[{_MARK[self.severity]}] {self.category}: {self.message}"
        if self.n is not None:
            head += f"（{self.n} 件）"
        cols = [str(c) for c in self.columns if c and str(c) not in self.message]
        if cols:
            shown = "、".join(cols[:10])
            if len(cols) > 10:
                shown += f" ほか{len(cols) - 10}列"
            head += f"\n      対象列: {shown}"
        return head + f"\n      → {self.action}"


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    checked: list = field(default_factory=list)       # 実施した検査の名前
    skipped: list = field(default_factory=list)       # (検査名, 実施できなかった理由)

    # ------------------------------------------------------------ 参照
    def by_severity(self, sev: str) -> list:
        return [f for f in self.findings if f.severity == sev]

    @property
    def errors(self) -> list:
        return self.by_severity(ERROR)

    @property
    def warnings(self) -> list:
        return self.by_severity(WARN)

    @property
    def ok(self) -> bool:
        """error が 1 件も無ければ True。warn は残っていてよい。"""
        return not self.errors

    # ------------------------------------------------------------ 報告
    def to_frame(self) -> pd.DataFrame:
        rows = [{
            "重大度": f.severity, "区分": f.category,
            "列": "、".join(map(str, f.columns)) if f.columns else "—",
            "件数": f.n, "所見": f.message, "対応": f.action,
        } for f in sorted(self.findings, key=lambda x: (_ORDER[x.severity], x.category))]
        return pd.DataFrame(rows, columns=["重大度", "区分", "列", "件数", "所見", "対応"])

    def report(self) -> str:
        lines = [f"データ品質監査  {self.n_rows} 行 × {self.n_cols} 列",
                 f"  致命的 {len(self.errors)} 件 / 要確認 {len(self.warnings)} 件 / "
                 f"記録 {len(self.by_severity(INFO))} 件"
                 f"（検査 {len(self.checked)} 種）"]
        if not self.findings:
            lines.append("\n  所見なし。")
        for sev in (ERROR, WARN, INFO):
            fs = self.by_severity(sev)
            if not fs:
                continue
            label = {ERROR: "致命的 — このまま解析すると誤った結論になる",
                     WARN: "要確認", INFO: "記録"}[sev]
            lines.append(f"\n── {label} ──")
            for f in fs:
                lines.append(str(f))
                if f.examples:
                    ex = "、".join(map(str, f.examples[:8]))
                    more = f" ほか{len(f.examples) - 8}件" if len(f.examples) > 8 else ""
                    lines.append(f"        該当例: {ex}{more}")
        if self.skipped:
            lines.append("\n── 実施できなかった検査 ──")
            for name, why in self.skipped:
                lines.append(f"  {name}: {why}")
        return "\n".join(lines)

    def show(self):
        print(self.report())


# ================================================================== 補助
def _examples(df: pd.DataFrame, mask, id_col: str | None, limit: int = 20) -> list:
    idx = df.index[mask] if not isinstance(mask, pd.Index) else mask
    if id_col and id_col in df.columns:
        return [df.at[i, id_col] for i in idx[:limit]]
    return list(idx[:limit])


def _columns_for(df: pd.DataFrame, amap: dict) -> dict:
    """列名 → 辞書キー。同じキーに複数列が当たる場合は全部返す。"""
    out: dict[str, list] = {}
    for c in df.columns:
        k = amap.get(_norm(c))
        if k:
            out.setdefault(k, []).append(c)
    return out


# ================================================================== 各検査
def _check_duplicate_rows(df, id_col, findings):
    n_all = int(df.duplicated().sum())
    if n_all:
        findings.append(Finding(
            ERROR, "重複", "全列が一致する行がある", n=n_all,
            action="入力の二重登録を疑う。残す 1 行を決めてから解析すること"))

    if id_col and id_col in df.columns:
        dup = df[id_col].duplicated(keep=False) & df[id_col].notna()
        if dup.any():
            n_id = int(df.loc[dup, id_col].nunique())
            findings.append(Finding(
                ERROR, "重複", f"同じ ID が複数行に現れる（{n_id} 人ぶん）", n=int(dup.sum()),
                columns=[id_col], examples=sorted(map(str, df.loc[dup, id_col].unique()))[:20],
                action="Cox・ロジスティック回帰・t 検定はいずれも 1 人 1 行を仮定している。"
                       "経時測定なら混合効果モデルか、1 人 1 行への集約が必要"))
        # ID を除くと全列一致する行（ID だけ振り直された二重登録）
        others = [c for c in df.columns if c != id_col]
        if others:
            n_body = int(df.duplicated(subset=others).sum()) - n_all
            if n_body > 0:
                findings.append(Finding(
                    WARN, "重複", "ID 以外の全列が一致する行がある", n=n_body,
                    action="ID を振り直した二重登録の可能性。元の記録に当たって確かめること"))


def _check_identifier_candidates(df, schema, id_col, findings):
    """schema があえて保留した「識別子かどうか」をここで人に問う。"""
    if id_col:
        return
    for c, sp in schema.columns.items():
        nn = int(df[c].notna().sum())
        if not nn:
            continue
        ratio = df[c].nunique(dropna=True) / nn
        if sp.role == HIGH_CARDINALITY and ratio > 0.95:
            findings.append(Finding(
                WARN, "識別子", f"'{c}' はほぼ全行で異なる。識別子か自由記載か決まらない",
                columns=[c], n=int(df[c].nunique(dropna=True)),
                action=f"識別子なら id_col='{c}' を指定する。自由記載なら schema で role を text に、"
                       f"意味のあるカテゴリなら nominal にすること（既定では解析から外している）"))
            continue
        # 数値列は連続値としてそのまま残す（schema はユニーク率で落とさない）。
        # そのため「整数で、全例が異なり、辞書のどの項目にも当たらない」ものだけを
        # 隠れた識別子の候補として挙げる。連続値はふつう整数にならない。
        if (sp.role == NUMERIC and sp.dict_key is None and ratio == 1.0 and nn >= 20):
            v = _num(df[c]).dropna()
            if len(v) and (v % 1 == 0).all():
                findings.append(Finding(
                    WARN, "識別子", f"'{c}' は全例で異なる整数。連番の識別子ではないか",
                    columns=[c], n=int(nn),
                    action=f"識別子なら id_col='{c}' を指定すること。"
                           f"このままでは連番が説明変数として学習に入る"))


def _check_near_duplicate_columns(df, schema, findings, low=0.95, high=0.999):
    """完全重複は schema が落とす。ここは「ほぼ同じ」を人に見せる。"""
    for c, sp in schema.columns.items():
        if sp.role == DUPLICATE:
            findings.append(Finding(
                INFO, "重複列", f"'{c}' は '{sp.duplicate_of}' と実質同じ情報のため外した",
                columns=[c, sp.duplicate_of or ""],
                action="片方だけ残す判断でよいか確認すること。残す側を変えるなら schema を直す"))

    keep = [c for c, sp in schema.columns.items()
            if sp.action == "keep" and sp.role in (NUMERIC, BINARY, ORDINAL)]
    num = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 2:
        return
    corr = df[num].corr(numeric_only=True).abs()
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and low <= r < high:
                findings.append(Finding(
                    WARN, "多重共線性", f"'{a}' と '{b}' の相関が |r| = {r:.3f}",
                    columns=[a, b],
                    action="回帰係数の符号が反転しうる。片方を落とすか、合成した指標を使うこと"))


# 列名が日付を示す語。**解釈に失敗した日付列を黙って落とさないための保険。**
#   日付として解釈できなかった列は role が datetime にならないので、
#   role だけを見ていると「日付のつもりの列が検査されない」という穴ができる。
_DATE_NAME = re.compile(
    r"年月日|日付|日時|年月|date|datetime|採取日|採血日|測定日|検査日|"
    r"開始日|終了日|導入日|入院日|退院日|手術日|死亡日|生年月日|打ち切り",
    re.IGNORECASE)


def _check_dates(df, schema, findings, date_cols=None):
    if date_cols is None:
        date_cols = [c for c, sp in schema.columns.items()
                     if sp.role == DATETIME or _DATE_NAME.search(str(c))]
    cols = date_cols
    today = pd.Timestamp.today().normalize()
    for c in cols:
        if c not in df.columns:
            continue
        try:
            r = parse_date_series(df[c])
        except Exception as e:                                    # noqa: BLE001
            findings.append(Finding(
                ERROR, "日付", f"'{c}' を日付として解釈できない（{e}）", columns=[c],
                action="表記を確かめること"))
            continue
        if r.order.startswith("ambiguous"):
            findings.append(Finding(
                ERROR, "日付", f"'{c}' の日と月の並びが確定できない", columns=[c],
                action="parse_date_series(..., order='dmy') のように並びを明示すること。"
                       "推測すると観察期間が最大 11 か月ずれ、しかも気づけない"))
            continue
        if r.n_failed:
            findings.append(Finding(
                WARN, "日付", f"'{c}' に解釈できない値がある（成功率 {r.success_rate:.1%}）",
                columns=[c], n=r.n_failed,
                examples=[repr(v) for _i, v in r.failures[:20]],
                action="表記のゆれを直すか、欠測として扱ってよいか判断すること"))
        v = r.values.dropna()
        if len(v) == 0:
            continue
        future = int((v > today).sum())
        if future:
            findings.append(Finding(
                ERROR, "日付", f"'{c}' に未来の日付がある（最大 {v.max().date()}）",
                columns=[c], n=future,
                action="入力ミスか、和暦・西暦の取り違えを疑う"))
        old = int((v < pd.Timestamp("1900-01-01")).sum())
        if old:
            findings.append(Finding(
                WARN, "日付", f"'{c}' に 1900 年より前の日付がある（最小 {v.min().date()}）",
                columns=[c], n=old,
                action="Excel シリアル値や 2 桁年の解釈ミスを疑う"))


def _check_survival(df, schema, id_col, findings):
    """日付 3 列から生存時間を導けるか、矛盾がどれだけあるかを見る。"""
    sv = schema.survival
    if not sv:
        return
    if sv.get("form") != "C":
        return
    start, ev, cens = sv["start_date"], sv["event_date"], sv["censor_date"]
    if not all(c in df.columns for c in (start, ev, cens)):
        return
    s = parse_date_series(df[start]).values
    e = parse_date_series(df[ev]).values
    c = parse_date_series(df[cens]).values

    both = e.notna() & c.notna()
    if both.any():
        findings.append(Finding(
            ERROR, "生存時間", "イベント発生日と打ち切り日の両方が入っている",
            n=int(both.sum()), columns=[ev, cens],
            examples=_examples(df, both, id_col),
            action="どちらが正しいかは機械には決まらない。元の記録で確かめること。"
                   "build_survival は既定でこれらを除外し、除外表に理由を残す"))
    neither = e.isna() & c.isna() & s.notna()
    if neither.any():
        findings.append(Finding(
            ERROR, "生存時間", "イベント発生日も打ち切り日も入っていない",
            n=int(neither.sum()), columns=[ev, cens],
            examples=_examples(df, neither, id_col),
            action="観察終了日が無いと観察期間が出せない。"
                   "最終観察日を打ち切り日として補えるか確認すること"))
    nostart = s.isna() & (e.notna() | c.notna())
    if nostart.any():
        findings.append(Finding(
            ERROR, "生存時間", "観察開始日が無い", n=int(nostart.sum()), columns=[start],
            examples=_examples(df, nostart, id_col),
            action="開始日が無ければ観察期間は出せない。透析導入日・登録日で補えるか確認すること"))

    end = e.fillna(c)
    rev = s.notna() & end.notna() & (end < s)
    if rev.any():
        findings.append(Finding(
            ERROR, "生存時間", "観察終了日が観察開始日より前になっている",
            n=int(rev.sum()), columns=[start, ev, cens],
            examples=_examples(df, rev, id_col),
            action="2 列の取り違えか入力ミス。逆転したまま解析すると観察期間が負になる"))
    dur = (end - s).dt.days
    zero = dur == 0
    if zero.any():
        findings.append(Finding(
            WARN, "生存時間", "観察期間が 0 日の症例がある", n=int(zero.sum()),
            examples=_examples(df, zero, id_col),
            action="lifelines は 0 を扱えるが Kaplan-Meier の最初の段が歪む。"
                   "build_survival は既定で 0.5 日に置き換える（zero_duration= で変更可）"))
    long = dur > 365.25 * 60
    if long.any():
        findings.append(Finding(
            WARN, "生存時間", "観察期間が 60 年を超える症例がある", n=int(long.sum()),
            examples=_examples(df, long, id_col),
            action="年の入力ミスを疑う"))

    n_event = int(e.notna().sum())
    if n_event == 0:
        findings.append(Finding(
            ERROR, "生存時間", "イベントが 1 件も無い",
            action="Cox 回帰も log-rank も計算できない。イベント定義と日付列の対応を確かめること"))
    elif n_event < 10:
        findings.append(Finding(
            WARN, "生存時間", f"イベント数が {n_event} 件しかない", n=n_event,
            action="多変量 Cox は共変量 1 つにつきイベント 10 件が目安（EPV≥10）。"
                   "単変量か、共変量を絞ること"))


def _check_outcome(df, schema, task, findings):
    name = schema.target["name"] if schema.target else None
    if not name or name not in df.columns:
        return
    s = df[name]
    miss = int(s.isna().sum())
    if miss:
        findings.append(Finding(
            WARN if miss / len(df) < 0.2 else ERROR, "目的変数",
            f"目的変数 '{name}' が欠測している", n=miss, columns=[name],
            action="目的変数の欠測は補完してはならない。該当行を解析から外すこと。"
                   "欠測が群に偏っていないかも確かめること"))
    nu = s.nunique(dropna=True)
    if nu <= 1:
        findings.append(Finding(
            ERROR, "目的変数", f"目的変数 '{name}' の値が {nu} 種類しかない",
            columns=[name], action="このままでは学習も検定もできない"))
        return
    t = task or ("classification" if nu <= 10 else "regression")
    if t.startswith("class"):
        vc = s.value_counts()
        minority, majority = int(vc.min()), int(vc.max())
        if minority < 10:
            findings.append(Finding(
                ERROR, "目的変数",
                f"最小クラス '{vc.idxmin()}' が {minority} 例しかない",
                n=minority, columns=[name],
                action="交差検証の分割で 0 例の fold ができる。クラスを統合するか、"
                       "層化分割と適切な評価指標（PR-AUC 等）を使うこと"))
        elif majority / max(minority, 1) >= 9:
            findings.append(Finding(
                WARN, "目的変数",
                f"クラス不均衡が {majority}:{minority}（{majority / minority:.1f} 倍）",
                columns=[name],
                action="正解率は無意味になる。PR-AUC・感度/特異度で評価し、"
                       "層化分割を必ず使うこと"))
    else:
        v = _num(s).dropna()
        if len(v) and v.std(ddof=0) == 0:
            findings.append(Finding(
                ERROR, "目的変数", f"目的変数 '{name}' の分散が 0", columns=[name],
                action="回帰できない"))


def _check_leakage(df, schema, findings, threshold=0.99):
    """目的変数とほぼ一対一で対応する説明変数を探す。

    「転帰」や「死亡日」のような列が残っていると、モデルは完璧な性能を出し、
    そして外部データでまったく動かない。これは前処理の失敗のうち最も
    気づきにくいものである。
    """
    if not schema.target:
        return
    y_name = schema.target["name"]
    if y_name not in df.columns:
        return
    y = df[y_name]
    y_num = pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10
    for c in schema.features():
        if c not in df.columns or c == y_name:
            continue
        x = df[c]
        try:
            if y_num and pd.api.types.is_numeric_dtype(x):
                r = abs(_num(x).corr(_num(y)))
                stat, label = r, f"|r| = {r:.4f}"
            else:
                if x.nunique(dropna=True) > 50:
                    continue
                v = cramers_v(x.astype(str), y.astype(str))
                if v is None:
                    continue
                stat, label = v, f"Cramér's V = {v:.4f}"
        except Exception:                                          # noqa: BLE001
            continue
        if pd.notna(stat) and stat >= threshold:
            findings.append(Finding(
                ERROR, "リーク", f"'{c}' が目的変数 '{y_name}' とほぼ一致する（{label}）",
                columns=[c, y_name],
                action="目的変数から導かれた列（転帰・死亡日・退院時の状態など）が"
                       "説明変数に残っていないか確かめること。残したままなら性能は偽物になる"))


def _check_groups(df, schema, group, findings, small=10):
    cols = [group] if group else schema.by_role(GROUP)
    for c in cols:
        if not c or c not in df.columns:
            continue
        vc = df[c].value_counts()
        tiny = vc[vc < small]
        if len(tiny):
            findings.append(Finding(
                WARN, "群", f"'{c}' に例数の少ない水準がある（"
                            + "、".join(f"{k}={v}" for k, v in tiny.items()) + f"、いずれも {small} 未満）",
                columns=[c], n=int(tiny.sum()),
                action="この水準を含む検定は不安定になる。統合するか『その他』にまとめるか、"
                       "解析から外すかを決めて schema に残すこと"))
        if len(vc) > 20:
            findings.append(Finding(
                WARN, "群", f"'{c}' の水準が {len(vc)} 個ある", columns=[c],
                action="one-hot にすると列が一気に増える。上位の水準だけ残すか、"
                       "min_frequency でまとめること"))


def _check_missing(df, schema, group, findings, high=0.5, mid=0.2):
    # ★生存時間の日付列は対象外★
    #   形式C では「イベント発生日が空欄」＝イベントが起きなかった、という意味を持つ。
    #   これを欠測率として報告すると、意味のある空欄を補完する誘導になる。
    sv = schema.survival or {}
    survival_dates = {sv.get("event_date"), sv.get("censor_date"), sv.get("start_date")} - {None}
    if survival_dates and sv.get("form") == "C":
        findings.append(Finding(
            INFO, "欠測", "生存時間の日付列は欠測率の対象から外した",
            columns=sorted(survival_dates),
            action="形式C では『イベント発生日が空欄』＝イベント無し、という意味を持つ。"
                   "これらの空欄を補完してはならない"))

    for c, sp in schema.columns.items():
        if sp.action != "keep" or c not in df.columns or c in survival_dates:
            continue
        m = float(df[c].isna().mean())
        if m >= high:
            findings.append(Finding(
                WARN, "欠測", f"'{c}' の欠測率が {m:.1%}", columns=[c],
                n=int(df[c].isna().sum()),
                action="半分以上を補完で埋めた列は、実質的に補完アルゴリズムの出力である。"
                       "列ごと外すか、欠測ありなしの 2 値として扱うことを検討すること"))
        elif m >= mid:
            findings.append(Finding(
                INFO, "欠測", f"'{c}' の欠測率が {m:.1%}", columns=[c],
                n=int(df[c].isna().sum()),
                action="補完すること自体は妥当だが、欠測がどこにあったかを必ず残すこと"
                       "（5_training_data・6_test_data（本コード専用） の「欠損値の位置」シート）"))

    rowmiss = df.isna().mean(axis=1)
    bad = rowmiss >= high
    if bad.any():
        findings.append(Finding(
            WARN, "欠測", "半分以上の列が欠測している行がある", n=int(bad.sum()),
            action="登録だけして検査が入っていない症例の可能性。解析対象に含めるか決めること"))

    # 欠測が群と関連しているか（MCAR でなければ、単純補完が群間差を作る）
    if group and group in df.columns:
        g = df[group]
        for c, sp in schema.columns.items():
            if (sp.action != "keep" or c not in df.columns or c == group
                    or c in survival_dates):
                continue
            miss = df[c].isna()
            if not (0.02 < miss.mean() < 0.98):
                continue
            t = pd.crosstab(g, miss)
            # ★観測度数 0 のセルで弾かない。★
            #   「A院は全例あり、B院は全例欠測」がいちばん強い偏りであり、
            #   そこには 0 のセルが必ずできる。周辺度数が 0 の行・列だけ除く。
            if t.shape[0] < 2 or t.shape[1] < 2:
                continue
            t = t.loc[t.sum(axis=1) > 0, t.sum(axis=0) > 0]
            if t.shape[0] < 2 or t.shape[1] < 2:
                continue
            try:
                p = chi2_contingency(t)[1]
            except Exception:                                      # noqa: BLE001
                continue
            if p < 1e-3:
                rates = (miss.groupby(g).mean().sort_values(ascending=False))
                detail = "、".join(f"{k}={v:.0%}" for k, v in rates.items())
                findings.append(Finding(
                    WARN, "欠測", f"'{c}' の欠測が '{group}' に偏っている（{detail}、p<0.001）",
                    columns=[c, group],
                    action="MCAR ではない。全体の中央値で埋めると群間差が人工的に作られる。"
                           "群別に補完するか、多重代入を使うこと"))


def _check_value_spellings(df, schema, findings):
    """『男』『男性』『ＭALE 』のような表記ゆれを見つける。

    正規化すると同じになる値が 2 通り以上あれば、別の水準として数えられている。
    """
    for c, sp in schema.columns.items():
        if c not in df.columns or sp.role not in (NOMINAL, BINARY, GROUP, HIGH_CARDINALITY):
            continue
        v = df[c].dropna()
        if len(v) == 0 or v.nunique() > 100 or pd.api.types.is_numeric_dtype(v):
            continue
        buckets: dict[str, set] = {}
        for raw in v.astype(str).unique():
            buckets.setdefault(_norm(raw), set()).add(raw)
        clashes = {k: s for k, s in buckets.items() if len(s) > 1}
        if clashes:
            ex = "；".join(" / ".join(sorted(map(repr, s))) for s in list(clashes.values())[:3])
            findings.append(Finding(
                WARN, "表記ゆれ", f"'{c}' に、正規化すると同じになる値が別々の水準として入っている",
                columns=[c], n=len(clashes),
                examples=[ex],
                action="全角半角・前後の空白・大文字小文字をそろえること。"
                       "そろえないと同じ群が 2 つに割れ、例数が減り、one-hot の列が増える"))


def _check_compositions(df, amap, dic, id_col, findings):
    """分画の検算。白血球分画が 100% にならない行を探す。"""
    colmap = _columns_for(df, amap)
    for gname, g in (dic.get("composition_groups") or {}).items():
        members = g.get("members") or []
        cols = [colmap[m][0] for m in members if m in colmap]
        if len(cols) < max(2, len(members) - 1):
            continue
        sub = df[cols].apply(_num)
        complete = sub.notna().all(axis=1)
        if not complete.any():
            continue
        total = sub[complete].sum(axis=1)
        target = g.get("sum_to")
        tol = g.get("tolerance", 3)
        if target is None:
            # 「好中球の内訳」のように、合計が別の列と一致するはずのもの
            ref = g.get("equals")
            if not ref or ref not in colmap:
                continue
            rv = _num(df[colmap[ref][0]])[complete]
            bad = (total - rv).abs() > tol
            if bad.any():
                findings.append(Finding(
                    WARN, "検算", f"{gname}: {' + '.join(cols)} が '{colmap[ref][0]}' と "
                                  f"±{tol} を超えてずれている",
                    columns=cols + [colmap[ref][0]], n=int(bad.sum()),
                    examples=_examples(df[complete], bad, id_col),
                    action="入力ミスか、別の検体の値が混ざっている可能性"))
            continue
        bad = (total - target).abs() > tol
        if bad.any():
            findings.append(Finding(
                WARN if bad.mean() < 0.1 else ERROR, "検算",
                f"{gname}の合計が {target}±{tol}% に収まらない"
                f"（中央値 {total.median():.1f}%）",
                columns=cols, n=int(bad.sum()),
                examples=_examples(df[complete], bad, id_col),
                action=f"検算に使う項目は {members}。"
                       + (f"{g['excluded_members']} は合計に含めない（"
                          "好中球＝桿状核球＋分葉核球のため二重計上になる）。"
                          if g.get("excluded_members") else "")
                       + "列の対応が正しいか確かめること"))
        if len(cols) < len(members):
            missing = [m for m in members if m not in colmap]
            findings.append(Finding(
                INFO, "検算", f"{gname}: {missing} の列が見つからず、残りだけで検算した",
                columns=cols, action="列名を辞書の別名に合わせれば全項目で検算できる"))


def _check_cross_item_consistency(df, amap, dic, id_col, findings):
    """項目どうしの関係から、単独では正常に見える誤りを見つける。"""
    colmap = _columns_for(df, amap)

    def col(k):
        return colmap[k][0] if k in colmap else None

    # --- TSAT > 100%（Fe > TIBC はあり得ない）
    fe, tibc = col("Fe"), col("TIBC")
    if fe and tibc:
        a, b = _num(df[fe]), _num(df[tibc])
        bad = a.notna() & b.notna() & (a > b)
        if bad.any():
            findings.append(Finding(
                ERROR, "検算", "血清鉄が TIBC を超えている（TSAT > 100% になる）",
                columns=[fe, tibc], n=int(bad.sum()),
                examples=_examples(df, bad, id_col),
                action="TIBC は Fe + UIBC であり、定義上 Fe を下回れない。"
                       "列の取り違えか単位違いを疑う"))

    # --- 補正Ca の再計算（Payne 式）
    cca, ca, alb = col("cCa"), col("Ca"), col("Alb")
    if cca and ca and alb:
        c_, a_, l_ = _num(df[cca]), _num(df[ca]), _num(df[alb])
        expect = np.where(l_ < 4.0, a_ + (4.0 - l_), a_)
        ok = c_.notna() & a_.notna() & l_.notna()
        bad = ok & (np.abs(c_ - expect) > 0.15)
        if bad.any():
            findings.append(Finding(
                WARN, "検算", "補正Ca（iCa）が Payne 式（Alb<4 のとき Ca+(4-Alb)）と一致しない",
                columns=[cca, ca, alb], n=int(bad.sum()),
                examples=_examples(df, bad, id_col),
                action="別の補正式を使っているなら、どの式かを記録すること。"
                       "式が違えば管理目標の達成率も変わる"))

    # --- 収縮期血圧 < 拡張期血圧
    sbp, dbp = col("SBP"), col("DBP")
    if sbp and dbp:
        a, b = _num(df[sbp]), _num(df[dbp])
        bad = a.notna() & b.notna() & (a <= b)
        if bad.any():
            findings.append(Finding(
                ERROR, "検算", "収縮期血圧が拡張期血圧以下になっている",
                columns=[sbp, dbp], n=int(bad.sum()),
                examples=_examples(df, bad, id_col),
                action="2 列の取り違えを疑う"))

    # --- 身長・体重から BMI があり得ない値になる
    h, w = col("height"), col("weight")
    if h and w:
        hh, ww = _num(df[h]) / 100.0, _num(df[w])
        with np.errstate(divide="ignore", invalid="ignore"):
            bmi = ww / (hh ** 2)
        bad = bmi.notna() & ((bmi < 10) | (bmi > 60))
        if bad.any():
            findings.append(Finding(
                WARN, "検算", "身長と体重から計算した BMI があり得ない範囲になる",
                columns=[h, w], n=int(bad.sum()),
                examples=_examples(df, bad, id_col),
                action="身長の単位（cm / m）と体重の単位（kg / g）を確かめること"))


def _check_timing(df, ts, schema, findings):
    """採血時点が不詳の列を、透析指標の算出可否とあわせて報告する。"""
    dict_keys = {c: sp.dict_key for c, sp in schema.columns.items() if sp.dict_key}
    unknown = [c for c in ts.unknown_columns(only_required=True, dict_keys=dict_keys)
               if c in df.columns]
    if unknown:
        findings.append(Finding(
            WARN, "採血時点", "透析前後で値が変わる項目なのに、採血時点が列名から読み取れない",
            columns=unknown, n=len(unknown),
            action="列名に『透析前』『post』等を入れるか、"
                   "データ全体が透析前値なら TimingSchema.infer(df, default_timing='pre') と"
                   "**人が明示する**こと。不詳のままでは spKt/V・nPCR・%CGR は算出しない"))

    # 前後の組を要する項目が片方しか無い場合（辞書キーで照合する）
    for base in PRE_POST_DIRECTION:
        pre_col, post_col = ts.get(base, PRE), ts.get(base, POST)
        if bool(pre_col) == bool(post_col):
            continue
        have, lack = ("前", "後") if pre_col else ("後", "前")
        findings.append(Finding(
            INFO, "採血時点", f"'{base}' は透析{have}の値だけがあり、透析{lack}の列が無い",
            columns=[pre_col or post_col],
            action="前後の組を要する指標（URR・spKt/V・nPCR・%CGR・クリアスペース率）は"
                   "算出できない。片方だけで評価できる指標に限ること"))


def _check_pre_post_direction(df, ts, findings, id_col=None):
    """透析前後の大小関係が生理と逆になっていないか。

    ★列の取り違えを見つけるいちばん確実な方法である。★
    BUN は透析で必ず下がる。下がっていない症例が過半数なら、
    それは病態ではなく列の取り違えである。
    """
    for base, direction in PRE_POST_DIRECTION.items():
        pre_col, post_col = ts.get(base, PRE), ts.get(base, POST)
        if not pre_col or not post_col:
            continue
        if pre_col not in df.columns or post_col not in df.columns:
            continue
        a, b = _num(df[pre_col]), _num(df[post_col])
        ok = a.notna() & b.notna()
        if ok.sum() < 10:
            continue
        viol = ok & ((b >= a) if direction == "decrease" else (b <= a))
        rate = viol.sum() / ok.sum()
        word = "下がる" if direction == "decrease" else "上がる"
        if rate > 0.5:
            findings.append(Finding(
                ERROR, "採血時点",
                f"'{base}' が透析後に{word}はずなのに、{rate:.0%} の症例で逆になっている",
                columns=[pre_col, post_col], n=int(viol.sum()),
                examples=_examples(df, viol, id_col),
                action=f"'{pre_col}' と '{post_col}' の取り違えを強く疑う。"
                       f"取り違えたまま spKt/V を計算すると、透析量が過大に評価される"))
        elif rate > 0.1:
            findings.append(Finding(
                WARN, "採血時点",
                f"'{base}' の透析前後の向きが逆の症例がある（{rate:.0%}）",
                columns=[pre_col, post_col], n=int(viol.sum()),
                examples=_examples(df, viol, id_col),
                action="個別の入力ミスか、採血の順序が守られなかった回の可能性。"
                       "該当症例を確かめること"))


# ================================================================== 測定法変更
def method_change_steps(
    df: pd.DataFrame,
    date_col: str,
    dic: dict | None = None,
    *,
    min_per_side: int = 20,
    ratio_tol: float = 0.15,
    alpha: float = 1e-3,
) -> list:
    """測定法が変わった日を挟んで、分布に段差があるかを調べる。

    ALP は 2020 年 4 月に JSCC 法から IFCC 法へ切り替わり、**同じ患者の同じ状態で
    値がおよそ 1/3 になった**。この段差を病態の変化として解析すると、
    「2020 年以降の症例は骨代謝が良い」という存在しない所見が出る。

    ここで見ているのは相関ではなく**日付を境にした分布の段差**である。
    段差があること自体は測定法変更の証拠ではないが、辞書に記録された
    変更日と一致していれば、まず測定法を疑うべきである。

    Parameters
    ----------
    date_col : 検体採取日（無ければ観察開始日など、値の時期を代表する日付列）
    ratio_tol : 前後の中央値の比がこの割合を超えてずれていたら段差とみなす
    """
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    colmap = _columns_for(df, amap)
    out: list = []

    if date_col not in df.columns:
        return [Finding(INFO, "測定法", f"日付列 '{date_col}' が無いため段差を調べられない",
                        action="検体採取日の列を渡すこと")]
    dates = parse_date_series(df[date_col]).values

    for ch in dic.get("measurement_changes") or []:
        item = ch.get("item")
        cols = colmap.get(item)
        if not cols:
            continue
        col = cols[0]
        raw_date = str(ch.get("date", ""))
        try:
            cut = pd.Timestamp(raw_date if len(raw_date) > 7 else raw_date + "-01")
        except Exception:                                          # noqa: BLE001
            out.append(Finding(
                INFO, "測定法", f"'{col}' は測定法変更のある項目（{ch.get('change')}）だが、"
                                f"変更日が『{raw_date}』で特定できない",
                columns=[col],
                action=(ch.get("action") or "施設に測定法と切替日を確認し、"
                        "測定法を列として持つこと")))
            continue

        v = _num(df[col])
        before = v[dates.notna() & (dates < cut) & v.notna()]
        after = v[dates.notna() & (dates >= cut) & v.notna()]
        if len(before) < min_per_side or len(after) < min_per_side:
            out.append(Finding(
                INFO, "測定法", f"'{col}' は {cut.date()} に {ch.get('change')} があるが、"
                                f"前後の例数（{len(before)} / {len(after)}）が足りず判定できない",
                columns=[col],
                action="データが変更日の片側に収まっているなら問題ない。"
                       "またいでいるなら測定法を確認すること"))
            continue

        mb, ma = float(before.median()), float(after.median())
        ratio = ma / mb if mb else float("nan")
        try:
            p = mannwhitneyu(before, after, alternative="two-sided").pvalue
        except Exception:                                          # noqa: BLE001
            p = float("nan")
        stepped = pd.notna(ratio) and abs(ratio - 1.0) > ratio_tol and pd.notna(p) and p < alpha
        if stepped:
            out.append(Finding(
                ERROR, "測定法",
                f"'{col}' は {cut.date()}（{ch.get('change')}）の前後で中央値が "
                f"{mb:g} → {ma:g}（{ratio:.2f} 倍、p={p:.1e}、前 {len(before)} 例 / "
                f"後 {len(after)} 例）と段差になっている",
                columns=[col],
                action=(ch.get("action")
                        or "期間を共変量に入れるか期間で層別すること。"
                           "換算して 1 本の列にするのは、個別値の厳密換算ができないため勧めない")
                + (f"（既知の影響: {ch['effect']}）" if ch.get("effect") else "")))
        else:
            out.append(Finding(
                INFO, "測定法",
                f"'{col}' は {cut.date()} に {ch.get('change')} があるが、"
                f"前後の中央値は {mb:g} → {ma:g}（{ratio:.2f} 倍）で大きな段差は見えない",
                columns=[col],
                action="施設が切替前から新法を使っていた可能性もある。測定法を記録しておくこと"))
    return out


# ================================================================== 入口
def audit(
    df: pd.DataFrame,
    schema: Schema | None = None,
    *,
    id_col: str | None = None,
    outcome: str | None = None,
    task: str | None = None,
    group: str | None = None,
    survival: tuple | None = None,
    survival_dates: tuple | None = None,
    date_col: str | None = None,
    default_timing: str | None = None,
    dic: dict | None = None,
) -> AuditReport:
    """データ品質監査を実行する。

    `schema` を渡さなければその場で推定する。**audit はデータを変更しない。**

    Parameters
    ----------
    date_col : 測定法変更の段差を調べるための日付列（検体採取日が望ましい）
    default_timing : 「このデータは全部透析前値である」と人が明示するときだけ渡す
    """
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    if schema is None:
        schema = Schema.infer(df, outcome=outcome, task=task, group=group,
                              survival=survival, survival_dates=survival_dates,
                              id_col=id_col, dic=dic)
    ts = TimingSchema.infer(df, default_timing=default_timing, alias_map=amap)

    findings: list = []
    rep = AuditReport(findings=findings, n_rows=len(df), n_cols=df.shape[1])

    steps = [
        ("重複行・重複ID", lambda: _check_duplicate_rows(df, id_col, findings)),
        ("識別子の候補", lambda: _check_identifier_candidates(df, schema, id_col, findings)),
        ("重複列・多重共線性", lambda: _check_near_duplicate_columns(df, schema, findings)),
        ("日付の整合性", lambda: _check_dates(df, schema, findings)),
        ("生存時間の整合性", lambda: _check_survival(df, schema, id_col, findings)),
        ("目的変数", lambda: _check_outcome(df, schema, task, findings)),
        ("リーク", lambda: _check_leakage(df, schema, findings)),
        ("群サイズ", lambda: _check_groups(df, schema, group, findings)),
        ("欠測の分布", lambda: _check_missing(df, schema, group, findings)),
        ("表記ゆれ", lambda: _check_value_spellings(df, schema, findings)),
        ("分画の検算", lambda: _check_compositions(df, amap, dic, id_col, findings)),
        ("項目間の整合性", lambda: _check_cross_item_consistency(df, amap, dic, id_col, findings)),
        ("採血時点", lambda: _check_timing(df, ts, schema, findings)),
        ("透析前後の向き", lambda: _check_pre_post_direction(df, ts, findings, id_col)),
    ]
    for name, fn in steps:
        fn()
        rep.checked.append(name)

    if date_col:
        findings.extend(method_change_steps(df, date_col, dic))
        rep.checked.append("測定法変更の段差")
    else:
        known = [c["item"] for c in (dic.get("measurement_changes") or [])
                 if c["item"] in _columns_for(df, amap)]
        if known:
            rep.skipped.append(
                ("測定法変更の段差",
                 f"日付列が指定されていない。{known} は測定法が変わった項目なので、"
                 f"audit(..., date_col='検体採取日') で調べること"))
    return rep
