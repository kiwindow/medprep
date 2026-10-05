"""medprep.outliers — 外れ値の検出と扱い。

**外れ値と入力ミスは別のものである。**

  Hb 0 g/dL は外れ値ではない。**入力ミス**である。winsorize して 6.5 に丸めては
  ならない。NaN にして補完へ回すのが正しい。この区別は医学領域辞書
  （`dict/ranges_ja.yaml` の `plausible`）が担い、`clean.py` が先に処理している。

  ここで扱うのは「生理学的にはあり得るが、この集団の中では極端」な値である。
  （0.13.0〜）ただし辞書に無い列では、入力ミスも素通りしてここまで来る。
  そこで `suspect_entry_errors` が、ほかの値から**桁違いに**離れた値を
  「入力ミスの可能性がある値」として人に返す（直すのは人。ここでは直さない）。
  CRP 28 mg/dL は入力ミスではない。敗血症の症例である。
  **削除してはならない。** 医学では外れ値こそが重要な症例でありうる。

したがって既定は `flag`（印をつけるだけ）と `winsorize`（上下限に丸める）であり、
`drop`（症例ごと捨てる）は人が明示したときにしか行わない。

★閾値は train でしか計算してはならない★
--------------------------------------
IQR の上下限も MAD も、**データから学習するパラメータ**である。
全データで閾値を決めてから分割すれば、test の情報が train に漏れる。
そこで winsorize は sklearn の transformer（`NanSafeWinsorizer`）として実装し、
`pipeline.py` で `ColumnTransformer` の中に閉じ込める。`fit` は train にしか呼べない。

なぜ自作するか
--------------
feature-engine の `Winsorizer` は **NaN を含む列で `ValueError` を投げて停止する**。
医学データでは「補完の前に外れ値を処理する」順序でなければならないのに、
補完前は必ず NaN がある。順序を曲げられないので、NaN を素通しする実装を自分で持つ。

  正しい順序: あり得ない値を NaN に（clean）→ winsorize → 補完 → スケーリング
  誤った順序: 補完 → winsorize   ← 補完値が外れ値に引きずられて汚染される
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .clean import excel_row as _excel_row
from .describe import group_levels
from .schema import NUMERIC, Schema
from .textfmt import frame_text

IQR, MAD, QUANTILE = "iqr", "mad", "quantile"

#: 入力ミスの可能性がある値の既定（0.13.0〜）。schema.yaml の policy の entry_error で変えられる。
#:   z     : ロバスト z（中央値からの距離を MAD で測ったもの）がこれ以上
#:   ratio : しかも、その次に極端な値の z の ratio 倍以上離れている
ENTRY_Z = 10.0
ENTRY_RATIO = 3.0
FLAG, WINSORIZE, NAN, DROP = "flag", "winsorize", "nan", "drop"


# ================================================================== 閾値
def iqr_limits(x, fold: float = 1.5) -> tuple[float, float]:
    """四分位範囲による上下限。歪んだ分布では下限が負になりうる点に注意。"""
    v = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(v) < 4:
        return float("-inf"), float("inf")
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        # ★IQR が 0 になる列がある。★ 値が数種類しかない列では Q1 = Q3 になる。
        #   透析時間は 4.0 時間が 6 割を占めるので、まさにこれに当たる。
        #   そのまま閾値にすると上下限が同じ値になり、**列全体が 1 つの値に潰れる**。
        #   分散 0 になるのでスケーリング後は全例きっかり 0 になり、
        #   **例外は出ないまま、その変数だけがモデルから消える。**
        #   閾値を作らないのが正しい（MAD 法も同じ理由で 0 を弾いている）。
        return float("-inf"), float("inf")
    return q1 - fold * iqr, q3 + fold * iqr


def mad_limits(x, z: float = 3.0) -> tuple[float, float]:
    """中央絶対偏差によるロバストな上下限。

    正規分布で標準偏差に一致するよう 1.4826 を掛ける。
    **MAD が 0 になる列がある**（半数以上が同じ値、たとえば検出限界未満が多い CRP）。
    その場合は閾値を作らない（無限大を返す）。0 で割ると全例が外れ値になる。
    """
    v = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(v) < 4:
        return float("-inf"), float("inf")
    med = float(v.median())
    mad = float((v - med).abs().median()) * 1.4826
    if mad <= 0:
        return float("-inf"), float("inf")
    return med - z * mad, med + z * mad


def quantile_limits(x, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    v = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(v) < 4:
        return float("-inf"), float("inf")
    return float(v.quantile(lower)), float(v.quantile(upper))


_METHODS = {IQR: iqr_limits, MAD: mad_limits, QUANTILE: quantile_limits}


def limits(x, method: str = IQR, **kw) -> tuple[float, float]:
    if method not in _METHODS:
        raise ValueError(f"method は {sorted(_METHODS)} のいずれか（渡されたのは {method!r}）")
    return _METHODS[method](x, **kw)


# ================================================================== 変換器
class NanSafeWinsorizer(BaseEstimator, TransformerMixin):
    """NaN を素通ししたまま上下限に丸める。**閾値は fit でしか決めない。**

    feature-engine の `Winsorizer` は NaN があると停止するため自作した。
    補完の前に外れ値を処理するという順序を曲げないために必要である。

    Parameters
    ----------
    method : 'iqr' | 'mad' | 'quantile'
    fold   : IQR 法の係数（1.5 が慣用。3.0 は「極端な外れ値」の慣用値）
    z      : MAD 法の係数
    bounds : {列名: (下限, 上限)} を渡すと、その列は学習せずこの値を使う
             （医学領域辞書の生理学的下限を下回らせたくない場合に使う）
    """

    def __init__(self, method: str = IQR, fold: float = 1.5, z: float = 3.0,
                 lower_q: float = 0.01, upper_q: float = 0.99, bounds: dict | None = None):
        self.method = method
        self.fold = fold
        self.z = z
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds = bounds

    def _limits_for(self, s: pd.Series):
        if self.bounds and s.name in self.bounds:
            lo, hi = self.bounds[s.name]
            return float(lo), float(hi)
        if self.method == IQR:
            return iqr_limits(s, self.fold)
        if self.method == MAD:
            return mad_limits(s, self.z)
        return quantile_limits(s, self.lower_q, self.upper_q)

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.limits_ = {c: self._limits_for(X[c]) for c in X.columns}
        # 学習に使った例数を残す。test に fit していないことを後から確かめられる。
        self.n_samples_fit_ = len(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "limits_")
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            lo, hi = self.limits_.get(c, (float("-inf"), float("inf")))
            v = pd.to_numeric(X[c], errors="coerce")
            # ★NaN は NaN のまま通す★ clip は NaN を保つので明示は不要だが、
            #   意図として残す（補完はこの後の段の仕事である）。
            X[c] = v.clip(lower=lo, upper=hi)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    def limits_frame(self) -> pd.DataFrame:
        check_is_fitted(self, "limits_")
        return pd.DataFrame(
            [{"列": c, "下限": lo, "上限": hi} for c, (lo, hi) in self.limits_.items()])


# ================================================================== 報告
@dataclass
class OutlierReport:
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = IQR
    action: str = FLAG
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)   # 列ごとの真偽値
    multivariate: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)
    #: 入力ミスの可能性がある値（0.13.0〜）。`suspect_entry_errors` の表
    suspects: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_flagged_rows(self) -> int:
        return int(self.flags.any(axis=1).sum()) if len(self.flags) else 0

    def report(self) -> str:
        lines = [f"外れ値の検出（方法: {self.method}、扱い: {self.action}）",
                 frame_text(self.table) if len(self.table) else "  （対象列なし）"]
        if len(self.flags):
            lines.append(f"\nいずれかの列で外れ値と判定された症例: {self.n_flagged_rows} 例")
        if len(self.multivariate):
            lines += ["\n多変量の外れ値:", frame_text(self.multivariate)]
        if len(self.suspects):
            lines += ["\n★入力ミスの可能性がある値★（ほかの値から桁違いに離れている）:",
                      frame_text(self.suspects)]
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    def show(self):
        print(self.report())


def detect(
    df: pd.DataFrame,
    schema: Schema | None = None,
    columns: list | None = None,
    *,
    method: str = IQR,
    fold: float = 1.5,
    z: float = 3.0,
    by: str | None = None,
    id_col: str | None = None,
    max_examples: int = 10,
    entry_z: float = ENTRY_Z,
    entry_ratio: float = ENTRY_RATIO,
) -> OutlierReport:
    """外れ値を検出する。**この関数はデータを変更しない。**

    `by` を渡すと群ごとに閾値を作る（施設差がある指標で、全体の閾値では
    小さい施設の値が丸ごと外れ値になるのを避けるため）。

    `entry_z` / `entry_ratio` は「入力ミスの可能性がある値」の判定に使う（0.13.0〜、
    `suspect_entry_errors` を参照）。結果は `rep.suspects` に入る。
    """
    cols = _numeric_columns(df, schema, columns)
    rep = OutlierReport(method=method, action=FLAG)
    if not cols:
        rep.notes.append("数値列が無いため検出しなかった")
        return rep

    kw = {"fold": fold} if method == IQR else ({"z": z} if method == MAD else {})
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if by and by in df.columns:
            lo_s = pd.Series(np.nan, index=df.index)
            hi_s = pd.Series(np.nan, index=df.index)
            for lv in group_levels(df[by]):
                m = df[by] == lv
                lo, hi = limits(v[m], method, **kw)
                lo_s[m], hi_s[m] = lo, hi
            mask = v.notna() & ((v < lo_s) | (v > hi_s))
            lo_txt = "群ごと"
            hi_txt = "群ごと"
        else:
            lo, hi = limits(v, method, **kw)
            mask = v.notna() & ((v < lo) | (v > hi))
            lo_txt, hi_txt = round(lo, 3), round(hi, 3)
        flags[c] = mask
        n = int(mask.sum())
        rows.append({
            "列": c, "下限": lo_txt, "上限": hi_txt, "外れ値": n,
            "割合": f"{n / max(int(v.notna().sum()), 1):.1%}",
            "最小": round(float(v.min()), 3) if v.notna().any() else np.nan,
            "最大": round(float(v.max()), 3) if v.notna().any() else np.nan,
            "該当例": _examples(df, mask, id_col, max_examples),
        })
    rep.table = pd.DataFrame(rows)
    rep.flags = flags
    rep.suspects = suspect_entry_errors(df, schema, columns, z=entry_z,
                                        ratio=entry_ratio, id_col=id_col)

    # ★外れ値が多すぎる列は、外れ値ではなく歪んだ分布である。★
    #   正規分布なら IQR 1.5 倍の外は 0.5% ほどしか出ない。5% を超えるようなら、
    #   それは「極端な症例」ではなく「分布の形」を見ている。
    heavy = rep.table[rep.table["外れ値"] / max(len(df), 1) > 0.05]
    if len(heavy):
        rep.notes.append(
            f"外れ値が 5% を超える列がある（{list(heavy['列'])}）。"
            f"正規分布なら IQR 法で外れるのは 0.5% ほどなので、"
            f"それは外れ値ではなく『分布が歪んでいる』だけの可能性が高い。"
            f"対数変換や method='mad' を検討すること")
    if by:
        rep.notes.append(f"閾値は '{by}' の群ごとに作った。"
                         f"全体の閾値だと、水準の違う群の値がまとめて外れ値になる")
    if len(rep.suspects):
        rep.notes.append(
            f"★入力ミスの可能性がある値が {len(rep.suspects)} 個ある★"
            f"（列 {list(dict.fromkeys(rep.suspects['列']))}）。"
            "ほかの値から桁違いに離れているので、IQR の印とは分けて rep.suspects に出した。"
            "誤りなら schema.yaml の列に plausible: [下限, 上限] を書いて再実行する"
            "（範囲の外は NaN にして補完へ回す。winsorize で丸めない）")
    rep.notes.append(
        "★生理学的にあり得ない値（Hb 0 など）はここには出ない。★ "
        "clean_numeric が先に NaN にしている（辞書に載っている列だけ）。ここに出るのは"
        "『あり得るが極端』な値であり、削除してはならない症例を含む")
    return rep


# ================================================================== 入力ミスの可能性（0.13.0〜）
def robust_z(x) -> pd.Series:
    """中央値と MAD によるロバスト z（= (x − 中央値) / (1.4826 × MAD)）。

    平均と標準偏差で測る z は、測りたい外れ値そのものに引きずられる
    （230 が 1 つ入るだけで標準偏差が 10 倍になり、230 の z が小さく見える）。
    中央値と MAD は 1 つの外れ値ではほとんど動かない。

    **MAD が 0 になる列がある**（半数以上が同じ値）。その場合は平均絶対偏差で代える。
    それも 0 なら（ほぼ定数の列）全例 NaN を返す。
    """
    v = pd.to_numeric(pd.Series(x), errors="coerce")
    d = v.dropna()
    nan = pd.Series(np.nan, index=v.index)
    if len(d) < 4:
        return nan
    med = float(d.median())
    scale = float((d - med).abs().median()) * 1.4826
    if scale <= 0:
        scale = float((d - med).abs().mean()) * 1.2533
    if scale <= 0:
        return nan
    return (v - med) / scale


def suspect_entry_errors(
    df: pd.DataFrame,
    schema: Schema | None = None,
    columns: list | None = None,
    *,
    z: float = ENTRY_Z,
    ratio: float = ENTRY_RATIO,
    id_col: str | None = None,
    min_unique: int = 10,
    max_k: int | None = None,
) -> pd.DataFrame:
    """**入力ミスの可能性がある値**を挙げる。データは変更しない。

    判定（列ごと、上側と下側それぞれ）
    --------------------------------
    1. ロバスト z（`robust_z`）が `z`（既定 10）以上
    2. しかも、その次に極端な値の z の `ratio`（既定 3）倍以上離れている

    1 だけでは足りない。右に裾の長い列（人口・CRP）では、実在する大きな値も
    z が 10 を超える。**本物の極端値は「なだらかに」並ぶ**（9842, 7651, 6675, …）が、
    桁の打ち間違いや 999 のような欠損コードは**ほかの値から飛び離れる**。2 はその差を見る。

    同じ誤りが 2 つ以上ある（999 が 2 例）と、1 番目と 2 番目は離れていない。
    そこで上位 k 個（k ≤ max(1, min(5, 1% の例数))）をまとめて見て、
    「上位 k 個の後ろに大きな切れ目がある」なら k 個とも挙げる。

    辞書に載っている列は、生理学的にあり得ない値が掃除の段で既に NaN になっている。
    残っているのは「あり得る範囲の中で飛び離れた値」なので、同じ基準で見てよい。

    返す表の列: 列 / Excel の行 / ID / 値 / ロバスト z / 次の値 / 次の z / 中央値 / 向き
    """
    cols = _numeric_columns(df, schema, columns)
    out_cols = ["列", "Excel の行", "ID", "値", "ロバスト z", "次の値", "次の z", "中央値", "向き"]
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.nunique(dropna=True) < min_unique:
            continue                       # コード値・少数の水準しかない列は対象外
        rz = robust_z(v)
        if rz.isna().all():
            continue
        n = int(v.notna().sum())
        kmax = max_k or max(1, min(5, int(np.ceil(0.01 * n))))
        med = float(v.median())
        for sign, side in ((1, "大きい側"), (-1, "小さい側")):
            s = (sign * rz).dropna()
            s = s[s > 0].sort_values(ascending=False, kind="mergesort")
            if s.empty or float(s.iloc[0]) < z:
                continue
            vals = s.to_numpy(dtype=float)
            k_found = 0
            for k in range(1, min(kmax, len(vals)) + 1):
                zk = vals[k - 1]
                nxt = vals[k] if k < len(vals) else 0.0
                # ★次の値が中央値のすぐ近く（z < 1）でも、比が無限大にならないよう 1 で抑える★
                if zk >= z and zk / max(nxt, 1.0) >= ratio:
                    k_found = k
            if not k_found:
                continue
            nxt_label = s.index[k_found] if k_found < len(s) else None
            for lab in s.index[:k_found]:
                rows.append({
                    "列": c,
                    "Excel の行": _excel_row(lab),
                    "ID": df.at[lab, id_col] if (id_col and id_col in df.columns) else None,
                    "値": v.at[lab],
                    "ロバスト z": round(float(rz.at[lab]), 1),
                    "次の値": v.at[nxt_label] if nxt_label is not None else np.nan,
                    "次の z": round(float(rz.at[nxt_label]), 1) if nxt_label is not None else np.nan,
                    "中央値": med,
                    "向き": side,
                })
    out = pd.DataFrame(rows, columns=out_cols)
    if not (id_col and id_col in df.columns):
        out = out.drop(columns="ID")
    return out


def entry_error_messages(suspects: pd.DataFrame) -> list:
    """`suspect_entry_errors` の表を「人の確認が要る事項」の文にする（列ごとに 1 文）。

    ★値と ID は「（例: [...]）」の中に書く。★ レポートは既定（show_values=False）で
    この部分を伏せる。行番号は伏せない（直すのに要る。症例の値ではない）。
    """
    if suspects is None or not len(suspects):
        return []
    msgs = []
    for c, g in suspects.groupby("列", sort=False):
        where = "、".join(str(r) for r in g["Excel の行"] if r is not None and pd.notna(r))
        vals = [f"{r['ID']}: {r['値']:g}" if "ID" in g.columns and pd.notna(r.get("ID"))
                else f"{r['値']:g}" for _, r in g.iterrows()]
        top = g.iloc[0]
        nxt = (f"次に極端な値は z = {top['次の z']:g}" if pd.notna(top["次の z"])
               else "ほかに極端な値は無い")
        msgs.append(
            f"★入力ミスの可能性がある値★ '{c}' の"
            + (f" Excel の行 {where}" if where else f" {len(g)} 個の値")
            + f"（例: [{', '.join(vals)}]）。ロバスト z = {top['ロバスト z']:g}（{nxt}）で、"
            "ほかの値から桁違いに離れている。元の記録と照らして、誤りなら schema.yaml の "
            f"columns: の '{c}' に plausible: [下限, 上限] を書いて再実行する"
            "（範囲の外は NaN にして補完へ回る）。正しい値なら何もしなくてよい")
    return msgs


def mahalanobis_outliers(df: pd.DataFrame, columns: list | None = None,
                         *, alpha: float = 0.001) -> pd.DataFrame:
    """多変量の外れ値（Mahalanobis 距離）。

    単変量では正常でも、組合せとしてあり得ない症例を見つける
    （身長 190cm・体重 40kg のように、どちらも単独では正常範囲）。
    """
    from scipy import stats
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) < len(cols) + 2 or len(cols) < 2:
        return pd.DataFrame()
    cov = np.cov(x.to_numpy(), rowvar=False)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.DataFrame()
    d = x.to_numpy() - x.to_numpy().mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", d, inv, d)
    thr = float(stats.chi2.ppf(1 - alpha, df=len(cols)))
    out = pd.DataFrame({"行": x.index, "Mahalanobis距離²": d2.round(3),
                        "閾値": round(thr, 3), "外れ値": d2 > thr})
    return out[out["外れ値"]].sort_values("Mahalanobis距離²", ascending=False)


def isolation_forest_outliers(df: pd.DataFrame, columns: list | None = None,
                              *, contamination: float = 0.02,
                              random_state: int = 0) -> pd.DataFrame:
    """非線形な多変量外れ値。**contamination は「これだけ外れ値がある」という仮定である。**"""
    from sklearn.ensemble import IsolationForest
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(x) < 20 or not cols:
        return pd.DataFrame()
    m = IsolationForest(contamination=contamination, random_state=random_state).fit(x)
    score = m.score_samples(x)
    pred = m.predict(x)
    out = pd.DataFrame({"行": x.index, "異常度": (-score).round(4), "外れ値": pred == -1})
    return out[out["外れ値"]].sort_values("異常度", ascending=False)


def apply_action(df: pd.DataFrame, rep: OutlierReport, action: str = FLAG,
                 *, suffix: str = "__outlier") -> tuple[pd.DataFrame, list]:
    """検出結果をデータに適用する。**`drop` は人が明示したときだけ。**

    `winsorize` をここで行うのは探索目的のときだけにすること。
    学習に使うデータでは `pipeline.py` の `NanSafeWinsorizer` を使う
    （閾値を train でしか学習しないため）。
    """
    out, notes = df.copy(), []
    if not len(rep.flags):
        return out, ["外れ値の検出結果が空のため何もしなかった"]

    if action == FLAG:
        for c in rep.flags.columns:
            if rep.flags[c].any():
                out[f"{c}{suffix}"] = rep.flags[c].astype(int)
        notes.append(f"外れ値の印を {int((rep.flags.any()).sum())} 列ぶん追加した（値は変えていない）")
    elif action == NAN:
        for c in rep.flags.columns:
            out.loc[rep.flags[c], c] = np.nan
        notes.append(f"外れ値 {int(rep.flags.to_numpy().sum())} 個を NaN にした（補完へ回す）")
    elif action == WINSORIZE:
        for c in rep.flags.columns:
            row = rep.table[rep.table["列"] == c]
            if not len(row):
                continue
            lo, hi = row.iloc[0]["下限"], row.iloc[0]["上限"]
            if isinstance(lo, str):      # 群ごとの閾値はここでは適用しない
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce").clip(lo, hi)
        notes.append("上下限に丸めた。**学習に使うなら pipeline の NanSafeWinsorizer を使うこと**"
                     "（ここで丸めると閾値が全データから作られ、リークになる）")
    elif action == DROP:
        keep = ~rep.flags.any(axis=1)
        notes.append(f"★{int((~keep).sum())} 例を削除した。★ "
                     f"医学では外れ値こそが重要な症例でありうる。"
                     f"削除した症例の一覧を必ず論文の補足に残すこと")
        out = out[keep]
    else:
        raise ValueError(f"action は flag / winsorize / nan / drop のいずれか（{action!r}）")
    return out, notes


# ================================================================== 補助
def _numeric_columns(df, schema, columns) -> list:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    if schema is not None:
        return [c for c in schema.kept()
                if c in df.columns and schema.columns[c].role == NUMERIC]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _examples(df, mask, id_col, limit) -> str:
    idx = df.index[mask][:limit]
    vals = [df.at[i, id_col] for i in idx] if (id_col and id_col in df.columns) else list(idx)
    s = "、".join(map(str, vals))
    return s + ("…" if int(mask.sum()) > limit else "")
