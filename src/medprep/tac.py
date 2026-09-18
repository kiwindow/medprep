"""medprep.tac — 時間平均濃度 TAC（time-averaged concentration）。

厳密な定義（対象物質によらず共通）

    TAC_X = (1/T) ∫[0,T] C_X(t) dt = AUC_X(0,T) / T

週間評価では T = 168 時間（透析中と非透析中の両方を含む）。
**検査値を全部足して回数で割る方法は、測定間隔が異なれば時間平均にならない。**

本モジュールは 4 通りを提供する。どれを使ったかは必ず結果に残る。

    simple     : 週初め透析後 BUN と 次回透析前 BUN の 2 点簡便法
    trapezoid  : 任意個の (時刻, 濃度) から台形則で数値積分
    linear     : 各回の透析前後＋次回透析前を、透析区間・非透析区間に分けて時間加重
    ―― simple と trapezoid/linear は別物として保存する（簡便値と AUC 由来値を区別する）

【取り違えの防止】
    TAC の簡便式は、次の2つと**まったく別のもの**である。
      × 同じ透析回の「透析前 BUN と 透析後 BUN の平均」  ← これは nPCR の BUN_mean
      × 「週初め透析前 BUN と 次回透析前 BUN の平均」
    どちらも数値としては計算できてしまうため、本モジュールは
    入力の大小関係からこの取り違えを自動で検出して警告する（`check_simple_inputs`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 尿素（60）と尿素窒素（28）の分子量・窒素部分質量（丸め値）
UREA_PER_BUN = 60.0 / 28.0
BUN_PER_UREA = 28.0 / 60.0
BUN_MGDL_TO_UREA_MMOLL = 1.0 / 2.8

WEEK_HOURS = 168.0


def _all_scalar(*args) -> bool:
    return all(np.ndim(a) == 0 for a in args)


def _out(x, scalar: bool):
    """スカラーを渡されたらスカラーを返す（medprep.hd と同じ約束）。"""
    a = np.asarray(x)
    if scalar and a.size == 1:
        return float(a.reshape(-1)[0])
    return a


@dataclass
class TACResult:
    """TAC の値と、算出に使った条件。"""
    value: float | np.ndarray
    solute: str = "BUN"
    unit: str = "mg/dL"
    method: str = ""
    kind: str = ""                     # 'simple' | 'auc'  ← 保存時に区別する
    warnings: list = field(default_factory=list)
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict = field(default_factory=dict)

    def to_urea_mg_dl(self):
        if self.solute != "BUN":
            raise ValueError("BUN の TAC からのみ換算できる。")
        return bun_to_urea_mg_dl(self.value)

    def to_urea_mmol_l(self):
        if self.solute != "BUN":
            raise ValueError("BUN の TAC からのみ換算できる。")
        return bun_to_urea_mmol_l(self.value)

    def __repr__(self):
        v = np.atleast_1d(self.value)
        s = f"{v[0]:.4g}" if v.size == 1 else f"n={v.size}"
        return f"<TAC {self.solute} {s} {self.unit} method={self.method} kind={self.kind}>"


# ====================================================================
# 1) 簡便式（2点）
# ====================================================================
def check_simple_inputs(bun_post_first, bun_pre_second) -> list:
    """簡便式の入力が取り違えられていないかを調べる。

    透析後 BUN は透析間に上昇するので、**次回透析前 BUN は週初め透析後 BUN より
    高いのが通常**である。逆転していれば、次のいずれかが疑われる。

      * 同じ透析回の (透析前, 透析後) を渡した  → 前 > 後 になるので必ず逆転する
      * 引数の順序を逆にした
      * 採血日時の対応がずれている

    数値としては計算できてしまうため、ここで捕まえる。
    """
    post = np.atleast_1d(np.asarray(bun_post_first, dtype=float))
    pre2 = np.atleast_1d(np.asarray(bun_pre_second, dtype=float))
    w = []
    both = np.isfinite(post) & np.isfinite(pre2)
    if not both.any():
        return w
    inv = both & (pre2 < post)
    if inv.any():
        n = int(inv.sum())
        w.append(
            f"{n} 件で「次回透析前 BUN < 週初め透析後 BUN」となっている。"
            f"通常は透析間に BUN が上昇するため逆転しない。"
            f"同じ透析回の透析前後を渡していないか、引数の順序、採血日時の対応を確認すること。")
    ratio = np.divide(pre2, post, out=np.full_like(pre2, np.nan), where=post > 0)
    odd = both & np.isfinite(ratio) & (ratio > 6.0)
    if odd.any():
        w.append(f"{int(odd.sum())} 件で次回透析前 BUN が週初め透析後 BUN の 6 倍を超える。単位と対応を確認すること。")
    return w


def tac_bun_simple(bun_post_first, bun_pre_second, validate: bool = True) -> TACResult:
    """TAC-BUN 簡便式 [mg/dL]。

        TAC_BUN = (週初め透析後 BUN + 次回透析開始前 BUN) / 2

    月・水・金透析なら (月曜透析後 + 水曜透析前) / 2。
    火・木・土透析なら (火曜透析後 + 木曜透析前) / 2。

    位置づけ：規則的な週3回血液透析における週間平均の**簡便な推定値**であり、
    168 時間の濃度曲線を積分した厳密値ではない。
    透析後リバウンドや体液量変化を補正する項は含まれていない。
    不規則な日程・週2回・頻回透析・急性病態に同じ精度を前提にしない。

    欠測は 0 として扱わず NaN（算出不能）にする。負値は入力エラーとして NaN にする。
    """
    sc = _all_scalar(bun_post_first, bun_pre_second)
    post = np.asarray(bun_post_first, dtype=float)
    pre2 = np.asarray(bun_pre_second, dtype=float)
    warns = check_simple_inputs(post, pre2) if validate else []

    bad = (post < 0) | (pre2 < 0)
    if np.any(bad):
        warns.append(f"{int(np.sum(bad))} 件に負値があった。入力エラーとして算出不能にした。")
    val = _out(np.where(bad, np.nan, (post + pre2) / 2.0), sc)

    return TACResult(
        value=val, solute="BUN", unit="mg/dL",
        method="週初め透析後＋次回透析前の2点簡便法", kind="simple",
        warnings=warns,
        provenance={
            "式": "TAC_BUN = (BUN_post_first + BUN_pre_second) / 2",
            "前提": "規則的な週3回血液透析",
            "位置づけ": "簡便な推定値。168時間の積分による厳密値ではない",
            "含まれない補正": "透析後リバウンド、体液量変化",
            "出典": "三島クリニック 透析室ニュース 第30号（2007-09-06）。医療機関の説明資料であり JSDT ガイドラインではない",
        })


# ====================================================================
# 2) 台形則（任意個の時点）
# ====================================================================
def tac_trapezoid(times_h, concentrations, solute="BUN", unit="mg/dL",
                  expect_hours: float | None = None) -> TACResult:
    """台形則による数値積分。

        AUC = Σ ((C[i] + C[i+1]) / 2) × (t[i+1] − t[i])
        TAC = AUC / (t[n] − t[0])

    時刻は**厳密な昇順**であること、評価時間が正であること、
    濃度と時間の単位が統一されていることを確認する。

    透析前後だけの粗いデータでは透析中の曲線や終了後リバウンドを表現できない。
    `expect_hours=168` を渡すと、観測区間が週全体を覆っているかを検算する。
    """
    t = np.asarray(times_h, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    warns = []

    if t.shape != c.shape:
        raise ValueError(f"時刻と濃度の個数が違う（{t.shape} と {c.shape}）。")
    if t.size < 2:
        raise ValueError("台形則には 2 点以上が必要。")
    if not np.all(np.diff(t) > 0):
        raise ValueError("時刻が厳密な昇順になっていない。")
    if np.any(~np.isfinite(c)):
        return TACResult(value=np.nan, solute=solute, unit=unit,
                         method="台形則", kind="auc",
                         warnings=["濃度に欠測または非数値がある。0 で埋めず算出不能とした。"])
    if np.any(c < 0):
        return TACResult(value=np.nan, solute=solute, unit=unit,
                         method="台形則", kind="auc",
                         warnings=["濃度に負値がある。入力エラーとして算出不能とした。"])

    span = t[-1] - t[0]
    if span <= 0:
        raise ValueError("評価時間が正でない。")
    auc = float(np.sum((c[:-1] + c[1:]) / 2.0 * np.diff(t)))
    tac = auc / span

    if expect_hours is not None and not np.isclose(span, expect_hours, atol=1e-6):
        warns.append(f"観測区間が {span:g} 時間で、想定する {expect_hours:g} 時間と一致しない。"
                     f"週間 TAC を出すなら区間全体を {expect_hours:g} 時間にすること。")
    if t.size <= 4:
        warns.append(f"時点が {t.size} 点しかない。透析中の曲線や終了後リバウンドを"
                     f"十分に表現できないため、粗い近似である。")

    return TACResult(value=float(tac), solute=solute, unit=unit,
                     method="台形則", kind="auc", warnings=warns,
                     detail=pd.DataFrame({"時刻[h]": t, "濃度": c}),
                     provenance={"式": "AUC = Σ((C[i]+C[i+1])/2)·Δt、TAC = AUC / (t[n]−t[0])",
                                 "区間[h]": span, "点数": int(t.size),
                                 "位置づけ": "定義に基づく数値積分。特定学会の専用推算式ではない"})


# ====================================================================
# 3) 時間加重近似（透析区間・非透析区間に分ける）
# ====================================================================
def interdialytic_hours(interval_h, td_h):
    """非透析時間 Ti = 開始間隔 − 透析時間。

    ★開始から次回開始まで 48 時間・透析 4 時間なら Ti = 44 時間。**Ti を 48 にしない。**
      長い間隔 72 時間・透析 4 時間なら Ti = 68 時間。
    """
    return np.asarray(interval_h, dtype=float) - np.asarray(td_h, dtype=float)


def tac_linear(c_pre, c_post, c_pre_next, td_h, ti_h,
               solute="BUN", unit="mg/dL", check_week=True) -> TACResult:
    """各回の透析前後と次回透析前を使う時間加重近似。

        AUC_dial[j]  = (C_pre[j]  + C_post[j])    / 2 × Td[j]
        AUC_inter[j] = (C_post[j] + C_pre[j+1])   / 2 × Ti[j]
        TAC_linear   = Σ(AUC_dial + AUC_inter) / Σ(Td + Ti)

    週3回透析の1週間では j = 1,2,3。C_pre[4] は**翌週第1回の実測透析前濃度**。
    週ごとの状態が同じという仮定を置く場合のみ C_pre[4] = C_pre[1] とできる。

    4時間透析を3回、開始間隔 48・48・72 時間なら全時間は 168 時間になる
    （`check_week=True` で検算する）。
    透析中の濃度低下は一般に直線ではないため、この式も近似値である。
    """
    pre = np.atleast_1d(np.asarray(c_pre, dtype=float))
    post = np.atleast_1d(np.asarray(c_post, dtype=float))
    nxt = np.atleast_1d(np.asarray(c_pre_next, dtype=float))
    td = np.atleast_1d(np.asarray(td_h, dtype=float))
    ti = np.atleast_1d(np.asarray(ti_h, dtype=float))
    warns = []

    n = pre.size
    for name, a in (("C_post", post), ("C_pre_next", nxt), ("Td", td), ("Ti", ti)):
        if a.size != n:
            raise ValueError(f"{name} の個数が C_pre と違う（{a.size} と {n}）。")
    if np.any(td <= 0) or np.any(ti <= 0):
        raise ValueError("Td と Ti は正であること。Ti = 開始間隔 − Td で求める。")
    vals = np.concatenate([pre, post, nxt])
    if np.any(~np.isfinite(vals)):
        return TACResult(value=np.nan, solute=solute, unit=unit,
                         method="時間加重近似", kind="auc",
                         warnings=["濃度に欠測がある。0 で埋めず算出不能とした。"])
    if np.any(vals < 0):
        return TACResult(value=np.nan, solute=solute, unit=unit,
                         method="時間加重近似", kind="auc",
                         warnings=["濃度に負値がある。入力エラーとして算出不能とした。"])

    auc_d = (pre + post) / 2.0 * td
    auc_i = (post + nxt) / 2.0 * ti
    total_h = float(np.sum(td + ti))
    tac = float(np.sum(auc_d + auc_i) / total_h)

    if check_week and n == 3 and not np.isclose(total_h, WEEK_HOURS, atol=1e-6):
        warns.append(f"週3回として合計 {total_h:g} 時間になった（168 時間でない）。"
                     f"Ti = 開始間隔 − Td になっているか確認すること"
                     f"（48時間間隔・4時間透析なら Ti = 44、長い間隔 72 時間なら Ti = 68）。")
    if np.any(post > pre):
        warns.append(f"{int(np.sum(post > pre))} 回で透析後濃度が透析前濃度より高い。"
                     f"前後の対応を確認すること。")

    detail = pd.DataFrame({"回": np.arange(1, n + 1), "C_pre": pre, "C_post": post,
                           "C_pre_next": nxt, "Td[h]": td, "Ti[h]": ti,
                           "AUC_透析中": auc_d, "AUC_非透析": auc_i})
    return TACResult(value=float(tac), solute=solute, unit=unit,
                     method="時間加重近似（透析区間・非透析区間の台形則）", kind="auc",
                     warnings=warns, detail=detail,
                     provenance={"式": "Σ((C_pre+C_post)/2·Td + (C_post+C_pre_next)/2·Ti) / Σ(Td+Ti)",
                                 "合計時間[h]": total_h, "回数": n,
                                 "近似": "透析中の濃度低下を直線とみなす"})


# ====================================================================
# 4) 尿素と BUN の換算
# ====================================================================
def bun_to_urea_mg_dl(tac_bun_mg_dl):
    """同一の評価期間・同一の時間平均法に対して TAC_urea ≈ TAC_BUN × (60/28)。"""
    return _out(np.asarray(tac_bun_mg_dl, dtype=float) * UREA_PER_BUN,
                _all_scalar(tac_bun_mg_dl))


def urea_to_bun_mg_dl(tac_urea_mg_dl):
    return _out(np.asarray(tac_urea_mg_dl, dtype=float) * BUN_PER_UREA,
                _all_scalar(tac_urea_mg_dl))


def bun_to_urea_mmol_l(tac_bun_mg_dl):
    """TAC_urea [mmol/L] ≈ TAC_BUN [mg/dL] / 2.8。"""
    return _out(np.asarray(tac_bun_mg_dl, dtype=float) * BUN_MGDL_TO_UREA_MMOLL,
                _all_scalar(tac_bun_mg_dl))
