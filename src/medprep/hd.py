"""medprep.hd — 血液透析に固有の派生指標。

実装した式はすべて岩藤先生から提示された定義に厳密に従い、
提示された計算例と一致することを検証してある（tests/test_hd.py）。

【重要な設計方針】
  1. これらの指標は **透析前値と透析後値の組** を必要とする。
     採血時点（pre / post）が不詳の値からは計算しない。推測して計算しない。
  2. 中間値は丸めない。表示時にのみ丸める。
  3. 異常な入力や負の産生速度が出ても **0 に丸めない**。NaN を返し、理由を記録する。
  4. nPCR の算出法が異なれば G_ext・G_int・%CGR も変わる。
     どの式を使ったかを必ず `provenance` に残す。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 1 g の食塩に含まれるナトリウム量 [mEq]（NaCl 58.44 g/mol → 17.1 mEq/g）
_MEQ_NA_PER_G_NACL = 17.0


def _all_scalar(*args) -> bool:
    """渡された入力がすべてスカラーか。列を渡されたのか 1 症例なのかを見分ける。"""
    return all(np.ndim(a) == 0 for a in args)


def _out(x, scalar: bool):
    """スカラーを渡されたらスカラーを返す。

    `float(result)` がそのまま書けないと、1 症例を確かめたいだけのときに
    毎回 `float(np.atleast_1d(...)[0])` を書かされることになる。
    numpy 2 では要素数 1 の配列を float() に渡すとエラーになるため、
    戻り値の形を入力に合わせておく。
    """
    a = np.asarray(x)
    if scalar and a.size == 1:
        return float(a.reshape(-1)[0])
    return a


# ====================================================================
# 基本指標
# ====================================================================
def urr(bun_pre, bun_post):
    """尿素除去率 URR [%] = 100 × (1 − BUN_post / BUN_pre)。"""
    sc = _all_scalar(bun_pre, bun_post)
    bun_pre = np.asarray(bun_pre, dtype=float)
    bun_post = np.asarray(bun_post, dtype=float)
    out = 100.0 * (1.0 - bun_post / bun_pre)
    return _out(np.where(bun_pre > 0, out, np.nan), sc)


def sp_ktv(bun_pre, bun_post, bw_pre, bw_post, td_hours):
    """尿素の single-pool Kt/V（Daugirdas 第2世代）。

        R      = BUN_post / BUN_pre
        spKt/V = −ln(R − 0.008·Td) + (4 − 3.5·R)·ΔBW / BW_post

    Td は **時間単位**（4時間なら 4。240 分を入れない）。
    R − 0.008·Td ≤ 0 のときは対数が定義できないので NaN を返す。
    """
    sc = _all_scalar(bun_pre, bun_post, bw_pre, bw_post, td_hours)
    bun_pre = np.asarray(bun_pre, dtype=float)
    bun_post = np.asarray(bun_post, dtype=float)
    bw_pre = np.asarray(bw_pre, dtype=float)
    bw_post = np.asarray(bw_post, dtype=float)
    td = np.asarray(td_hours, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = bun_post / bun_pre
        inner = r - 0.008 * td
        delta_bw = bw_pre - bw_post
        out = -np.log(inner) + (4.0 - 3.5 * r) * delta_bw / bw_post
    ok = (bun_pre > 0) & (bun_post > 0) & (bw_post > 0) & (td > 0) & (td < 72) & (inner > 0)
    return _out(np.where(ok, out, np.nan), sc)


def npcr(ktv, bun_pre, bun_post):
    """標準化蛋白異化率 nPCR [g/kg/日] — Kaynar 2012 の簡便推算式。

        nPCR = 0.0136 × Kt/V × (BUN_pre + BUN_post)/2 + 0.251

    BUN_mean は透析前後2点の **算術平均** であり、時間平均 BUN ではない。
    出典: Kaynar K, et al. Hippokratia. 2012;16(3):236-240.
    """
    sc = _all_scalar(ktv, bun_pre, bun_post)
    ktv = np.asarray(ktv, dtype=float)
    bun_mean = (np.asarray(bun_pre, dtype=float) + np.asarray(bun_post, dtype=float)) / 2.0
    return _out(0.0136 * ktv * bun_mean + 0.251, sc)


# ====================================================================
# %CGR（クレアチニン産生速度比）
# ====================================================================
@dataclass
class CGRResult:
    """%CGR の全中間値。監査できるよう途中経過をすべて保持する。"""
    delta_bw: np.ndarray
    R: np.ndarray
    spKtV: np.ndarray
    nPCR: np.ndarray
    Cr_corr: np.ndarray
    L: np.ndarray
    D: np.ndarray
    A: np.ndarray
    G_total: np.ndarray
    G_ext: np.ndarray
    G_int: np.ndarray
    G_ref: np.ndarray
    percent_CGR: np.ndarray
    invalid: pd.DataFrame = field(default_factory=pd.DataFrame)
    provenance: dict = field(default_factory=dict)

    def to_frame(self, index=None) -> pd.DataFrame:
        return pd.DataFrame({
            "ΔBW": self.delta_bw, "R": self.R, "spKt/V": self.spKtV, "nPCR": self.nPCR,
            "Cr_corr": self.Cr_corr, "L": self.L, "D": self.D, "A": self.A,
            "G_total": self.G_total, "G_ext": self.G_ext, "G_int": self.G_int,
            "G_ref": self.G_ref, "%CGR": self.percent_CGR}, index=index)


def percent_cgr(sex, age, bun_pre, bun_post, cr_pre, cr_post,
                bw_pre, bw_post, td_hours, male_values=("男", "男性", "M", "Male", 1)):
    """%CGR を元の測定値から算出する（11 ステップ）。

    適用条件（提示資料より）
      * 通常の週3回血液透析。G_total の式には 72 時間が固定値として含まれる。
        中2日の長い透析間隔に対応する **週初めの採血** を想定する。
        任意の透析曜日・連日透析・週2回透析にそのまま適用しない。
      * 尿中クレアチニン排泄の補正を含まない。残腎機能があると内因性産生を過小評価しうる。
      * 透析前後の採血は同一セッションのもの。

    nPCR には Kaynar の簡便式（`npcr`）を用いる。**算出法が異なれば
    G_ext・G_int・%CGR も異なる**ため、日本透析医学会の統計調査や
    施設の透析管理ソフトと同一の算出法とみなしてはならない。
    """
    sc = _all_scalar(age, bun_pre, bun_post, cr_pre, cr_post, bw_pre, bw_post, td_hours)

    def to_f(x):
        return np.asarray(x, dtype=float)

    age_a = to_f(age)
    bun_pre_a, bun_post_a = to_f(bun_pre), to_f(bun_post)
    cr_pre_a, cr_post_a = to_f(cr_pre), to_f(cr_post)
    bw_pre_a, bw_post_a = to_f(bw_pre), to_f(bw_post)
    td = to_f(td_hours)

    with np.errstate(divide="ignore", invalid="ignore"):
        # STEP 1-2
        delta_bw = bw_pre_a - bw_post_a
        R = bun_post_a / bun_pre_a
        # STEP 3
        spktv = sp_ktv(bun_pre_a, bun_post_a, bw_pre_a, bw_post_a, td)
        # STEP 4
        npcr_v = npcr(spktv, bun_pre_a, bun_post_a)
        # STEP 5   分母は必ず (60 * Td) 全体
        cr_corr = (-81.622 * np.log(cr_post_a / cr_pre_a) / (60.0 * td) + 0.942) * cr_post_a
        # STEP 6
        L = np.log(cr_corr / cr_pre_a)
        D = (0.019 * td + 0.999) * L - (0.00367 * td - 0.0219)
        A = 3864.0 + (7.8 * td + 411.0) * L - 1.5 * td - 1449.0 / D
        # STEP 7
        G_total = cr_pre_a * (7056.0 / A + (delta_bw / bw_post_a) * 240.0 / (72.0 - td))
        # STEP 8-9
        G_ext = 7.79 * npcr_v ** 2 - 7.91 * npcr_v + 1.93
        G_int = G_total - G_ext
        # STEP 10
        is_male = pd.Series(list(np.atleast_1d(sex))).isin(list(male_values)).to_numpy()
        if is_male.size == 1 and age_a.size > 1:
            is_male = np.repeat(is_male, age_a.size)
        # スカラー入力のときに形が (1,) へ広がらないよう、age と同じ形に揃える
        if age_a.ndim == 0 and is_male.size == 1:
            is_male = is_male.reshape(())
        G_ref = np.where(is_male, 23.53 - 0.15 * age_a, 19.58 - 0.12 * age_a)
        # STEP 11
        pcgr = 100.0 * G_int / G_ref

    # ---- 妥当性の検査（自動で 0 に丸めない。NaN にして理由を残す）
    checks = {
        "BUN_pre ≤ 0": ~(bun_pre_a > 0),
        "BUN_post ≤ 0": ~(bun_post_a > 0),
        "Cr_pre ≤ 0": ~(cr_pre_a > 0),
        "Cr_post ≤ 0": ~(cr_post_a > 0),
        "BW_pre ≤ 0": ~(bw_pre_a > 0),
        "BW_post ≤ 0": ~(bw_post_a > 0),
        "Td が 0 < Td < 72 の外": ~((td > 0) & (td < 72)),
        "R − 0.008·Td ≤ 0": ~((R - 0.008 * td) > 0),
        "Cr_corr ≤ 0": ~(cr_corr > 0),
        "D = 0": D == 0,
        "A = 0": A == 0,
        "G_ref ≤ 0": ~(G_ref > 0),
        "G_int < 0（内因性産生が負）": G_int < 0,
    }
    bad = np.zeros_like(pcgr, dtype=bool)
    rows = []
    for msg, m in checks.items():
        m = np.atleast_1d(m)
        if m.any():
            rows.append((msg, int(m.sum())))
            bad |= m
    pcgr = np.where(bad, np.nan, pcgr)

    return CGRResult(
        delta_bw=_out(delta_bw, sc), R=_out(R, sc), spKtV=_out(spktv, sc),
        nPCR=_out(npcr_v, sc), Cr_corr=_out(cr_corr, sc),
        L=_out(L, sc), D=_out(D, sc), A=_out(A, sc), G_total=_out(G_total, sc),
        G_ext=_out(G_ext, sc), G_int=_out(G_int, sc), G_ref=_out(G_ref, sc),
        percent_CGR=_out(pcgr, sc),
        invalid=pd.DataFrame(rows, columns=["理由", "件数"]),
        provenance={
            "nPCR式": "Kaynar 2012 簡便式: 0.0136·spKt/V·(BUN_pre+BUN_post)/2 + 0.251",
            "Kt/V": "Daugirdas 第2世代 spKt/V",
            "G_total の 72 時間": "週3回・中2日の長い透析間隔（週初め採血）を前提とする固定値",
            "残腎機能": "尿中クレアチニン排泄の補正を含まない",
            "出典": "滋賀腎・透析研究会（spKt/V・Cr_corr・A・G_total・G_ext・G_int・G_ref）／Kaynar 2012（nPCR）",
        })


# ====================================================================
# クリアスペース率 A/V
# ====================================================================
def clear_space_ratio(m_mg, c_pre_mg_dl, v_post_l):
    """実測法（定義式）。

        A = M / C_pre            等価な浄化容積 [L]
        A/V = A / V_post
        A/V[%] = 100 · M_mg / (10 · C_pre[mg/dL] · V_post[L])

    BUN を対象にする場合、M も **尿素窒素の除去量 [mg-N]** とする。
    尿素の質量と尿素窒素の質量を混在させない。分母に透析後 BUN を入れない。
    透析後 V で標準化した指標は条件により 100% を超えうる。
    **出力を一律に 0〜100% へ切り詰めない。**
    """
    sc = _all_scalar(m_mg, c_pre_mg_dl, v_post_l)
    m = np.asarray(m_mg, dtype=float)
    c = np.asarray(c_pre_mg_dl, dtype=float)
    v = np.asarray(v_post_l, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_l = m / (10.0 * c)
        ratio = a_l / v
    ok = np.isfinite(m) & np.isfinite(c) & np.isfinite(v) & (m >= 0) & (c > 0) & (v > 0)
    return {
        "A_L": _out(np.where(ok, a_l, np.nan), sc),
        "AV_ratio": _out(np.where(ok, ratio, np.nan), sc),
        "AV_percent": _out(np.where(ok, 100.0 * ratio, np.nan), sc),
    }


def removed_mass_from_effluent(c_effluent_mg_dl, v_effluent_l):
    """全排液から除去量 M [mg] を求める。M = 10 · C_effluent · V_effluent。

    新鮮透析液に対象溶質が含まれないことを前提とする。
    分割採取の場合は区間ごとに呼んで合計すること。
    **排液の一部だけを採取した場合、採取液量を全排液量として使わない。**
    この M は排液に回収された量であり、膜への吸着量等は含まない。
    """
    sc = _all_scalar(c_effluent_mg_dl, v_effluent_l)
    return _out(10.0 * np.asarray(c_effluent_mg_dl, dtype=float)
                * np.asarray(v_effluent_l, dtype=float), sc)


def clear_space_ratio_estimated(c_pre_mg_dl, c_post_mg_dl, v_pre_l, v_post_l,
                                g_mg_min=0.0, t_min=None):
    """推算法（物質収支に基づく単一プール近似）。**実測値と同一視しないこと。**

        A/V[%] = 100 · (V_pre/V_post − C_post/C_pre + G·t / (10·C_pre·V_post))

    仮定: 単一プールで濃度が均一。透析以外への排泄・外部からの流入なし。
    `g_mg_min = 0` は「産生を無視する近似を採用する」という明示的な選択であり、
    **産生速度が不明であることと産生速度が 0 であることは同じではない。**
    透析直後の血液濃度を平衡化後濃度の代用にすると、リバウンドで結果が変わる。
    この式にリバウンドの経験的補正は含まれていない。
    """
    sc = _all_scalar(c_pre_mg_dl, c_post_mg_dl, v_pre_l, v_post_l)
    c_pre = np.asarray(c_pre_mg_dl, dtype=float)
    c_post = np.asarray(c_post_mg_dl, dtype=float)
    v_pre = np.asarray(v_pre_l, dtype=float)
    v_post = np.asarray(v_post_l, dtype=float)
    g = np.asarray(g_mg_min, dtype=float)
    t = 0.0 if t_min is None else np.asarray(t_min, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (v_pre / v_post - c_post / c_pre + g * t / (10.0 * c_pre * v_post))
    ok = (c_pre > 0) & (v_post > 0)
    return _out(np.where(ok, pct, np.nan), sc)


def clear_space_ratio_from_weight(c_pre_mg_dl, c_post_mg_dl, bw_pre_kg, bw_post_kg, v_post_l):
    """体液量減少を前後体重差で近似し、産生を無視した推算。

        A/V[%] = 100 · (1 + ΔBW / V_post − C_post / C_pre)

    体重差 1 kg を体液量差 1 L と近似している。
    """
    sc = _all_scalar(c_pre_mg_dl, c_post_mg_dl, bw_pre_kg, bw_post_kg, v_post_l)
    delta = np.asarray(bw_pre_kg, dtype=float) - np.asarray(bw_post_kg, dtype=float)
    c_pre = np.asarray(c_pre_mg_dl, dtype=float)
    c_post = np.asarray(c_post_mg_dl, dtype=float)
    v_post = np.asarray(v_post_l, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * (1.0 + delta / v_post - c_post / c_pre)
    return _out(np.where((c_pre > 0) & (v_post > 0), pct, np.nan), sc)


# ====================================================================
# 推定食塩摂取量
# ====================================================================
def salt_intake(delta_bw_kg, na_meq_l, interdialytic_days=2.0):
    """透析間体重増加と血清 Na から推定する食塩摂取量 [g/日]。

        摂取 Na [mEq] = 透析間体重増加 [kg → L] × 血清 Na [mEq/L]
        食塩 [g]      = 摂取 Na / 17          （1 g NaCl ≒ 17 mEq Na）
        1日あたり     = 食塩 [g] / 透析間隔日数

    ★この推定に含まれる仮定（レポートに明記される）★
      * 透析間に増加した体液の Na 濃度が血清 Na 濃度に等しいとみなしている。
        実際には飲水（Na をほとんど含まない）と食塩摂取の比率で変わる。
        自由水を多く摂った患者では食塩摂取量を過大評価する。
      * 不感蒸泄・残腎からの Na 排泄・便中排泄を無視している。
      * 透析間隔日数は中1日なら 2、中2日なら 3 を与える。
        週初めの採血（中2日後）に 2 を使うと 1.5 倍の過大評価になる。
        **列があれば実日数を使い、無い場合は既定 2 とするが警告を出す。**

    残腎機能がない維持透析患者では実務上よく使われる近似だが、
    厳密な摂取量ではないことを結果の名称（「推定」）で示すこと。
    """
    sc = _all_scalar(delta_bw_kg, na_meq_l, interdialytic_days)
    d = np.asarray(delta_bw_kg, dtype=float)
    na = np.asarray(na_meq_l, dtype=float)
    days = np.asarray(interdialytic_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_total = d * na / _MEQ_NA_PER_G_NACL
        out = g_total / days
    return _out(np.where((na > 0) & (days > 0) & (d >= 0), out, np.nan), sc)


# ====================================================================
# 栄養・体格
# ====================================================================
def bmi(weight_kg, height_cm):
    sc = _all_scalar(weight_kg, height_cm)
    w = np.asarray(weight_kg, dtype=float)
    h = np.asarray(height_cm, dtype=float) / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = w / h ** 2
    return _out(np.where((w > 0) & (h > 0), out, np.nan), sc)


def ideal_body_weight(height_cm, bmi_target=22.0):
    """理想体重 = 身長(m)² × 22（日本の透析領域で慣用）。"""
    sc = _all_scalar(height_cm)
    h = np.asarray(height_cm, dtype=float) / 100.0
    return _out(np.where(h > 0, bmi_target * h ** 2, np.nan), sc)


def select_weight_for_gnri(df, dw_col=None, post_col=None, pre_col=None):
    """GNRI に使う体重を選ぶ。**ドライウェイト優先、無ければ透析後体重。**
    （2026-09-18 岩藤先生決定）

    症例ごとに選び、どちらを使ったかを列で返すので、レポートに内訳を出せる。
    透析前体重は使わない（体液貯留分が栄養指標に混入するため）。使える体重が
    無い症例は NaN とし、GNRI も NaN になる。

    Returns
    -------
    (体重の Series, 出所ラベルの Series)
    """
    idx = df.index
    w = pd.Series(np.nan, index=idx, dtype=float)
    src = pd.Series("なし", index=idx, dtype=object)
    for col, label in ((dw_col, "ドライウェイト"), (post_col, "透析後体重")):
        if col is None or col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        take = w.isna() & v.notna()
        w[take] = v[take]
        src[take] = label
    if pre_col is not None and pre_col in df.columns:
        n = int((src == "なし").sum())
        if n:
            src[src == "なし"] = f"なし（透析前体重は使わない方針のため {n} 例は算出せず）"
    return w, src


def gnri(alb_g_dl, weight_kg, height_cm, cap_ratio=True, bmi_target=22.0):
    """GNRI = 14.89 × Alb + 41.7 × (体重 / 理想体重)。

    体重は `select_weight_for_gnri` で選ぶ（ドライウェイト優先、無ければ透析後体重）。

    既定として採用した規約（変更可能。使った値は provenance に残す）
      * `cap_ratio=True` — 体重/理想体重 > 1 のとき 1 に丸める（原法の規約）。
      * `bmi_target=22.0` — 理想体重 = 身長(m)² × 22（日本の透析領域で慣用）。
        Lorentz 式を使う施設もあるので、揃える必要があれば引数で変える。
    リスク分類: <82 高度 / 82–<92 中等度 / 92–<98 軽度 / ≥98 なし
    """
    sc = _all_scalar(alb_g_dl, weight_kg, height_cm)
    alb = np.asarray(alb_g_dl, dtype=float)
    ibw = np.asarray(ideal_body_weight(height_cm, bmi_target), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(weight_kg, dtype=float) / ibw
    if cap_ratio:
        ratio = np.minimum(ratio, 1.0)
    return _out(14.89 * alb + 41.7 * ratio, sc)


def corrected_ca(ca_mg_dl, alb_g_dl):
    """Payne 式。Alb < 4.0 g/dL のときのみ補正する。

        補正Ca = 実測Ca + 4 − Alb   (Alb < 4.0)
        補正Ca = 実測Ca             (Alb ≥ 4.0)

    低 Alb 時にイオン化 Ca との乖離が生じるため、補正値は **目安** として扱う。
    """
    sc = _all_scalar(ca_mg_dl, alb_g_dl)
    ca = np.asarray(ca_mg_dl, dtype=float)
    alb = np.asarray(alb_g_dl, dtype=float)
    return _out(np.where(alb < 4.0, ca + 4.0 - alb, ca), sc)


def tsat(fe_ug_dl, tibc_ug_dl):
    """TSAT [%] = Fe / TIBC × 100。100% 超は Fe > TIBC であり入力ミス。"""
    sc = _all_scalar(fe_ug_dl, tibc_ug_dl)
    fe = np.asarray(fe_ug_dl, dtype=float)
    tibc = np.asarray(tibc_ug_dl, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = fe / tibc * 100.0
    return _out(np.where(tibc > 0, out, np.nan), sc)
