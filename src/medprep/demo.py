"""演習用の合成データ（実データは一切含まない）。

第3回の演習は、自分のデータを持ち込めない受講者のために合成透析コホートを使う。
**教材の要はここに仕込んだ「汚れ」である。** きれいなデータで前処理を教えても、
受講者は自分のデータに戻った瞬間に立ち往生する。

仕込んである汚れ（すべて実務で実際に見るもの）:

  1. 日付が 8 通りの表記で混在（和暦・Excel シリアル値・全角・時刻付きを含む）
  2. イベント日と打ち切り日の**両方**が入っている矛盾例
  3. どちらも空欄の例、観察開始日が空欄の例
  4. 観察開始日 > 終了日 の逆転例
  5. 欠損コード 999 ／「未測定」
  6. 検出限界表記 `<0.1`（CRP）
  7. 単位混在（Hb を一部 g/L で記録）
  8. 生理学的にあり得ない値（Hb 0 g/dL、年齢 250 歳）
  9. ID 的な高カーディナリティ文字列列（備考）
 10. 完全相関する重複列（施設 と 施設コード）
 11. **2020 年 4 月の ALP 測定法変更（JSCC → IFCC）による段差**
 12. **透析前後を取り違えた症例**（透析後 BUN のほうが高い）

透析前後の両方を持つ項目
------------------------
BUN・クレアチニン・カリウム・体重は**透析前と透析後の 2 列**を持つ。
これがあると URR・spKt/V・nPCR・%CGR が計算でき、
`quality.audit()` の「前後の方向」検査（BUN は透析で下がるはず）も働く。

真の生存モデルの係数も返せるので、推定が正しいことを受講者が自分で確かめられる。

    import medprep as mp
    df = mp.demo.dialysis_cohort()
    mp.demo.TRUE_COEFFICIENTS          # Cox の真の係数
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 合成データを作るときに使った Cox の真の係数（演習で推定値と照合する）。
TRUE_COEFFICIENTS = {
    "年齢": 0.045, "Alb": -0.85, "Hb": -0.18, "logCRP": 0.30,
    "糖尿病": 0.35, "vintage": 0.002,
}

#: ALP の測定法が変わった日（JSCC 法 → IFCC 法。値がおよそ 1/3 になる）。
ALP_METHOD_CHANGE = pd.Timestamp("2020-04-01")

_DATE_STYLE_P = [.30, .18, .10, .06, .10, .08, .10, .08]


def _fmt(ts, style, rng):
    """1 つの日付を、style で指定した表記に崩す。"""
    if pd.isna(ts):
        return ""
    y, m, d = ts.year, ts.month, ts.day
    if style == 0:
        return f"{y}/{m}/{d}"
    if style == 1:
        return f"{y}-{m:02d}-{d:02d}"
    if style == 2:
        return f"{y}.{m}.{d}"
    if style == 3:
        return "{" + f"{y}, {m}, {d}" + "}"
    if style == 4:
        return (f"{y}-{m}-{d} {rng.integers(0, 24):02d}:"
                f"{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}")
    if style == 5:                                    # 和暦
        return (f"H{y - 1988}.{m}.{d}" if y < 2019 or (y == 2019 and m < 5)
                else f"R{y - 2018}.{m}.{d}")
    if style == 6:                                    # Excel シリアル値
        return int((ts - pd.Timestamp("1899-12-30")).days)
    if style == 7:                                    # 全角
        return str(f"{y}/{m}/{d}").translate(
            str.maketrans("0123456789/", "０１２３４５６７８９／"))
    return str(ts.date())


def _hhmm(hours: float, rng) -> str:
    """小数時間を時刻の文字列にする。**表記ゆれを混ぜる。**

    実務の時刻列は `9:30`・`09:30`・`9時30分`・全角コロンが平気で混ざる。
    ここで混ぜておかないと、掃除の手順を演習で試せない。
    """
    h = int(hours) % 24
    m = int(round((hours - int(hours)) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    style = rng.integers(0, 10)
    if style < 6:
        return f"{h:02d}:{m:02d}"
    if style < 8:
        return f"{h}:{m:02d}"
    if style < 9:
        return f"{h}時{m:02d}分"
    return f"{h:02d}：{m:02d}"          # 全角コロン


def dialysis_cohort(n: int = 600, seed: int = 20260918) -> pd.DataFrame:
    """演習用の合成透析コホートを作る。

    既定（`n=600, seed=20260918`）は教材の図表と一致する。
    """
    rng = np.random.default_rng(seed)

    fac = rng.choice(["A院", "B院", "C院", "D院"], n, p=[.4, .3, .2, .1])
    sex = rng.choice(["男", "女"], n, p=[.62, .38])
    age = np.clip(rng.normal(68, 12, n), 20, 98).round(0)
    dm = rng.choice([0, 1], n, p=[.6, .4])
    alb = np.clip(rng.normal(3.6, 0.45, n), 1.5, 5.2).round(1)
    hb = np.clip(rng.normal(10.8, 1.2, n), 6, 15).round(1)
    crp = np.exp(rng.normal(-1.2, 1.3, n)).round(2)
    p_ = np.clip(rng.normal(5.2, 1.3, n), 1.5, 12).round(1)
    ca = np.clip(rng.normal(8.9, 0.7, n), 5, 12).round(1)
    ipth = np.exp(rng.normal(4.9, 0.8, n)).round(0)
    b2mg = np.clip(rng.normal(28, 6, n), 10, 60).round(1)
    vintage = np.clip(rng.exponential(60, n), 1, 400).round(0)     # 透析歴（月）

    # ---- 透析前後の検査値と体格 ------------------------------------------
    #   BUN・Cr・K は透析で下がり、体重は除水で下がる。
    #   これがあると URR・spKt/V・nPCR・%CGR が計算できる。
    height = np.where(sex == "男",
                      rng.normal(165, 7, n), rng.normal(152, 6, n)).round(1)
    dw = np.clip(rng.normal(21.5, 3.5, n) * (height / 100) ** 2, 30, 110).round(1)
    idwg = np.clip(rng.normal(2.6, 0.9, n), 0.2, 6.0).round(1)      # 透析間体重増加
    w_pre = (dw + idwg).round(1)
    w_post = dw

    # 透析時間。★中央値を 4 時間あたりに置き、5 時間前後の症例も含める。★
    #   以前は 3.5 / 4.0 / 4.5 / 5.0 の 4 値しか取らず、4.0 が 6 割を占めていた。
    #   すると **Q1 = Q3 になって IQR が 0** になり、外れ値処理がこの列を
    #   1 つの値に潰していた（0.10.0 で iqr_limits を直した）。
    #   実務でも 15 分刻みでばらつくので、そちらに合わせる。
    td = np.clip(rng.normal(4.05, 0.42, n), 3.0, 5.5)
    td = (np.round(td * 4) / 4)                    # 15 分刻み

    # ★透析時間は列として渡さない。開始時刻と終了時刻から作らせる。★
    #   実務のデータはたいてい時刻で入っており、時間そのものは入っていない。
    #   午前・午後・夜間の 3 シフト。表記ゆれ（全角コロン、"9時30分"）も混ぜる。
    shift = rng.choice([0, 1, 2], n, p=[.50, .30, .20])
    base_h = np.select([shift == 0, shift == 1], [8.5, 13.0], 17.0)
    t_start = base_h + rng.choice([0.0, 0.25, 0.5], n)
    t_end = t_start + td

    # 鉄関連（TSAT を算出させるため）
    fe = np.clip(rng.normal(58, 24, n), 8, 200).round(0)
    tibc = np.clip(rng.normal(255, 45, n), 120, 450).round(0)
    urr = np.clip(rng.normal(0.68, 0.07, n), 0.45, 0.85)            # 尿素除去率
    bun_pre = np.clip(rng.normal(62, 15, n), 25, 120).round(0)
    bun_post = (bun_pre * (1 - urr)).round(0)
    cr_pre = np.clip(rng.normal(11.0, 2.6, n), 3, 20).round(1)
    cr_post = (cr_pre * (1 - np.clip(rng.normal(0.60, 0.07, n), 0.3, 0.8))).round(1)
    k_pre = np.clip(rng.normal(5.2, 0.8, n), 3.0, 8.0).round(1)
    k_post = (k_pre * (1 - np.clip(rng.normal(0.35, 0.07, n), 0.1, 0.6))).round(1)

    # 真の生存モデル: 高齢・低Alb・高CRP・低Hb・糖尿病で予後不良
    lp = (0.045 * (age - 68) - 0.85 * (alb - 3.6) + 0.30 * np.log(crp + 0.1)
          - 0.18 * (hb - 10.8) + 0.35 * dm + 0.002 * (vintage - 60))
    lam = 0.00045 * np.exp(lp)
    t_event = rng.exponential(1 / np.clip(lam, 1e-6, None))        # 日
    t_cens = rng.uniform(200, 2200, n)
    obs = np.minimum(t_event, t_cens)
    evt = (t_event <= t_cens).astype(int)

    # 登録期間を 2011-01-01 〜 2020-01 に取る。観察期間は最長 2200 日なので、
    # 観察終了は最も遅くても 2026 年初めで、未来日付は生じない。
    start = (pd.Timestamp("2011-01-01")
             + pd.to_timedelta(rng.integers(0, 3300, n), "D"))
    end = start + pd.to_timedelta(obs.round(0), "D")

    # 検査値は観察期間中のある 1 回の採血パネルから取ったものとする。
    # ★この日が 2020-04 をまたぐので、ALP に本当に段差が生じる。★
    draw = start + pd.to_timedelta((obs * rng.uniform(0, 1, n)).round(0), "D")

    styles_s = rng.choice(8, n, p=_DATE_STYLE_P)
    styles_e = rng.choice(8, n, p=_DATE_STYLE_P)
    styles_h = rng.choice(8, n, p=_DATE_STYLE_P)

    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(1, n + 1)],
        "施設": fac,
        "施設コード": pd.Series(fac).map({"A院": 1, "B院": 2, "C院": 3, "D院": 4}),
        "性別": sex,
        "年齢": age,
        "糖尿病": dm,
        "身長": height,
        "透析開始時刻": [_hhmm(x, rng) for x in t_start],
        "透析終了時刻": [_hhmm(x, rng) for x in t_end],
        "透析前体重": w_pre,
        "透析後体重": w_post,
        "透析前BUN": bun_pre,
        "透析後BUN": bun_post,
        "透析前クレアチニン(Cr)": cr_pre,
        "透析後クレアチニン(Cr)": cr_post,
        "透析前カリウム(K)": k_pre,
        "透析後カリウム(K)": k_post,
        "アルブミン(Alb)": alb,
        "末梢血｜血色素量(Hb)": hb,
        "CRP定量": crp,
        "無機リン(P)": p_,
        "カルシウム(Ca)": ca,
        "インタクトPTH(iPTH)": ipth,
        "β2マイクログロブリン(β2MG)": b2mg,
        "血清鉄(Fe)": fe,
        "総鉄結合能(TIBC)": tibc,
        # ★測定法変更（JSCC→IFCC）で値がおよそ 1/3 になる。★
        #   病態ではなく測定法の段差であり、audit がこれを検出する。
        "アルカリフォスファターゼ(ALP)": np.where(
            draw < ALP_METHOD_CHANGE,
            np.clip(rng.normal(250, 80, n), 60, 900).round(0),
            np.clip(rng.normal(85, 28, n), 20, 300).round(0)),
        "備考": [f"{rng.choice(['特記なし', 'シャント再建', '入院歴あり', '転院'])}_{i}"
               for i in range(n)],
    })

    # ★透析歴そのものは渡さない。透析開始年月日から作らせる。★
    #   実務のデータは導入日が入っていて、年数は入っていないことが多い。
    #   検体採取日から遡った日を導入日とする（= そのときの透析歴が vintage 月）。
    hd_start = draw - pd.to_timedelta((vintage * 30.44).round(0), "D")

    df["透析開始年月日"] = [_fmt(t, s_, rng) for t, s_ in zip(hd_start, styles_h)]
    df["検体採取日"] = [_fmt(t, 0, rng) for t in draw]
    df["観察開始年月日"] = [_fmt(t, s, rng) for t, s in zip(start, styles_s)]
    df["event発生年月日"] = [_fmt(t, s, rng) if e else ""
                        for t, s, e in zip(end, styles_e, evt)]
    df["観察打ち切り年月日"] = ["" if e else _fmt(t, s, rng)
                        for t, s, e in zip(end, styles_e, evt)]

    _inject_dirt(df, end, rng, n)
    df.attrs["medprep_demo"] = {"n": n, "seed": seed, "真のイベント数": int(evt.sum())}
    return df


def _inject_dirt(df: pd.DataFrame, end, rng, n: int) -> None:
    """実務で実際に見る汚れを入れる。★教材の要はここである。★"""
    i = rng.permutation(n)

    def cut(a, b):
        return i[int(n * a / 600):int(n * b / 600)]

    df.loc[cut(0, 8), "観察打ち切り年月日"] = df.loc[cut(0, 8), "event発生年月日"]   # 両方記入
    df.loc[cut(8, 14), ["event発生年月日", "観察打ち切り年月日"]] = ""               # 両方空欄
    df.loc[cut(14, 18), "観察開始年月日"] = ""                                    # 開始日なし
    for k in cut(18, 22):                                                       # 逆転
        df.at[k, "観察開始年月日"] = _fmt(end[k] + pd.Timedelta(days=30), 0, rng)

    # 文字列の汚れを入れる列は、あらかじめ object にしておく
    #（float の列に文字列を代入すると pandas 3 でエラーになる）
    for c in ["CRP定量", "β2マイクログロブリン(β2MG)"]:
        df[c] = df[c].astype(object)

    df.loc[cut(22, 60), "CRP定量"] = "<0.1"                           # 検出限界
    df.loc[cut(60, 95), "インタクトPTH(iPTH)"] = 999                              # 欠損コード
    df.loc[cut(95, 120), "β2マイクログロブリン(β2MG)"] = "未測定"
    df.loc[cut(120, 135), "末梢血｜血色素量(Hb)"] = (                              # g/L 単位混在
        df.loc[cut(120, 135), "末梢血｜血色素量(Hb)"].astype(float) * 10).round(0)
    df.loc[cut(135, 138), "末梢血｜血色素量(Hb)"] = 0                              # あり得ない値
    df.loc[cut(138, 141), "年齢"] = 250                                          # あり得ない値
    df.loc[cut(141, 190), "アルブミン(Alb)"] = np.nan                             # 通常の欠損
    df.loc[cut(190, 205), "透析開始年月日"] = ""      # 導入日が不明な症例

    # ★透析前後の取り違え（C院の検体だけ列の順序が逆）★
    #   施設ごとにエクスポートの仕様が違い、1 施設だけ前後が入れ替わっていた——
    #   実務でいちばんよくある形である。**検定でも例外でも捕まらない。**
    #   透析後 BUN のほうが高いのは生理学的にあり得ず、URR が負になる。
    #   audit の「前後の方向」検査が拾い、施設別に見ると 1 施設に偏っていると分かる。
    swap = df.index[df["施設"] == "C院"]
    for a, b in (("透析前BUN", "透析後BUN"),
                 ("透析前クレアチニン(Cr)", "透析後クレアチニン(Cr)"),
                 ("透析前カリウム(K)", "透析後カリウム(K)"),
                 ("透析前体重", "透析後体重")):
        va, vb = df.loc[swap, a].copy(), df.loc[swap, b].copy()
        df.loc[swap, a], df.loc[swap, b] = vb, va


def save(path="synthetic_dialysis_cohort.xlsx", **kwargs) -> str:
    """合成データを Excel に書き出す（演習では受講者がこれを読み込む）。"""
    df = dialysis_cohort(**kwargs)
    df.to_excel(path, index=False)
    return str(path)
