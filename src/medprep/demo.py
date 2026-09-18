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

    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(1, n + 1)],
        "施設": fac,
        "施設コード": pd.Series(fac).map({"A院": 1, "B院": 2, "C院": 3, "D院": 4}),
        "性別": sex,
        "年齢": age,
        "透析歴_月": vintage,
        "糖尿病": dm,
        "アルブミン(Alb)": alb,
        "末梢血｜血色素量(Hb)": hb,
        "C反応性蛋白(CRP)定量": crp,
        "無機リン(P)": p_,
        "カルシウム(Ca)": ca,
        "インタクトPTH(iPTH)": ipth,
        "β2マイクログロブリン(β2MG)": b2mg,
        # ★測定法変更（JSCC→IFCC）で値がおよそ 1/3 になる。★
        #   病態ではなく測定法の段差であり、audit がこれを検出する。
        "アルカリフォスファターゼ(ALP)": np.where(
            draw < ALP_METHOD_CHANGE,
            np.clip(rng.normal(250, 80, n), 60, 900).round(0),
            np.clip(rng.normal(85, 28, n), 20, 300).round(0)),
        "備考": [f"{rng.choice(['特記なし', 'シャント再建', '入院歴あり', '転院'])}_{i}"
               for i in range(n)],
    })

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
    for c in ["C反応性蛋白(CRP)定量", "β2マイクログロブリン(β2MG)"]:
        df[c] = df[c].astype(object)

    df.loc[cut(22, 60), "C反応性蛋白(CRP)定量"] = "<0.1"                           # 検出限界
    df.loc[cut(60, 95), "インタクトPTH(iPTH)"] = 999                              # 欠損コード
    df.loc[cut(95, 120), "β2マイクログロブリン(β2MG)"] = "未測定"
    df.loc[cut(120, 135), "末梢血｜血色素量(Hb)"] = (                              # g/L 単位混在
        df.loc[cut(120, 135), "末梢血｜血色素量(Hb)"].astype(float) * 10).round(0)
    df.loc[cut(135, 138), "末梢血｜血色素量(Hb)"] = 0                              # あり得ない値
    df.loc[cut(138, 141), "年齢"] = 250                                          # あり得ない値
    df.loc[cut(141, 190), "アルブミン(Alb)"] = np.nan                             # 通常の欠損
    df.loc[cut(190, 205), "透析歴_月"] = np.nan


def save(path="synthetic_dialysis_cohort.xlsx", **kwargs) -> str:
    """合成データを Excel に書き出す（演習では受講者がこれを読み込む）。"""
    df = dialysis_cohort(**kwargs)
    df.to_excel(path, index=False)
    return str(path)
