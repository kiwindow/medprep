"""第3回演習用の合成透析コホートを作る（実データは一切含まない）。

意図的に仕込んだ「汚れ」:
  1. 日付が 7 通りの表記で混在（和暦・Excelシリアル値・全角・時刻付きを含む）
  2. イベント日と打ち切り日の両方が入っている矛盾例
  3. どちらも空欄の例、観察開始日が空欄の例
  4. 観察開始日 > 終了日 の逆転例
  5. 欠損コード 999 / 「未測定」
  6. 検出限界表記 "<0.1"（CRP）
  7. 単位混在（Hb を一部 g/L で記録）
  8. 生理学的にあり得ない値（Hb 0、年齢 250）
  9. ID 的な高カーディナリティ文字列列（備考）
 10. 完全相関する重複列（施設 と 施設コード）
 11. 2020年4月のALP測定法変更による段差
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20260918)
N = 600

fac = rng.choice(["A院", "B院", "C院", "D院"], N, p=[.4, .3, .2, .1])
sex = rng.choice(["男", "女"], N, p=[.62, .38])
age = np.clip(rng.normal(68, 12, N), 20, 98).round(0)
dm = rng.choice([0, 1], N, p=[.6, .4])
alb = np.clip(rng.normal(3.6, 0.45, N), 1.5, 5.2).round(1)
hb = np.clip(rng.normal(10.8, 1.2, N), 6, 15).round(1)
crp = np.exp(rng.normal(-1.2, 1.3, N)).round(2)
p_ = np.clip(rng.normal(5.2, 1.3, N), 1.5, 12).round(1)
ca = np.clip(rng.normal(8.9, 0.7, N), 5, 12).round(1)
ipth = np.exp(rng.normal(4.9, 0.8, N)).round(0)
b2mg = np.clip(rng.normal(28, 6, N), 10, 60).round(1)
vintage = np.clip(rng.exponential(60, N), 1, 400).round(0)   # 透析歴（月）

# 真の生存モデル: 高齢・低Alb・高CRP・低Hb・糖尿病で予後不良
lp = (0.045*(age-68) - 0.85*(alb-3.6) + 0.30*np.log(crp+0.1)
      - 0.18*(hb-10.8) + 0.35*dm + 0.002*(vintage-60))
lam = 0.00045 * np.exp(lp)
t_event = rng.exponential(1/np.clip(lam, 1e-6, None))          # 日
t_cens = rng.uniform(200, 2200, N)
obs = np.minimum(t_event, t_cens)
evt = (t_event <= t_cens).astype(int)

start = pd.Timestamp("2013-01-01") + pd.to_timedelta(rng.integers(0, 1600, N), "D")
end = start + pd.to_timedelta(obs.round(0), "D")


def fmt(ts, style):
    if pd.isna(ts):
        return ""
    y, m, d = ts.year, ts.month, ts.day
    if style == 0: return f"{y}/{m}/{d}"
    if style == 1: return f"{y}-{m:02d}-{d:02d}"
    if style == 2: return f"{y}.{m}.{d}"
    if style == 3: return "{" + f"{y}, {m}, {d}" + "}"
    if style == 4: return f"{y}-{m}-{d} {rng.integers(0,24):02d}:{rng.integers(0,60):02d}:{rng.integers(0,60):02d}"
    if style == 5:                                    # 和暦
        return (f"H{y-1988}.{m}.{d}" if y < 2019 or (y == 2019 and m < 5)
                else f"R{y-2018}.{m}.{d}")
    if style == 6:                                    # Excel シリアル値
        return int((ts - pd.Timestamp("1899-12-30")).days)
    if style == 7:                                    # 全角
        return str(f"{y}/{m}/{d}").translate(str.maketrans("0123456789/", "０１２３４５６７８９／"))
    return str(ts.date())


styles_s = rng.choice(8, N, p=[.30, .18, .10, .06, .10, .08, .10, .08])
styles_e = rng.choice(8, N, p=[.30, .18, .10, .06, .10, .08, .10, .08])

df = pd.DataFrame({
    "仮名ID": [f"P{i:04d}" for i in range(1, N+1)],
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
    "アルカリフォスファターゼ(ALP)": np.where(
        start.year < 2020, np.clip(rng.normal(250, 80, N), 60, 900).round(0),
        np.clip(rng.normal(85, 28, N), 20, 300).round(0)),
    "備考": [f"{rng.choice(['特記なし','シャント再建','入院歴あり','転院'])}_{i}" for i in range(N)],
})

df["観察開始年月日"] = [fmt(t, s) for t, s in zip(start, styles_s)]
df["event発生年月日"] = [fmt(t, s) if e else "" for t, s, e in zip(end, styles_e, evt)]
df["観察打ち切り年月日"] = ["" if e else fmt(t, s) for t, s, e in zip(end, styles_e, evt)]

# ------------------------------------------------ 汚れを注入
i = rng.permutation(N)
df.loc[i[:8], "観察打ち切り年月日"] = df.loc[i[:8], "event発生年月日"]        # 両方記入（矛盾）
df.loc[i[8:14], ["event発生年月日", "観察打ち切り年月日"]] = ""               # 両方空欄
df.loc[i[14:18], "観察開始年月日"] = ""                                      # 開始日なし
for k in i[18:22]:                                                           # 逆転
    df.at[k, "観察開始年月日"] = fmt(end[k] + pd.Timedelta(days=30), 0)
df.loc[i[22:60], "C反応性蛋白(CRP)定量"] = "<0.1"                            # 検出限界
df.loc[i[60:95], "インタクトPTH(iPTH)"] = 999                                # 欠損コード
df.loc[i[95:120], "β2マイクログロブリン(β2MG)"] = "未測定"
df.loc[i[120:135], "末梢血｜血色素量(Hb)"] = (df.loc[i[120:135], "末梢血｜血色素量(Hb)"]
                                             .astype(float) * 10).round(0)   # g/L 単位混在
df.loc[i[135:138], "末梢血｜血色素量(Hb)"] = 0                                # あり得ない値
df.loc[i[138:141], "年齢"] = 250                                             # あり得ない値
df.loc[i[141:190], "アルブミン(Alb)"] = np.nan                                # 通常の欠損
df.loc[i[190:205], "透析歴_月"] = np.nan

df.to_excel("synthetic_dialysis_cohort.xlsx", index=False)
print(f"{len(df)} 行 × {df.shape[1]} 列 を synthetic_dialysis_cohort.xlsx に保存")
print(f"真のイベント数: {evt.sum()}（注入した汚れにより解析対象は減る）")
