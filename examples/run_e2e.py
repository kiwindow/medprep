"""第3回・副次到達目標の通し検証:
   汚れた4列形式データ → 掃除 → (duration, event) → Table 1 → KM → log-rank → Cox → PH 検定
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401  日本語フォント
import numpy as np
import pandas as pd

from medprep.clean import clean_numeric, derive
from medprep.survival_input import build_survival

pd.set_option("display.width", 200)
STEP = lambda t: print("\n" + "=" * 78 + f"\n■ {t}\n" + "=" * 78)

# ---------------------------------------------------------------- 1) 読む
STEP("1) データ読み込み")
df = pd.read_excel("synthetic_dialysis_cohort.xlsx")
print(f"{df.shape[0]} 行 × {df.shape[1]} 列")
print("dtypes（汚れているため object のままの列に注目）:")
print(df.dtypes.to_string())

# ---------------------------------------------------------------- 2) 掃除
STEP("2) 辞書駆動の掃除（検出限界・欠損コード・単位混在・あり得ない値）")
clean, rep = clean_numeric(df)
rep.show()

STEP("3) 派生指標の自動算出")
clean, notes = derive(clean)
for n in notes:
    print(" -", n)

# ---------------------------------------------------------------- 4) 生存時間
STEP("4) 4列の日付 → (duration, event) への変換")
sf = build_survival(
    clean,
    id_col="仮名ID",
    start_date="観察開始年月日",
    event_date="event発生年月日",
    censor_date="観察打ち切り年月日",
    covariates=["施設", "性別", "年齢", "透析歴_月", "糖尿病",
                "アルブミン(Alb)", "末梢血｜血色素量(Hb)", "C反応性蛋白(CRP)定量",
                "無機リン(P)", "補正Ca"],
    unit="years",
)
print(sf.report())
print("\n除外された症例:")
print(sf.excluded.to_string(index=False))

d = sf.data.rename(columns={
    "アルブミン(Alb)": "Alb", "末梢血｜血色素量(Hb)": "Hb",
    "C反応性蛋白(CRP)定量": "CRP", "無機リン(P)": "P", "透析歴_月": "vintage"})
d["logCRP"] = np.log(d["CRP"] + 0.1)
d["低Alb"] = np.where(d["Alb"] < 3.5, "Alb<3.5", "Alb≥3.5")

# ---------------------------------------------------------------- 5) Table 1
STEP("5) Table 1（低アルブミン群 vs 非低アルブミン群）")
from tableone import TableOne

cols = ["年齢", "性別", "糖尿病", "vintage", "Hb", "CRP", "P", "補正Ca", "duration", "event"]
t1 = TableOne(d[cols + ["低Alb"]], columns=cols,
              categorical=["性別", "糖尿病", "event"],
              nonnormal=["CRP", "vintage", "duration"],
              groupby="低Alb", pval=True, smd=True, htest_name=True, missing=True)
print(t1.tabulate(tablefmt="github"))
t1.to_excel("table1.xlsx")

# ---------------------------------------------------------------- 6) KM
STEP("6) Kaplan-Meier 曲線 + log-rank")
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fitters = []
for g, sub in d.groupby("低Alb"):
    km = KaplanMeierFitter(label=f"{g} (n={len(sub)})")
    km.fit(sub["duration"], sub["event"])
    km.plot_survival_function(ax=axes[0], ci_show=True)
    fitters.append(km)
    med = km.median_survival_time_
    print(f"  {g:10s} n={len(sub):3d} events={int(sub.event.sum()):3d} "
          f"生存期間中央値={'NR' if not np.isfinite(med) else f'{med:.2f} 年'}")
add_at_risk_counts(*fitters, ax=axes[0])
axes[0].set_title("アルブミン値による生存曲線"); axes[0].set_xlabel("観察期間（年）")

a = d[d["低Alb"] == "Alb<3.5"]; b = d[d["低Alb"] == "Alb≥3.5"]
lr = logrank_test(a["duration"], b["duration"], a["event"], b["event"])
print(f"\n  log-rank 検定: χ² = {lr.test_statistic:.2f}, p = {lr.p_value:.3g}")

for g, sub in d.groupby("施設"):
    km = KaplanMeierFitter(label=f"{g} (n={len(sub)})")
    km.fit(sub["duration"], sub["event"]); km.plot_survival_function(ax=axes[1], ci_show=False)
axes[1].set_title("施設別の生存曲線"); axes[1].set_xlabel("観察期間（年）")
mlr = multivariate_logrank_test(d["duration"], d["施設"], d["event"])
print(f"  多群 log-rank（施設 4 群）: χ² = {mlr.test_statistic:.2f}, p = {mlr.p_value:.3g}")
plt.tight_layout(); plt.savefig("km_curves.png", dpi=150); plt.close()
print("  → km_curves.png を保存")

# ---------------------------------------------------------------- 7) Cox
STEP("7) Cox 比例ハザード回帰")
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

cov = ["年齢", "Alb", "Hb", "logCRP", "糖尿病", "vintage"]
print("単変量スクリーニング:")
uni = []
for c in cov:
    sub = d[["duration", "event", c]].dropna()
    m = CoxPHFitter().fit(sub, "duration", "event")
    r = m.summary.loc[c]
    uni.append((c, r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"], r["p"]))
u = pd.DataFrame(uni, columns=["変数", "HR", "95%CI下限", "95%CI上限", "p"])
print(u.round(3).to_string(index=False))

fit = d[["duration", "event"] + cov].dropna()
n_ev = int(fit["event"].sum())
print(f"\n多変量 Cox: n = {len(fit)}, イベント = {n_ev}, 共変量 = {len(cov)}, "
      f"EPV = {n_ev/len(cov):.1f}" + ("  ← EPV<10 なら警告" if n_ev/len(cov) < 10 else "  （EPV≥10: 可）"))
cph = CoxPHFitter().fit(fit, "duration", "event")
s = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
s.columns = ["HR", "95%CI下限", "95%CI上限", "p"]
print(s.round(3).to_string())
print(f"\n  C-index = {cph.concordance_index_:.3f}")

print("\n比例ハザード仮定の検定（Schoenfeld 残差, time_transform='rank'）:")
ph = proportional_hazard_test(cph, fit, time_transform="rank")
pht = ph.summary.round(3)
print(pht.to_string())
viol = pht[pht["p"] < 0.05].index.tolist()
print("  → " + ("違反なし（全て p ≥ 0.05）" if not viol
                else f"★PH 仮定違反: {viol} — 層別化または時間依存項を検討すること"))

# フォレストプロット
fig, ax = plt.subplots(figsize=(7, 4))
cph.plot(ax=ax); ax.set_title("多変量 Cox 回帰（log HR と 95%CI）")
plt.tight_layout(); plt.savefig("cox_forest.png", dpi=150); plt.close()
print("  → cox_forest.png を保存")

# ---------------------------------------------------------------- 8) 真値との照合
STEP("8) 合成データの真の係数との照合（推定が正しいことの確認）")
truth = {"年齢": 0.045, "Alb": -0.85, "Hb": -0.18, "logCRP": 0.30,
         "糖尿病": 0.35, "vintage": 0.002}
chk = pd.DataFrame({
    "真の係数": pd.Series(truth),
    "推定係数": cph.params_,
    "95%CI下限": cph.confidence_intervals_.iloc[:, 0],
    "95%CI上限": cph.confidence_intervals_.iloc[:, 1]})
chk["CIに真値を含む"] = (chk["95%CI下限"] <= chk["真の係数"]) & (chk["真の係数"] <= chk["95%CI上限"])
print(chk.round(3).to_string())
print(f"\n  {int(chk['CIに真値を含む'].sum())}/{len(chk)} の変数で 95%CI が真値を含む")
print("\n✅ 通し検証 完了")
