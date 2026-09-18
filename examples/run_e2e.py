"""第3回・副次到達目標の通し検証:
   汚れた4列形式データ → 列の役割の推定 → 品質監査 → 掃除 → (duration, event)
   → Table 1 → KM → log-rank → Cox → PH 検定
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import matplotlib

matplotlib.use("Agg")
import matplotlib_fontja  # noqa: F401  日本語フォント
import numpy as np
import pandas as pd

import medprep as mp
from medprep import viz
from medprep.clean import clean_numeric, derive
from medprep.describe import compare_groups, table_one, target_achievement, target_summary
from medprep.missing import analyze as analyze_missing
from medprep.missing import drop_missing_outcome
from medprep.outliers import detect as detect_outliers
from medprep.pipeline import LeakageError, Preprocessor, leak_check, prepare
from medprep.quality import audit
from medprep.report import build_report
from medprep.schema import Schema
from medprep.splitting import split
from medprep.survival import Survival
from medprep.survival_input import build_survival

pd.set_option("display.width", 200)
STEP = lambda t: print("\n" + "=" * 78 + f"\n■ {t}\n" + "=" * 78)

# ---------------------------------------------------------------- 1) 読む
STEP("1) データ読み込み")
df = pd.read_excel("synthetic_dialysis_cohort.xlsx")
print(f"{df.shape[0]} 行 × {df.shape[1]} 列")
print("dtypes（汚れているため object のままの列に注目）:")
print(df.dtypes.to_string())

# ---------------------------------------------------------------- 2) 列の役割
STEP("2) 列の役割の推定（schema）— 判断を理由つきで書き出す")
SURV = ("観察開始年月日", "event発生年月日", "観察打ち切り年月日")
sch = Schema.infer(df, id_col="仮名ID", group="施設", survival_dates=SURV)
print(sch.report())
sch.to_yaml("schema.yaml")
print("\n  → schema.yaml を保存（人が直して再実行できる）")

# ---------------------------------------------------------------- 3) 監査
STEP("3) データ品質監査（audit）— このまま解析してよいかを問う")
aud = audit(df, sch, id_col="仮名ID", group="施設", date_col="検体採取日")
aud.show()
print(f"\n  致命的な所見 {len(aud.errors)} 件。"
      + ("このまま解析してはならない。" if not aud.ok else "無し。")
      + "以降は、除外と掃除でこれらをどう扱うかを見ていく。")

# ---------------------------------------------------------------- 4) 掃除
STEP("4) 辞書駆動の掃除（検出限界・欠損コード・単位混在・あり得ない値）")
clean, rep = clean_numeric(df)
rep.show()

STEP("5) 派生指標の自動算出")
clean, notes = derive(clean)
for n in notes:
    print(" -", n)

# ---------------------------------------------------------------- 6) 生存時間
STEP("6) 4列の日付 → (duration, event) への変換")
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

# ---------------------------------------------------------------- 7) Table 1
STEP("7) Table 1（低アルブミン群 vs 非低アルブミン群）")
cols = ["年齢", "性別", "糖尿病", "vintage", "Hb", "CRP", "P", "補正Ca", "duration", "event"]
t1 = table_one(d, groupby="低Alb", columns=cols)
print(t1.report())
t1.to_excel("table1.xlsx"); t1.to_html("table1.html")
print("  → table1.xlsx / table1.html を保存")

print("\n群間比較の詳細（検定の選択理由と効果量）:")
print(t1.comparison.to_frame()[["項目", "検定", "p", "q(BH)", "効果量", "SMD", "判定の根拠"]]
      .to_string(index=False))

print("\n施設4群での事後比較（3群以上なら Tukey / Dunn まで出す）:")
g4 = compare_groups(d, "施設", ["Alb", "Hb", "CRP"])
print(g4.to_frame()[["項目", "検定", "p", "q(BH)", "効果量"]].to_string(index=False))
ph = g4.posthoc_frame()
print(ph.to_string(index=False) if len(ph) else "  （事後比較なし）")

# ---------------------------------------------------------------- 8) 管理目標
STEP("8) 管理目標の達成率（開区間と閉区間を区別する）")
print(target_summary(clean).to_string(index=False))
ach = target_achievement(clean, by="施設")
print("\n" + ach.to_string(index=False))
print("\n  ★境界値ちょうどの行に注目。P 5.5・補正Ca 9.5・Hb 12.0・iPTH 240 は")
print("    いずれも目標範囲に『含まれない』。<= と < の取り違えが達成率を何％動かすか:")
for key, col in [("P", "無機リン(P)"), ("cCa", "補正Ca"), ("Hb", "末梢血｜血色素量(Hb)")]:
    spec = mp.load_dict()["items"][key]; tg = spec["target"]
    v = pd.to_numeric(clean[col], errors="coerce"); ok = v.notna()
    right = mp.in_target(v, tg)
    wrong = mp.in_target(v, {**tg, "high_inclusive": True, "low_inclusive": True})
    a, b = right[ok].mean(), wrong[ok].mean()
    print(f"    {spec['name_ja']:<12s} 正しく開区間 {a:6.1%} / 誤って閉区間 {b:6.1%}"
          f"  → 差 {b - a:+.1%}（{int((wrong[ok] != right[ok]).sum())} 例）")

# ---------------------------------------------------------------- 9) KM
STEP("9) Kaplan-Meier 曲線 + log-rank（medprep.survival）")
surv = Survival(d, unit="years")
km = surv.km(by="低Alb", title="アルブミン値による生存曲線", save="km_alb.png")
print(km.report())
print("\n" + surv.logrank(by="低Alb").report())
print("  → km_alb.png を保存")

km2 = surv.km(by="施設", title="施設別の生存曲線", save="km_facility.png")
print("\n" + km2.report())
print("\n" + surv.logrank(by="施設").report())
print("  → km_facility.png を保存")

# ---------------------------------------------------------------- 10) Cox
STEP("10) Cox 比例ハザード回帰（medprep.survival）")
cov = ["年齢", "Alb", "Hb", "logCRP", "糖尿病", "vintage"]
cox = surv.cox(covariates=cov)
print(cox.report())
surv.forest(cox, save="cox_forest.png")
surv.schoenfeld_plot(cox, save="cox_schoenfeld.png")
print("\n  → cox_forest.png / cox_schoenfeld.png を保存")

if cox.ph_violations:
    print("\n  PH 違反があったので RMST（比例ハザードを仮定しない指標）も出す:")
    print(surv.rmst(by="低Alb", t=3.0).to_string(index=False))

# ---------------------------------------------------------------- 11) 真値との照合
STEP("11) 合成データの真の係数との照合（推定が正しいことの確認）")
truth = {"年齢": 0.045, "Alb": -0.85, "Hb": -0.18, "logCRP": 0.30,
         "糖尿病": 0.35, "vintage": 0.002}
m = cox.model
chk = pd.DataFrame({
    "真の係数": pd.Series(truth),
    "推定係数": m.params_,
    "95%CI下限": m.confidence_intervals_.iloc[:, 0],
    "95%CI上限": m.confidence_intervals_.iloc[:, 1]})
chk["CIに真値を含む"] = (chk["95%CI下限"] <= chk["真の係数"]) & (chk["真の係数"] <= chk["95%CI上限"])
print(chk.round(3).to_string())
print(f"\n  {int(chk['CIに真値を含む'].sum())}/{len(chk)} の変数で 95%CI が真値を含む")

# ---------------------------------------------------------------- 12) 欠損と外れ値
STEP("12) 欠損と外れ値（補完はここではしない）")
ms = analyze_missing(clean, sch, group="施設")
print(ms.report())
print()
ol = detect_outliers(clean, sch, id_col="仮名ID")
print(ol.report())

# ---------------------------------------------------------------- 13) 分割と前処理
STEP("13) 分割と前処理 — リークを構造的に不可能にする")

# 目的変数: 1 年以内のイベント発生。
#   1 年経たずに打ち切られた症例は「1 年以内に起きたか」を判定できないので除く。
#   ★除いた数を必ず報告する。★
d1 = d.copy()
d1["1年以内イベント"] = np.where(d1["event"] == 1, (d1["duration"] <= 1.0).astype(float), 0.0)
d1.loc[(d1["event"] == 0) & (d1["duration"] < 1.0), "1年以内イベント"] = np.nan
d1, info = drop_missing_outcome(d1, "1年以内イベント")
d1["1年以内イベント"] = d1["1年以内イベント"].astype(int)
print(f"目的変数『1年以内イベント』: {info['残り']} 例（判定できない {info['除外']} 例を除外。"
      f"{info['理由'].splitlines()[0]}）")
print(f"  陽性率 {d1['1年以内イベント'].mean():.1%}")

sch1 = Schema.infer(d1, id_col="仮名ID", group="施設",
                    outcome="1年以内イベント", task="classification")
sp = split(d1, sch1, test_size=0.25, seed=0)
print("\n" + sp.report())

prep_out = prepare(sp, sch1, columns=[c for c in sch1.features()
                                      if c not in ("duration", "event", "低Alb")])
print("\n" + prep_out.report())

print("\nリークの検査:")
print(leak_check(prep_out.preprocessor, sp.train, sp.test).to_string(index=False))

print("\ntest に fit しようとすると止まる:")
try:
    Preprocessor(sch1).fit(sp.test)
except LeakageError as e:
    print("  ✗ " + str(e).split("。")[0] + "。")

# 実際に 1 つモデルを通してみる（前処理が学習にそのまま渡せることの確認）
STEP("14) 前処理済みの行列をそのままモデルに渡す")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

m = LogisticRegression(max_iter=2000).fit(prep_out.X_train, prep_out.y_train)
auc_tr = roc_auc_score(prep_out.y_train, m.predict_proba(prep_out.X_train)[:, 1])
auc_te = roc_auc_score(prep_out.y_test, m.predict_proba(prep_out.X_test)[:, 1])
print(f"ロジスティック回帰  train AUC = {auc_tr:.3f} / test AUC = {auc_te:.3f}")
coef = (pd.Series(m.coef_[0], index=prep_out.X_train.columns)
        .sort_values(key=abs, ascending=False).head(8))
print("\n係数の大きい順（★列名が残っているので読める★）:")
print(coef.round(3).to_string())


# ---------------------------------------------------------------- 15) レポート
STEP("15) 図とレポート — すべてを HTML 1 枚にまとめる")
figs = viz.overview(clean, sch, achievement=ach, balance=sp.balance)
print(f"図 {len(figs)} 枚"
      + (f"（描けなかったもの: {[t for t, _ in figs.skipped]}）" if figs.skipped else ""))

rep = build_report(
    clean, sch,
    title="第3回演習 前処理レポート（合成データ）",
    audit=aud, missing=ms, outliers=ol,
    table1=t1, comparison=t1.comparison, achievement=ach,
    survival_summary=surv.summary(), logrank=surv.logrank(by="低Alb"), cox=cox,
    split=sp, preprocessor=prep_out.preprocessor, figures=figs,
    path="prep_report.html",
)
rep.to_excel("prep_tables.xlsx")
viz.close_all()
size = os.path.getsize("prep_report.html") / 1024 / 1024
print(f"  → prep_report.html（{size:.1f} MB、図は base64 埋め込みの単一ファイル）")
print("  → prep_tables.xlsx（表をシート別に）")
print("  ★症例レベルの値は既定では出していない。"
      "必要なら show_values=True を明示すること。★")



# ---------------------------------------------------------------- 16) 層1
STEP("16) 層1 — ここまでの全段を 1 行で（mp.autoprep）")
print("層2を順に呼ぶだけの薄い層である。層1にしか無い処理は無い。\n")

from medprep import paths as mp_paths

mp_paths.OUTPUT_DIR = os.path.join(os.getcwd(), "_lab_output")   # 検証用の保存先
mp_paths.WORK_DIR = os.path.join(os.getcwd(), "_lab_work")

auto = mp.autoprep(
    "synthetic_dialysis_cohort.xlsx",
    outcome=None,                       # 目的変数は 13) で作った派生列なので層1では使わない
    group="施設", id_col="仮名ID", date_col="検体採取日",
    survival_dates=SURV,
    save=True, method="Preprocessing",
)
print()
print(auto.report())

print("\n  ★run 番号は既存教材と同じ規約で採られている★"
      "（table / model / figure が空でない run は決して上書きしない）")
print(f"  ★{len(auto.warnings)} 件の『人の確認が要る事項』が残っている。"
      "空で返ってくるほうを疑うこと。★")
viz.close_all()

print("\n✅ 通し検証 完了（全 16 段）")
