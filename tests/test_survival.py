"""生存時間解析。

確かめたいのは、数字が出ることではなく **黙って壊れないこと** である。

  * 共変量を 1 本足すと Cox の n が黙って減る（完全ケース解析）
  * 生存期間中央値に到達しない群を NaN のまま流す
  * 比例ハザード違反を「p<0.05 でした」で終わらせる
  * 順序の無い群に傾向検定を当てる

どれも例外を出さずに通ってしまう。
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from lifelines.statistics import logrank_test  # noqa: E402

from medprep.survival import (  # noqa: E402
    NR,
    CoxResult,
    Survival,
    logrank_trend,
)
from medprep.survival_input import build_survival  # noqa: E402

rng = np.random.default_rng(20260918)


def frame(n=300, censor=0.35, beta=0.8):
    """x が大きいほど予後が悪いデータ。打ち切りは独立に入れる。"""
    x = rng.normal(0, 1, n)
    g = rng.choice(["A", "B", "C"], n)
    t_event = rng.exponential(1 / (0.15 * np.exp(beta * x)))
    t_cens = rng.exponential(1 / (0.15 * censor / (1 - censor)) if censor else 1e9, n)
    d = np.minimum(t_event, t_cens)
    e = (t_event <= t_cens).astype(int)
    return pd.DataFrame({"duration": d, "event": e, "x": x, "群": g})


def s_(df=None, **kw):
    return Survival(df if df is not None else frame(), unit="years", **kw)


# ---------------------------------------------------------------- 構築
def test_event_column_must_be_zero_or_one():
    df = frame(30)
    df.loc[0, "event"] = 2
    with pytest.raises(ValueError, match="0/1 以外"):
        Survival(df)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="がデータに無い"):
        Survival(pd.DataFrame({"a": [1]}), time="t", event="e")


def test_rows_without_duration_or_event_are_dropped_with_a_note():
    df = frame(50)
    df.loc[:4, "duration"] = np.nan
    s = Survival(df)
    assert s.n == 45
    assert any("欠測" in n for n in s.notes)


def test_from_survival_frame_carries_the_unit():
    raw = pd.DataFrame({
        "id": [f"P{i}" for i in range(6)],
        "s": ["2013/1/1"] * 6,
        "e": ["2015/1/1", "", "2016/1/1", "", "2017/1/1", ""],
        "c": ["", "2018/1/1", "", "2019/1/1", "", "2020/1/1"],
        "年齢": [60, 70, 65, 72, 68, 55],
    })
    sf = build_survival(raw, id_col="id", start_date="s", event_date="e", censor_date="c",
                        covariates=["年齢"], unit="years")
    s = Survival.from_survival_frame(sf)
    assert s.unit == "years" and s.n == 6 and s.n_events == 3


def test_summary_counts_match():
    df = frame(200)
    s = s_(df)
    assert s.n == 200
    assert s.n_events == int(df["event"].sum())
    assert "イベント率" in set(s.summary()["項目"])


# ---------------------------------------------------------------- KM
def test_km_has_one_row_per_group():
    s = s_()
    km = s.km(by="群")
    assert set(km.table["群"]) == {"A", "B", "C"}
    assert km.table["n"].sum() == s.n
    assert km.figure is not None


def test_km_reports_nr_when_the_median_is_not_reached():
    """打ち切りばかりなら中央値は出ない。NaN で流さず NR と書く。"""
    n = 60
    df = pd.DataFrame({"duration": rng.uniform(1, 5, n), "event": np.zeros(n, dtype=int)})
    km = Survival(df, unit="years").km()
    assert NR in km.table.iloc[0]["生存期間中央値 [years]"]
    assert any("到達しなかった" in x for x in km.notes)


def test_a_group_that_did_reach_its_median_is_not_called_not_reached():
    """★信頼区間の上限が NR なのと、中央値が出ないのは別の話である。★

    文字列で 'NR' を探すと、'4.6 [3.0–NR]' を『到達せず』と誤判定する。
    """
    n = 400
    df = pd.DataFrame({"duration": rng.exponential(2, n),
                       "event": np.ones(n, dtype=int)})
    km = Survival(df, unit="years").km()
    assert NR not in km.table.iloc[0]["生存期間中央値 [years]"].split("[")[0]
    assert not any("到達しなかった" in x for x in km.notes)


def test_follow_up_median_is_not_the_survival_median():
    s = s_(frame(400))
    km = s.km()
    row = km.table.iloc[0]
    assert row["生存期間中央値 [years]"] != row["追跡期間中央値 [years]"]
    assert any("reverse Kaplan-Meier" in n for n in km.notes)


# ---------------------------------------------------------------- log-rank
def test_two_group_logrank_matches_lifelines():
    df = frame(300)
    df["群2"] = np.where(df["x"] > 0, "高", "低")
    s = s_(df)
    r = s.logrank(by="群2")
    a, b = df["群2"] == "高", df["群2"] == "低"
    ref = logrank_test(df["duration"][a], df["duration"][b],
                       df["event"][a], df["event"][b])
    assert r.statistic == pytest.approx(float(ref.test_statistic))
    assert r.p == pytest.approx(float(ref.p_value))
    assert r.df == 1


def test_three_group_logrank_adds_pairwise_with_holm():
    r = s_().logrank(by="群")
    assert r.df == 2 and r.pairwise is not None
    assert len(r.pairwise) == 3
    assert (r.pairwise["p補正"] >= r.pairwise["p"] - 1e-12).all()
    assert set(r.pairwise["補正法"]) == {"HOLM"}
    assert any("対比較は" in n for n in r.notes)


def test_logrank_warns_about_groups_without_events():
    df = frame(120)
    df.loc[df.index[:30], "群"] = "D"
    df.loc[df["群"] == "D", "event"] = 0
    r = s_(df).logrank(by="群")
    assert any("イベントが 1 件も無い群" in n for n in r.notes)


def test_logrank_needs_at_least_two_levels():
    df = frame(50)
    df["群"] = "A"
    with pytest.raises(ValueError, match="水準が"):
        s_(df).logrank(by="群")


# ---------------------------------------------------------------- 傾向検定
def test_trend_test_with_two_groups_equals_the_logrank_test():
    """★スコア (0,1) の傾向検定は、通常の log-rank と一致しなければならない。★

    自前実装の正しさをこれで固定する。
    """
    df = frame(300)
    g = np.where(df["x"] > 0, "高", "低")
    st, pt = logrank_trend(df["duration"], g, df["event"], levels=["低", "高"], scores=[0, 1])
    ref = logrank_test(df["duration"][g == "低"], df["duration"][g == "高"],
                       df["event"][g == "低"], df["event"][g == "高"])
    assert st == pytest.approx(float(ref.test_statistic), rel=1e-9)
    assert pt == pytest.approx(float(ref.p_value), rel=1e-9)


def test_trend_test_detects_a_monotone_gradient():
    rows = []
    for i, g in enumerate(["G1", "G2", "G3", "G4"]):
        t = rng.exponential(1 / (0.1 * (1 + i)), 120)
        rows.append(pd.DataFrame({"duration": t, "event": 1, "stage": g}))
    df = pd.concat(rows, ignore_index=True)
    st, p = logrank_trend(df["duration"], df["stage"], df["event"])
    assert p < 0.001 and st > 10


def test_trend_is_announced_as_requiring_ordered_groups():
    r = s_().logrank(by="群")
    assert r.trend is not None
    assert any("順序があるときにだけ" in n for n in r.notes)


def test_trend_can_be_turned_off():
    assert s_().logrank(by="群", trend=False).trend is None


# ---------------------------------------------------------------- Cox
def test_cox_recovers_the_true_coefficient():
    s = s_(frame(1200, beta=0.8))
    c = s.cox(covariates=["x"])
    lo = float(c.summary.loc["x", "95%CI下限"])
    hi = float(c.summary.loc["x", "95%CI上限"])
    assert lo < np.exp(0.8) < hi
    assert 0.5 < c.c_index <= 1.0


def test_cox_reports_how_many_cases_the_missing_covariates_removed():
    """★これがこのモジュールの主目的である。★

    Cox は共変量に欠測のある症例を黙って捨てる。共変量を 1 本足しただけで
    n が減り、それに気づかないまま「ハザード比が変わった」と読むのが
    実データの解析で最も多い事故のひとつ。
    """
    df = frame(300)
    df["y"] = rng.normal(0, 1, len(df))
    df.loc[df.index[:80], "y"] = np.nan
    c = s_(df).cox(covariates=["x", "y"], impute=None)      # 補完しない（完全ケース）
    assert c.n == 220 and c.n_dropped == 80
    assert c.events_dropped >= 0
    w = " ".join(c.warnings)
    assert "80 例" in w and "黙って減る" in w and "y" in w


def test_cox_imputes_by_default_and_says_what_it_filled():
    """★0.11 から既定で補完する。★ 何を・いくつ・何で埋めたかを必ず出す。"""
    df = frame(300)
    df["y"] = rng.normal(0, 1, len(df))
    df.loc[df.index[:80], "y"] = np.nan
    c = s_(df).cox(covariates=["x", "y"])
    assert c.n == 300 and c.n_dropped == 0
    t = c.imputation
    assert list(t["列"]) == ["y"] and int(t["欠損数"].iloc[0]) == 80
    assert t["欠損率"].iloc[0] == round(80 / 300, 4) and t["方法"].iloc[0] == "中央値"
    assert "補完" in c.report() and "欠損率" in c.report()


def test_cox_warns_when_epv_is_below_ten():
    df = frame(200)
    df.loc[df.index[20:], "event"] = 0          # イベントを 20 件に絞る
    for i in range(5):
        df[f"v{i}"] = rng.normal(0, 1, len(df))
    c = s_(df).cox(covariates=["x"] + [f"v{i}" for i in range(5)])
    assert c.epv < 10
    assert any("EPV" in w for w in c.warnings)


def test_categorical_covariates_are_dummied_with_a_recorded_reference():
    """基準水準が分からなければハザード比は読めない。"""
    c = s_().cox(covariates=["x", "群"])
    assert "群" in c.reference_levels
    ref = c.reference_levels["群"]
    assert f"群={ref}" not in c.covariates
    assert sum(1 for v in c.covariates if v.startswith("群=")) == 2
    assert any("基準" in n for n in c.notes)


def test_constant_covariates_are_refused_with_a_reason():
    df = frame(150)
    df["定数"] = 1.0
    c = s_(df).cox(covariates=["x", "定数"])
    assert any("定数" in w for w in c.warnings)
    assert "定数" not in c.covariates


def test_high_cardinality_categoricals_are_refused():
    df = frame(150)
    df["ID的"] = [f"S{i}" for i in range(len(df))]
    c = s_(df).cox(covariates=["x", "ID的"])
    assert any("水準" in w for w in c.warnings)


def test_auto_screening_records_the_verdict_for_each_variable():
    df = frame(400)
    df["noise"] = rng.normal(0, 1, len(df))
    c = s_(df).cox(covariates="auto")
    assert len(c.univariate) >= 2
    assert "判定" in c.univariate.columns
    assert set(c.univariate["判定"]) & {"多変量へ"}


def test_auto_screening_says_it_is_not_stepwise():
    c = s_(frame(400)).cox(covariates="auto")
    assert any("逐次選択" in n and "stepwise" in n for n in c.notes)


def test_unknown_covariate_raises():
    with pytest.raises(ValueError, match="共変量がデータに無い"):
        s_().cox(covariates=["無い列"])


def test_separation_is_reported_rather_than_silently_producing_a_huge_hr():
    """あるレベルでイベントが 0 件だと係数は発散する。数字だけ出して黙らない。"""
    n = 200
    x = np.array([0] * (n // 2) + [1] * (n // 2))
    d = np.where(x == 1, rng.uniform(0.1, 1.0, n), rng.uniform(5, 10, n))
    e = (x == 1).astype(int)                     # x=0 は全例打ち切り
    df = pd.DataFrame({"duration": d, "event": e, "x": x.astype(float)})
    c = Survival(df, unit="years").cox(covariates=["x"], check_ph=False)
    w = " ".join(c.warnings)
    assert c.failed or "完全分離" in w or "極端に広い" in w
    # lifelines 自身の収束警告も結果に取り込む（標準エラーに流して消さない）
    assert c.failed or "lifelines からの収束警告" in w


def test_proportional_hazards_violation_is_explained_not_just_flagged():
    """★『p<0.05 でした』では何の役にも立たない。どうするかを言う。★"""
    n = 400
    x = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
    # x=1 は早期に、x=0 は遅れてイベントが起こる（ハザード比が時間で反転する）
    d = np.where(x == 1, rng.exponential(1.0, n), 4 + rng.exponential(1.0, n))
    df = pd.DataFrame({"duration": d, "event": np.ones(n, dtype=int), "x": x})
    c = Survival(df, unit="years").cox(covariates=["x"])
    assert "x" in c.ph_violations
    w = " ".join(c.warnings)
    assert "層別" in w and "RMST" in w


def test_check_ph_uses_the_frame_that_was_fitted():
    c = s_().cox(covariates=["x", "群"])
    assert list(c.ph.index.sort_values()) == sorted(c.covariates)
    assert len(c.fit_frame) == c.n


def test_strata_is_accepted():
    s = s_()
    c = s.cox(covariates=["x"], strata="群")
    assert c.strata == ["群"] and c.model is not None


def test_cox_report_is_printable():
    txt = s_().cox(covariates=["x"]).report()
    assert "Cox 比例ハザード回帰" in txt and "C-index" in txt


def test_failed_cox_reports_instead_of_raising():
    df = frame(50)
    c = s_(df).cox(covariates=[])
    assert c.failed and "共変量" in c.failed
    assert "実行できなかった" in c.report()


# ---------------------------------------------------------------- RMST
def test_rmst_matches_the_hand_calculation_without_censoring():
    """打ち切りが無ければ RMST = mean(min(T, τ))。"""
    t = np.array([1.0, 2.0, 3.0, 8.0, 9.0])
    df = pd.DataFrame({"duration": t, "event": np.ones(5, dtype=int)})
    r = Survival(df, unit="years").rmst(t=5.0)
    assert float(r.iloc[0]["RMST [years]"]) == pytest.approx(np.minimum(t, 5).mean(), abs=1e-6)


def test_rmst_by_group_returns_one_row_per_group():
    r = s_().rmst(by="群", t=3.0)
    assert set(r["群"]) == {"A", "B", "C"} and (r["τ [years]"] == 3.0).all()


# ---------------------------------------------------------------- 図
def test_forest_plot_is_drawn(tmp_path):
    s = s_(frame(500))
    c = s.cox(covariates=["x", "群"])
    fig = s.forest(c, save=tmp_path / "forest.png")
    assert (tmp_path / "forest.png").exists() and fig is not None


def test_forest_refuses_when_cox_did_not_run():
    with pytest.raises(ValueError, match="描けない"):
        s_().forest(CoxResult())


def test_km_figure_is_saved(tmp_path):
    s_().km(by="群", save=tmp_path / "km.png")
    assert (tmp_path / "km.png").exists()


def test_schoenfeld_plot_is_drawn(tmp_path):
    s = s_(frame(400))
    c = s.cox(covariates=["x"])
    s.schoenfeld_plot(c, save=tmp_path / "sch.png")
    assert (tmp_path / "sch.png").exists()


# ---------------------------------------------------------------- 通し
def test_report_runs_everything():
    txt = s_(frame(400)).report(by="群", covariates=["x"])
    assert "Kaplan-Meier" in txt and "log-rank" in txt and "Cox" in txt
