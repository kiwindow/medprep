"""欠損の分析。**補完しないこと**と、**欠損の偏りを名指しすること**を確かめる。"""
import numpy as np
import pandas as pd
import pytest

from medprep.missing import (
    analyze,
    drop_missing_outcome,
    mcar_signals,
    missingness_by_group,
)
from medprep.schema import Schema

rng = np.random.default_rng(20260918)


def frame(n=300):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "アルブミン(Alb)": rng.normal(3.6, 0.45, n).round(1),
        "末梢血｜血色素量(Hb)": rng.normal(10.8, 1.2, n).round(1),
    })
    df.loc[df.index[:40], "アルブミン(Alb)"] = np.nan          # 完全にランダムな欠損
    return df


# ---------------------------------------------------------------- 分析
def test_analyze_does_not_modify_the_data():
    df = frame()
    before = df.copy(deep=True)
    analyze(df)
    pd.testing.assert_frame_equal(df, before)


def test_column_and_row_summaries():
    df = frame()
    r = analyze(df)
    row = r.columns[r.columns["列"] == "アルブミン(Alb)"].iloc[0]
    assert int(row["欠損"]) == 40
    assert r.n_complete == 260
    assert r.complete_rate == pytest.approx(260 / 300)


def test_patterns_name_the_columns_that_go_missing_together():
    df = frame()
    df.loc[df.index[:40], "末梢血｜血色素量(Hb)"] = np.nan     # Alb と同時に欠ける
    r = analyze(df)
    top = r.patterns.iloc[1]["欠損している列"]
    assert "アルブミン(Alb)" in top and "Hb" in top
    assert int(r.patterns.iloc[1]["症例数"]) == 40


def test_it_always_says_that_imputation_happens_elsewhere():
    """★補完は train でしか fit できない。だからここではやらない。★"""
    r = analyze(frame())
    assert any("補完はこのモジュールでは行わない" in n for n in r.notes)
    assert any("add_indicator" in n for n in r.notes)


def test_a_half_empty_column_is_called_out():
    df = frame()
    df.loc[df.index[:200], "年齢"] = np.nan
    r = analyze(df)
    assert any("補完アルゴリズムの出力" in n for n in r.notes)


def test_survival_date_columns_are_excluded_from_the_missing_rate():
    """形式C では『イベント発生日が空欄』＝イベント無し。欠損ではない。"""
    df = pd.DataFrame({
        "仮名ID": [f"P{i}" for i in range(6)],
        "観察開始年月日": ["2013/1/1"] * 6,
        "死亡年月日": ["2015/1/1", "", "2016/1/1", "", "2017/1/1", ""],
        "打切年月日": ["", "2018/1/1", "", "2019/1/1", "", "2020/1/1"],
        "年齢": [60, 70, 65, 72, 68, 55],
    })
    sch = Schema.infer(df, id_col="仮名ID",
                       survival_dates=("観察開始年月日", "死亡年月日", "打切年月日"))
    r = analyze(df, sch)
    assert "死亡年月日" not in set(r.columns["列"])
    assert "打切年月日" not in set(r.columns["列"])


# ---------------------------------------------------------------- MCAR
def test_mcar_signals_is_empty_when_the_missingness_is_random():
    r = mcar_signals(frame(), ["年齢", "アルブミン(Alb)", "末梢血｜血色素量(Hb)"])
    assert not len(r)


def test_mcar_signals_names_the_column_the_missingness_is_tied_to():
    """★欠損が群に偏っていれば、全体の中央値で埋めると群間差が人工的に作られる。★"""
    n = 400
    df = pd.DataFrame({
        "施設": ["A院"] * (n // 2) + ["B院"] * (n // 2),
        "Alb": rng.normal(3.6, 0.4, n).round(1),
        "年齢": rng.normal(68, 12, n).round(0),
    })
    df.loc[df["施設"] == "B院", "Alb"] = np.nan          # B 院では測っていない
    r = mcar_signals(df, ["Alb", "年齢", "施設"])
    assert len(r)
    assert set(r["欠損した列"]) == {"Alb"}
    assert "施設" in set(r["関連する列"])


def test_mcar_signal_detected_in_the_report_adds_a_note():
    n = 400
    df = pd.DataFrame({
        "施設": ["A院"] * (n // 2) + ["B院"] * (n // 2),
        "Alb": rng.normal(3.6, 0.4, n).round(1),
    })
    df.loc[df["施設"] == "B院", "Alb"] = np.nan
    r = analyze(df, group="施設")
    assert len(r.signals)
    assert any("MCAR ではない" in n for n in r.notes)


def test_columns_that_are_almost_never_missing_are_not_tested():
    df = frame()
    df.loc[df.index[:1], "年齢"] = np.nan          # 0.3% しか欠損していない
    r = mcar_signals(df, ["年齢", "アルブミン(Alb)"])
    assert "年齢" not in set(r["欠損した列"]) if len(r) else True


# ---------------------------------------------------------------- 目的変数
def test_missing_outcome_is_dropped_not_imputed():
    """★補完した目的変数で学習した結果は解釈できない。★"""
    df = frame()
    df["転帰"] = rng.choice([0, 1], len(df))
    df.loc[df.index[:25], "転帰"] = np.nan
    out, info = drop_missing_outcome(df, "転帰")
    assert info["除外"] == 25 and info["残り"] == len(df) - 25
    assert out["転帰"].notna().all()
    assert "補完してはならない" in info["理由"]


def test_dropping_a_missing_outcome_needs_the_column_to_exist():
    with pytest.raises(ValueError, match="目的変数"):
        drop_missing_outcome(frame(), "存在しない列")


def test_analyze_warns_when_the_outcome_has_missing_values():
    df = frame()
    df["転帰"] = rng.choice([0, 1], len(df))
    df.loc[df.index[:5], "転帰"] = np.nan
    r = analyze(df, outcome="転帰")
    assert any("目的変数" in n and "補完してはならない" in n for n in r.notes)


# ---------------------------------------------------------------- 群別
def test_missingness_by_group_shows_each_level():
    n = 200
    df = pd.DataFrame({
        "施設": ["A院"] * 100 + ["B院"] * 100,
        "Alb": rng.normal(3.6, 0.4, n),
    })
    df.loc[df["施設"] == "B院", "Alb"] = np.nan
    t = missingness_by_group(df, "施設", columns=["Alb"])
    assert t.iloc[0]["A院"] == "0.0%" and t.iloc[0]["B院"] == "100.0%"


def test_missingness_by_group_needs_the_column():
    with pytest.raises(ValueError, match="群分け列"):
        missingness_by_group(frame(), "存在しない列")


# ---------------------------------------------------------------- 図・報告
def test_plot_runs_for_every_kind(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    df = frame()
    r = analyze(df, columns=["年齢", "アルブミン(Alb)", "末梢血｜血色素量(Hb)"])
    for kind in ("matrix", "bar"):
        r.plot(df, kind, save=tmp_path / f"{kind}.png")
        assert (tmp_path / f"{kind}.png").exists()
    with pytest.raises(ValueError, match="kind は"):
        r.plot(df, "存在しない図")


def test_report_is_printable():
    assert "欠損の分析" in analyze(frame()).report()
