"""生存時間データの 3 形式。矛盾を黙って解決しないことを確かめる。"""
import numpy as np
import pandas as pd
import pytest

from medprep.survival_input import build_survival


def four_column_frame():
    return pd.DataFrame({
        "仮名ID": [f"A{i:03d}" for i in range(1, 9)],
        "観察開始年月日": ["2013/3/3", "{2014, 5, 20}", "2015.7.1", "2016,1,15",
                      "H29.4.1", 41336, "2018/6/1", ""],
        "event発生年月日": ["2015/6/30", "", "2017.12.25", "", "", "2014/1/1", "2018/6/1", "2015/1/1"],
        "観察打ち切り年月日": ["", "2018/3/31", "", "2021,6,30", "2023/3/31", "2014/1/1", "", ""],
        "年齢": [65, 72, 58, 80, 69, 55, 77, 70],
    })


def test_form_c_derives_event_and_duration():
    sf = build_survival(four_column_frame(), id_col="仮名ID", start_date="観察開始年月日",
                        event_date="event発生年月日", censor_date="観察打ち切り年月日",
                        covariates=["年齢"], unit="days")
    assert sf.form.startswith("C")
    got = dict(zip(sf.data["仮名ID"], sf.data["event"]))
    assert got["A001"] == 1 and got["A002"] == 0 and got["A003"] == 1 and got["A004"] == 0
    assert (sf.data["duration"] > 0).all()


def test_both_dates_present_is_excluded_not_silently_resolved():
    sf = build_survival(four_column_frame(), id_col="仮名ID", start_date="観察開始年月日",
                        event_date="event発生年月日", censor_date="観察打ち切り年月日")
    reasons = dict(zip(sf.excluded["ID"], sf.excluded["理由"]))
    assert "A006" in reasons and "両方" in reasons["A006"]
    assert "A006" not in set(sf.data["仮名ID"])


def test_missing_start_or_end_is_excluded_with_reason():
    sf = build_survival(four_column_frame(), id_col="仮名ID", start_date="観察開始年月日",
                        event_date="event発生年月日", censor_date="観察打ち切り年月日")
    reasons = dict(zip(sf.excluded["ID"], sf.excluded["理由"]))
    assert "観察開始日" in reasons["A008"]


def test_excluded_table_keeps_the_original_input_values():
    """なぜ除かれたかを後から説明できるよう、元の文字列を残すこと。"""
    sf = build_survival(four_column_frame(), id_col="仮名ID", start_date="観察開始年月日",
                        event_date="event発生年月日", censor_date="観察打ち切り年月日")
    assert "入力値:観察開始日" in sf.excluded.columns
    assert "入力値:イベント発生日" in sf.excluded.columns


def test_reversed_dates_are_excluded():
    df = pd.DataFrame({"id": ["x"], "s": ["2014/6/7"], "e": ["2014/5/8"], "c": [""]})
    sf = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c")
    assert len(sf.data) == 0
    assert "逆転" in sf.excluded["理由"].iloc[0]


def test_zero_duration_is_handled_not_left_to_break_lifelines():
    df = pd.DataFrame({"id": ["x"], "s": ["2018/6/1"], "e": [""], "c": ["2018/6/1"]})
    sf = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c",
                        unit="days")
    assert sf.data["duration"].iloc[0] == pytest.approx(0.5)
    assert any("0.5 日" in w for w in sf.warnings)

    sf = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c",
                        zero_duration="drop")
    assert len(sf.data) == 0


@pytest.mark.parametrize("unit,factor", [("days", 1.0), ("weeks", 7.0),
                                         ("months", 30.4375), ("years", 365.25)])
def test_units(unit, factor):
    df = pd.DataFrame({"id": ["x"], "s": ["2013/1/1"], "e": ["2014/1/1"], "c": [""]})
    sf = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c",
                        unit=unit)
    assert sf.data["duration"].iloc[0] == pytest.approx(365.0 / factor)


def test_inclusive_counting_adds_one_day():
    df = pd.DataFrame({"id": ["x"], "s": ["2013/1/1"], "e": ["2013/1/31"], "c": [""]})
    a = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c",
                       unit="days")
    b = build_survival(df, id_col="id", start_date="s", event_date="e", censor_date="c",
                       unit="days", inclusive=True)
    assert b.data["duration"].iloc[0] - a.data["duration"].iloc[0] == 1.0


def test_form_a_duration_event():
    df = pd.DataFrame({"t": [10.0, 20.0], "e": [1, 0]})
    sf = build_survival(df, duration="t", event="e")
    assert sf.form.startswith("A") and sf.n == 2 and sf.n_events == 1


def test_form_b_start_end_event():
    df = pd.DataFrame({"s": ["2013/1/1", "2013/1/1"], "t": ["2014/1/1", "2015/1/1"], "e": [1, 0]})
    sf = build_survival(df, start_date="s", end_date="t", event="e", unit="days")
    assert sf.form.startswith("B") and sf.n == 2 and sf.n_events == 1


def test_unknown_form_raises_with_guidance():
    with pytest.raises(ValueError, match="入力形式を特定できない"):
        build_survival(pd.DataFrame({"a": [1]}))


def test_epv_and_warnings():
    n = 40
    df = pd.DataFrame({"t": np.arange(1.0, n + 1), "e": [1] * 5 + [0] * (n - 5)})
    sf = build_survival(df, duration="t", event="e")
    assert sf.epv(5) == pytest.approx(1.0)
    assert any("イベント数" in w for w in sf.warnings)


def test_all_events_is_flagged():
    df = pd.DataFrame({"t": [1.0, 2.0], "e": [1, 1]})
    sf = build_survival(df, duration="t", event="e")
    assert any("打ち切りが無い" in w for w in sf.warnings)
