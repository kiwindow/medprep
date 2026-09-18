"""日付パーサ。多様な表記を受け入れ、決まらないときは推測しないことを確かめる。"""
import pandas as pd
import pytest

from medprep.dates import parse_date_series

EXPECT_20130303 = [
    "2013/3/3", "2013-3-3", "2013.3.3", "2013,3,3",
    "{2013, 3, 3}", "(2013,3,3)", "  2013 / 3 / 3  ",
    "2013-3-3 14:32:11", "2013-03-03T14:32:11", "2013年3月3日 14時32分",
    "2013/03/03", "20130303", "2013年3月3日", "２０１３／３／３",
    "H25.3.3", "平成25年3月3日", "平成25.3.3",
    41336, "41336", 41336.6,
]


@pytest.mark.parametrize("raw", EXPECT_20130303)
def test_various_notations_resolve_to_same_date(raw):
    r = parse_date_series(pd.Series([raw], dtype=object))
    assert r.values.iloc[0] == pd.Timestamp("2013-03-03")


@pytest.mark.parametrize(
    "raw,expected",
    [("R5.3.3", "2023-03-03"), ("令和元年5月1日", "2019-05-01"), ("S48.11.23", "1973-11-23")],
)
def test_wareki_eras(raw, expected):
    r = parse_date_series(pd.Series([raw], dtype=object))
    assert r.values.iloc[0] == pd.Timestamp(expected)


def test_time_of_day_is_dropped():
    r = parse_date_series(pd.Series(["2013-3-3 14:32:11"], dtype=object))
    assert r.values.iloc[0].hour == 0 and r.values.iloc[0].minute == 0


@pytest.mark.parametrize("raw", ["", None, "-", "未測定", "不明", "9999"])
def test_null_markers_become_nat_and_are_counted_as_null(raw):
    r = parse_date_series(pd.Series([raw], dtype=object))
    assert pd.isna(r.values.iloc[0])
    assert r.n_null == 1 and r.n_failed == 0


@pytest.mark.parametrize("raw", ["abc", "2013/13/45"])
def test_unparseable_is_reported_not_silently_dropped(raw):
    r = parse_date_series(pd.Series([raw], dtype=object))
    assert pd.isna(r.values.iloc[0])
    assert r.n_failed == 1
    assert r.failures and r.failures[0][1] == raw


def test_day_month_order_inferred_when_a_value_exceeds_12():
    r = parse_date_series(pd.Series(["25/03/2013", "14/07/2013", "3/3/2013"], dtype=object))
    assert r.order == "dmy"
    assert list(r.values.dt.strftime("%Y-%m-%d")) == ["2013-03-25", "2013-07-14", "2013-03-03"]

    r = parse_date_series(pd.Series(["03/25/2013", "07/14/2013", "3/3/2013"], dtype=object))
    assert r.order == "mdy"
    assert list(r.values.dt.strftime("%Y-%m-%d")) == ["2013-03-25", "2013-07-14", "2013-03-03"]


def test_ambiguous_order_is_refused_not_guessed():
    """年が末尾で日・月とも 12 以下しかない列は、誤れば観察期間が最大 11 か月ずれる。
    黙って ymd を仮定せず、列を未解析のまま返して指定を求めること。"""
    r = parse_date_series(pd.Series(["03/04/2013", "05/06/2013"], dtype=object))
    assert r.order == "ambiguous_dmy_mdy"
    assert r.values.isna().all()
    assert any("確定できない" in n for n in r.notes)


def test_explicit_order_resolves_the_ambiguous_case():
    r = parse_date_series(pd.Series(["03/04/2013"], dtype=object), order="dmy")
    assert r.values.iloc[0] == pd.Timestamp("2013-04-03")
    r = parse_date_series(pd.Series(["03/04/2013"], dtype=object), order="mdy")
    assert r.values.iloc[0] == pd.Timestamp("2013-03-04")


def test_year_month_only_is_refused_by_default():
    """日が無いと観察期間に最大 30 日の誤差が入る。黙って 1 日にしない。"""
    r = parse_date_series(pd.Series(["2013/3"], dtype=object))
    assert pd.isna(r.values.iloc[0]) and r.n_failed == 1

    r = parse_date_series(pd.Series(["2013/3"], dtype=object), allow_year_month=True)
    assert r.values.iloc[0] == pd.Timestamp("2013-03-01")
    assert any("月初" in n for n in r.notes)


def test_datetime_dtype_passes_through_normalized():
    s = pd.to_datetime(["2013-03-03 14:32:11", "2014-07-14 09:00:00"])
    r = parse_date_series(pd.Series(s))
    assert list(r.values.dt.strftime("%Y-%m-%d")) == ["2013-03-03", "2014-07-14"]
    assert (r.values.dt.hour == 0).all()


def test_two_digit_year_convention():
    assert parse_date_series(pd.Series(["69/3/3"], dtype=object), order="ymd").values.iloc[0].year == 2069
    assert parse_date_series(pd.Series(["70/3/3"], dtype=object), order="ymd").values.iloc[0].year == 1970
