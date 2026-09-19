"""外れ値。**削除してよい値と、削除してはいけない値の区別**を確かめる。"""
import numpy as np
import pandas as pd
import pytest

from medprep.outliers import (
    DROP,
    FLAG,
    NAN,
    NanSafeWinsorizer,
    apply_action,
    detect,
    iqr_limits,
    isolation_forest_outliers,
    limits,
    mad_limits,
    mahalanobis_outliers,
    quantile_limits,
)

rng = np.random.default_rng(20260918)


# ---------------------------------------------------------------- 閾値
def test_iqr_limits_match_the_hand_calculation():
    x = list(range(1, 101))            # Q1=25.75, Q3=75.25, IQR=49.5
    lo, hi = iqr_limits(x, fold=1.5)
    assert lo == pytest.approx(25.75 - 1.5 * 49.5)
    assert hi == pytest.approx(75.25 + 1.5 * 49.5)


def test_mad_limits_move_less_than_iqr_when_the_data_is_contaminated():
    """汚染が入ったときに閾値がどれだけ動くかで比べる。

    四分位点は汚染が 2 割を超えると引きずられるが、中央絶対偏差は動きにくい。
    """
    clean = rng.normal(10, 1, 200)
    dirty = np.r_[clean, rng.normal(1000, 10, 70)]      # 約 26% の汚染
    shift_iqr = abs(iqr_limits(dirty)[1] - iqr_limits(clean)[1])
    shift_mad = abs(mad_limits(dirty)[1] - mad_limits(clean)[1])
    assert shift_mad < shift_iqr


def test_mad_returns_no_limits_when_the_mad_is_zero():
    """★半数以上が同じ値だと MAD が 0 になる。★

    検出限界未満が多い CRP などで実際に起きる。0 で割ると全例が外れ値になるので、
    閾値を作らないのが正しい。
    """
    x = [0.05] * 60 + [0.1, 0.2, 0.3]
    lo, hi = mad_limits(x)
    assert lo == float("-inf") and hi == float("inf")


def test_quantile_limits_and_unknown_method():
    lo, hi = quantile_limits(list(range(100)), 0.05, 0.95)
    assert lo == pytest.approx(4.95) and hi == pytest.approx(94.05)
    with pytest.raises(ValueError, match="method は"):
        limits([1, 2, 3, 4], "存在しない方法")


def test_too_few_values_give_no_limits():
    assert iqr_limits([1, 2]) == (float("-inf"), float("inf"))


# ---------------------------------------------------------------- winsorize
def frame(n=200):
    return pd.DataFrame({
        "a": np.r_[rng.normal(10, 2, n - 4), [60, 65, -30, -35]],
        "b": rng.normal(5, 1, n),
    })


def test_winsorizer_keeps_nan_as_nan():
    """★feature-engine の Winsorizer は NaN で止まる。だから自作した。★

    「あり得ない値を NaN に → winsorize → 補完」という順序を曲げないために、
    NaN を素通しできなければならない。
    """
    df = frame()
    df.loc[:9, "a"] = np.nan
    out = NanSafeWinsorizer().fit_transform(df)
    assert out["a"].isna().sum() == 10
    assert out["b"].isna().sum() == 0


def test_winsorizer_clips_to_the_learned_limits():
    df = frame()
    w = NanSafeWinsorizer(fold=1.5).fit(df)
    out = w.transform(df)
    lo, hi = w.limits_["a"]
    assert out["a"].max() == pytest.approx(hi)
    assert out["a"].min() == pytest.approx(lo)


def test_limits_are_learned_on_fit_and_never_relearned():
    """★これがリーク防止の要である。★

    test を transform するときに test の分布から閾値を作り直してはならない。
    """
    train = frame(300)
    w = NanSafeWinsorizer().fit(train)
    learned = dict(w.limits_)
    test = pd.DataFrame({"a": rng.normal(1000, 5, 50), "b": rng.normal(5, 1, 50)})
    out = w.transform(test)
    assert w.limits_ == learned                     # 閾値は変わらない
    assert out["a"].max() == pytest.approx(learned["a"][1])   # train の上限に丸められる


def test_explicit_bounds_override_the_learned_ones():
    df = frame()
    w = NanSafeWinsorizer(bounds={"a": (0.0, 20.0)}).fit(df)
    out = w.transform(df)
    assert out["a"].min() >= 0.0 and out["a"].max() <= 20.0


def test_winsorizer_keeps_column_names():
    df = frame()
    w = NanSafeWinsorizer().fit(df)
    assert list(w.get_feature_names_out()) == ["a", "b"]
    assert list(w.transform(df).columns) == ["a", "b"]
    assert set(w.limits_frame()["列"]) == {"a", "b"}


# ---------------------------------------------------------------- 検出
def test_detect_does_not_modify_the_data():
    df = frame()
    before = df.copy(deep=True)
    detect(df, columns=["a", "b"])
    pd.testing.assert_frame_equal(df, before)


def test_detect_counts_the_outliers_per_column():
    df = frame()
    r = detect(df, columns=["a", "b"])
    assert set(r.table["列"]) == {"a", "b"}
    assert int(r.table.set_index("列").loc["a", "外れ値"]) >= 4
    assert r.n_flagged_rows >= 4


def test_detect_always_says_that_impossible_values_are_handled_elsewhere():
    """Hb 0 は外れ値ではなく入力ミス。ここには出てこないことを明示する。"""
    r = detect(frame(), columns=["a"])
    assert any("あり得ない値" in n and "clean_numeric" in n for n in r.notes)


def test_a_skewed_column_is_called_out_rather_than_trimmed():
    df = pd.DataFrame({"CRP": rng.lognormal(-1, 1.5, 400)})
    r = detect(df, columns=["CRP"])
    assert any("分布が歪んでいる" in n for n in r.notes)


def test_group_wise_thresholds_find_what_the_whole_data_threshold_misses():
    """施設ごとに水準が違う指標を、全体の閾値で見ない。

    A 院は 10 前後、B 院は 100 前後という指標では、全体の四分位範囲が
    **群間の差そのもの**になって閾値が広がりすぎ、群内の外れ値を全部見逃す。
    """
    df = pd.DataFrame({
        "施設": ["A"] * 200 + ["B"] * 200,
        "値": np.r_[rng.normal(10, 1, 200), rng.normal(100, 10, 200)],
    })
    df.loc[0, "値"] = 30.0            # A 院の中では明らかに外れている
    whole = detect(df, columns=["値"])
    per = detect(df, columns=["値"], by="施設")
    assert int(whole.table.iloc[0]["外れ値"]) == 0        # 全体では見逃す
    assert int(per.table.iloc[0]["外れ値"]) >= 1          # 群ごとなら見つかる
    assert per.flags["値"].iloc[0]
    assert any("群ごと" in n for n in per.notes)


def test_examples_use_the_id_column():
    df = frame(100)
    df["仮名ID"] = [f"P{i:03d}" for i in range(len(df))]
    r = detect(df, columns=["a"], id_col="仮名ID")
    assert "P" in r.table.iloc[0]["該当例"]


# ---------------------------------------------------------------- 適用
def test_flag_adds_columns_without_changing_values():
    df = frame()
    r = detect(df, columns=["a"])
    out, notes = apply_action(df, r, FLAG)
    assert "a__outlier" in out.columns
    pd.testing.assert_series_equal(out["a"], df["a"])


def test_nan_action_sends_them_to_imputation():
    df = frame()
    r = detect(df, columns=["a"])
    out, _ = apply_action(df, r, NAN)
    assert out["a"].isna().sum() == int(r.table.iloc[0]["外れ値"])


def test_drop_says_loudly_what_it_did():
    """★医学では外れ値こそが重要な症例でありうる。★"""
    df = frame()
    r = detect(df, columns=["a"])
    out, notes = apply_action(df, r, DROP)
    assert len(out) < len(df)
    assert any("重要な症例" in n and "補足に残す" in n for n in notes)


def test_winsorize_here_warns_that_it_is_not_for_training():
    df = frame()
    r = detect(df, columns=["a"])
    _out, notes = apply_action(df, r, "winsorize")
    assert any("リークになる" in n for n in notes)


def test_unknown_action_raises():
    df = frame()
    r = detect(df, columns=["a"])
    with pytest.raises(ValueError, match="action は"):
        apply_action(df, r, "存在しない")


# ---------------------------------------------------------------- 多変量
def test_mahalanobis_finds_an_impossible_combination():
    """身長 190cm・体重 40kg は、どちらも単独では正常範囲にある。"""
    n = 300
    h = rng.normal(165, 8, n)
    w = (h - 100) * 0.9 + rng.normal(0, 4, n)
    df = pd.DataFrame({"身長": h, "体重": w})
    df.loc[0] = [190.0, 40.0]
    out = mahalanobis_outliers(df, ["身長", "体重"])
    assert len(out) and 0 in set(out["行"])


def test_isolation_forest_returns_a_ranking():
    df = pd.DataFrame({"x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300)})
    df.loc[0] = [10.0, -10.0]
    out = isolation_forest_outliers(df, ["x", "y"], contamination=0.02)
    assert len(out) and out.iloc[0]["行"] == 0


def test_multivariate_helpers_return_empty_when_they_cannot_run():
    assert not len(mahalanobis_outliers(pd.DataFrame({"x": [1, 2, 3]})))
    assert not len(isolation_forest_outliers(pd.DataFrame({"x": [1.0, 2.0]})))


def test_report_is_printable():
    r = detect(frame(), columns=["a", "b"])
    assert "外れ値の検出" in r.report()


def test_iqr_of_zero_does_not_collapse_the_column():
    """★IQR が 0 の列を winsorize すると、列が 1 つの値に潰れる。★

    値が数種類しかない列（透析時間は 4.0 時間が 6 割）では Q1 = Q3 になる。
    そのまま閾値にすると上下限が同じ値になり、分散 0 の列ができる。
    スケーリング後は全例きっかり 0 になり、**例外は出ないまま、その変数だけが
    モデルから消える。** 実際に合成データの透析時間がこれで消えていた。
    """
    from medprep.outliers import NanSafeWinsorizer, iqr_limits

    x = pd.Series([4.0] * 500 + [3.5] * 50 + [4.5] * 40 + [5.0] * 10)
    assert x.quantile(0.25) == x.quantile(0.75)          # IQR = 0 の状況
    lo, hi = iqr_limits(x)
    assert lo == float("-inf") and hi == float("inf")    # 閾値を作らない

    out = NanSafeWinsorizer().fit_transform(pd.DataFrame({"透析時間": x}))
    assert out["透析時間"].nunique() == 4                 # 潰れていない
    assert float(out["透析時間"].std()) > 0
