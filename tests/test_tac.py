"""TAC-BUN。取り違えを数値だけでは見抜けないため、検出できることを確かめる。"""
import numpy as np
import pytest

from medprep.tac import (
    bun_to_urea_mg_dl,
    bun_to_urea_mmol_l,
    interdialytic_hours,
    tac_bun_simple,
    tac_linear,
    tac_trapezoid,
    urea_to_bun_mg_dl,
)


def test_simple_matches_worked_example():
    """月曜透析後 20、水曜透析前 60 → 40 mg/dL。"""
    r = tac_bun_simple(20, 60)
    assert float(np.atleast_1d(r.value)[0]) == pytest.approx(40.0)
    assert r.kind == "simple" and r.solute == "BUN" and r.unit == "mg/dL"
    assert not r.warnings


@pytest.mark.parametrize("fn,want", [
    (bun_to_urea_mg_dl, 40 * 60 / 28),
    (bun_to_urea_mmol_l, 40 / 2.8),
])
def test_urea_conversion_matches_worked_example(fn, want):
    assert float(np.atleast_1d(fn(40))[0] if not np.isscalar(fn(40)) else fn(40)) == pytest.approx(want)


def test_urea_roundtrip():
    assert float(urea_to_bun_mg_dl(bun_to_urea_mg_dl(40))) == pytest.approx(40.0)


def test_result_conversion_methods():
    r = tac_bun_simple(20, 60)
    assert float(np.atleast_1d(r.to_urea_mg_dl())[0]) == pytest.approx(85.714285714)
    assert float(np.atleast_1d(r.to_urea_mmol_l())[0]) == pytest.approx(14.285714286)


def test_confusion_with_same_session_pre_post_is_detected():
    """同じ透析回の (前60, 後20) を渡すと 40 になり、正解と同じ数字が出てしまう。
    数字では気づけないので、大小関係で捕まえること。"""
    bad = tac_bun_simple(60, 20)
    assert float(np.atleast_1d(bad.value)[0]) == pytest.approx(40.0)   # 数字は正しく見える
    assert bad.warnings and "逆転" in bad.warnings[0]


def test_missing_is_not_zero_filled():
    r = tac_bun_simple([20, np.nan, -5], [60, 55, 50])
    assert float(r.value[0]) == pytest.approx(40.0)
    assert np.isnan(r.value[1]), "欠測を 0 で埋めない"
    assert np.isnan(r.value[2]), "負値は入力エラー"
    assert any("負値" in w for w in r.warnings)


def test_interdialytic_hours_is_interval_minus_td():
    ti = interdialytic_hours([48, 48, 72], [4, 4, 4])
    assert list(ti) == [44.0, 44.0, 68.0], "Ti を開始間隔そのものにしない"
    assert float(np.sum(ti) + 12) == pytest.approx(168.0)


def _week():
    pre = np.array([60.0, 55.0, 58.0])
    post = np.array([20.0, 18.0, 19.0])
    nxt = np.array([55.0, 58.0, 60.0])
    td = np.array([4.0, 4.0, 4.0])
    ti = interdialytic_hours([48, 48, 72], td)
    return pre, post, nxt, td, ti


def test_linear_weighting_and_week_total():
    pre, post, nxt, td, ti = _week()
    r = tac_linear(pre, post, nxt, td, ti)
    manual = ((pre + post) / 2 * td + (post + nxt) / 2 * ti).sum() / (td + ti).sum()
    assert float(np.atleast_1d(r.value)[0]) == pytest.approx(float(manual))
    assert r.provenance["合計時間[h]"] == pytest.approx(168.0)
    assert r.kind == "auc" and not r.warnings


def test_wrong_ti_is_caught_by_the_168_hour_check():
    pre, post, nxt, td, _ = _week()
    r = tac_linear(pre, post, nxt, td, np.array([48.0, 48.0, 72.0]))
    assert r.provenance["合計時間[h]"] == pytest.approx(180.0)
    assert any("168" in w for w in r.warnings)


def test_trapezoid_agrees_with_linear_on_the_same_polyline():
    pre, post, nxt, td, ti = _week()
    t, c, cur = [0.0], [pre[0]], 0.0
    for j in range(3):
        cur += td[j]
        t.append(cur)
        c.append(post[j])
        cur += ti[j]
        t.append(cur)
        c.append(nxt[j])
    tr = tac_trapezoid(t, c, expect_hours=168)
    lin = tac_linear(pre, post, nxt, td, ti)
    assert float(np.atleast_1d(tr.value)[0]) == pytest.approx(float(np.atleast_1d(lin.value)[0]))


def test_time_average_differs_from_naive_mean():
    """検査値を足して回数で割る方法は、測定間隔が異なれば時間平均にならない。"""
    pre, post, nxt, td, ti = _week()
    t, c, cur = [0.0], [pre[0]], 0.0
    for j in range(3):
        cur += td[j]
        t.append(cur)
        c.append(post[j])
        cur += ti[j]
        t.append(cur)
        c.append(nxt[j])
    tr = float(np.atleast_1d(tac_trapezoid(t, c).value)[0])
    assert abs(tr - float(np.mean(c))) > 2.0


def test_trapezoid_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="昇順"):
        tac_trapezoid([0, 10, 5], [1, 2, 3])


def test_trapezoid_refuses_missing_rather_than_filling_zero():
    r = tac_trapezoid([0, 10], [50, np.nan])
    assert np.isnan(float(np.atleast_1d(r.value)[0]))
    assert any("0 で埋めず" in w for w in r.warnings)
