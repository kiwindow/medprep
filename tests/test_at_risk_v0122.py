"""0.12.2 — KM 曲線の at-risk 表が lifelines・NumPy の版に関係なく出る。

lifelines 0.30.2 以前の ``add_at_risk_counts`` は NumPy 2.4 以降で
``TypeError: only 0-dimensional arrays can be converted to Python scalars`` になり、
``Survival.km`` が図を作れずに止まっていた（SetupLab の ~/lab: lifelines 0.30.0・NumPy 2.4.6）。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

import medprep as mp  # noqa: E402
from medprep._at_risk import add_at_risk_counts  # noqa: E402

rng = np.random.default_rng(122)


def _data(n=240):
    return pd.DataFrame({
        "duration": rng.exponential(3.0, n).round(2) + 0.01,
        "event": rng.integers(0, 2, n),
        "群": rng.choice(["A", "B", "C"], n),
    })


def _labels(ax):
    """at-risk 表は twiny で足した 2 本目の軸の目盛りの文字に入る。"""
    extra = [a for a in ax.figure.axes if a is not ax]
    assert extra, "at-risk 表の軸が無い"
    return [t.get_text() for t in extra[-1].get_xticklabels()]


def test_km_with_at_risk_table_does_not_stop():
    surv = mp.Survival(_data(), unit="years")
    for by in (None, "群"):
        km = surv.km(by=by)
        assert len(km.table) == (1 if by is None else 3)
        plt.close("all")


def test_all_three_rows_can_be_shown():
    surv = mp.Survival(_data(), unit="years")
    km = surv.km(by="群", at_risk_rows=("At risk", "Censored", "Events"))
    ax = km.figure.axes[0]
    first = _labels(ax)[0]
    for row in ("At risk", "Censored", "Events"):
        assert row in first
    plt.close("all")


def test_counts_at_time_zero_are_everyone():
    d = _data()
    fig, ax = plt.subplots()
    k = KaplanMeierFitter(label="全体").fit(d["duration"], d["event"])
    k.plot_survival_function(ax=ax)
    ax.set_xlim(left=0)
    add_at_risk_counts(k, ax=ax, rows_to_show=["At risk"])
    first = _labels(ax)[0]
    assert first.split()[-1] == str(len(d))       # 0 の時点では全員が at risk
    plt.close(fig)


def test_a_wrong_row_name_is_rejected_with_a_message():
    d = _data()
    fig, ax = plt.subplots()
    k = KaplanMeierFitter().fit(d["duration"], d["event"])
    k.plot_survival_function(ax=ax)
    with pytest.raises(ValueError, match="at_risk_rows"):
        add_at_risk_counts(k, ax=ax, rows_to_show=["at risk"])
    plt.close(fig)


def _lifelines_version():
    import lifelines
    return tuple(int(x) for x in lifelines.__version__.split(".")[:3])


@pytest.mark.skipif(_lifelines_version() < (0, 30, 3),
                    reason="lifelines 0.30.3 以降でだけ元の関数と比べられる")
@pytest.mark.parametrize("rows", [["At risk"], ["At risk", "Censored", "Events"]])
def test_same_table_as_lifelines_0_30_3(rows):
    from lifelines.plotting import add_at_risk_counts as original
    d = _data()
    out = []
    for fn in (original, add_at_risk_counts):
        fig, ax = plt.subplots(figsize=(8, 5))
        ks = []
        for g in ("A", "B"):
            m = d["群"] == g
            k = KaplanMeierFitter(label=g).fit(d.loc[m, "duration"], d.loc[m, "event"])
            k.plot_survival_function(ax=ax)
            ks.append(k)
        ax.set_xlim(left=0)
        fn(*ks, ax=ax, rows_to_show=rows)
        out.append(_labels(ax))
        plt.close(fig)
    assert out[0] == out[1]
