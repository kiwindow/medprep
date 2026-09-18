"""分割。**リークはここで始まる。**

いちばん確かめたいのは「同じ患者が train と test の両方に入らないこと」である。
行単位で分けると、同じ患者の別の測定が両側に入り、モデルは患者を覚えるだけで
高い精度を出す。そして新しい患者ではまったく動かない。
"""
import numpy as np
import pandas as pd
import pytest

from medprep.schema import Schema
from medprep.splitting import (
    TEST,
    TRAIN,
    balance_table,
    cv_splitter,
    fold_summary,
    group_overlap,
    levels_only_in_test,
    mark_as,
    split,
    split_kind,
)

rng = np.random.default_rng(20260918)


def frame(n=400, positive=0.2):
    return pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院", "C院"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "Alb": rng.normal(3.6, 0.45, n).round(1),
        "転帰": (rng.random(n) < positive).astype(int),
    })


def long_frame(n_patients=120, per=3):
    """1 人が複数行ある縦持ちのデータ。"""
    rows = []
    for i in range(n_patients):
        for k in range(per):
            rows.append({"仮名ID": f"P{i:04d}", "回": k,
                         "値": rng.normal(10 + i * 0.05, 0.5),
                         "転帰": int(i % 5 == 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 基本
def test_sizes_and_no_row_appears_twice():
    r = split(frame(), test_size=0.25, seed=0)
    assert r.n_train + r.n_test == 400
    assert not (set(r.train.index) & set(r.test.index))


def test_test_size_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match="test_size は"):
        split(frame(), test_size=1.5)


def test_train_and_test_are_marked():
    """印は pipeline が『test に fit していないか』を見るために使う。"""
    r = split(frame(), test_size=0.25)
    assert split_kind(r.train) == TRAIN and split_kind(r.test) == TEST
    assert split_kind(frame()) is None
    assert split_kind(mark_as(frame(), TRAIN)) == TRAIN
    with pytest.raises(ValueError, match="kind は"):
        mark_as(frame(), "その他")


# ---------------------------------------------------------------- 層化
def test_classification_is_stratified_by_the_outcome():
    df = frame(positive=0.1)
    sch = Schema.infer(df, id_col="仮名ID", outcome="転帰", task="classification")
    r = split(df, sch, test_size=0.25, seed=0)
    assert r.stratify_by == "転帰"
    assert r.train["転帰"].mean() == pytest.approx(r.test["転帰"].mean(), abs=0.02)
    assert any("層化" in n for n in r.notes)


def test_survival_is_stratified_by_the_event():
    df = frame()
    df["duration"] = rng.exponential(3, len(df))
    sch = Schema.infer(df, id_col="仮名ID", survival=("duration", "転帰"))
    r = split(df, sch, test_size=0.25, seed=0)
    assert r.stratify_by == "転帰"
    assert any("イベント" in n and "層化" in n for n in r.notes)


def test_regression_says_why_it_did_not_stratify():
    df = frame()
    df["eGFR"] = rng.normal(30, 10, len(df))
    sch = Schema.infer(df, id_col="仮名ID", outcome="eGFR", task="regression")
    r = split(df, sch, test_size=0.25)
    assert r.stratify_by is None
    assert any("回帰なので層化していない" in n for n in r.notes)


def test_stratify_can_be_named_or_turned_off():
    df = frame()
    a = split(df, stratify="施設", test_size=0.25, seed=0)
    assert a.stratify_by == "施設"
    b = split(df, stratify=False, test_size=0.25, seed=0)
    assert b.stratify_by is None
    with pytest.raises(ValueError, match="層化に使う列"):
        split(df, stratify="無い列")


# ---------------------------------------------------------------- グループ
def test_the_same_patient_never_lands_on_both_sides():
    """★これが分割でいちばん大事な性質である。★"""
    df = long_frame()
    sch = Schema.infer(df, id_col="仮名ID")
    r = split(df, sch, test_size=0.25, seed=0)
    assert r.group_by == "仮名ID"
    assert group_overlap(r.train, r.test, "仮名ID") == []


def test_duplicated_ids_are_detected_without_being_told():
    df = long_frame()
    sch = Schema.infer(df, id_col="仮名ID")
    r = split(df, sch, test_size=0.25)
    assert any("重複がある" in w and "両方に入る" in w for w in r.warnings)


def test_group_splitting_can_be_turned_off_explicitly():
    df = long_frame()
    sch = Schema.infer(df, id_col="仮名ID")
    r = split(df, sch, test_size=0.25, group=False)
    assert r.group_by is None
    assert group_overlap(r.train, r.test, "仮名ID") != []      # 行単位だと混ざる


def test_group_column_can_be_named():
    df = frame()
    r = split(df, test_size=0.25, group="施設")
    assert r.group_by == "施設"
    assert group_overlap(r.train, r.test, "施設") == []
    with pytest.raises(ValueError, match="グループ列"):
        split(df, group="無い列")


# ---------------------------------------------------------------- 時間順
def test_time_order_puts_the_past_in_train_and_the_future_in_test():
    df = frame(200)
    df["検体採取日"] = pd.date_range("2015-01-01", periods=200, freq="10D").astype(str)
    r = split(df, test_size=0.25, time_order="検体採取日")
    assert r.strategy == "時間順"
    assert pd.to_datetime(r.train["検体採取日"]).max() < pd.to_datetime(r.test["検体採取日"]).min()
    assert any("楽観的" in n for n in r.notes)


def test_time_order_refuses_an_ambiguous_date_column():
    df = frame(30)
    df["日付"] = ["3/4/2013", "5/6/2014", "7/8/2015"] * 10
    with pytest.raises(ValueError, match="並びが確定できない"):
        split(df, test_size=0.25, time_order="日付")


def test_time_order_needs_the_column():
    with pytest.raises(ValueError, match="時間順に使う列"):
        split(frame(), time_order="無い列")


# ---------------------------------------------------------------- バランス
def test_balance_table_reports_smd_per_column():
    r = split(frame(), test_size=0.25, seed=0)
    assert {"列", "train", "test", "SMD", "判定"} <= set(r.balance.columns)
    assert r.balance["SMD"].notna().any()


def test_an_unbalanced_split_is_warned_about():
    df = frame(120)
    df["偏る"] = np.r_[np.zeros(60), np.ones(60)]
    # わざと偏る分割を作る（前半 90 行を train に）
    tr, te = df.iloc[:90], df.iloc[90:]
    b = balance_table(tr, te, columns=["偏る"])
    assert abs(float(b.iloc[0]["SMD"])) >= 0.2
    assert b.iloc[0]["判定"] == "偏り"


def test_a_tiny_test_set_is_warned_about():
    r = split(frame(60), test_size=0.2, seed=0)
    assert any("test が" in w and "交差検証" in w for w in r.warnings)


def test_few_events_in_test_is_warned_about():
    df = frame(200, positive=0.03)
    df["duration"] = rng.exponential(3, len(df))
    sch = Schema.infer(df, id_col="仮名ID", survival=("duration", "転帰"))
    r = split(df, sch, test_size=0.2, seed=0)
    assert any("イベント" in w and "不安定" in w for w in r.warnings)


# ---------------------------------------------------------------- 交差検証
def test_cv_splitter_picks_stratified_when_it_can():
    df = frame()
    sch = Schema.infer(df, id_col="仮名ID", outcome="転帰", task="classification")
    sp, y, g, notes = cv_splitter(df, sch, n_splits=5)
    assert type(sp).__name__ == "StratifiedKFold" and g is None
    assert any("fold ごとに train で fit" in n for n in notes)


def test_cv_splitter_uses_groups_when_ids_repeat():
    df = long_frame()
    sch = Schema.infer(df, id_col="仮名ID")
    sp, y, g, notes = cv_splitter(df, sch, n_splits=4)
    assert "Group" in type(sp).__name__ and g is not None
    s = fold_summary(df, sp, y, g)
    assert (s["両側に出るグループ"] == 0).all()


def test_fold_summary_shows_the_outcome_balance():
    df = frame()
    sch = Schema.infer(df, id_col="仮名ID", outcome="転帰", task="classification")
    sp, y, g, _ = cv_splitter(df, sch, n_splits=5)
    s = fold_summary(df, sp, y, g, outcome="転帰")
    assert len(s) == 5 and "test の陽性率" in s.columns


# ---------------------------------------------------------------- 未知の水準
def test_levels_only_in_test_are_listed():
    df = frame(100)
    tr = df.iloc[:80].copy()
    te = df.iloc[80:].copy()
    te.loc[te.index[0], "施設"] = "Z院"
    out = levels_only_in_test(tr, te, columns=["施設"])
    assert len(out) and "Z院" in out.iloc[0]["test にしか無い水準"]


def test_report_is_printable():
    assert "分割" in split(frame(), test_size=0.25).report()
