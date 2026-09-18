"""前処理。**リークが構造的に起こらないこと**を確かめる。

ここで守りたい性質は 1 つに尽きる。

    前処理の統計量（補完の中央値、スケーラの平均と分散、one-hot のカテゴリ集合、
    winsorize の閾値）は、train からしか学習されない。

手で書くと必ずどこかで破れる性質なので、破れたら止まるようにしてある。
"""
import numpy as np
import pandas as pd
import pytest

from medprep.pipeline import (
    LeakageError,
    Preprocessor,
    RenameFeatures,
    build_preprocessor,
    leak_check,
    prepare,
)
from medprep.schema import Schema
from medprep.splitting import mark_as, split

rng = np.random.default_rng(20260918)


def frame(n=400):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院", "C院"], n, p=[0.6, 0.3, 0.1]),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "Alb": rng.normal(3.6, 0.45, n).round(1),
        "CRP": rng.lognormal(-1, 1.2, n).round(2),
        "転帰": (rng.random(n) < 0.3).astype(int),
    })
    df.loc[df.index[:30], "Alb"] = np.nan
    return df


def schema_of(df):
    return Schema.infer(df, id_col="仮名ID", group="施設",
                        outcome="転帰", task="classification")


def prepared(df=None, **kw):
    df = frame() if df is None else df
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25, seed=0)
    return sp, prepare(sp, sch, **kw)


# ---------------------------------------------------------------- 見張り 1
def test_fitting_on_the_test_set_is_refused():
    """★これがこのモジュールの存在理由である。★"""
    df = frame()
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25)
    with pytest.raises(LeakageError, match="リーク"):
        Preprocessor(sch).fit(sp.test)


def test_fitting_on_unsplit_data_warns():
    """全データに fit してから分割するのは、最も多いリークの形である。"""
    df = frame()
    sch = schema_of(df)
    with pytest.warns(UserWarning, match="分割を経ていない"):
        Preprocessor(sch).fit(df)


def test_marking_by_hand_silences_the_warning():
    df = frame()
    sch = schema_of(df)
    prep = Preprocessor(sch).fit(mark_as(df, "train"))
    assert not any("分割を経ていない" in w for w in prep.record.warnings)


def test_transforming_a_mixture_of_train_and_test_warns():
    df = frame()
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25, seed=0)
    prep = Preprocessor(sch).fit(sp.train)
    mixed = pd.concat([sp.train.iloc[:20], sp.test.iloc[:20]])
    with pytest.warns(UserWarning, match="混ざっている"):
        prep.transform(mixed)


def test_transform_before_fit_is_refused():
    df = frame()
    with pytest.raises(LeakageError, match="まだ fit していない"):
        Preprocessor(schema_of(df)).transform(df)


# ---------------------------------------------------------------- 統計量の由来
def test_the_scaler_is_centred_on_train_not_on_test():
    """★test を transform するときに平均を計算し直していないことを数値で示す。★

    test の平均が train と大きく違えば、変換後の test の平均は 0 から離れる。
    もし 0 になっていたら、それは test で fit し直している証拠である。
    """
    n = 300
    df = pd.DataFrame({"x": np.r_[rng.normal(0, 1, n), rng.normal(50, 1, n)],
                       "転帰": np.r_[np.zeros(n), np.ones(n)].astype(int)})
    sch = Schema.infer(df, outcome="転帰", task="classification")
    train = mark_as(df.iloc[:n], "train")
    test = mark_as(df.iloc[n:], "test")
    # winsorize を切って、スケーラだけの効果を見る
    prep = Preprocessor(sch, {"outlier": {"action": "none"}})
    xtr = prep.fit_transform(train)
    xte = prep.transform(test)
    assert abs(float(xtr["x"].mean())) < 0.05        # train は中心 0
    assert float(xte["x"].mean()) > 5                # test は大きくずれたまま


def test_a_test_set_outside_the_train_range_is_reported_as_a_shift():
    """★train の範囲外の値は winsorize で丸められ、違いが失われる。★

    前処理の不具合ではなく、train と test の分布がずれている合図である。
    黙って潰さず、どの列が何割潰れたかを言う。
    """
    n = 300
    df = pd.DataFrame({"x": np.r_[rng.normal(0, 1, n), rng.normal(50, 1, n)]})
    sch = Schema.infer(df)
    prep = Preprocessor(sch)
    prep.fit(mark_as(df.iloc[:n], "train"))
    prep.transform(mark_as(df.iloc[n:], "test"))
    shift = prep.distribution_shift(df.iloc[n:])
    assert len(shift) and shift.iloc[0]["列"] == "x"
    assert shift.iloc[0]["割合"] == "100.0%"
    assert any("分布がずれている" in w for w in prep.record.warnings)


def test_a_skewed_column_alone_does_not_trigger_the_shift_warning():
    """★train 自身の丸め率と比べる。★

    CRP のようにもともと歪んだ列は train でも 1 割ほど丸められる。
    それを毎回「分布がずれている」と言えば、本当のずれが埋もれる。
    """
    n = 400
    x = rng.lognormal(-1, 1.2, 2 * n)
    df = pd.DataFrame({"CRP": x})
    sch = Schema.infer(df)
    prep = Preprocessor(sch)
    prep.fit(mark_as(df.iloc[:n], "train"))
    prep.transform(mark_as(df.iloc[n:], "test"))
    assert not any("分布がずれている" in w for w in prep.record.warnings)


def test_the_imputed_value_comes_from_train():
    n = 200
    df = pd.DataFrame({"x": np.r_[np.full(n, 10.0), np.full(n, 100.0)]})
    df.loc[df.index[n:n + 10], "x"] = np.nan          # test 側だけ欠損させる
    sch = Schema.infer(df)
    sch.columns["x"].role = "numeric"
    prep = Preprocessor(sch, {"scale": {"method": "none"},
                              "outlier": {"action": "none"}})
    prep.fit(mark_as(df.iloc[:n], "train"))
    out = prep.transform(mark_as(df.iloc[n:], "test"))
    # train の中央値 10 で埋まる。test の中央値 100 ではない。
    assert float(out["x"].iloc[0]) == pytest.approx(10.0)


def test_winsorize_limits_come_from_train():
    n = 300
    df = pd.DataFrame({"x": np.r_[rng.normal(10, 1, n), rng.normal(10, 1, n)]})
    df.loc[df.index[n], "x"] = 1000.0                 # test 側の極端値
    sch = Schema.infer(df)
    prep = Preprocessor(sch, {"scale": {"method": "none"}})
    prep.fit(mark_as(df.iloc[:n], "train"))
    out = prep.transform(mark_as(df.iloc[n:], "test"))
    assert float(out["x"].max()) < 20                 # train の上限で丸められる


# ---------------------------------------------------------------- 列名
def test_output_keeps_readable_column_names():
    """★列名が消えると係数プロットも SHAP も x0, x1, … になる。★"""
    _sp, p = prepared()
    cols = list(p.X_train.columns)
    assert not any(c.startswith("x") and c[1:].isdigit() for c in cols)
    assert "年齢" in cols
    assert any(c.startswith("施設=") for c in cols)
    assert list(p.X_train.columns) == list(p.X_test.columns)


def test_missing_indicators_are_named_in_japanese():
    _sp, p = prepared()
    assert "欠損あり_Alb" in p.X_train.columns
    assert not any(c.startswith("missingindicator_") for c in p.X_train.columns)


def test_rename_features_is_a_plain_transformer():
    r = RenameFeatures().fit(pd.DataFrame({"missingindicator_年齢": [1], "年齢": [2]}))
    assert list(r.get_feature_names_out()) == ["欠損あり_年齢", "年齢"]
    out = r.transform(pd.DataFrame({"missingindicator_年齢": [1], "年齢": [2]}))
    assert list(out.columns) == ["欠損あり_年齢", "年齢"]


def test_reference_levels_are_recorded():
    """基準水準が分からなければ、係数もオッズ比も読めない。"""
    _sp, p = prepared()
    refs = p.preprocessor.reference_levels()
    assert "施設" in refs and "性別" in refs
    assert f"施設={refs['施設']}" not in p.X_train.columns
    assert "基準水準" in p.preprocessor.report()


# ---------------------------------------------------------------- 設計
def test_dates_are_not_fed_in_as_features():
    """暦の効果（診療の変化、測定法の変更）をモデルに学習させない。"""
    df = frame()
    df["観察開始年月日"] = pd.date_range("2015-01-01", periods=len(df), freq="D").astype(str)
    sch = schema_of(df)
    _ct, rec = build_preprocessor(sch)
    dropped = dict(rec["dropped"])
    assert "観察開始年月日" in dropped and "日付" in dropped["観察開始年月日"]


def test_the_outcome_and_the_id_never_enter_the_feature_matrix():
    _sp, p = prepared()
    cols = list(p.X_train.columns)
    assert "転帰" not in cols and "仮名ID" not in cols
    assert p.y_train is not None and len(p.y_train) == len(p.X_train)


def test_ordinal_columns_use_the_order_from_the_schema():
    df = frame(200)
    df["CKD病期"] = rng.choice(["G3a", "G3b", "G4", "G5"], len(df))
    sch = schema_of(df)
    assert sch.columns["CKD病期"].role == "ordinal"
    sp = split(df, sch, test_size=0.25, seed=0)
    p = prepare(sp, sch)
    assert "CKD病期" in p.X_train.columns
    assert p.X_train["CKD病期"].max() <= 3


def test_a_category_seen_only_in_test_does_not_raise():
    df = frame()
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25, seed=0)
    prep = Preprocessor(sch).fit(sp.train)
    test = sp.test.copy()
    test.loc[test.index[0], "施設"] = "Z院"
    out = prep.transform(mark_as(test, "test"))
    assert len(out) == len(test)
    assert any("train に無いカテゴリ" in n for n in prep.record.notes)


def test_missing_required_columns_raise():
    df = frame()
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25, seed=0)
    with pytest.raises(ValueError, match="必要な列がデータに無い"):
        Preprocessor(sch).fit(sp.train.drop(columns=["年齢"]))


# ---------------------------------------------------------------- 検査・出力
def test_leak_check_passes_on_a_correct_flow():
    sp, p = prepared()
    t = leak_check(p.preprocessor, sp.train, sp.test)
    assert set(t["結果"]) == {"OK"}, t.to_string(index=False)


def test_leak_check_catches_overlapping_rows():
    sp, p = prepared()
    bad = pd.concat([sp.test, sp.train.iloc[:5]])
    t = leak_check(p.preprocessor, sp.train, bad)
    assert "★問題" in set(t["結果"])


def test_prepare_warns_when_the_outcome_still_has_missing_values():
    df = frame()
    df.loc[df.index[:10], "転帰"] = np.nan
    sch = Schema.infer(df, id_col="仮名ID", group="施設", outcome="転帰", task="classification")
    sp = split(df, sch, test_size=0.25, seed=0)
    p = prepare(sp, sch)
    assert any("補完してはならない" in n for n in p.notes)


def test_report_and_save(tmp_path):
    _sp, p = prepared()
    txt = p.report()
    assert "前処理" in txt and "fit に使ったデータ" in txt
    path = p.preprocessor.save(tmp_path / "prep.joblib")
    assert path.exists()
    import joblib
    loaded = joblib.load(path)
    assert loaded["features"] == p.preprocessor.feature_names_


def test_a_second_fit_is_recorded():
    df = frame()
    sch = schema_of(df)
    sp = split(df, sch, test_size=0.25, seed=0)
    prep = Preprocessor(sch).fit(sp.train)
    prep.fit(sp.train)
    assert any("2 回目の fit" in n for n in prep.record.notes)
