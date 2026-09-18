"""演習用の合成データ。

**教材の要は、ここに仕込んだ「汚れ」である。**
きれいなデータで前処理を教えても、受講者は自分のデータに戻った瞬間に立ち往生する。
汚れが 1 つでも消えていれば、その回の演習は成立しない。
"""
import numpy as np
import pandas as pd

from medprep import demo


def test_the_default_cohort_is_reproducible():
    a, b = demo.dialysis_cohort(), demo.dialysis_cohort()
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (600, 20)


def test_a_different_seed_gives_a_different_cohort():
    a = demo.dialysis_cohort(seed=1)
    b = demo.dialysis_cohort(seed=2)
    assert not a["年齢"].equals(b["年齢"])


def test_it_scales_to_other_sizes():
    df = demo.dialysis_cohort(n=120, seed=3)
    assert len(df) == 120 and df["仮名ID"].nunique() == 120


# ---------------------------------------------------------------- 汚れ
def test_missing_codes_are_present():
    df = demo.dialysis_cohort()
    assert (df["インタクトPTH(iPTH)"] == 999).sum() > 0
    assert (df["β2マイクログロブリン(β2MG)"].astype(str) == "未測定").sum() > 0


def test_the_limit_of_detection_notation_is_present():
    df = demo.dialysis_cohort()
    assert (df["C反応性蛋白(CRP)定量"].astype(str) == "<0.1").sum() > 0


def test_physiologically_impossible_values_are_present():
    """★Hb 0 g/dL は外れ値ではなく入力ミスである。winsorize してはならない。★"""
    df = demo.dialysis_cohort()
    hb = pd.to_numeric(df["末梢血｜血色素量(Hb)"], errors="coerce")
    assert (hb == 0).sum() > 0
    assert (df["年齢"] == 250).sum() > 0


def test_mixed_units_are_present():
    """Hb の一部が g/L で入っている（10 倍）。"""
    hb = pd.to_numeric(demo.dialysis_cohort()["末梢血｜血色素量(Hb)"], errors="coerce")
    assert (hb > 50).sum() > 0


def test_the_duplicate_column_is_perfectly_correlated():
    df = demo.dialysis_cohort()
    assert df.groupby("施設")["施設コード"].nunique().max() == 1


def test_the_free_text_column_looks_like_an_identifier():
    df = demo.dialysis_cohort()
    assert df["備考"].nunique() == len(df)


def test_dates_come_in_many_notations():
    """★和暦・Excel シリアル値・全角・時刻付きが混ざっている。★"""
    s = demo.dialysis_cohort()["観察開始年月日"].astype(str)
    assert s.str.match(r"^[HR]\d").any()                 # 和暦
    assert s.str.match(r"^\d{5}$").any()                 # Excel シリアル値
    assert s.str.contains("０|１|／").any()               # 全角
    assert s.str.contains(":").any()                     # 時刻付き


def test_the_survival_dates_contain_the_four_contradictions():
    df = demo.dialysis_cohort()
    ev = df["event発生年月日"].astype(str).str.strip()
    cn = df["観察打ち切り年月日"].astype(str).str.strip()
    assert ((ev != "") & (cn != "")).sum() > 0           # 両方入っている
    assert ((ev == "") & (cn == "")).sum() > 0           # どちらも空
    assert (df["観察開始年月日"].astype(str).str.strip() == "").sum() > 0


def test_the_alp_method_change_makes_a_real_step():
    """★2020-04 の JSCC → IFCC。値がおよそ 1/3 になる。★

    病態ではなく測定法の段差である。audit がこれを検出できなければ、
    受講者は「2020 年以降 ALP が下がった」と誤って解釈する。
    """
    df = demo.dialysis_cohort()
    draw = pd.to_datetime(df["検体採取日"])
    alp = pd.to_numeric(df["アルカリフォスファターゼ(ALP)"])
    before = alp[draw < demo.ALP_METHOD_CHANGE].median()
    after = alp[draw >= demo.ALP_METHOD_CHANGE].median()
    assert after / before < 0.5
    assert (draw >= demo.ALP_METHOD_CHANGE).sum() >= 20   # 後期の例数が足りている


def test_ordinary_missing_values_are_present():
    df = demo.dialysis_cohort()
    assert df["アルブミン(Alb)"].isna().sum() > 0
    assert df["透析歴_月"].isna().sum() > 0


# ---------------------------------------------------------------- 真値
def test_the_true_coefficients_are_published():
    """受講者が Cox の推定値と突き合わせられるように、真の係数を出しておく。"""
    assert set(demo.TRUE_COEFFICIENTS) == {"年齢", "Alb", "Hb", "logCRP", "糖尿病", "vintage"}
    assert demo.TRUE_COEFFICIENTS["Alb"] < 0 < demo.TRUE_COEFFICIENTS["年齢"]


def test_save_writes_an_excel_file(tmp_path):
    p = demo.save(tmp_path / "c.xlsx", n=50)
    assert len(pd.read_excel(p)) == 50


def test_the_event_count_is_recorded():
    df = demo.dialysis_cohort()
    assert 0 < df.attrs["medprep_demo"]["真のイベント数"] < len(df)
    assert np.isfinite(df.attrs["medprep_demo"]["seed"])
