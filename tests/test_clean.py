"""辞書駆動の掃除。実測の正常値を「補正」してしまわないことを確かめる。"""
import numpy as np
import pandas as pd

from medprep.clean import clean_numeric, derive


def test_detection_limit_becomes_half_lod_with_a_flag():
    df = pd.DataFrame({"C反応性蛋白(CRP)定量": ["<0.1", 0.5, "＜0.1"]})
    out, rep = clean_numeric(df)
    assert out["C反応性蛋白(CRP)定量"].iloc[0] == 0.05
    assert out["C反応性蛋白(CRP)定量__censored_low"].tolist() == [1, 0, 1]


def test_normal_values_are_never_rescaled_as_unit_confusion():
    """CRP の基準範囲上限は 0.14 mg/dL。実測 0.3 を「桁違い」と判定してはならない。"""
    df = pd.DataFrame({"C反応性蛋白(CRP)定量": [0.3] * 50})
    out, rep = clean_numeric(df)
    assert (out["C反応性蛋白(CRP)定量"] == 0.3).all()
    assert not any(a[1] == "単位混在を換算" for a in rep.actions)


def test_only_out_of_range_values_are_unit_converted():
    df = pd.DataFrame({"末梢血｜血色素量(Hb)": [10.8, 108.0, 11.2]})
    out, _ = clean_numeric(df)
    assert out["末梢血｜血色素量(Hb)"].tolist() == [10.8, 10.8, 11.2]


def test_physiologically_impossible_becomes_nan_not_winsorized():
    df = pd.DataFrame({"末梢血｜血色素量(Hb)": [10.8, 0.0], "年齢": [65, 250]})
    out, _ = clean_numeric(df)
    assert np.isnan(out["末梢血｜血色素量(Hb)"].iloc[1])
    assert np.isnan(out["年齢"].iloc[1])


def test_missing_codes_and_text_markers():
    df = pd.DataFrame({"インタクトPTH(iPTH)": [120, 999, 999, 999],
                       "β2マイクログロブリン(β2MG)": [28.0, "未測定", 30.0, 31.0]})
    out, _ = clean_numeric(df)
    assert out["インタクトPTH(iPTH)"].isna().sum() == 3
    assert np.isnan(out["β2マイクログロブリン(β2MG)"].iloc[1])


def test_derive_corrected_ca_and_product():
    df = pd.DataFrame({"カルシウム(Ca)": [8.5, 8.5], "アルブミン(Alb)": [3.0, 4.5],
                       "無機リン(P)": [5.0, 5.0]})
    out, notes = derive(df)
    assert out["補正Ca"].tolist() == [9.5, 8.5]
    assert out["補正Ca×P"].tolist() == [47.5, 42.5]
    assert any("Payne" in n for n in notes)


def test_unmatched_columns_are_reported_not_silently_ignored():
    df = pd.DataFrame({"謎の列": [1, 2, 3], "年齢": [60, 70, 80]})
    _, rep = clean_numeric(df)
    assert "謎の列" in rep.unmatched and "年齢" in rep.matched
