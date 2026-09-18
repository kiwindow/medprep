"""辞書駆動の掃除。実測の正常値を「補正」してしまわないことを確かめる。"""
import numpy as np
import pandas as pd
import pytest

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
    """iCa は **Payne 式**。★補正するのは Alb < 4.0 のときだけ★

    Alb ≥ 4.0 では補正せず実測 Ca をそのまま使う（日本の運用）。
    無条件に `Ca + (4 − Alb)` とすると、Alb 4.5 の症例で実測より 0.5 低くなる。
    列名は `補正Ca` → `iCa(mg/dL)`（単位を列名に書く）。
    """
    df = pd.DataFrame({"カルシウム(Ca)": [8.5, 8.5], "アルブミン(Alb)": [3.0, 4.5],
                       "無機リン(P)": [5.0, 5.0]})
    out, notes = derive(df)
    assert out["iCa(mg/dL)"].tolist() == [9.5, 8.5]
    assert out["iCa×P"].tolist() == [47.5, 42.5]
    assert any("Payne" in n for n in notes)


def test_unmatched_columns_are_reported_not_silently_ignored():
    df = pd.DataFrame({"謎の列": [1, 2, 3], "年齢": [60, 70, 80]})
    _, rep = clean_numeric(df)
    assert "謎の列" in rep.unmatched and "年齢" in rep.matched


# ---------------------------------------------------------------- 派生指標
def test_corrected_calcium_is_not_computed_when_albumin_is_missing():
    """★Alb が欠測なら補正Ca は「算出不能」であって Ca ではない。★

    `np.where(np.nan < 4.0, ca + (4.0 - alb), ca)` は NaN の比較が False に落ちるため、
    **補正されていない Ca が iCa として黙って混ざる**。
    合成データでは 49 例がこれに当たっていた。管理目標の達成率も Cox の係数も、
    その分だけ静かにずれる。missing.mcar_signals が
    「Alb の欠測と iCa が関連している」として検出した。
    """
    df = pd.DataFrame({
        "カルシウム(Ca)": [9.0, 8.0, 9.0],
        "アルブミン(Alb)": [4.2, 3.0, np.nan],
    })
    out, notes = derive(df)
    assert out["iCa(mg/dL)"].iloc[0] == pytest.approx(9.0)   # Alb≥4 → 補正しない
    assert out["iCa(mg/dL)"].iloc[1] == pytest.approx(9.0)   # 8.0 + (4.0-3.0)
    assert pd.isna(out["iCa(mg/dL)"].iloc[2])                # ★Ca の 9.0 を返さない★
    assert any("算出不能" in n for n in notes)


def test_corrected_calcium_times_p_inherits_the_missingness():
    df = pd.DataFrame({
        "カルシウム(Ca)": [9.0, 9.0],
        "アルブミン(Alb)": [3.0, np.nan],
        "無機リン(P)": [5.0, 5.0],
    })
    out, _ = derive(df)
    assert out["iCa×P"].iloc[0] == pytest.approx(50.0)
    assert pd.isna(out["iCa×P"].iloc[1])


# ---------------------------------------------- 透析の派生指標（式と単位を確かめる）
def _hd_frame(n=6):
    return pd.DataFrame({
        "透析開始時刻": ["08:30", "9:00", "8時45分", "１３：００", "17:30", None],
        "透析終了時刻": ["12:30", "13:00", "13:15", "17:00", "21:30", "12:00"],
        "透析前体重": [60.0, 55.0, 70.0, 48.0, 62.0, 50.0],
        "透析後体重": [58.0, 52.5, 67.0, 46.0, 59.0, 48.0],
        "透析前BUN": [60.0, 70.0, 50.0, 80.0, 65.0, 60.0],
        "透析後BUN": [18.0, 21.0, 20.0, 24.0, 90.0, 18.0],
        "血清鉄(Fe)": [60.0, 50.0, 80.0, 40.0, 70.0, 55.0],
        "総鉄結合能(TIBC)": [250.0, 200.0, 320.0, 160.0, 280.0, 220.0],
        "カルシウム(Ca)": [8.8, 9.0, 9.2, 8.5, 9.1, 8.9],
        "アルブミン(Alb)": [3.5, 4.2, 3.0, None, 3.8, 3.6],
    })[:n]


def test_session_hours_reads_the_written_forms_people_actually_use():
    """`9:00`・`8時45分`・全角コロンが混ざるのが実務である。**読めないものは NaN。**"""
    from medprep.clean import session_hours, to_hours
    assert to_hours("08:30") == pytest.approx(8.5)
    assert to_hours("8時45分") == pytest.approx(8.75)
    assert to_hours("１３：００") == pytest.approx(13.0)
    assert to_hours(0.5) == pytest.approx(12.0)          # Excel のシリアル小数
    assert np.isnan(to_hours("未実施"))
    h = session_hours(pd.Series(["22:00"]), pd.Series(["02:00"]))   # 日またぎ
    assert float(h.iloc[0]) == pytest.approx(4.0)


def test_dialysis_indices_use_the_stated_formulas():
    from medprep.clean import derive_dialysis
    df = _hd_frame()
    out, notes = derive_dialysis(df)

    assert float(out.loc[0, "透析時間(hr)"]) == pytest.approx(4.0)
    assert float(out.loc[0, "除水量(kg)"]) == pytest.approx(2.0)
    assert float(out.loc[0, "URR(%)"]) == pytest.approx((60 - 18) / 60 * 100)
    assert float(out.loc[0, "TSAT(%)"]) == pytest.approx(60 / 250 * 100)
    assert float(out.loc[0, "iCa(mg/dL)"]) == pytest.approx(8.8 + (4 - 3.5))  # Alb 3.5

    # spKt/V = -ln(R - 0.008t) + (4 - 3.5R)·UF/W   ★UF は L（= kg）★
    r = 18 / 60
    want = -np.log(r - 0.008 * 4.0) + (4 - 3.5 * r) * 2.0 / 58.0
    assert float(out.loc[0, "spKt/V"]) == pytest.approx(want, abs=1e-3)
    assert any("Daugirdas" in n for n in notes)


def test_negative_urr_is_kept_not_hidden():
    """前後が入れ替わっていると URR は負になる。**これは消してはならない合図である。**"""
    from medprep.clean import derive_dialysis
    out, notes = derive_dialysis(_hd_frame())
    assert float(out.loc[4, "URR(%)"]) < 0
    assert any("前後が逆" in n for n in notes)


def test_missing_albumin_makes_ica_unknown_not_uncorrected_calcium():
    """★Alb が欠測なら iCa は算出不能。★ Ca をそのまま入れない。"""
    from medprep.clean import derive_dialysis
    out, _ = derive_dialysis(_hd_frame())
    assert pd.isna(out.loc[3, "iCa(mg/dL)"])


def test_existing_columns_are_not_overwritten():
    """人が入れた値を計算値で上書きしない。"""
    from medprep.clean import derive_dialysis
    df = _hd_frame()
    df["URR(%)"] = 99.0
    out, _ = derive_dialysis(df)
    assert (out["URR(%)"] == 99.0).all()


def test_nothing_is_created_when_the_inputs_are_not_paired():
    """片方しか無い組からは作らない（推測した値を混ぜない）。"""
    from medprep.clean import derive_dialysis
    df = _hd_frame().drop(columns=["透析後BUN", "総鉄結合能(TIBC)"])
    out, _ = derive_dialysis(df)
    for c in ("URR(%)", "spKt/V", "TSAT(%)"):
        assert c not in out.columns
    assert "除水量(kg)" in out.columns        # 体重の組は揃っているので作る
