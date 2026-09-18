"""透析指標。提示された計算例との一致を回帰テストとして固定する。"""
import numpy as np
import pandas as pd
import pytest

from medprep.hd import (
    clear_space_ratio,
    clear_space_ratio_from_weight,
    corrected_ca,
    gnri,
    ideal_body_weight,
    percent_cgr,
    removed_mass_from_effluent,
    salt_intake,
    select_weight_for_gnri,
    sp_ktv,
    tsat,
    urr,
)

# 提示された計算例（男性60歳）— 中間値をすべて固定する
CGR_CASE = {"sex": "男性", "age": 60, "bun_pre": 60, "bun_post": 20,
            "cr_pre": 12, "cr_post": 4, "bw_pre": 63, "bw_post": 60, "td_hours": 4}
CGR_EXPECT = {
    "delta_bw": 3.0, "R": 0.3333333333, "spKtV": 1.3412048739, "nPCR": 0.9806154514,
    "Cr_corr": 5.2625155371, "L": -0.8242974984, "A": 5142.1477249759,
    "G_total": 18.5839184553, "G_ext": 1.6642476884, "G_int": 16.9196707669,
    "G_ref": 14.53, "percent_CGR": 116.4464608869,
}


@pytest.mark.parametrize("name,want", CGR_EXPECT.items())
def test_percent_cgr_matches_reference_worked_example(name, want):
    r = percent_cgr(**CGR_CASE)
    got = float(np.atleast_1d(getattr(r, name))[0])
    assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


def test_percent_cgr_records_provenance():
    """nPCR の算出法が変われば %CGR も変わる。どの式で出したかを残すこと。"""
    r = percent_cgr(**CGR_CASE)
    assert "Kaynar" in r.provenance["nPCR式"]
    assert "週3回" in r.provenance["G_total の 72 時間"]
    assert "残腎機能" in r.provenance


def test_percent_cgr_returns_nan_not_zero_for_bad_input():
    """異常な入力や負の産生速度を自動で 0 に丸めない。"""
    r = percent_cgr(sex=["男性", "男性", "女性"], age=[60, 60, 60],
                    bun_pre=[60, 60, 60], bun_post=[20, 0, 20], cr_pre=[12, 12, 12],
                    cr_post=[4, 4, 4], bw_pre=[63, 63, 63], bw_post=[60, 60, 60],
                    td_hours=[4, 4, 200])
    assert not np.isnan(r.percent_CGR[0])
    assert np.isnan(r.percent_CGR[1]) and np.isnan(r.percent_CGR[2])
    assert len(r.invalid) > 0


def test_clear_space_ratio_matches_worked_example():
    cs = clear_space_ratio(15120, 60, 36)
    assert float(cs["A_L"]) == pytest.approx(25.2)
    assert float(cs["AV_ratio"]) == pytest.approx(0.7)
    assert float(cs["AV_percent"]) == pytest.approx(70.0)


def test_removed_mass_from_effluent_matches_worked_example():
    m = removed_mass_from_effluent(12, 126)
    assert float(m) == pytest.approx(15120.0)
    assert float(clear_space_ratio(m, 60, 36)["AV_percent"]) == pytest.approx(70.0)


def test_clear_space_ratio_is_not_clipped_to_100():
    """透析後 V で標準化した指標は条件により 100% を超える。切り詰めない。"""
    assert float(clear_space_ratio(30000, 60, 36)["AV_percent"]) > 100.0


def test_clear_space_ratio_from_weight():
    got = clear_space_ratio_from_weight(60, 20, 63, 60, 36)
    assert float(got) == pytest.approx(100 * (1 + 3 / 36 - 20 / 60))


def test_sp_ktv_requires_positive_log_argument():
    assert np.isnan(float(np.atleast_1d(sp_ktv(60, 0, 63, 60, 4))[0]))


def test_urr():
    assert float(np.atleast_1d(urr(60, 20))[0]) == pytest.approx(66.6666666667)


@pytest.mark.parametrize("ca,alb,want", [(8.5, 3.0, 9.5), (8.5, 4.5, 8.5), (8.5, 4.0, 8.5)])
def test_corrected_ca_payne(ca, alb, want):
    assert float(np.atleast_1d(corrected_ca(ca, alb))[0]) == pytest.approx(want)


def test_tsat():
    assert float(np.atleast_1d(tsat(60, 300))[0]) == pytest.approx(20.0)


def test_salt_intake_and_its_denominator():
    """透析間隔日数を取り違えると 1.5 倍の過大評価になる。"""
    two = float(np.atleast_1d(salt_intake(3.0, 140, 2.0))[0])
    three = float(np.atleast_1d(salt_intake(3.0, 140, 3.0))[0])
    assert three == pytest.approx(3 * 140 / 17 / 3)
    assert two / three == pytest.approx(1.5)


def test_gnri_uses_dry_weight_first_then_post_dialysis():
    df = pd.DataFrame({
        "ドライウェイト": [58.0, np.nan, np.nan, np.nan],
        "透析後体重": [57.5, 62.0, np.nan, np.nan],
        "透析前体重": [60.0, 64.5, 55.0, np.nan],
        "アルブミン": [3.8, 3.2, 4.0, 3.5],
        "身長": [165, 170, 158, 172],
    })
    w, src = select_weight_for_gnri(df, "ドライウェイト", "透析後体重", "透析前体重")
    assert src.iloc[0] == "ドライウェイト" and w.iloc[0] == 58.0
    assert src.iloc[1] == "透析後体重" and w.iloc[1] == 62.0
    g = gnri(df["アルブミン"], w, df["身長"])
    assert np.isnan(g[2]) and np.isnan(g[3]), "透析前体重しか無い症例は算出しない"
    assert g[0] == pytest.approx(14.89 * 3.8 + 41.7 * min(58.0 / ideal_body_weight(165), 1.0))


def test_ideal_body_weight_bmi22():
    assert float(np.atleast_1d(ideal_body_weight(170))[0]) == pytest.approx(22 * 1.7 ** 2)
