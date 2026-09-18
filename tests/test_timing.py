"""採血時点。不詳の値から透析指標を計算しないことを確かめる。"""
import pandas as pd
import pytest

from medprep.clean import build_alias_map, load_dict
from medprep.timing import POST, PRE, UNKNOWN, TimingSchema, check_requirements, detect_timing


@pytest.mark.parametrize("col,tag,base", [
    ("透析前BUN", PRE, "BUN"), ("透析後BUN", POST, "BUN"),
    ("BUN_pre", PRE, "BUN"), ("BUN_post", POST, "BUN"),
    ("BUN(透析前)", PRE, "BUN"), ("BUN（透析後）", POST, "BUN"),
    ("HD前K", PRE, "K"), ("HD後K", POST, "K"),
    ("透前体重", PRE, "体重"), ("透後体重", POST, "体重"),
    ("開始前Hb", PRE, "Hb"), ("終了後Hb", POST, "Hb"),
    ("pre-Cr", PRE, "Cr"), ("Cr_POST", POST, "Cr"),
    ("収縮期血圧_pre", PRE, "収縮期血圧"), ("週初めHb", PRE, "Hb"),
])
def test_timing_is_read_from_column_name(col, tag, base):
    got_tag, got_base = detect_timing(col)
    assert got_tag == tag and got_base == base


@pytest.mark.parametrize("col", ["尿素窒素(BUN)", "アルブミン(Alb)", "年齢", "体重",
                                 "ドライウェイト", "前回透析日", "透析歴_月"])
def test_columns_without_a_timing_word_stay_unknown(col):
    assert detect_timing(col)[0] == UNKNOWN


def _amap():
    return build_alias_map(load_dict())


def _avail(**kw):
    base = {"Td": None, "age": None, "sex": None, "V_post": None,
            "weight_gain": None, "interdialytic_hours": None}
    base.update(kw)
    return base


def test_indices_are_computable_when_pre_post_pairs_exist():
    df = pd.DataFrame(columns=["年齢", "性別", "透析時間", "透析前BUN", "透析後BUN",
                               "透析前Cr", "透析後Cr", "透析前体重", "透析後体重"])
    ts = TimingSchema.infer(df, alias_map=_amap())
    r = check_requirements(ts, _avail(Td="透析時間", age="年齢", sex="性別"))
    verdict = dict(zip(r["指標"], r["判定"]))
    for k in ("URR", "spKt/V", "nPCR", "%CGR"):
        assert verdict[k] == "算出可"


def test_indices_are_refused_when_timing_is_unknown():
    """列はあるが採血時点が分からない場合、推測して計算しない。"""
    df = pd.DataFrame(columns=["年齢", "性別", "透析時間", "BUN", "Cr", "体重"])
    ts = TimingSchema.infer(df, alias_map=_amap())
    r = check_requirements(ts, _avail(Td="透析時間", age="年齢", sex="性別"))
    row = r[r["指標"] == "URR"].iloc[0]
    assert row["判定"] == "算出不可"
    assert "採血時点が不詳" in row["不足している入力"]


def test_tac_needs_the_next_session_pre_not_the_same_session_pre():
    """同一回の前後だけでは TAC は出せない。黙って別のものを計算しない。"""
    df = pd.DataFrame(columns=["透析前BUN", "透析後BUN", "透析時間"])
    ts = TimingSchema.infer(df, alias_map=_amap())
    r = check_requirements(ts, _avail(Td="透析時間"))
    row = r[r["指標"] == "TAC-BUN(簡便式)"].iloc[0]
    assert row["判定"] == "算出不可" and "BUN_next_pre" in row["不足している入力"]

    df2 = pd.DataFrame(columns=["透析前BUN", "透析後BUN", "次回透析前BUN", "透析時間", "非透析時間"])
    ts2 = TimingSchema.infer(df2, alias_map=_amap())
    r2 = check_requirements(ts2, _avail(Td="透析時間", interdialytic_hours="非透析時間"))
    assert dict(zip(r2["指標"], r2["判定"]))["TAC-BUN(簡便式)"] == "算出可"


def test_default_timing_is_applied_only_when_a_human_says_so():
    df = pd.DataFrame(columns=["BUN", "Cr"])
    assert TimingSchema.infer(df).timing_of("BUN") == UNKNOWN
    ts = TimingSchema.infer(df, default_timing=PRE)
    assert ts.timing_of("BUN") == PRE
    assert any("人の指定" in n for n in ts.notes)


def test_pairs_report():
    df = pd.DataFrame(columns=["透析前BUN", "透析後BUN", "透析前Na", "アルブミン(Alb)"])
    p = TimingSchema.infer(df, alias_map=_amap()).pairs().set_index("項目")["状態"]
    assert p["BUN"] == "揃っている"
    assert p["Na"] == "片方のみ"
    assert p["アルブミン(Alb)"] == "時点不詳のみ"
