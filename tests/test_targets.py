"""管理目標。開区間と閉区間の取り違えが達成率を数％動かすため、境界値を固定する。"""
import numpy as np
import pandas as pd
import pytest

from medprep.clean import load_dict
from medprep.targets import achievement, describe_target, in_target

DIC = load_dict()


@pytest.mark.parametrize("key,value,expected", [
    ("P", 3.4, False), ("P", 3.5, True), ("P", 5.4, True), ("P", 5.5, False), ("P", 5.6, False),
    ("cCa", 8.3, False), ("cCa", 8.4, True), ("cCa", 9.4, True), ("cCa", 9.5, False),
    ("Hb", 9.9, False), ("Hb", 10.0, True), ("Hb", 11.9, True), ("Hb", 12.0, False),
    ("iPTH", 60, True), ("iPTH", 239, True), ("iPTH", 240, False), ("iPTH", 59, True),
    ("KtV", 1.2, False), ("KtV", 1.39, False), ("KtV", 1.4, True),
    ("B2MG", 29, True), ("B2MG", 30, False),
])
def test_open_and_closed_interval_boundaries(key, value, expected):
    t = DIC["items"][key]["target"]
    assert bool(in_target([value], t).iloc[0]) is expected


def test_ipth_has_no_universal_lower_bound():
    """2025年改訂版の要点。下限を全患者共通で 60 にしない。"""
    t = DIC["items"]["iPTH"]["target"]
    assert t.get("low") is None
    assert t["conditional"][0]["low"] == 60


def test_ktv_separates_target_from_minimum():
    t = DIC["items"]["KtV"]["target"]
    assert t["low"] == 1.4 and t["minimum"]["low"] == 1.2


def test_b2mg_stretch_goal_is_marked_as_opinion():
    t = DIC["items"]["B2MG"]["target"]
    assert t["high"] == 30 and t["stretch"]["high"] == 25
    assert "オピニオン" in t["stretch"]["evidence"]


def test_missing_values_stay_na_not_false():
    t = DIC["items"]["P"]["target"]
    got = in_target([np.nan, 4.0], t)
    assert pd.isna(got.iloc[0]) and got.iloc[1]


def test_describe_target_wording():
    assert describe_target(DIC["items"]["P"]["target"]) == "3.5 以上、5.5 未満"
    assert "週初め" in describe_target(DIC["items"]["Hb"]["target"])


def test_achievement_reports_boundary_cases_separately():
    df = pd.DataFrame({"無機リン(P)": [3.5, 5.5, 5.5, 4.0, 3.4]})
    a = achievement(df, DIC, {"P": "無機リン(P)"})
    edge = a[a["群"].str.contains("境界")]
    assert len(edge) >= 1
    assert int(edge[edge["群"].str.contains("5.5")]["評価可能例数"].iloc[0]) == 2
