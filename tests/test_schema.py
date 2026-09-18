"""列の役割推定。**推測してはいけないところで推測していないこと**を確かめる。

ここで守りたい性質は 3 つ。

  1. 判断には必ず理由がつく（`reason` が空の ColumnSpec を作らない）
  2. 決められないものは決めない（ユニーク率だけで識別子と断定しない、等）
  3. YAML に往復できる（人が直して再実行できる）
"""
import numpy as np
import pandas as pd
import pytest

from medprep.schema import (
    BINARY,
    CONSTANT,
    DATETIME,
    DUPLICATE,
    GROUP,
    HIGH_CARDINALITY,
    ID,
    NOMINAL,
    NUMERIC,
    ORDINAL,
    OUTCOME,
    Schema,
    cramers_v,
    infer_column,
)


def frame(n=120):
    rng = np.random.default_rng(0)
    fac = rng.choice(["A院", "B院", "C院"], n)
    return pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": fac,
        "施設コード": pd.Series(fac).map({"A院": 1, "B院": 2, "C院": 3}),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.integers(30, 90, n),
        "観察開始年月日": [f"201{i % 8}/{i % 12 + 1}/{i % 28 + 1}" for i in range(n)],
        "備考": [f"特記なし_{i}" for i in range(n)],
        "施設内番号": rng.integers(1, 5, n),
        "研究名": ["鹿鳴館コホート"] * n,
        "ECOG": rng.integers(0, 5, n),
        "転帰": rng.choice([0, 1], n),
    })


# ---------------------------------------------------------------- 理由
def test_every_decision_carries_a_reason():
    """★理由の無い判断を作らない。★ これがこのモジュールの存在理由である。"""
    sch = Schema.infer(frame())
    for c, sp in sch.columns.items():
        assert sp.reason and sp.reason.strip(), f"{c}: reason が空"
        assert sp.role, f"{c}: role が空"
        assert sp.action in ("keep", "drop")


# ---------------------------------------------------------------- 識別子
def test_id_col_given_by_human_wins():
    sch = Schema.infer(frame(), id_col="仮名ID")
    assert sch.columns["仮名ID"].role == ID
    assert sch.columns["仮名ID"].action == "drop"
    assert "id_col" in sch.columns["仮名ID"].reason


def test_unique_ratio_alone_does_not_make_a_column_an_identifier():
    """備考（自由記載）はユニーク率 1.00 になるが、識別子ではない。

    ユニーク率からは「識別子」と「自由記載」を区別できない。
    区別できないものを断定せず、解析から外すにとどめる。
    """
    sch = Schema.infer(frame())
    assert sch.columns["備考"].role == HIGH_CARDINALITY
    assert sch.columns["備考"].role != ID
    assert "id_col" in sch.columns["備考"].reason      # 人への問いかけを残す


def test_column_named_like_an_id_is_detected_without_being_told():
    sch = Schema.infer(frame())
    assert sch.columns["仮名ID"].role == ID


# ---------------------------------------------------------------- 日付
def test_dates_are_detected_before_the_unique_ratio_rule():
    """★順序が逆だと、観察開始日が『ほぼ全行で異なる列』として識別子にされる。★"""
    sch = Schema.infer(frame())
    assert sch.columns["観察開始年月日"].role == DATETIME


def test_survival_dates_given_by_human_are_kept():
    df = frame()
    df["死亡年月日"] = ""
    df["打切年月日"] = "2020/1/1"
    sch = Schema.infer(df, survival_dates=("観察開始年月日", "死亡年月日", "打切年月日"))
    assert sch.survival["form"] == "C"
    for c in ("観察開始年月日", "死亡年月日", "打切年月日"):
        assert sch.columns[c].action == "keep"
    assert "形式C" in sch.report()


# ---------------------------------------------------------------- 尺度
def test_small_integer_codes_are_not_mistaken_for_an_ordinal_scale():
    """施設内番号 {1,2,3,4} は ECOG {0,1,2,3,4} の部分集合になってしまう。

    小さい整数の集合から順序尺度かどうかは決まらない。列名の裏づけを要求する。
    """
    sch = Schema.infer(frame())
    assert sch.columns["施設内番号"].role != ORDINAL
    assert sch.columns["ECOG"].role == ORDINAL        # 列名が尺度名と一致する場合のみ


def test_binary_map_records_which_level_became_one():
    sch = Schema.infer(frame())
    sp = sch.columns["性別"]
    assert sp.role == BINARY
    assert sp.value_map == {"男": 1, "女": 0}          # どちらを 1 にしたかを残す


def test_binary_without_a_known_map_asks_the_human():
    s = pd.Series(["甲"] * 50 + ["乙"] * 50)
    sp = infer_column(s, "型")
    assert sp.role == BINARY and sp.value_map is None
    assert "人が決める" in sp.reason


def test_constant_and_near_constant_are_dropped():
    sch = Schema.infer(frame())
    assert sch.columns["研究名"].role == CONSTANT
    assert sch.columns["研究名"].action == "drop"
    s = pd.Series([1] * 199 + [2])
    assert infer_column(s, "ほぼ定数").role == CONSTANT


def test_all_missing_column_is_constant_not_numeric():
    s = pd.Series([np.nan] * 50)
    assert infer_column(s, "未入力").role == CONSTANT


def test_numeric_with_symbols_is_still_numeric():
    s = pd.Series(["<0.1", "0.3", "1,200", "２．５"] * 10)
    assert infer_column(s, "CRP").role == NUMERIC


# ---------------------------------------------------------------- 重複列
def test_duplicate_columns_are_found_across_dtypes():
    """施設（文字列）と 施設コード（整数）は完全に対応する。

    数値どうし・カテゴリどうしだけを見ていると気づけない組み合わせである。
    """
    sch = Schema.infer(frame(), group="施設")
    assert sch.columns["施設コード"].role == DUPLICATE
    assert sch.columns["施設コード"].duplicate_of == "施設"
    assert sch.columns["施設"].action == "keep"        # 読める側を残す


def test_cramers_v_is_one_for_a_perfect_correspondence():
    a = pd.Series(list("aabbcc") * 20)
    b = a.map({"a": 1, "b": 2, "c": 3})
    assert cramers_v(a, b) == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_returns_none_when_undefined():
    assert cramers_v(pd.Series(["a"] * 10), pd.Series([1] * 10)) is None


# ---------------------------------------------------------------- 目的変数
def test_outcome_and_features():
    sch = Schema.infer(frame(), outcome="転帰", task="classification", id_col="仮名ID")
    assert sch.columns["転帰"].role == OUTCOME
    assert sch.target["task"] == "classification"
    feats = sch.features()
    assert "転帰" not in feats and "仮名ID" not in feats
    assert "年齢" in feats


def test_group_given_by_human_is_marked():
    sch = Schema.infer(frame(), group="施設")
    assert sch.columns["施設"].role == GROUP


# ---------------------------------------------------------------- 往復
def test_yaml_round_trip_changes_nothing(tmp_path):
    """★往復で差分 0。★ ここが崩れると schema.yaml を配る意味が無くなる。"""
    sch = Schema.infer(frame(), id_col="仮名ID", group="施設", outcome="転帰")
    path = tmp_path / "schema.yaml"
    sch.to_yaml(path)
    back = Schema.from_yaml(str(path))
    assert list(back.columns) == list(sch.columns)
    assert len(sch.diff(back)) == 0


def test_diff_shows_what_a_human_changed(tmp_path):
    sch = Schema.infer(frame())
    path = tmp_path / "s.yaml"
    sch.to_yaml(path)
    edited = Schema.from_yaml(str(path))
    edited.columns["備考"].role = NOMINAL
    edited.columns["備考"].action = "keep"
    d = sch.diff(edited)
    assert set(d["項目"]) == {"role", "action"}
    assert set(d["列名"]) == {"備考"}


def test_yaml_header_tells_the_reader_what_to_do():
    text = Schema.infer(frame()).to_yaml()
    assert "人が直して再実行できる" in text
    assert "reason" in text


def test_report_is_printable_for_both_survival_forms():
    df = frame()
    df["t"] = 1.0
    df["e"] = 0
    a = Schema.infer(df, survival=("t", "e"))
    assert "形式A" in a.report()
    c = Schema.infer(df, survival_dates=("観察開始年月日", "観察開始年月日", "観察開始年月日"))
    assert "形式C" in c.report()


def test_to_frame_has_one_row_per_column():
    sch = Schema.infer(frame())
    assert len(sch.to_frame()) == len(sch.columns)
    assert "判断の根拠" in sch.to_frame().columns
