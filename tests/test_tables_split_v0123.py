"""0.12.3 — Table 1 と Table 2 を別のファイルに書き出す。欠損の一覧は安定ソート。

以前は run{N}/table/Table1_2.xlsx・.docx に 2 枚を並べていた。Finder のプレビュー
（1 ページ目だけ）では Table 2 が無いように見えたので、表ごとに分ける。

  * Table1.xlsx・Table1.docx … 全症例の背景（日本語・英語）
  * Table2.xlsx・Table2.docx … GROUP 別の比較（GROUP を選んだときだけ）
"""
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

import medprep as mp
from medprep.missing import analyze
from test_outputs_v011 import _cohort


def _docx_text(path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("word/document.xml").decode("utf-8")


def _table_dir(rep):
    return os.path.join(rep.run.run, "table")


def _binary(n=160, seed=3):
    """二値の目的変数をそのまま GROUP にする形（halddata と同じ使い方）。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({
        "身長": (160 + 8 * x).round(),
        "体重": (60 + 10 * x + rng.normal(0, 5, n)).round(),
        "年齢": rng.normal(60, 10, n).round(),
        "男性": rng.integers(0, 2, n),
        "長時間手術": (x + rng.normal(0, 1, n) > 0.5).astype(int),
    })


@pytest.fixture
def with_group(tmp_path):
    return mp.autoprep(_cohort(), id_col="仮名ID", group="施設",
                       save=True, out_dir=str(tmp_path), method="T", verbose=False)


def test_two_tables_are_written_separately(with_group):
    d = _table_dir(with_group)
    names = set(os.listdir(d))
    for f in ("Table1.xlsx", "Table1.docx", "Table2.xlsx", "Table2.docx"):
        assert f in names, f
        assert os.path.join(d, f) in with_group.saved
    assert not any(n.startswith("Table1_2") for n in names)
    assert "table1_詳細.xlsx" in names                     # 詳細はこれまでどおり


def test_each_word_file_holds_only_its_own_table(with_group):
    d = _table_dir(with_group)
    t1 = _docx_text(os.path.join(d, "Table1.docx"))
    t2 = _docx_text(os.path.join(d, "Table2.docx"))
    assert "表1." in t1 and "表2." not in t1
    assert "表2." in t2 and "表1." not in t2
    assert "Table 1" in t1 and "Table 2" in t2              # 英語版も同じファイルに
    assert t1.count("<w:tbl>") == 2 and t2.count("<w:tbl>") == 2   # 日・英 1 枚ずつ


def test_each_excel_file_holds_only_its_own_table(with_group):
    d = _table_dir(with_group)
    for name, title, other in (("Table1.xlsx", "表1.", "表2."),
                               ("Table2.xlsx", "表2.", "表1.")):
        book = pd.read_excel(os.path.join(d, name), sheet_name=None, header=None)
        assert list(book) == ["日本語", "English"]
        ja = [str(v) for v in book["日本語"].to_numpy().ravel()]
        assert any(v.startswith(title) for v in ja)
        assert not any(v.startswith(other) for v in ja)


def test_without_group_only_table1(tmp_path):
    rep = mp.autoprep(_cohort(), id_col="仮名ID",
                      save=True, out_dir=str(tmp_path), method="T", verbose=False)
    names = set(os.listdir(_table_dir(rep)))
    assert {"Table1.xlsx", "Table1.docx"} <= names
    assert not any(n.startswith("Table2") for n in names)


def test_outcome_as_group_gives_table2_by_outcome(tmp_path):
    rep = mp.autoprep(_binary(), group="長時間手術", outcome="長時間手術",
                      task="classification",
                      save=True, out_dir=str(tmp_path), method="T", verbose=False)
    t2 = _docx_text(os.path.join(_table_dir(rep), "Table2.docx"))
    assert "長時間手術" in t2


def test_which_is_checked_and_default_keeps_both(with_group, tmp_path):
    with pytest.raises(ValueError):
        with_group.gt.to_excel(str(tmp_path / "x.xlsx"), which="table3")
    both = with_group.gt.to_docx(str(tmp_path / "both.docx"))   # 引数なしは従来どおり 2 枚
    s = _docx_text(both)
    assert "表1." in s and "表2." in s


def test_missing_table_keeps_column_order_for_ties():
    df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [np.nan, np.nan, 1, 2],
                       "c": [1, np.nan, 3, 4], "d": [np.nan, np.nan, 1, 2],
                       "e": [1, 2, 3, 4]})
    order = analyze(df).columns["列"].tolist()
    assert order == ["b", "d", "a", "c", "e"]
