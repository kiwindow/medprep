"""表を文字で出すときの桁揃え。

**`to_string()` は桁を「文字数」で揃える。** 日本語は等幅フォントで 2 桁ぶんの幅を
取るので、日本語の列名が混ざると表がずれる。医学データの列名はほぼ日本語なので、
ずれないほうが珍しい。表示幅で揃っていることを、ここで機械に見張らせる。
"""
import unicodedata

import pandas as pd
import pytest

from medprep import frame_text


def width(s: str) -> int:
    """等幅フォントでの表示幅（全角は 2 桁ぶん）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def frame():
    return pd.DataFrame({
        "列名": ["年齢", "アルブミン(Alb)", "末梢血｜血色素量(Hb)", "ID"],
        "役割": ["numeric", "numeric", "numeric", "id"],
        "判断の根拠": ["数値型", "数値型", "前処理で 100% が数値化できる", "人が指定した"],
    })


def test_every_line_has_the_same_display_width():
    """★桁が揃うとは、全行の表示幅が同じということ。★"""
    assert len({width(line) for line in frame_text(frame()).split("\n")}) == 1


def test_the_plain_to_string_is_the_one_that_breaks():
    """比較用。既定の to_string は日本語でずれる（この性質があるから本モジュールがある）。"""
    lines = frame().to_string(index=False).split("\n")
    assert len({width(line) for line in lines}) > 1


def test_long_values_are_not_truncated():
    """★「判断の根拠」は切り詰めたら意味を失う。★"""
    long = "あ" * 120
    df = pd.DataFrame({"列": ["x"], "根拠": [long]})
    assert long in frame_text(df)


def test_it_can_be_narrowed_on_purpose():
    df = pd.DataFrame({"列": ["x"], "根拠": ["あ" * 120]})
    assert "..." in frame_text(df, max_colwidth=20)


def test_the_index_can_be_shown():
    df = pd.DataFrame({"値": [1.0, 2.0]}, index=["年齢", "アルブミン"])
    out = frame_text(df, index=True)
    assert "アルブミン" in out
    assert len({width(line) for line in out.split("\n")}) == 1


def test_it_does_not_change_the_users_pandas_settings():
    """★報告のために全体の設定を書き換えてはならない。★"""
    before = pd.get_option("display.unicode.east_asian_width")
    frame_text(frame())
    assert pd.get_option("display.unicode.east_asian_width") == before


def test_all_columns_are_shown_even_when_there_are_many():
    df = pd.DataFrame({f"列{i}": [i] for i in range(25)})
    out = frame_text(df)
    assert "列24" in out and "..." not in out


@pytest.mark.parametrize("report", ["schema", "missing", "outliers"])
def test_the_reports_line_up(report):
    """★実際の報告でも揃っていること。★"""
    import matplotlib
    matplotlib.use("Agg")
    import medprep as mp
    df = mp.demo.dialysis_cohort(n=120, seed=1)
    sch = mp.Schema.infer(df, id_col="仮名ID", group="施設")
    text = {
        "schema": lambda: sch.report(),
        "missing": lambda: mp.analyze_missing(df, sch).report(),
        "outliers": lambda: mp.detect_outliers(df, sch, id_col="仮名ID").report(),
    }[report]()
    # 表の部分（複数の空白で区切られた行）だけを見る
    table = [ln for ln in text.split("\n") if ln.count("  ") >= 2 and ln.strip()]
    assert len(table) >= 3
    assert len({width(ln) for ln in table}) <= 2      # 表は 1 つとは限らない
