"""ファイルを読む。**推測したことを黙らせない。**

文字コードを当てたのも、見出しが 3 行目だと判断したのも推測である。
推測を記録に残さないと、読み間違えたことに誰も気づかない。
"""
import pandas as pd
import pytest

from medprep.loading import read_any


def frame():
    return pd.DataFrame({"仮名ID": ["P001", "P002"], "年齢": [68, 72],
                         "施設": ["A院", "B院"]})


# ---------------------------------------------------------------- 形式
def test_csv_and_excel_and_json(tmp_path):
    df = frame()
    df.to_csv(tmp_path / "a.csv", index=False)
    df.to_excel(tmp_path / "a.xlsx", index=False)
    df.to_json(tmp_path / "a.json")
    for name in ("a.csv", "a.xlsx", "a.json"):
        assert list(read_any(tmp_path / name).columns) == list(df.columns)


def test_an_unknown_extension_says_so(tmp_path):
    p = tmp_path / "a.sas7bdat"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="読み方が分からない"):
        read_any(p)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="ファイルが無い"):
        read_any(tmp_path / "無い.csv")


# ---------------------------------------------------------------- 文字コード
def test_shift_jis_is_read_and_the_guess_is_recorded(tmp_path):
    """★日本の医療データの csv は cp932 で出てくる。★"""
    p = tmp_path / "a.csv"
    frame().to_csv(p, index=False, encoding="cp932")
    df = read_any(p)
    assert "A院" in df["施設"].tolist()
    assert any("cp932" in n for n in df.attrs["medprep_read"])


def test_utf8_needs_no_note(tmp_path):
    p = tmp_path / "a.csv"
    frame().to_csv(p, index=False, encoding="utf-8-sig")
    assert not any("文字コード" in n for n in read_any(p).attrs["medprep_read"])


# ---------------------------------------------------------------- 見出し
def test_a_title_row_above_the_header_is_found(tmp_path):
    """★1 行目が『○○病院 患者一覧』の Excel。★

    header=0 のまま読むと列名が全部 `Unnamed: 1` になり、以降のすべてが狂う。
    """
    p = tmp_path / "a.xlsx"
    raw = pd.DataFrame([["○○病院 透析患者一覧（2025年3月）", None, None],
                        [None, None, None],
                        ["仮名ID", "年齢", "施設"],
                        ["P001", 68, "A院"]])
    raw.to_excel(p, index=False, header=False)
    df = read_any(p)
    assert list(df.columns) == ["仮名ID", "年齢", "施設"]
    assert any("見出しは 3 行目" in n for n in df.attrs["medprep_read"])


def test_a_normal_table_is_left_alone(tmp_path):
    p = tmp_path / "a.xlsx"
    frame().to_excel(p, index=False)
    assert not any("見出し" in n for n in read_any(p).attrs["medprep_read"])


def test_an_explicit_header_wins(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame([["表題", None], ["列A", "列B"], [1, 2]]).to_csv(
        p, index=False, header=False)
    assert list(read_any(p, header=1).columns) == ["列A", "列B"]


# ---------------------------------------------------------------- 掃除
def test_empty_columns_are_dropped_and_reported(tmp_path):
    """Excel の余白がそのまま列になることがある。"""
    p = tmp_path / "a.xlsx"
    df = frame()
    df["余白"] = None
    df.to_excel(p, index=False)
    out = read_any(p)
    assert "余白" not in out.columns
    assert any("空の列" in n for n in out.attrs["medprep_read"])


def test_empty_rows_are_dropped_and_reported(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("仮名ID,年齢,施設\nP001,68,A院\n,,\nP002,72,B院\n", encoding="utf-8")
    out = read_any(p)
    assert len(out) == 2
    assert any("空の行" in n for n in out.attrs["medprep_read"])


def test_duplicate_column_names_are_flagged(tmp_path):
    """★同名の列は pandas が `.1` を付けて黙って区別する。★"""
    p = tmp_path / "a.csv"
    p.write_text("Alb,Alb,年齢\n3.5,3.6,68\n", encoding="utf-8")
    df = read_any(p)
    assert any("同じ名前の列" in n for n in df.attrs["medprep_read"])


def test_the_source_path_is_kept(tmp_path):
    p = tmp_path / "a.csv"
    frame().to_csv(p, index=False)
    assert read_any(p).attrs["medprep_source"] == str(p)
