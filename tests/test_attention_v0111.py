"""0.11.1 — 「人の確認が要る事項」を画面に流すだけで終わらせない。

`rep.warnings` はノートブックを閉じれば消える。前書きに「schema.yaml と
人の確認が要る事項に残る」と書いた以上、ファイルに残らなければならない。
同じ一覧を、同じ順で、3 か所に書き出す。

  * run{N}/report/人の確認が要る事項.txt
  * run{N}/table/prep_tables.xlsx の 1 枚目のシート
  * run{N}/report/prep_report.html の「まず読むこと」の先頭
"""
import os
from pathlib import Path

import pandas as pd
import pytest

import medprep as mp
from medprep.report import ATTENTION, ATTENTION_TXT, build_report
from test_outputs_v011 import _cohort


@pytest.fixture
def saved(tmp_path):
    rep = mp.autoprep(_cohort(), id_col="仮名ID", group="施設",
                      survival_dates=("開始日", "発生日", "打切日"),
                      save=True, out_dir=str(tmp_path), method="T", verbose=False)
    return rep


def _items_of(rep):
    """伏せたあとの一覧（HTML・Excel・txt に載るもの）。"""
    return rep.html.attention


def test_there_is_something_to_check(saved):
    # ★空でないのがふつうである。★ 空だと以下の検査が何も確かめなくなる。
    assert len(saved.warnings) >= 2
    assert len(_items_of(saved)) == len(saved.warnings)


def test_txt_is_written_in_the_report_folder(saved):
    path = saved.run.file("report", ATTENTION_TXT)
    assert os.path.exists(path) and path in saved.saved
    raw = Path(path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                 # BOM：Windows のメモ帳で化けない
    text = raw.decode("utf-8-sig")
    assert text.splitlines()[0] == f"■ {ATTENTION}"
    assert f"全 {len(saved.warnings)} 件" in text
    for i, t in enumerate(_items_of(saved), 1):
        assert f"{i}. {t}" in text, t
    assert os.sep not in text.splitlines()[1]              # ★フルパスを書かない★


def test_excel_has_the_same_list_on_the_first_sheet(saved):
    path = saved.run.file("table", "prep_tables.xlsx")
    book = pd.ExcelFile(path)
    assert book.sheet_names[0] == ATTENTION
    sh = pd.read_excel(path, sheet_name=ATTENTION)
    assert list(sh.columns) == ["番号", "内容"]
    assert sh["内容"].tolist() == _items_of(saved)          # ★同じ一覧、同じ順★
    assert sh["番号"].tolist() == list(range(1, len(saved.warnings) + 1))


def test_html_puts_the_list_at_the_top_of_read_first(saved):
    html = Path(saved.run.file("report", "prep_report.html")).read_text(encoding="utf-8")
    top = html.index("<h2>まず読むこと</h2>")
    att = html.index(f">{ATTENTION}（{len(saved.warnings)} 件）</h3>")
    nav = html.index("<nav>")
    assert top < att < nav                                  # ★目次より前、まず読むことの先頭★
    from markupsafe import escape as _esc
    pos = [html.index(f"<li>{_esc(t)}</li>") for t in _items_of(saved)]
    assert pos == sorted(pos)                               # 同じ順
    if saved.audit.errors:                                  # 監査の詳しい所見はその下
        assert att < html.index("監査の致命的な所見（詳しく）")


def test_three_places_agree(saved):
    txt = Path(saved.run.file("report", ATTENTION_TXT)).read_text(encoding="utf-8-sig")
    sh = pd.read_excel(saved.run.file("table", "prep_tables.xlsx"), sheet_name=ATTENTION)
    html = Path(saved.run.file("report", "prep_report.html")).read_text(encoding="utf-8")
    from markupsafe import escape as _esc
    for t in sh["内容"]:
        assert t in txt and f"<li>{_esc(t)}</li>" in html


def test_an_empty_list_says_none_in_all_three(tmp_path):
    df = pd.DataFrame({"年齢": [50, 60, 70], "Alb": [3.5, 3.8, 4.0]})
    rep = build_report(df, attention=[])
    assert rep.attention_frame()["内容"].tolist() == ["なし"]
    assert "なし" in rep.attention_text()
    html = rep.to_html()
    assert f">{ATTENTION}</h3>" in html and "<p>なし</p>" in html
    rep.to_excel(tmp_path / "t.xlsx")
    assert pd.ExcelFile(tmp_path / "t.xlsx").sheet_names[0] == ATTENTION


def test_case_values_are_masked_unless_asked():
    df = pd.DataFrame({"年齢": [50, 60, 70]})
    w = ["日付 '検体採取日' の 2 例を解釈できず空欄にした（例: ['2021/13/45', '不明']）"]
    hidden = build_report(df, attention=w)
    assert "2021/13/45" not in hidden.to_html()
    assert "2021/13/45" not in hidden.attention_text()
    assert "show_values=True" in hidden.attention[0]
    shown = build_report(df, attention=w, show_values=True)
    assert "2021/13/45" in shown.to_html() and "2021/13/45" in shown.attention_text()


def test_dictionary_examples_are_not_masked():
    # 「例:」の後が症例の値のリストでなければ伏せない（測定法の説明など）
    df = pd.DataFrame({"年齢": [50, 60, 70]})
    w = ["測定法  ALP の段差  → 期間で層別すること（既知の影響: 値がおよそ 1/3 になる"
         "（例: 240 U/L → 80 U/L））"]
    assert "240 U/L → 80 U/L" in build_report(df, attention=w).attention[0]


def test_without_attention_the_report_is_as_before():
    # autoprep を通さずに build_report を呼んだとき（層2）。勝手に「なし」と書かない。
    df = pd.DataFrame({"年齢": [50, 60, 70]})
    rep = build_report(df)
    assert rep.attention is None
    assert ATTENTION not in rep.to_html()
