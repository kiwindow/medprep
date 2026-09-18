"""HTML 1 枚のレポート。

確かめたいのは体裁ではなく、次の 3 つである。

  1. **警告が先頭にある**（10 枚目にある致命的所見は読まれないのと同じ）
  2. **症例レベルの値が既定で出ない**（レポートはメールで回る）
  3. **表と図が崩れずに出る**（崩れた表は「欠けている」ように見える）
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from medprep import viz  # noqa: E402
from medprep.quality import audit  # noqa: E402
from medprep.report import Report, build_report  # noqa: E402
from medprep.schema import Schema  # noqa: E402

rng = np.random.default_rng(20260918)


def frame(n=120):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "Alb": rng.normal(3.6, 0.45, n).round(1),
    })
    df.loc[df.index[:10], "Alb"] = np.nan
    return df


def with_findings():
    """致命的な所見が出るデータ（同じ ID が 2 行）。"""
    df = frame()
    df.loc[df.index[1], "仮名ID"] = df.loc[df.index[0], "仮名ID"]
    sch = Schema.infer(df, id_col="仮名ID", group="施設")
    return df, sch, audit(df, sch, id_col="仮名ID", group="施設")


# ---------------------------------------------------------------- 骨格
def test_the_report_leads_with_the_warnings():
    """★致命的な所見は先頭に出す。★"""
    df, sch, aud = with_findings()
    html = build_report(df, sch, audit=aud).to_html()
    assert "まず読むこと" in html
    assert html.index("まず読むこと") < html.index("データ品質監査")
    assert "致命的" in html


def test_counts_banner_matches_the_audit():
    df, sch, aud = with_findings()
    rep = build_report(df, sch, audit=aud)
    assert rep.counts["error"] == len(aud.errors)
    assert rep.counts["warn"] == len(aud.warnings)


def test_table_of_contents_links_to_every_section():
    df, sch, aud = with_findings()
    rep = build_report(df, sch, audit=aud)
    html = rep.to_html()
    for s in rep.sections:
        assert f'href="#{s.anchor}"' in html
        assert f'id="{s.anchor}"' in html


def test_it_works_with_nothing_but_a_dataframe():
    html = build_report(frame()).to_html()
    assert "再現性" in html and "<table" not in html.split("再現性")[1]


# ---------------------------------------------------------------- 個人情報
def test_case_level_values_are_hidden_by_default():
    """★レポートはメールで回る。仮名 ID でも既定では出さない。★"""
    df, sch, aud = with_findings()
    html = build_report(df, sch, audit=aud).to_html()
    dup_id = df.loc[df.index[0], "仮名ID"]
    assert dup_id not in html
    assert "show_values=True" in html
    assert "値は show_values=True で表示" in html


def test_case_level_values_appear_only_when_asked_for():
    df, sch, aud = with_findings()
    html = build_report(df, sch, audit=aud, show_values=True).to_html()
    assert df.loc[df.index[0], "仮名ID"] in html
    assert "症例レベルの値を含む" in html


def test_outlier_examples_are_dropped_unless_asked_for():
    from medprep.outliers import detect
    df = frame()
    df.loc[df.index[0], "年齢"] = 250.0
    ol = detect(df, columns=["年齢"], id_col="仮名ID")
    hidden = build_report(df, outliers=ol).to_html()
    shown = build_report(df, outliers=ol, show_values=True).to_html()
    assert "該当例" not in hidden
    assert "該当例" in shown


# ---------------------------------------------------------------- 体裁
def test_the_css_is_not_html_escaped():
    """★autoescape が引用符を壊すと、font-family ごと落ちて本文が明朝になる。★

    見た目だけの問題に見えて、実際には表の行が 4 倍の高さになるなど崩れる。
    """
    html = build_report(frame()).to_html()
    assert 'font-family: system-ui, -apple-system, "Hiragino Sans"' in html
    assert "&quot;Hiragino" not in html and "&#34;Hiragino" not in html


def test_table_cell_classes_do_not_collide_with_the_page_wrapper():
    """★クラス名の衝突。★

    ページ外枠の `.wrap`（padding 80px）がセルにも当たると、
    行が 4 倍の高さに膨らむ。表のクラスには接頭辞を付ける。
    """
    df, sch, aud = with_findings()
    html = build_report(df, sch, audit=aud).to_html()
    assert '<td class="t-wrap"' in html or '<td class="t-num"' in html
    assert '<td class="wrap"' not in html


def test_figures_are_embedded_as_data_uris():
    """図を別ファイルにすると、メールに添付した時点でばらける。"""
    df = frame()
    fs = viz.FigureSet()
    fs.add("分布", "説明", lambda: viz.distributions(df, columns=["年齢"]))
    html = build_report(df, figures=fs).to_html()
    assert "data:image/png;base64," in html
    assert 'alt="分布"' in html


def test_long_text_wraps_and_short_text_does_not():
    rep = Report()
    s = rep.section("試験")
    s.table(pd.DataFrame({"短": ["12.3"], "長": ["あ" * 40]}))
    html = rep.to_html()
    assert '<td class="t-num">12.3</td>' in html
    assert '<td class="t-wrap">' in html


def test_content_is_escaped():
    rep = Report()
    rep.section("試験").table(pd.DataFrame({"列": ["<script>alert(1)</script>"]}))
    html = rep.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_missing_values_in_a_table_do_not_print_nan():
    rep = Report()
    rep.section("試験").table(pd.DataFrame({"a": [1.0, np.nan]}))
    html = rep.to_html()
    assert "nan" not in html.lower().split("<tbody>")[1].split("</tbody>")[0]


# ---------------------------------------------------------------- 出力
def test_html_is_written_to_disk(tmp_path):
    p = tmp_path / "r.html"
    build_report(frame(), path=p)
    assert p.exists() and p.stat().st_size > 2000
    assert p.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_excel_gets_one_sheet_per_table(tmp_path):
    df, sch, aud = with_findings()
    rep = build_report(df, sch, audit=aud)
    p = rep.to_excel(tmp_path / "t.xlsx")
    sheets = pd.read_excel(p, sheet_name=None)
    assert len(sheets) >= 1


def test_the_footer_states_what_reproduces_the_result():
    html = build_report(frame()).to_html()
    assert "schema.yaml" in html and "再現" in html


def test_sections_can_be_built_by_hand():
    rep = Report(title="手書き")
    s = rep.section("節")
    s.text("本文").pre("そのまま出す\n2 行目").text("注記", kind="note")
    html = rep.to_html()
    assert "手書き" in html and "<pre>" in html and 'class="note"' in html
