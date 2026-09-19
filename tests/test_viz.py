"""図。**色の役割を取り違えていないこと**を確かめる。

  相関（符号あり）  → 発散配色、中点は灰色
  関連の強さ（0〜1）→ 順次配色（単一色相の明→暗）

ここを取り違えると、中点の灰色が「関連なし」ではなく「負の関連」に見える。
色の選び方は検証済みの配色に固定してあり、**9 色目を作らない**。
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from medprep import viz  # noqa: E402
from medprep.schema import Schema  # noqa: E402

rng = np.random.default_rng(20260918)


def frame(n=300):
    return pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院", "C院"], n, p=[0.6, 0.3, 0.1]),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "Alb": rng.normal(3.6, 0.45, n).round(1),
        "CRP": rng.lognormal(-1, 1.2, n).round(2),
        "転帰": (rng.random(n) < 0.3).astype(int),
    })


def schema_of(df):
    return Schema.infer(df, id_col="仮名ID", group="施設",
                        outcome="転帰", task="classification")


# ---------------------------------------------------------------- 配色
def test_categorical_order_is_fixed_and_never_cycled():
    """★9 色目を作らない。★ 循環させた色は色覚特性のある読者に区別できない。"""
    p = viz.PALETTE
    assert len(p.categorical) == 8
    assert p.series(0) == p.categorical[0]
    with pytest.raises(ValueError, match="8 色まで"):
        p.series(8)


def test_scatter_forms_are_capped_at_three_colours():
    """散布図はすべての対が同時に見えるので 3 色まで。4 色目で黄と橙が並ぶ。"""
    p = viz.PALETTE
    cols = p.colors_for(["a", "b", "c", "d", "e"], all_pairs=True)
    used = [cols[k] for k in "abc"]
    assert used == list(p.categorical[:3])
    assert cols["d"] == cols["e"] == p.muted        # 4 つ目以降は色を割り当てない


def test_levels_are_folded_instead_of_adding_a_ninth_colour():
    s = pd.Series(list("abcdefghij") * 10)
    out, levels, folded = viz._fold_levels(s, 3)
    assert folded and levels[-1] == "その他" and len(levels) == 3
    assert set(out.unique()) == set(levels)


def test_few_levels_are_left_alone():
    s = pd.Series(["a", "b"] * 20)
    _out, levels, folded = viz._fold_levels(s, 8)
    assert not folded and set(levels) == {"a", "b"}


def test_the_diverging_midpoint_is_neutral_not_a_hue():
    """★中点に色相を置かない。★ 中点は『関連なし』でなければならない。"""
    mid = viz.diverging_cmap()(0.5)[:3]
    assert max(mid) - min(mid) < 0.05              # ほぼ無彩色
    lo, hi = viz.diverging_cmap()(0.0)[:3], viz.diverging_cmap()(1.0)[:3]
    assert lo[2] > lo[0] and hi[0] > hi[2]         # 片方が寒色、もう片方が暖色


def test_the_sequential_ramp_is_one_hue_getting_darker():
    cm = viz.sequential_cmap()
    lums = [sum(cm(x)[:3]) for x in np.linspace(0, 1, 7)]
    assert all(a > b for a, b in zip(lums, lums[1:]))   # 単調に暗くなる


# ---------------------------------------------------------------- 図
def test_distributions(tmp_path):
    fig = viz.distributions(frame(), schema_of(frame()), save=tmp_path / "d.png")
    assert (tmp_path / "d.png").exists() and fig is not None


def test_distributions_by_group():
    df = frame()
    fig = viz.distributions(df, schema_of(df), by="施設")
    assert fig is not None


def test_distributions_needs_numeric_columns():
    with pytest.raises(ValueError, match="連続変数が無い"):
        viz.distributions(pd.DataFrame({"a": ["x", "y"]}), columns=[])


def test_category_bars(tmp_path):
    df = frame()
    viz.category_bars(df, schema_of(df), save=tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()


def test_correlation_uses_the_diverging_map_and_is_symmetric():
    df = frame()
    fig = viz.correlation_heatmap(df, schema_of(df))
    im = fig.axes[0].images[0]
    assert im.get_cmap().name == "medprep_div"
    assert im.get_clim() == (-1, 1)                # 符号があるので −1〜1


def test_association_uses_the_sequential_map_on_zero_to_one():
    df = frame()
    fig = viz.association_heatmap(df, schema_of(df))
    im = fig.axes[0].images[0]
    assert im.get_cmap().name == "medprep_seq"
    assert im.get_clim() == (0, 1)                 # 大きさなので 0〜1


def test_association_handles_mixed_types():
    """数値×カテゴリは η²、カテゴリ×カテゴリは Cramér's V。"""
    df = frame()
    v = viz._association(df, "年齢", "施設", ["年齢", "Alb", "CRP", "転帰"])
    assert 0 <= v <= 1
    v2 = viz._association(df, "施設", "性別", ["年齢"])
    assert 0 <= v2 <= 1


def test_correlation_needs_two_numeric_columns():
    with pytest.raises(ValueError, match="2 つ以上"):
        viz.correlation_heatmap(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))


def test_outcome_relations_classification_and_regression():
    df = frame()
    assert viz.outcome_relations(df, "転帰", schema_of(df), task="classification")
    df["eGFR"] = rng.normal(30, 9, len(df))
    assert viz.outcome_relations(df, "eGFR", task="regression")


def test_outcome_relations_needs_the_column():
    with pytest.raises(ValueError, match="目的変数"):
        viz.outcome_relations(frame(), "無い列")


def test_univariate_auc_is_centred_on_one_half():
    df = frame()
    fig = viz.univariate_auc(df, "転帰", schema_of(df))
    ax = fig.axes[0]
    assert ax.get_xlim()[0] < 0.5 < ax.get_xlim()[1]


def test_univariate_auc_needs_a_binary_outcome():
    df = frame()
    df["連続"] = rng.normal(0, 1, len(df))
    with pytest.raises(ValueError, match="2 値でない"):
        viz.univariate_auc(df, "連続")


def test_pca_scatter(tmp_path):
    df = frame()
    viz.pca_scatter(df, schema_of(df), by="施設", save=tmp_path / "p.png")
    assert (tmp_path / "p.png").exists()


def test_smd_forest_marks_the_ones_over_the_threshold():
    b = pd.DataFrame({"列": ["a", "b", "c"], "SMD": [0.02, -0.31, 0.08]})
    fig = viz.smd_forest(b, threshold=0.1)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("偏り" in x for x in labels)        # 色だけでなくラベルでも示す


def test_smd_forest_needs_values():
    with pytest.raises(ValueError, match="SMD の値が無い"):
        viz.smd_forest(pd.DataFrame({"列": ["a"], "SMD": [np.nan]}))


def test_achievement_bars():
    a = pd.DataFrame({
        "項目": ["無機リン", "無機リン", "補正Ca", "補正Ca"],
        "群": ["全体", "A院", "全体", "A院"],
        "達成率": ["46.2%", "48.3%", "44.2%", "44.3%"],
    })
    fig = viz.achievement_bars(a)
    assert fig.axes[0].get_ylim() == (0, 100)


def test_missing_map(tmp_path):
    df = frame()
    df.loc[:20, "Alb"] = np.nan
    viz.missing_map(df, columns=["年齢", "Alb", "CRP"], save=tmp_path / "m.png")
    assert (tmp_path / "m.png").exists()
    with pytest.raises(ValueError, match="kind は"):
        viz.missing_map(df, kind="存在しない図")


# ---------------------------------------------------------------- まとめ
def test_overview_builds_what_it_can_and_records_what_it_cannot():
    df = frame()
    fs = viz.overview(df, schema_of(df))
    assert len(fs) >= 5
    titles = [t for t, _c, _f in fs.figures]
    assert "相関" in titles and "関連の強さ" in titles
    assert all(c for _t, c, _f in fs.figures)      # 説明が空の図を作らない


def test_overview_records_the_reason_a_figure_could_not_be_drawn():
    fs = viz.FigureSet()
    fs.add("壊れる図", "説明", lambda: (_ for _ in ()).throw(ValueError("理由")))
    assert len(fs) == 0 and fs.skipped[0][0] == "壊れる図"
    assert "理由" in fs.skipped[0][1]


def test_to_base64_produces_a_png():
    import base64
    fig = viz.distributions(frame(), columns=["年齢"])
    raw = base64.b64decode(viz.to_base64(fig))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_close_all_runs():
    viz.distributions(frame(), columns=["年齢"])
    viz.close_all()


# ---------------------------------------------------------------- 文字
def test_characters_missing_from_the_japanese_font_are_replaced():
    """★フォントに無い文字は例外を出さず、黙って豆腐（□）になる。★

    IPAexGothic には ≥ ≤ ✗ µ が無い。描く前に ≧ ≦ × μ に置き換える。
    """
    assert viz.safe_text("|SMD| ≥ 0.1") == "|SMD| ≧ 0.1"
    assert viz.safe_text("×10³/µL") == "×10³/μL"


def test_no_figure_contains_a_glyph_the_font_cannot_draw():
    b = pd.DataFrame({"列": ["a", "b"], "SMD": [0.02, -0.31]})
    for fig in (viz.smd_forest(b, threshold=0.1),
                viz.distributions(frame(), columns=["年齢"]),
                viz.correlation_heatmap(frame(), schema_of(frame()))):
        assert viz.missing_glyphs(fig) == set()


def test_missing_map_shrinks_its_labels_so_they_do_not_overlap():
    """★40 列の日本語の列名は、既定の大きさ・45 度では重なって読めない。★

    図としては描けているので例外は出ない。人が見るまで分からない壊れ方である。
    列数から幅と字の大きさを決め、多いときは縦書きにする。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from medprep.viz import fit_tick_labels, missing_map

    df = pd.DataFrame({f"とても長い検査項目名(No{i:02d})": [1.0, None, 2.0] * 10
                       for i in range(40)})
    fig = missing_map(df)
    # missingno は行列の軸とスパークラインの軸を作る。ラベルを持つほうを見る。
    ax = next(a for a in fig.axes if a.get_xticklabels())
    labels = ax.get_xticklabels()
    assert fig.get_size_inches()[0] >= 13          # 列数に応じて広げている
    assert all(t.get_rotation() == 90 for t in labels)
    sizes = {t.get_fontsize() for t in labels}
    assert len(sizes) == 1 and max(sizes) <= 13
    plt.close(fig)

    # 列が少なければ 45 度のままで、字も小さくしすぎない
    small = pd.DataFrame({"年齢": [1.0, None], "Alb": [2.0, 3.0]})
    fig2 = missing_map(small)
    ax2 = next(a for a in fig2.axes if a.get_xticklabels())
    assert fit_tick_labels(ax2) >= 4.5
    plt.close(fig2)
