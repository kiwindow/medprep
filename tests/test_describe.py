"""記述統計・Table 1・群間比較。

確かめたいのは「表が出ること」ではなく、**正しい検定が選ばれること**と
**選んだ理由が残ること**である。医学論文の Table 1 は、数字より先に
「なぜこの検定か」を問われる。
"""
import numpy as np
import pandas as pd
import pytest

from medprep.describe import (
    CATEGORICAL,
    CONTINUOUS,
    _adjust,
    cliffs_delta,
    compare_groups,
    dunn_test,
    epsilon_squared,
    eta_squared,
    hedges_g,
    normality,
    smd_categorical,
    smd_continuous,
    table_one,
    target_achievement,
    target_summary,
)
from medprep.schema import Schema

rng = np.random.default_rng(20260918)


def two_group(n=60, shift=0.0, lognormal=False):
    g = ["A"] * n + ["B"] * n
    if lognormal:
        v = np.concatenate([rng.lognormal(0, 1, n), rng.lognormal(shift, 1, n)])
    else:
        v = np.concatenate([rng.normal(10, 2, n), rng.normal(10 + shift, 2, n)])
    return pd.DataFrame({"群": g, "値": v})


def three_group(n=50, lognormal=False):
    rows = []
    for i, g in enumerate("ABC"):
        v = rng.lognormal(i * 0.5, 0.8, n) if lognormal else rng.normal(10 + i * 2, 2, n)
        rows.append(pd.DataFrame({"群": g, "値": v}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 正規性
def test_normal_data_is_called_normal():
    r = normality(rng.normal(10, 2, 300))
    assert r.is_normal and r.reason


def test_lognormal_data_is_called_nonnormal():
    r = normality(rng.lognormal(0, 1, 300))
    assert not r.is_normal
    assert "歪度" in r.reason


def test_the_reason_names_only_the_criterion_that_actually_fired():
    """歪んでいるが尖ってはいない分布に「尖度も大きい」と書かない。"""
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(3, 1, 120)])   # 右に裾
    r = normality(x)
    if not r.is_normal and r.kurtosis < 1.0:
        assert "過剰尖度" not in r.reason


def test_large_n_with_a_tiny_deviation_stays_normal():
    """★検定の棄却だけで非正規にしない。★

    n が大きいと Shapiro-Wilk は実用上どうでもよい歪みでも棄却する。
    そこで中央値[Q1,Q3] に切り替えてしまうと、読みにくいだけで何も改善しない。
    決めたいのは「平均と中央値のどちらが忠実か」である。
    """
    x = np.concatenate([rng.normal(10, 2, 3000), [10.05] * 400])   # 山を少し尖らせる
    r = normality(x)
    assert r.p < 0.05          # 検定は棄却する
    assert r.is_normal         # それでも平均(SD) で表示する
    assert "大標本では" in r.reason


def test_constant_and_tiny_samples_do_not_crash():
    assert normality([5, 5, 5, 5]).is_normal
    assert normality([1, 2]).n == 2
    r = normality([1, 2, 3, 100])
    assert "検出力" in r.reason


# ---------------------------------------------------------------- 効果量
def test_hedges_g_matches_the_hand_calculation():
    g = hedges_g([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert g == pytest.approx(-1.1425, abs=1e-3)


def test_cliffs_delta_is_one_when_groups_do_not_overlap():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_eta_and_epsilon_squared_are_between_zero_and_one():
    e = eta_squared([[1, 2, 3], [7, 8, 9], [14, 15, 16]])
    assert 0.9 < e <= 1.0
    assert eta_squared([[1, 2, 3], [1, 2, 3]]) == pytest.approx(0.0, abs=1e-12)
    assert epsilon_squared(10.0, 30, 3) == pytest.approx(8 / 27)
    # 群間差が偶然より小さいと分子が負になる。効果量に負の値は出さない。
    assert epsilon_squared(0.5, 30, 3) == 0.0


# ---------------------------------------------------------------- SMD
def test_smd_continuous_matches_the_definition():
    a, b = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
    assert smd_continuous(a, b) == pytest.approx(-2 / np.sqrt(2.5))


def test_smd_binary_matches_the_closed_form():
    a = pd.Series(["y"] * 60 + ["n"] * 40)
    b = pd.Series(["y"] * 30 + ["n"] * 70)
    assert smd_categorical(a, b) == pytest.approx(0.3 / np.sqrt(0.225), rel=1e-6)


def test_smd_is_zero_for_identical_groups():
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    assert smd_categorical(a, a.copy()) == pytest.approx(0.0, abs=1e-9)
    assert smd_continuous([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)


# ---------------------------------------------------------------- 多重比較
def test_bh_q_is_never_smaller_than_p():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    q = _adjust(p, "bh")
    assert (q >= p - 1e-12).all() and (q <= 1.0).all()


def test_adjust_keeps_nan_as_nan():
    q = _adjust([0.01, np.nan, 0.5], "bh")
    assert np.isnan(q[1]) and np.isfinite(q[0])


def test_dunn_separates_the_groups_that_differ():
    d = dunn_test({"A": list(range(20)), "B": list(range(20)),
                   "C": list(range(100, 120))})
    ph = {(r["群1"], r["群2"]): r["p補正"] for _, r in d.iterrows()}
    assert ph[("A", "B")] > 0.5          # 同じ分布
    assert ph[("A", "C")] < 0.01         # はっきり違う


# ---------------------------------------------------------------- 検定の選択
def _one(df, **kw):
    return compare_groups(df, "群", ["値"], **kw).comparisons[0]


def test_two_normal_groups_use_welch():
    c = _one(two_group(shift=1.5))
    assert c.test == "Welch の t 検定"
    assert "Welch" in c.reason and c.effect_name == "Hedges' g"


def test_two_nonnormal_groups_use_mann_whitney():
    c = _one(two_group(shift=1.0, lognormal=True))
    assert c.test == "Mann-Whitney U 検定"
    assert c.effect_name == "Cliff's δ"


def test_three_normal_groups_use_anova_then_tukey():
    c = _one(three_group())
    assert c.test == "一元配置分散分析" and c.effect_name == "η²"
    assert c.posthoc is not None and set(c.posthoc["検定"]) == {"Tukey HSD"}
    assert len(c.posthoc) == 3                       # 3 群なら 3 対


def test_three_nonnormal_groups_use_kruskal_then_dunn():
    c = _one(three_group(lognormal=True))
    assert c.test == "Kruskal-Wallis 検定" and c.effect_name == "ε²"
    assert set(c.posthoc["検定"]) == {"Dunn"}
    assert set(c.posthoc["補正法"]) == {"HOLM"}


def test_categorical_uses_chi_square():
    df = pd.DataFrame({"群": ["A"] * 100 + ["B"] * 100,
                       "値": ["x"] * 60 + ["y"] * 40 + ["x"] * 30 + ["y"] * 70})
    c = _one(df)
    assert c.test == "χ² 検定" and c.kind == CATEGORICAL
    assert c.effect_name == "Cramér's V"


def test_sparse_table_switches_to_fisher():
    df = pd.DataFrame({"群": ["A"] * 10 + ["B"] * 10,
                       "値": ["x"] * 9 + ["y"] + ["x"] * 10})
    c = _one(df)
    assert c.test in ("Fisher 正確検定", "χ² 検定")
    assert "期待度数" in c.reason


def test_cramers_v_of_zero_is_reported_as_zero_not_missing():
    """★`cramers_v(...) or nan` と書くと 0.0 が NaN になる。★

    関連がまったく無い（V=0）ことと、算出できなかったことは別である。
    """
    # 2×2 のどのセルも 25 例（＝まったく関連が無い）
    df = pd.DataFrame({"群": ["A"] * 50 + ["B"] * 50,
                       "値": (["x"] * 25 + ["y"] * 25) * 2})
    c = _one(df)
    assert np.isfinite(c.effect)
    assert c.effect == pytest.approx(0.0, abs=1e-9)


def test_paired_comparison_uses_paired_tests():
    n = 40
    base = rng.normal(10, 2, n)
    df = pd.DataFrame({
        "ID": list(range(n)) * 2,
        "群": ["前"] * n + ["後"] * n,
        "値": np.concatenate([base, base + 1.0 + rng.normal(0, 0.3, n)]),
    })
    c = compare_groups(df, "群", ["値"], paired_by="ID").comparisons[0]
    assert "対応のある" in c.test or "Wilcoxon" in c.test
    assert "対応" in c.reason and "40 組" in c.note


def test_unpaired_fallback_is_announced_when_no_pairs_match():
    df = two_group(n=30)
    df["ID"] = range(len(df))            # 対応がつかない
    c = compare_groups(df, "群", ["値"], paired_by="ID").comparisons[0]
    assert "対応なしとして扱った" in c.note


def test_every_comparison_carries_a_reason():
    df = three_group()
    df["cat"] = rng.choice(["x", "y"], len(df))
    r = compare_groups(df, "群", ["値", "cat"])
    for c in r.comparisons:
        assert c.reason and c.reason.strip(), f"{c.name}: reason が空"


def test_q_values_are_added_across_the_columns():
    df = three_group()
    for i in range(5):
        df[f"v{i}"] = rng.normal(0, 1, len(df))
    r = compare_groups(df, "群", ["値"] + [f"v{i}" for i in range(5)])
    assert all(np.isfinite(c.q) for c in r.comparisons)
    assert all(c.q >= c.p - 1e-12 for c in r.comparisons)
    assert any("補正" in n for n in r.notes)


def test_missing_group_column_raises():
    with pytest.raises(ValueError, match="群分け列"):
        compare_groups(two_group(), "無い列")


def test_single_level_group_is_reported_not_crashed():
    df = pd.DataFrame({"群": ["A"] * 10, "値": range(10)})
    r = compare_groups(df, "群", ["値"])
    assert r.comparisons == [] and any("水準が" in n for n in r.notes)


def test_posthoc_frame_stacks_tukey_and_dunn_with_the_same_columns():
    df = three_group()
    df["skewed"] = rng.lognormal(0, 1, len(df)) + (df["群"] == "C") * 5
    r = compare_groups(df, "群", ["値", "skewed"])
    ph = r.posthoc_frame()
    assert set(ph["検定"]) == {"Tukey HSD", "Dunn"}
    assert list(ph.columns)[:3] == ["項目", "群1", "群2"]
    assert ph["p補正"].notna().all()


# ---------------------------------------------------------------- Table 1
def cohort(n=200):
    fac = rng.choice(["A院", "B院"], n)
    return pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": fac,
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "C反応性蛋白(CRP)定量": rng.lognormal(-1.2, 1.3, n).round(2),
        "無機リン(P)": rng.normal(5.2, 1.3, n).round(1),
    })


def test_table_one_has_an_n_row_and_group_columns():
    df = cohort()
    t = table_one(df, groupby="施設")
    assert t.table.iloc[0]["項目"] == "n"
    assert set(t.groups) == {"A院", "B院"}
    assert t.table.iloc[0]["A院"] == str(int((df["施設"] == "A院").sum()))
    assert t.table.iloc[0]["全体"] == str(len(df))


def test_table_one_switches_the_summary_by_normality():
    t = table_one(cohort(), groupby="施設")
    items = list(t.table["項目"])
    assert any("年齢" in i and "平均 (SD)" in i for i in items)
    assert any("CRP" in i and "中央値 [Q1, Q3]" in i for i in items)


def test_table_one_lists_each_category_level_on_its_own_row():
    t = table_one(cohort(), groupby="施設")
    items = list(t.table["項目"])
    assert any(i.startswith("性別") and "n (%)" in i for i in items)
    assert "　男" in items and "　女" in items


def test_table_one_reports_missing_counts():
    df = cohort()
    df.loc[:9, "年齢"] = np.nan
    t = table_one(df, groupby="施設")
    row = t.table[t.table["項目"].str.startswith("年齢")].iloc[0]
    assert int(row["欠測"]) == 10


def test_table_one_always_warns_about_baseline_p_values():
    """★ベースライン表の p 値の解釈は、表と一緒に配らないと誤読される。★"""
    t = table_one(cohort(), groupby="施設")
    assert any("SMD" in n and "p" in n for n in t.notes)
    assert "SMD" in t.table.columns


def test_table_one_without_a_group_has_no_test_columns():
    t = table_one(cohort())
    assert "p" not in t.table.columns and "SMD" not in t.table.columns
    assert t.comparison is None


def test_table_one_unit_and_timing_appear_in_the_label():
    """透析前 BUN と透析後 BUN が同じ表に並ぶので、時点を書かないと読み違える。"""
    df = cohort()
    df["透析前BUN"] = rng.normal(65, 10, len(df))
    df["透析後BUN"] = rng.normal(20, 5, len(df))
    t = table_one(df, groupby="施設")
    items = " ".join(t.table["項目"])
    assert "（透析前）" in items and "（透析後）" in items
    assert "[mg/dL]" in items


def test_table_one_writes_excel_and_html(tmp_path):
    t = table_one(cohort(), groupby="施設")
    p = t.to_excel(tmp_path / "table1.xlsx")
    assert p.exists()
    sheets = pd.read_excel(p, sheet_name=None)
    assert {"Table1", "群間比較", "注記"} <= set(sheets)
    html = t.to_html(tmp_path / "table1.html")
    assert "<table" in html and "Table 1" in html


def test_table_one_report_is_printable():
    s = str(table_one(cohort(), groupby="施設"))
    assert "Table 1" in s and "施設" in s


# ---------------------------------------------------------------- 管理目標
def test_target_achievement_finds_the_columns_by_itself():
    df = cohort()
    a = target_achievement(df)
    assert len(a) and "無機リン" in set(a["項目"])
    assert (a["出典"].astype(str).str.len() > 0).all()


def test_target_achievement_always_shows_the_boundary_cases():
    """P 5.5 ちょうどは目標範囲に**含まれない**。境界の扱いを必ず見せる。"""
    df = pd.DataFrame({"無機リン(P)": [5.5] * 10 + [4.0] * 10})
    a = target_achievement(df)
    edge = a[a["群"].str.contains("境界")]
    assert len(edge)
    row = edge[edge["群"].str.contains("5.5")].iloc[0]
    assert row["目標"] == "含まない" and row["達成率"] == "0.0%"


def test_target_achievement_by_group():
    df = cohort()
    a = target_achievement(df, by="施設")
    assert {"全体", "A院", "B院"} <= set(a["群"])


def test_target_summary_lists_the_targets_with_sources():
    s = target_summary(cohort())
    assert "管理目標" in s.columns and len(s)
    assert all("未満" in v or "以上" in v or "以下" in v for v in s["管理目標"])


# ---------------------------------------------------------------- schema 連携
def test_table_one_drops_what_the_schema_dropped():
    df = cohort()
    df["備考"] = [f"メモ{i}" for i in range(len(df))]
    sch = Schema.infer(df, id_col="仮名ID", group="施設")
    t = table_one(df, sch, groupby="施設")
    items = " ".join(t.table["項目"])
    assert "仮名ID" not in items and "備考" not in items
    assert "年齢" in items


def test_compare_groups_uses_the_schema_kinds():
    df = cohort()
    sch = Schema.infer(df, id_col="仮名ID", group="施設")
    r = compare_groups(df, "施設", schema=sch)
    kinds = {c.name.split(" ")[0]: c.kind for c in r.comparisons}
    assert kinds["性別"] == CATEGORICAL
    assert any(k.startswith("年齢") and v == CONTINUOUS for k, v in kinds.items())


def test_a_free_text_column_does_not_explode_table_one():
    """★水準ごとに 1 行ずつ出すと、自由記載の列で表が数百行になる。★

    そうなると Table 1 は読めず、χ² も意味を持たない。1 行にまとめて、
    何が起きたかを注記に残す。
    """
    n = 120
    df = pd.DataFrame({
        "群": ["A"] * 60 + ["B"] * 60,
        "備考": [f"自由記載_{i}" for i in range(n)],
        "年齢": np.linspace(50, 85, n),
    })
    t = table_one(df, groupby="群", columns=["備考", "年齢"])
    assert len(t.table) < 10
    assert any("水準" in n_ and "備考" in n_ for n_ in t.notes)


def test_a_normal_categorical_column_still_shows_its_levels():
    df = pd.DataFrame({"群": ["A"] * 30 + ["B"] * 30,
                       "施設": (["X", "Y", "Z"] * 20)})
    t = table_one(df, groupby="群", columns=["施設"])
    items = t.table["項目"].tolist()
    assert any("X" in str(x) for x in items)
