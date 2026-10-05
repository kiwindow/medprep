"""0.13.0 — 入力ミスの可能性がある値（A）と、人が決めた範囲 plausible（B）。

きっかけ: California Housing（N=500）の AveOccup に 230（ほかは 1.4〜5.8）が 1 つ入っていた。
辞書に無い列なので掃除の段を素通りし、IQR の印 8 件の中に埋もれた。k-fold CV の 1 fold で
線形回帰が −96.8 を予測し、R² が −64 になった。

  A. ほかの値から桁違いに離れた値を「人の確認が要る事項」に出す（直さない）
     - 右に裾の長い本物の極端値（人口・CRP）は出さない
     - 同じ誤りが 2 つ（999 が 2 例）でも出す。小さい側（−999）も出す
     - 値は既定で伏せる。行番号は出す
  B. 人が schema.yaml（または plausible=）に書いた範囲の外を NaN にして補完へ回す
     - 行は消さない。境界ちょうどは範囲の中。数値でない文字列には触らない
     - schema.yaml に残り、autoprep(schema=...) と mp.reproduce で再現できる
     - 引数と schema.yaml が食い違えば schema.yaml を使い、そう知らせる
"""
import numpy as np
import pandas as pd
import pytest
import yaml

import medprep as mp
from medprep import paths
from medprep.clean import USER_RANGE_ACTION, apply_ranges, excel_row, normalize_range
from medprep.outliers import detect, entry_error_messages, robust_z, suspect_entry_errors
from medprep.report import _EXAMPLE_RE

COL = "通院距離km"          # 辞書に無い列（掃除の段では範囲を見ない）


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "lab_output"))
    monkeypatch.setattr(paths, "WORK_DIR", str(tmp_path / "lab_work"))
    monkeypatch.setattr(paths, "IN_COLAB", False)
    monkeypatch.setattr(paths, "_LAST", {})
    return tmp_path


def frame(n=200, typo_at=37, typo=230.0):
    rng = np.random.default_rng(20261005)     # ★毎回同じデータ（再実行の比較に要る）★
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        COL: rng.gamma(4.0, 1.5, n).round(2),           # 右に裾が長いが、なだらか
        "世帯人口": rng.lognormal(7, 0.7, n).round(0),     # 裾がもっと長い（本物の極端値）
        "転帰": rng.normal(10, 2, n).round(2),
    })
    if typo_at is not None:
        df.loc[typo_at, COL] = typo
    return df


def run(df=None, **kw):
    kw.setdefault("verbose", False)
    kw.setdefault("do_figures", False)
    return mp.autoprep(df if df is not None else frame(), id_col="仮名ID",
                       outcome="転帰", task="regression", **kw)


# ================================================================ A. 判定
def test_robust_z_is_not_pulled_by_the_outlier_itself():
    x = pd.Series(list(np.linspace(1, 5, 99)) + [230.0])
    z = robust_z(x)
    assert z.iloc[-1] > 100            # 平均と SD の z なら 10 程度にしかならない
    assert abs(z.iloc[:-1]).max() < 2


def test_robust_z_falls_back_when_mad_is_zero_and_gives_up_on_constants():
    x = pd.Series([1.0] * 60 + [2.0, 3.0, 50.0])     # 半数以上が同じ値 → MAD = 0
    assert robust_z(x).iloc[-1] > 10
    assert robust_z(pd.Series([3.0] * 50)).isna().all()


def test_a_single_typo_is_found_with_its_excel_row():
    df = frame()
    t = suspect_entry_errors(df, columns=[COL, "世帯人口", "年齢"])
    assert list(t["列"]) == [COL]
    assert t["Excel の行"].iloc[0] == 37 + 2          # 見出しが 1 行目
    assert t["値"].iloc[0] == 230.0
    assert t["向き"].iloc[0] == "大きい側"


def test_a_long_but_smooth_tail_is_not_called_an_error():
    """人口のような本物の裾は、z が大きくても次の値となだらかに続く。"""
    df = frame(typo_at=None)
    df.loc[0, "世帯人口"] = df["世帯人口"].max() * 1.3        # 少し飛び出た本物の値
    assert suspect_entry_errors(df, columns=["世帯人口", COL]).empty


def test_two_identical_codes_are_both_found_and_the_low_side_too():
    df = frame(typo_at=None)
    df.loc[[5, 9], "年齢"] = 999
    df.loc[12, COL] = -999
    t = suspect_entry_errors(df, columns=["年齢", COL])
    got = set(zip(t["列"], t["Excel の行"]))
    assert got == {("年齢", 7), ("年齢", 11), (COL, 14)}
    assert set(t.loc[t["列"] == COL, "向き"]) == {"小さい側"}


def test_columns_with_few_distinct_values_are_skipped():
    df = pd.DataFrame({"コード": [1, 2, 3] * 30 + [99]})
    assert suspect_entry_errors(df).empty


def test_thresholds_can_be_changed():
    df = frame()
    assert suspect_entry_errors(df, columns=[COL], z=1e6).empty
    assert suspect_entry_errors(df, columns=[COL], ratio=1e6).empty


def test_detect_carries_the_suspects_and_says_so():
    rep = detect(frame(), columns=[COL, "年齢"])
    assert len(rep.suspects) == 1
    assert any("入力ミスの可能性がある値" in n for n in rep.notes)
    assert "入力ミスの可能性がある値" in rep.report()


def test_the_message_puts_values_where_the_report_can_hide_them():
    t = suspect_entry_errors(frame(), columns=[COL])
    (m,) = entry_error_messages(t)
    assert "Excel の行 39" in m and "plausible" in m
    hidden = _EXAMPLE_RE.sub("（例は show_values=True で表示）", m)
    assert "230" not in hidden and "Excel の行 39" in hidden


# ================================================================ A. autoprep
def test_autoprep_lists_the_typo_among_the_things_to_check():
    rep = run()
    hits = [w for w in rep.warnings if "入力ミスの可能性がある値" in w]
    assert len(hits) == 1 and f"'{COL}'" in hits[0]
    assert rep.schema.policy["entry_error"] == {"z": 10.0, "ratio": 3.0}


def test_clean_data_raises_no_entry_warning():
    rep = run(frame(typo_at=None))
    assert not [w for w in rep.warnings if "入力ミスの可能性がある値" in w]


def test_the_report_hides_the_value_by_default_but_keeps_the_row():
    rep = run()
    html = rep.html.to_html() if hasattr(rep.html, "to_html") else str(rep.html)
    assert "入力ミスの可能性がある値" in html
    assert "Excel の行 39" in html
    assert "例: [230" not in html                       # 人の確認が要る事項では伏せる
    i = html.index("<caption>入力ミスの可能性がある値")    # 新しい表にも値の列は無い
    table = html[i:html.index("</table>", i)]
    assert "230" not in table and ">39<" in table


def test_policy_can_turn_the_threshold_up():
    rep = run(policy={"entry_error": {"z": 1e6, "ratio": 3.0}})
    assert not [w for w in rep.warnings if "入力ミスの可能性がある値" in w]


# ================================================================ B. 範囲
@pytest.mark.parametrize("spec,want", [
    ([0.5, 20], (0.5, 20.0)), ((0, None), (0.0, None)), ([None, 20], (None, 20.0)),
    ({"low": 1, "high": 2}, (1.0, 2.0)),
])
def test_normalize_range_accepts_the_usual_forms(spec, want):
    assert normalize_range(spec) == want


@pytest.mark.parametrize("spec", [[20, 0.5], [1, 1], [None, None], [1], "0.5-20", ["a", 2],
                                  [True, 3]])
def test_normalize_range_refuses_what_it_cannot_read(spec):
    with pytest.raises(ValueError):
        normalize_range(spec)


def test_apply_ranges_nans_only_what_is_outside_and_keeps_rows():
    df = pd.DataFrame({"x": [0.5, 3.0, 20.0, 230.0, 0.1, np.nan, "12 km"]})
    out, acts, msgs = apply_ranges(df, {"x": [0.5, 20]})
    assert len(out) == len(df)                                   # 行は消えない
    assert out["x"].tolist()[:3] == [0.5, 3.0, 20.0]             # 境界は範囲の中
    assert pd.isna(out.at[3, "x"]) and pd.isna(out.at[4, "x"])
    assert out.at[6, "x"] == "12 km"                             # 文字列には触らない
    assert acts == [("x", USER_RANGE_ACTION, 2, acts[0][3])]
    assert "Excel の行 5、6" in msgs[0]


def test_apply_ranges_reports_problems_instead_of_stopping():
    df = pd.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"]})
    _, acts, msgs = apply_ranges(df, {"無い列": [0, 1], "s": [0, 1], "x": [5, 1]})
    assert acts == []
    assert len(msgs) == 3 and all("★" in m for m in msgs)


def test_excel_row_only_for_integer_labels():
    assert excel_row(0) == 2 and excel_row(np.int64(332)) == 334
    assert excel_row("P01") is None and excel_row(True) is None


# ================================================================ B. autoprep
def test_plausible_argument_nans_the_typo_and_silences_the_suspect():
    rep = run(plausible={COL: [0, 100]})
    assert pd.isna(rep.df_clean.at[37, COL])
    assert not [w for w in rep.warnings if "入力ミスの可能性がある値" in w]
    assert any("NaN にした" in w and "Excel の行 39" in w for w in rep.warnings)
    assert rep.schema.columns[COL].plausible == [0.0, 100.0]
    r = rep.removed
    row = r[(r["種類"] == "値") & (r["対象"] == COL)]
    assert len(row) == 1 and row["件数"].iloc[0] == 1
    assert "人が決めた範囲" in row["段"].iloc[0]
    # 補完される（train か test のどちらかに入り、NaN のまま残らない）
    assert not rep.prepared.X_train.isna().any().any()
    assert not rep.prepared.X_test.isna().any().any()


def test_plausible_is_written_to_schema_yaml_and_used_again(tmp_path):
    rep = run(plausible={COL: [0, 100]})
    path = tmp_path / "schema.yaml"
    rep.schema.to_yaml(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert d["columns"][COL]["plausible"] == [0.0, 100.0]
    rep2 = mp.autoprep(frame(), schema=str(path), verbose=False, do_figures=False)
    assert pd.isna(rep2.df_clean.at[37, COL])
    pd.testing.assert_frame_equal(rep2.prepared.X_train, rep.prepared.X_train)


def test_a_range_written_by_hand_in_schema_yaml_is_used(tmp_path):
    """人が schema.yaml の列に 1 行書き足して再実行する、という本来の使い方。"""
    rep = run()
    path = tmp_path / "schema.yaml"
    rep.schema.to_yaml(path)
    text = path.read_text(encoding="utf-8").replace(
        f"  {COL}:\n", f"  {COL}:\n    plausible: [null, 100]\n", 1)
    path.write_text(text, encoding="utf-8")
    rep2 = mp.autoprep(frame(), schema=str(path), verbose=False, do_figures=False)
    assert pd.isna(rep2.df_clean.at[37, COL])
    assert any("plausible" in w and "schema.yaml で直した判断" in w for w in rep2.warnings)


def test_schema_yaml_wins_over_the_argument_and_says_so(tmp_path):
    rep = run(plausible={COL: [0, 100]})
    path = tmp_path / "schema.yaml"
    rep.schema.to_yaml(path)
    rep2 = mp.autoprep(frame(), schema=str(path), plausible={COL: [0, 1000]},
                       verbose=False, do_figures=False)
    assert rep2.schema.columns[COL].plausible == [0.0, 100.0]
    assert any("plausible の引数" in w for w in rep2.warnings)


def test_a_bad_range_is_reported_and_skipped():
    rep = run(plausible={COL: [100, 0], "無い列": [0, 1]})
    assert not pd.isna(rep.df_clean.at[37, COL])
    assert any(f"'{COL}' の plausible を使わなかった" in w for w in rep.warnings)
    assert any("'無い列' がデータに無い" in w for w in rep.warnings)


def test_ranges_apply_even_without_dictionary_cleaning():
    rep = run(clean=False, plausible={COL: [0, 100]})
    assert pd.isna(rep.df_clean.at[37, COL])


def test_reproduce_is_identical_with_a_range():
    rep = run(plausible={COL: [0, 100]}, save=True)
    rr = mp.reproduce(rep.run.run)
    assert rr.identical, rr.report()


def test_schema_diff_shows_the_range():
    a = run().schema
    b = run(plausible={COL: [0, 100]}).schema
    d = a.diff(b)
    assert ((d["列名"] == COL) & (d["項目"] == "plausible")).any()
