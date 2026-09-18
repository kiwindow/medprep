"""層1（`mp.autoprep()`）。

確かめたいのは「1 行で走ること」ではない。次の 3 つである。

  1. **層1に固有のロジックが無い**（同じことを層2で 1 段ずつ再現できる）
  2. **判断がすべて外に出る**（`schema.yaml` と `rep.warnings`。空でないのがふつう）
  3. **リークしない**（train でしか fit していない）
"""
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import medprep as mp  # noqa: E402
from medprep import paths  # noqa: E402

rng = np.random.default_rng(20260918)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "lab_output"))
    monkeypatch.setattr(paths, "WORK_DIR", str(tmp_path / "lab_work"))
    monkeypatch.setattr(paths, "IN_COLAB", False)
    monkeypatch.setattr(paths, "_LAST", {})
    return tmp_path


def frame(n=200):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "アルブミン(Alb)": rng.normal(3.6, 0.45, n).round(1),
        "末梢血｜血色素量(Hb)": rng.normal(10.8, 1.2, n).round(1),
        "無機リン(P)": rng.normal(5.2, 1.1, n).round(1),
        "転帰": (rng.random(n) < 0.3).astype(int),
    })
    df.loc[df.index[:15], "アルブミン(Alb)"] = np.nan
    return df


def run(df=None, **kw):
    kw.setdefault("verbose", False)
    return mp.autoprep(df if df is not None else frame(),
                       id_col="仮名ID", group="施設", **kw)


# ---------------------------------------------------------------- 骨格
def test_one_line_gets_a_model_ready_matrix():
    rep = run(outcome="転帰", task="classification")
    assert rep.X_train is not None and rep.X_test is not None
    assert list(rep.X_train.columns) == list(rep.X_test.columns)
    assert len(rep.X_train) + len(rep.X_test) == 200
    assert rep.pipeline is rep.prepared.preprocessor


def test_every_step_is_recorded_with_its_outcome():
    """★飛ばした段は黙って消さない。★"""
    rep = run(outcome="転帰", task="classification")
    names = [n for n, _ok, _d in rep.steps]
    for expected in ("列の役割を推定する", "品質を監査する", "辞書で掃除する",
                     "train / test に分ける", "リークを検査する"):
        assert expected in names


def test_without_an_outcome_it_stops_before_splitting():
    rep = run()
    assert rep.split is None and rep.prepared is None
    assert rep.schema is not None and rep.audit is not None


def test_quicklook_does_not_split():
    rep = mp.quicklook(frame(), id_col="仮名ID", group="施設", verbose=False)
    assert rep.split is None and rep.table1 is not None


def test_it_reads_a_file_and_says_what_it_guessed(tmp_path):
    p = tmp_path / "c.csv"
    frame().to_csv(p, index=False, encoding="cp932")
    rep = run(str(p))
    assert len(rep.df_raw) == 200
    assert any("cp932" in d for _n, _ok, d in rep.steps)


# ---------------------------------------------------------------- 判断の明示化
def test_the_warnings_are_not_empty_for_dirty_data():
    """★『確認事項なし』で返ってくるほうを疑うこと。★"""
    df = frame()
    df.loc[df.index[1], "仮名ID"] = df.loc[df.index[0], "仮名ID"]     # 同じ ID が 2 行
    rep = run(df, outcome="転帰", task="classification")
    assert rep.warnings
    assert any("重複" in w or "識別子" in w for w in rep.warnings)


def test_every_fatal_finding_reaches_the_warnings():
    df = frame()
    df.loc[df.index[1], "仮名ID"] = df.loc[df.index[0], "仮名ID"]
    rep = run(df)
    assert len(rep.warnings) >= len(rep.audit.errors)


def test_the_schema_can_be_written_out_and_read_back(tmp_path):
    rep = run(outcome="転帰", task="classification")
    p = tmp_path / "schema.yaml"
    rep.schema.to_yaml(p)
    back = mp.Schema.from_yaml(p)
    assert set(back.columns) == set(rep.schema.columns)


def test_dropping_missing_outcome_is_reported_not_silent():
    """★目的変数は補完してはならない。除いた数を必ず言う。★"""
    df = frame()
    df.loc[df.index[:20], "転帰"] = np.nan
    rep = run(df, outcome="転帰", task="classification")
    assert len(rep.split.train) + len(rep.split.test) == 180
    assert any("目的変数" in w and "20" in w for w in rep.warnings)


# ---------------------------------------------------------------- リーク
def test_the_preprocessor_was_fitted_on_train_only():
    rep = run(outcome="転帰", task="classification")
    lc = mp.leak_check(rep.pipeline, rep.split.train, rep.split.test)
    assert (lc["結果"] == "OK").all()


def test_the_identifier_never_becomes_a_feature():
    rep = run(outcome="転帰", task="classification")
    assert "仮名ID" not in rep.X_train.columns
    assert "転帰" not in rep.X_train.columns


# ---------------------------------------------------------------- 飛ばし方
def test_a_failing_optional_step_is_skipped_with_its_reason(monkeypatch):
    """付随する段が落ちても止まらない。ただし理由を残す。"""
    from medprep import auto
    monkeypatch.setattr(auto, "target_achievement",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("わざと")))
    rep = run()
    skipped = [(n, d) for n, ok, d in rep.steps if not ok]
    assert any("管理目標" in n and "わざと" in d for n, d in skipped)


def test_a_failing_backbone_step_stops_everything(monkeypatch):
    """★背骨で失敗したら止まる。前処理だけ抜けた結果を返してはならない。★"""
    from medprep import auto
    monkeypatch.setattr(auto, "prepare",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("止まる")))
    with pytest.raises(ValueError, match="止まる"):
        run(outcome="転帰", task="classification")


# ---------------------------------------------------------------- 保存
def test_save_writes_the_run_folder(isolated):
    rep = run(outcome="転帰", task="classification", save=True)
    p = rep.run
    assert p.runnumber == 1
    for rel in (("model", "schema.yaml"), ("model", "pipeline.pkl"),
                ("model", "medprep_version.txt"), ("table", "run_info.json"),
                ("table", "prep_tables.xlsx"), ("report", "prep_report.html")):
        assert os.path.exists(p.file(*rel)), rel
    assert len(os.listdir(p.figure)) >= 5


def test_the_run_folder_is_gitignored(isolated):
    """★出力には個票由来の値が入りうる。git に載せない。★"""
    run(save=True)
    assert (isolated / "lab_output" / ".gitignore").exists()


def test_two_runs_do_not_collide(isolated):
    a = run(save=True)
    b = run(save=True)
    assert a.run.runnumber != b.run.runnumber


def test_the_log_records_the_run(isolated):
    run(outcome="転帰", task="classification", save=True)
    hist = isolated / "lab_work" / "log" / "history.csv"
    assert hist.exists()
    row = pd.read_csv(hist).iloc[0]
    assert row["手法"] == "Preprocessing" and row["状態"] == "完了"


def test_nothing_is_written_without_save(isolated):
    run(outcome="転帰", task="classification")
    assert not (isolated / "lab_output").exists()


# ---------------------------------------------------------------- レポート
def test_case_level_values_are_hidden_by_default():
    df = frame()
    df.loc[df.index[1], "仮名ID"] = df.loc[df.index[0], "仮名ID"]
    rep = run(df)
    assert df.loc[df.index[0], "仮名ID"] not in rep.html.to_html()
    assert df.loc[df.index[0], "仮名ID"] in rep.to_html(show_values=True).to_html()


def test_the_report_leads_with_the_warnings():
    df = frame()
    df.loc[df.index[1], "仮名ID"] = df.loc[df.index[0], "仮名ID"]
    rep = run(df)
    html = rep.html.to_html()
    assert html.index("まず読むこと") < html.index("データ品質監査")


def test_report_text_lists_the_steps_and_the_warnings():
    rep = run(outcome="転帰", task="classification")
    text = rep.report()
    assert "段:" in text and "人の確認が要る事項" in text


def test_to_excel_writes_one_sheet_per_table(tmp_path):
    rep = run()
    p = rep.to_excel(tmp_path / "t.xlsx")
    assert len(pd.read_excel(p, sheet_name=None)) >= 1
