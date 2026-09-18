"""保存先 — ★既存教材の `run{N}` 規約と 1 つもずれていないこと。★

受講者の PC には既に run1, run2, ... が溜まっている。
採番がノートブックと 1 つでもずれれば、**過去の計算結果を上書きする。**
ここで確かめるのは体裁ではなく、その一致である。
"""
import csv
import json
import os

import pytest

from medprep import paths


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "lab_output"))
    monkeypatch.setattr(paths, "WORK_DIR", str(tmp_path / "lab_work"))
    monkeypatch.setattr(paths, "IN_COLAB", False)
    monkeypatch.setattr(paths, "_LAST", {})
    return tmp_path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")


# ---------------------------------------------------------------- 採番
def test_the_first_run_is_run1():
    assert paths.new_run("Preprocessing").runnumber == 1


def test_the_same_session_never_reuses_a_run(monkeypatch):
    """★同じセッションで 2 回呼べば必ず run が進む。★

    ノートブックの `runnumber += 1` と同じ振る舞いにしてある。
    1 回目の結果をまだ書いていなくても、2 回目は別の run に入れる。
    """
    a = paths.new_run("Preprocessing")
    b = paths.new_run("Preprocessing")
    assert (a.runnumber, b.runnumber) == (1, 2)


def test_a_new_session_reuses_an_empty_run(monkeypatch):
    """逆に、中身の無い run はセッションをまたげば使い回す（空の run を増やさない）。"""
    a = paths.new_run("Preprocessing")
    monkeypatch.setattr(paths, "_LAST", {})          # ＝カーネルを作り直した状態
    assert paths.new_run("Preprocessing").runnumber == a.runnumber


def test_a_used_run_is_never_overwritten():
    a = paths.new_run("Preprocessing")
    touch(a.file("table", "x.xlsx"))
    assert paths.new_run("Preprocessing").runnumber == 2


def test_it_starts_after_the_highest_existing_run(isolated):
    """★Colab はランタイムを作り直すたび runnumber が 1 に戻る。★

    「1 から空きを探す」と、途中に空の run が残っていればそこへ戻って上書きする。
    実際にある run 番号の最大値を下限にする。
    """
    d = isolated / "lab_output" / "Preprocessing"
    for n in (1, 2, 3):
        touch(str(d / f"run{n}" / "table" / "a.xlsx"))
    assert paths.new_run("Preprocessing").runnumber == 4


def test_a_run_the_notebook_made_is_seen_as_used(isolated):
    """D&Dアプリ・ノートブックが作った run を medprep が踏まないこと。"""
    touch(str(isolated / "lab_output" / "Preprocessing" / "run5"
              / "table" / "run_info.json"))
    assert paths.new_run("Preprocessing").runnumber == 6


def test_report_alone_does_not_mark_a_run_as_used(isolated):
    """★採番に report を数えてはならない。★

    既存ノートブックは table / model / figure だけを見る。medprep が report を
    数えると、同じ run を片方は「使用済み」、片方は「空き」と判断する。
    """
    d = str(isolated / "lab_output" / "Preprocessing")
    touch(os.path.join(d, "run3", "report", "a.html"))
    assert "report" not in paths.RUN_FOLDERS
    assert paths.determine_runnumber(d, paths.RUN_FOLDERS, 3) == 3


def test_unreadable_run_is_treated_as_used(tmp_path):
    """読めないときは上書きを避ける（`_used` が True を返す）。"""
    d = tmp_path / "p" / "run1" / "table"
    d.mkdir(parents=True)
    (d / "a").write_text("x")
    assert paths.determine_runnumber(str(tmp_path / "p"), ["table"], 1) == 2


def test_runs_are_separate_per_method():
    a = paths.new_run("Preprocessing")
    touch(a.file("table", "x"))
    assert paths.new_run("Classification").runnumber == 1


# ---------------------------------------------------------------- 形
def test_the_four_folders_are_created():
    p = paths.new_run("Preprocessing")
    for name in ("table", "model", "figure", "report"):
        assert os.path.isdir(getattr(p, name))
    assert p.run.endswith(os.path.join("Preprocessing", "run1"))


def test_create_false_touches_nothing():
    p = paths.new_run("Preprocessing", create=False)
    assert not os.path.exists(p.run)


def test_file_rejects_an_unknown_folder():
    p = paths.new_run("Preprocessing")
    with pytest.raises(ValueError, match="kind は"):
        p.file("そんなフォルダ", "a.txt")


def test_colab_puts_everything_under_the_ai_folder(monkeypatch):
    monkeypatch.setattr(paths, "IN_COLAB", True)
    monkeypatch.setattr(paths, "OUTPUT_DIR", "")
    monkeypatch.setattr(paths, "WORK_DIR", "")
    assert paths.resolve_output_dir() == "/content/drive/MyDrive/AI/lab_output"
    assert paths.resolve_log_dir() == "/content/drive/MyDrive/AI/lab_work/log"


def test_the_results_folder_is_gitignored(isolated):
    """★run には個票由来の出力が集まる。git に載せる事故を防ぐ。★"""
    paths.new_run("Preprocessing")
    gi = isolated / "lab_output" / ".gitignore"
    assert gi.exists() and gi.read_text(encoding="utf-8").strip().endswith("*")


# ---------------------------------------------------------------- 互換
def test_run_info_keeps_the_keys_the_launcher_reads():
    """★D&Dアプリ（papermill ランチャ）が読むキー。減らしてはならない。★"""
    p = paths.new_run("Preprocessing")
    info = load(paths.write_run_info(p))
    for key in ("runnumber", "project_folder", "project_name", "project_directory",
                "path_run", "path_table", "path_model", "path_figure"):
        assert key in info
    assert info["runnumber"] == p.runnumber


def test_history_csv_keeps_the_notebook_columns():
    """★history.csv は log/*.json から毎回作り直す。★

    medprep の列だけで書き直すと、**ノートブックの『最良R2』が消える。**
    列は既存定義 ＋ JSON にある未知の列、の順にする。
    """
    p = paths.new_run("Preprocessing")
    with open(os.path.join(p.log_dir, "RegressionHoldOut_run3_x.json"),
              "w", encoding="utf-8") as f:
        json.dump({"開始日時": "2026-09-01 10:00", "手法": "RegressionHoldOut",
                   "run": 3, "最良モデル(R2)": "RandomForest", "最良R2": 0.81}, f,
                  ensure_ascii=False)
    log = paths.RunLog.start(p, 入力ファイル名="t.xlsx")
    log.finish(**{"所要時間(秒)": 1.2})

    with open(os.path.join(p.log_dir, "history.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0][:len(paths.LOG_CSV_COLUMNS)] == paths.LOG_CSV_COLUMNS
    body = {r[1]: r for r in rows[1:]}
    assert body["RegressionHoldOut"][rows[0].index("最良R2")] == "0.81"
    assert body["Preprocessing"][rows[0].index("所要時間(秒)")] == "1.2"


def test_history_csv_is_excel_readable():
    """utf-8-sig でないと Excel が文字化けする。"""
    p = paths.new_run("Preprocessing")
    paths.RunLog.start(p).finish()
    with open(os.path.join(p.log_dir, "history.csv"), "rb") as f:
        raw = f.read()
    assert raw.startswith(b"\xef\xbb\xbf")


def test_the_log_records_the_run_and_its_state():
    p = paths.new_run("Preprocessing")
    log = paths.RunLog.start(p, 入力ファイル名="t.xlsx")
    assert load(log.path)["状態"] == "実行中"
    log.finish()
    d = load(log.path)
    assert d["状態"] == "完了" and d["run"] == p.runnumber
    assert d["出力フォルダ"] == p.run


def test_the_copied_functions_keep_their_signature():
    """★ノートブックからの写し。引数を変えると D&Dアプリが壊れる。★"""
    import inspect
    assert list(inspect.signature(paths.determine_runnumber).parameters) == [
        "project_directory", "folder_names", "runnumber"]
    assert list(inspect.signature(paths.create_directory).parameters) == [
        "path_run", "folder_name"]


def test_colab_paths_use_forward_slashes_on_every_os(monkeypatch):
    """★`os.path.join` で Colab のパスを組んではならない。★

    Windows の Python から呼ぶと `…/AI\\lab_output` になる。
    Colab の保存先は常に POSIX である。
    """
    monkeypatch.setattr(paths, "IN_COLAB", True)
    monkeypatch.setattr(paths, "OUTPUT_DIR", "")
    monkeypatch.setattr(paths, "WORK_DIR", "")
    for f in (paths.resolve_output_dir, paths.resolve_work_dir, paths.resolve_log_dir):
        assert "\\" not in f(), f
