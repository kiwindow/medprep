"""0.12.0 — schema.yaml による再実行と再現の検査。

  * schema.yaml に autoprep の設定（run 欄）がすべて残る
  * mp.reproduce(run) で、0〜6 のデータと列の判断が元と完全に一致する
  * 直した schema.yaml を autoprep(schema=...) に渡すと、直した所だけが変わる
  * schema.yaml の設定は引数より優先し、食い違いを「人の確認が要る事項」に出す
  * 0.11.x の schema.yaml（run 欄なし）も止まらずに読む
"""
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use("Agg")

import medprep as mp  # noqa: E402
from medprep import paths  # noqa: E402

rng = np.random.default_rng(20261003)
DATES = ("開始日", "発生日", "打切日")
ALB = "アルブミン(Alb)"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "lab_output"))
    monkeypatch.setattr(paths, "WORK_DIR", str(tmp_path / "lab_work"))
    monkeypatch.setattr(paths, "IN_COLAB", False)
    monkeypatch.setattr(paths, "_LAST", {})
    return tmp_path


def _frame(n=200):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院", "C院"], n),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        ALB: rng.normal(3.6, 0.45, n).round(1),
        "末梢血｜血色素量(Hb)": rng.normal(10.8, 1.2, n).round(1),
        "CRP定量": rng.lognormal(-1, 1, n).round(2),
        "備考": [f"メモ{i}" for i in range(n)],
    })
    df.loc[df.index[:15], ALB] = np.nan
    base = pd.Timestamp("2015-01-01")
    df["開始日"] = base + pd.to_timedelta(rng.integers(0, 500, n), "D")
    df["発生日"] = df["開始日"] + pd.to_timedelta(rng.integers(30, 900, n), "D")
    df["打切日"] = pd.Series([""] * len(df), index=df.index, dtype=object)  # pandas 3 でも日付を代入できる
    df.loc[df.index[::2], "発生日"] = ""
    df.loc[df.index[::2], "打切日"] = df.loc[df.index[::2], "開始日"] + pd.Timedelta(days=700)
    return df


KW = {"id_col": "仮名ID", "group": "施設", "survival_dates": DATES, "outcome": ALB,
      "task": "regression", "seed": 7, "test_size": 0.25, "do_figures": False,
      "verbose": False}


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(d, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True)


def _first(df=None, **kw):
    return mp.autoprep(_frame() if df is None else df, save=True, **{**KW, **kw})


def _schema_path(rep):
    return rep.run.file("model", "schema.yaml")


def test_run_block_records_every_setting():
    rep = _first()
    run = mp.Schema.from_yaml(_schema_path(rep)).run
    assert run["seed"] == 7 and run["test_size"] == 0.25
    assert run["group"] == "施設" and run["id_col"] == "仮名ID"
    assert run["outcome"] == ALB and run["task"] == "regression"
    assert run["survival_dates"] == list(DATES)
    assert run["medprep"] == f"medprep {mp.__version__}"
    for k in mp.rerun.RUN_DEFAULTS:
        assert k in run, k
    with open(rep.run.file("model", "medprep_version.txt"), encoding="utf-8") as f:
        ver = f.read()
    assert f"@v{mp.__version__}" in ver


def test_reproduce_is_identical():
    rep = _first()
    rr = mp.reproduce(rep.run.run)
    assert rr.identical, rr.report()
    files = set(rr.table["対象"])
    assert "data/3_機械学習用データ.xlsx" in files
    assert "data/5_training_data_本コード専用.xlsx" in files
    assert rr.reproduced != rr.original


def test_edited_schema_changes_only_what_was_edited():
    rep = _first()
    d = _read_yaml(_schema_path(rep))
    d["columns"]["末梢血｜血色素量(Hb)"]["action"] = "drop"
    d["columns"]["末梢血｜血色素量(Hb)"]["reason"] = "人が外した（測定法が途中で変わった）"
    edited = os.path.join(os.path.dirname(_schema_path(rep)), "schema_edited.yaml")
    _write_yaml(d, edited)

    rep2 = mp.autoprep(_frame(), schema=edited, save=True, verbose=False, do_figures=False)
    assert not any("血色素" in c for c in rep2.X_train.columns)
    assert any("血色素" in c for c in rep.X_train.columns)
    assert rep2.schema.columns["末梢血｜血色素量(Hb)"].reason.startswith("人が外した")
    edit = [w for w in rep2.warnings if "schema.yaml で直した判断" in w]
    assert len(edit) == 1 and "血色素" in edit[0]
    # 観察期間・イベントの役割は autoprep が付け直すもので、人が直したものではない
    assert "duration" not in edit[0] and "'event'" not in edit[0]
    assert any("schema.yaml の設定（run 欄）で実行した" in w and "seed=7" in w
               for w in rep2.warnings)
    # 直していない設定は元のまま
    assert rep2.split.n_train == rep.split.n_train
    assert rep2.schema.run["seed"] == 7


def test_schema_wins_over_arguments_and_says_so():
    rep = _first()
    rep2 = mp.autoprep(_frame(), schema=_schema_path(rep), seed=1, test_size=0.5,
                       save=False, verbose=False, do_figures=False)
    assert rep2.schema.run["seed"] == 7 and rep2.schema.run["test_size"] == 0.25
    w = [x for x in rep2.warnings if "schema.yaml の値を使った" in x]
    assert w and "seed" in w[0] and "test_size" in w[0]
    # 既定値のままの引数（group=None など）は食い違いとして挙げない
    assert "group" not in w[0]


def test_old_schema_without_run_block_still_works():
    rep = _first()
    d = _read_yaml(_schema_path(rep))
    d.pop("run")
    old = os.path.join(os.path.dirname(_schema_path(rep)), "schema_0_11.yaml")
    _write_yaml(d, old)
    rep2 = mp.autoprep(_frame(), schema=old, seed=7, test_size=0.25, save=False,
                       verbose=False, do_figures=False)
    assert rep2.schema.target["name"] == ALB            # 目的変数は schema.yaml から読んだ
    assert rep2.survival is not None                     # 生存時間の 3 列も
    assert any("run 欄" in w for w in rep2.warnings)
    assert rep2.split.n_train == rep.split.n_train


def test_bad_role_is_rejected_with_a_clear_message():
    rep = _first()
    d = _read_yaml(_schema_path(rep))
    d["columns"]["年齢"]["role"] = "numerci"
    bad = os.path.join(os.path.dirname(_schema_path(rep)), "schema_bad.yaml")
    _write_yaml(d, bad)
    with pytest.raises(ValueError, match="numerci"):
        mp.autoprep(_frame(), schema=bad, verbose=False)
    with pytest.raises(FileNotFoundError):
        mp.autoprep(_frame(), schema="/nowhere/schema.yaml", verbose=False)


def test_keep_on_a_text_column_is_flagged():
    rep = _first()
    d = _read_yaml(_schema_path(rep))
    assert d["columns"]["備考"]["action"] == "drop"
    d["columns"]["備考"]["action"] = "keep"
    p = os.path.join(os.path.dirname(_schema_path(rep)), "schema_keep.yaml")
    _write_yaml(d, p)
    rep2 = mp.autoprep(_frame(), schema=p, save=False, verbose=False, do_figures=False)
    assert any("特徴量にならない" in w and "備考" in w for w in rep2.warnings)


def test_policy_argument_is_kept_in_schema():
    rep = mp.autoprep(_frame(), save=True,
                      policy={"missing": {"numeric": "median", "categorical": "most_frequent",
                                          "add_indicator": True}},
                      **KW)
    pol = mp.Schema.from_yaml(_schema_path(rep)).policy
    assert pol["missing"]["add_indicator"] is True
    rr = mp.reproduce(rep.run.run)
    assert rr.identical, rr.report()
