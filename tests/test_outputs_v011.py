"""0.11.0 — 出力を 0〜6 の番号で分けた版の検査。

  * 時刻を HH:MM に揃え、日をまたいだ終了は 25:35 と書く
  * 2_解析用データに ID を残す
  * 3_機械学習用データ（分割・標準化・補完なし、ダミー変数）
  * 4_生存時間用データ
  * 目的変数がない行は 1・2 に残し、3〜6 から除いて一覧を出す
  * 目的変数の指定が無く 3 つの日付だけがあれば、τ 以内のイベントを目的変数にする
  * SMD の向き（train − test）と、カテゴリは役割で決めること
"""
import os

import numpy as np
import pandas as pd
import pytest

import medprep as mp
from medprep.clean import normalize_times, to_hours
from medprep.horizon import binary_at, choose_horizon
from medprep.schema import TIME_OF_DAY, Schema
from medprep.splitting import balance_table


# ------------------------------------------------------------------ 時刻
def test_times_are_written_as_hhmm_and_midnight_is_25_35():
    df = pd.DataFrame({"透析開始時刻": ["9:00", "13：15", "21時00分", "未記入", "08:45"],
                       "透析終了時刻": ["13:00", "17:30", "1:35", "12:00", "12時45分"]})
    out, notes, info = normalize_times(df)
    assert list(out["透析開始時刻"]) == ["09:00", "13:15", "21:00", "未記入", "08:45"]
    assert list(out["透析終了時刻"]) == ["13:00", "17:30", "25:35", "12:00", "12:45"]
    assert info["crossed"] == [2]
    assert info["unreadable"] == {"透析開始時刻": 1}      # 読めないものは元のまま残す
    assert any("25:35" in n for n in notes)
    assert to_hours("25:35") == pytest.approx(25 + 35 / 60)   # 自分の出力を読み直せる


def test_a_duration_column_named_time_is_not_treated_as_a_clock():
    df = pd.DataFrame({"time": [0.5, 1.2, 3.0], "event": [1, 0, 1]})
    out, notes, _ = normalize_times(df)
    assert out["time"].tolist() == [0.5, 1.2, 3.0] and not notes


def test_time_columns_get_their_own_role():
    df = pd.DataFrame({"透析開始時刻": [f"{h}:{m:02d}" for h in range(6, 20) for m in (0, 15, 30)],
                       "年齢": range(42)})
    sch = Schema.infer(df)
    sp = sch.columns["透析開始時刻"]
    assert sp.role == TIME_OF_DAY and sp.action == "drop" and "時刻" in sp.reason


# ------------------------------------------------------------------ τ
def test_binary_at_splits_into_one_zero_and_undetermined():
    d = pd.Series([0.5, 2.0, 0.3, 1.0, 3.0])
    e = pd.Series([1, 1, 0, 1, 0])
    y = binary_at(d, e, 1.0)
    assert y.tolist()[:2] == [1.0, 0.0]          # 2 年のイベントは「1 年以内」では 0
    assert np.isnan(y.iloc[2])                    # 0.3 年で打ち切り → 判定できない
    assert y.iloc[3] == 1.0 and y.iloc[4] == 0.0


def test_choose_horizon_keeps_undetermined_under_the_limit():
    rng = np.random.default_rng(0)
    n = 500
    t_ev = rng.exponential(3, n)
    t_c = rng.uniform(0.2, 6, n)
    d = np.minimum(t_ev, t_c)
    e = (t_ev <= t_c).astype(int)
    hz = choose_horizon(d, e, unit="years")
    assert hz.n_undetermined / n <= 0.20
    assert hz.tau <= hz.upper + 1e-9
    assert (hz.table["選んだ"] == "★").sum() == 1
    assert hz.name.startswith("イベント_") and hz.name.endswith("以内")
    assert hz.reason


# ------------------------------------------------------------------ 出力
def _cohort(n=300, seed=1):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    t = rng.exponential(900, n).astype(int) + 10
    c = rng.integers(60, 2200, n)
    ev = t <= c
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院", "C院"], n),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 10, n).round(),
        "アルブミン(Alb)": rng.normal(3.6, 0.4, n).round(1),
        "透析開始時刻": rng.choice(["9:00", "13:00", "17時30分", "21:00"], n),
        "開始日": start.strftime("%Y-%m-%d"),
        "発生日": np.where(ev, (start + pd.to_timedelta(t, unit="D")).strftime("%Y-%m-%d"), ""),
        "打切日": np.where(~ev, (start + pd.to_timedelta(c, unit="D")).strftime("%Y-%m-%d"), ""),
    })
    df["透析終了時刻"] = df["透析開始時刻"].map(
        {"9:00": "13:00", "13:00": "17:15", "17時30分": "21:30", "21:00": "1:15"})
    df.loc[df.index[:5], "アルブミン(Alb)"] = np.nan
    return df


@pytest.fixture
def auto_run(tmp_path):
    df = _cohort()
    rep = mp.autoprep(df, id_col="仮名ID", group="施設",
                      survival_dates=("開始日", "発生日", "打切日"),
                      save=True, out_dir=str(tmp_path), method="T", verbose=False)
    return df, rep, os.path.join(rep.run.run, "data")


def test_files_are_numbered_0_to_6(auto_run):
    _df, _rep, d = auto_run
    names = set(os.listdir(d))
    for f in ("0_元データ.xlsx", "1_掃除済みデータ.xlsx", "2_解析用データ.xlsx",
              "3_機械学習用データ.xlsx", "4_生存時間用データ.xlsx",
              "5_training_data_本コード専用.xlsx", "6_test_data_本コード専用.xlsx",
              "目的変数がないため削除した行.xlsx"):
        assert f in names, f


def test_outcome_is_made_from_the_dates_when_none_was_given(auto_run):
    _df, rep, _d = auto_run
    assert rep.horizon is not None
    name = rep.horizon.name
    assert rep.schema.target["name"] == name
    assert any("自動で目的変数" in w for w in rep.warnings)
    # ★観察期間とイベントが説明変数に紛れ込んでいない★（ほぼ目的変数そのもの）
    feats = set(rep.prepared.preprocessor.input_columns())
    assert not ({"duration", "event", name} & feats)


def test_rows_without_outcome_stay_in_1_2_and_leave_3_to_6(auto_run):
    df, rep, d = auto_run
    use = pd.read_excel(os.path.join(d, "2_解析用データ.xlsx"))
    ml = pd.read_excel(os.path.join(d, "3_機械学習用データ.xlsx"))
    gone = pd.read_excel(os.path.join(d, "目的変数がないため削除した行.xlsx"))
    assert len(use) == len(df) and "仮名ID" in use.columns     # ★ID を残す★
    assert len(ml) == len(df) - rep.horizon.n_undetermined - int(
        rep.df_clean["duration"].isna().sum())
    assert set(gone["ID"]).isdisjoint(set(ml["仮名ID"]))
    assert len(ml) + len(gone[gone["除いたファイル"].str.contains("3")]) == len(df)
    assert (gone["理由"].str.len() > 0).all()
    assert len(rep.X_train) + len(rep.X_test) == len(ml)


def test_ml_file_is_not_split_scaled_or_imputed(auto_run):
    _df, rep, d = auto_run
    ml = pd.read_excel(os.path.join(d, "3_機械学習用データ.xlsx"))
    assert {"元の行", "仮名ID"} <= set(ml.columns)
    assert "施設=B院" in ml.columns and "施設" not in ml.columns     # ダミー変数
    assert "男性" in ml.columns                                        # 二値は 0/1 の 1 本
    assert ml["年齢"].mean() > 30                                      # ★標準化していない★
    assert ml["アルブミン(Alb)"].isna().any()                          # ★補完していない★
    assert "透析開始時刻" not in ml.columns


def test_training_file_says_what_was_imputed(auto_run):
    _df, rep, d = auto_run
    imp = pd.read_excel(os.path.join(d, "5_training_data_本コード専用.xlsx"),
                        sheet_name="補完の記録")
    row = imp[imp["列"] == "アルブミン(Alb)"].iloc[0]
    assert row["欠損率_training"] > 0 and "中央値" in row["方法"]


def test_survival_file_keeps_units(auto_run):
    _df, rep, d = auto_run
    sv = pd.read_excel(os.path.join(d, "4_生存時間用データ.xlsx"))
    assert {"仮名ID", "duration", "event"} <= set(sv.columns)
    assert sv["年齢"].mean() > 30


def test_cleaned_file_has_hhmm_and_25_hour_ends(auto_run):
    _df, rep, d = auto_run
    cl = pd.read_excel(os.path.join(d, "1_掃除済みデータ.xlsx"), sheet_name="データ")
    assert set(cl["透析開始時刻"]) <= {"09:00", "13:00", "17:30", "21:00"}
    assert "25:15" in set(cl["透析終了時刻"])
    assert rep.df_clean["透析時間(hr)"].between(3.9, 4.6).all()


# ------------------------------------------------------------------ SMD
def test_facility_codes_get_categorical_smd_when_schema_says_group():
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"施設": rng.choice([1, 2, 3, 4], 200)})
    te = pd.DataFrame({"施設": rng.choice([1, 2, 3, 4], 60)})
    sch = Schema.infer(pd.concat([tr, te]), group="施設")
    b = balance_table(tr, te, schema=sch, columns=["施設"])
    assert "カテゴリ" in b["要約"].iloc[0] and b["SMD"].iloc[0] >= 0
