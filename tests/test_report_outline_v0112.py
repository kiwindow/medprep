"""0.11.2 — レポートの節番号をノートブックと揃えた版の検査。

  * 目的変数に選んだ列（Alb など）が、生存時間の共変量からも消えないこと
    （消えると、ノートブックの「Alb の 2 群で KM」が KeyError で止まっていた）
  * レポートの節が 1. 〜 11. の順に並び、小節に番号が付くこと
  * 欠損の地図の列名が傾かず、縦（列の帯と平行）に置かれること
"""
import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import medprep as mp  # noqa: E402
from medprep import paths  # noqa: E402

rng = np.random.default_rng(20261002)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", str(tmp_path / "lab_output"))
    monkeypatch.setattr(paths, "WORK_DIR", str(tmp_path / "lab_work"))
    monkeypatch.setattr(paths, "IN_COLAB", False)
    monkeypatch.setattr(paths, "_LAST", {})
    return tmp_path


def _frame(n=240):
    df = pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "性別": rng.choice(["男", "女"], n),
        "年齢": rng.normal(68, 12, n).round(0),
        "アルブミン(Alb)": rng.normal(3.6, 0.45, n).round(1),
        "末梢血｜血色素量(Hb)": rng.normal(10.8, 1.2, n).round(1),
    })
    base = pd.Timestamp("2015-01-01")
    df["開始日"] = base + pd.to_timedelta(rng.integers(0, 500, n), "D")
    df["発生日"] = df["開始日"] + pd.to_timedelta(rng.integers(30, 900, n), "D")
    df["打切日"] = ""
    df.loc[df.index[::2], "発生日"] = ""
    df.loc[df.index[::2], "打切日"] = df.loc[df.index[::2], "開始日"] + pd.Timedelta(days=700)
    return df


DATES = ("開始日", "発生日", "打切日")


def test_the_chosen_outcome_stays_a_survival_covariate():
    rep = mp.autoprep(_frame(), id_col="仮名ID", survival_dates=DATES,
                      outcome="アルブミン(Alb)", task="regression", verbose=False)
    assert rep.schema.target["name"] == "アルブミン(Alb)"
    cols = set(rep.survival.data.columns)
    assert "アルブミン(Alb)" in cols                       # ★ここが消えていた★
    assert {"年齢", "duration", "event"} <= cols
    # 生存時間の解析が最後まで通る（ノートブックの 8. と同じ流れ）
    d = rep.survival.data.rename(columns={"アルブミン(Alb)": "Alb"})
    d["低Alb"] = np.where(d["Alb"] < 3.5, "Alb<3.5", "Alb≧3.5")
    surv = mp.Survival(d, unit="years")
    assert surv.logrank(by="低Alb").report()
    assert surv.cox(covariates=["年齢", "Alb"]).report()


def test_the_auto_horizon_outcome_is_not_a_survival_covariate():
    """τ から作る目的変数は event そのものなので、共変量に入ってはならない（リーク）。"""
    rep = mp.autoprep(_frame(), id_col="仮名ID", survival_dates=DATES, verbose=False)
    assert rep.horizon is not None
    assert rep.horizon.name not in rep.survival.data.columns


def _h2(html):
    return [re.sub("<[^>]+>", "", t) for t in re.findall(r"<h2>(.*?)</h2>", html)]


def test_report_sections_are_numbered_1_to_11_in_order():
    rep = mp.autoprep(_frame(), id_col="仮名ID", group="施設", survival_dates=DATES,
                      outcome="アルブミン(Alb)", task="regression", save=True, verbose=False)
    titles = _h2(rep.html.to_html())
    nums = [int(m.group(1)) for t in titles if (m := re.match(r"(\d+)\. ", t))]
    assert nums == sorted(nums) and nums[0] == 1 and nums[-1] == 11
    for want in ("1. データ品質監査", "2. 列の役割", "3. 行と列の処理（減らしたもの）",
                 "4. 書き出したデータ", "5. 欠損", "6. 外れ値", "7. Table 1",
                 "8. 生存時間", "9. 分割と前処理", "10. 図", "11. 再現性"):
        assert any(t.startswith(want) for t in titles), want
    # 旧い番号（1b / 1c / 1d / 5a / 5b）が残っていない
    assert not any(re.match(r"\d+[a-z]\. ", t) for t in titles)


def test_subsections_carry_the_section_number():
    rep = mp.autoprep(_frame(), id_col="仮名ID", group="施設", survival_dates=DATES,
                      save=True, verbose=False)        # ノートブックと同じく保存する
    html = rep.html.to_html()
    h3 = [re.sub("<[^>]+>", "", t) for t in re.findall(r"<h3>(.*?)</h3>", html)]
    for want in ("3.1 行の処理", "3.2 列の処理", "3.3 値の処理",
                 "4.2 目的変数を自動で作った場合", "9.1 分割", "9.2 前処理", "10.1 "):
        assert any(t.startswith(want) for t in h3), want
    assert "Kaplan-Meier・log-rank・Cox はノートブックの 8. で行う" in html


def test_missing_map_labels_are_vertical():
    df = _frame()
    df.loc[df.index[:30], "末梢血｜血色素量(Hb)"] = np.nan
    rep = mp.autoprep(df, id_col="仮名ID", verbose=False)
    fig = rep.missing.plot(rep.df_clean, kind="matrix")
    rots = {round(t.get_rotation()) for ax in fig.axes for t in ax.get_xticklabels()
            if t.get_text()}
    assert rots == {90}, rots


def test_reading_guide_names_parts_3_and_4():
    """0.11.3 — ノートブック Ver3_5 の部立て（第3部 1.〜7.・第4部 8.〜11.）を案内する。"""
    rep = mp.autoprep(_frame(), id_col="仮名ID", survival_dates=DATES, save=True, verbose=False)
    html = rep.html.to_html()
    assert "「第3部 結果の確認」（1.〜7.）" in html
    assert "「第4部 生存時間分析と機械学習」（8.〜11.）" in html
    assert "9. 分割と前処理 ―― 機械学習の準備（モデルに渡す行列）" in html
