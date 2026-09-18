"""データ品質監査。**単独の列を見ても正常に見える壊れ方**を捕まえられることを確かめる。

ここで検査しているのは、いずれも実データで実際に起きる型の決まった事故である。
  * 同じ患者が 2 行ある
  * 透析前と透析後の列が入れ替わっている
  * 白血球分画が 100% にならない
  * 血清鉄が TIBC を超えている
  * 2020年4月を挟んで ALP に段差がある
どれも例外を出さずに通ってしまうので、通ってしまう前に見つける。
"""
import numpy as np
import pandas as pd

from medprep.quality import (
    ERROR,
    INFO,
    WARN,
    Finding,
    audit,
    method_change_steps,
)
from medprep.schema import Schema


def base(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "仮名ID": [f"P{i:04d}" for i in range(n)],
        "施設": rng.choice(["A院", "B院"], n),
        "年齢": rng.integers(40, 85, n),
        "性別": rng.choice(["男", "女"], n),
    })


def cats(rep, category):
    return [f for f in rep.findings if f.category == category]


def msgs(rep, severity=None):
    return " / ".join(f.message for f in rep.findings
                      if severity is None or f.severity == severity)


# ---------------------------------------------------------------- 不変条件
def test_every_finding_says_what_to_do():
    """★『警告だけ出して放り出す』ことをしない。★"""
    df = base()
    df.loc[0, "仮名ID"] = df.loc[1, "仮名ID"]
    rep = audit(df, id_col="仮名ID", group="施設")
    assert rep.findings
    for f in rep.findings:
        assert f.action and f.action.strip(), f"{f.message}: action が空"
        assert f.severity in (ERROR, WARN, INFO)


def test_audit_does_not_modify_the_data():
    """★audit は直さない。報告するだけ。★"""
    df = base()
    before = df.copy(deep=True)
    audit(df, id_col="仮名ID", group="施設")
    pd.testing.assert_frame_equal(df, before)


def test_clean_data_produces_no_errors():
    rep = audit(base(), id_col="仮名ID", group="施設")
    assert rep.ok, msgs(rep, ERROR)


# ---------------------------------------------------------------- 重複
def test_duplicate_patient_id_is_an_error():
    """1 人 2 行のデータで Cox を回すと、独立性の仮定が壊れる。"""
    df = base()
    df.loc[5, "仮名ID"] = df.loc[4, "仮名ID"]
    rep = audit(df, id_col="仮名ID")
    assert not rep.ok
    assert any("同じ ID" in f.message for f in rep.errors)


def test_fully_duplicated_rows_are_an_error():
    df = pd.concat([base(20), base(20).iloc[:3]], ignore_index=True)
    rep = audit(df, id_col="仮名ID")
    assert any("全列が一致" in f.message for f in rep.errors)


def test_rows_identical_except_the_id_are_flagged():
    df = base(20)
    df.loc[19] = df.loc[3].copy()
    df.loc[19, "仮名ID"] = "P9999"
    rep = audit(df, id_col="仮名ID")
    assert any("ID 以外の全列が一致" in f.message for f in rep.warnings)


# ---------------------------------------------------------------- 識別子
def test_high_cardinality_column_is_asked_about_when_no_id_given():
    df = base()
    df["受付コード"] = [f"X{i}" for i in range(len(df))]
    rep = audit(df)                      # id_col を渡さない
    assert any(f.category == "識別子" for f in rep.warnings)


def test_no_identifier_question_when_the_human_already_named_the_id():
    df = base()
    df["備考"] = [f"メモ{i}" for i in range(len(df))]
    rep = audit(df, id_col="仮名ID")
    assert not cats(rep, "識別子")


# ---------------------------------------------------------------- 透析前後
def pre_post(n=40, swap=False, bad=0):
    rng = np.random.default_rng(7)
    pre = rng.normal(65, 10, n).round(1)
    post = (pre * rng.uniform(0.25, 0.4, n)).round(1)
    if swap:
        pre, post = post, pre
    else:
        for i in range(bad):
            pre[i], post[i] = post[i], pre[i]
    return pd.DataFrame({
        "仮名ID": [f"P{i:03d}" for i in range(n)],
        "透析前BUN": pre, "透析後BUN": post,
    })


def test_swapped_pre_post_columns_are_caught():
    """★BUN は透析で必ず下がる。下がらない症例が過半数なら取り違えである。★

    取り違えたまま spKt/V を計算すると透析量が過大に評価され、
    しかもどの値も単独では正常範囲に見える。
    """
    rep = audit(pre_post(swap=True), id_col="仮名ID")
    e = [f for f in rep.errors if f.category == "採血時点"]
    assert e, msgs(rep)
    assert "取り違え" in e[0].action


def test_a_few_reversed_cases_are_a_warning_not_an_error():
    rep = audit(pre_post(bad=8), id_col="仮名ID")
    assert not any(f.category == "採血時点" and f.severity == ERROR for f in rep.findings)
    assert any(f.category == "採血時点" and f.severity == WARN for f in rep.findings)


def test_correct_pre_post_data_raises_nothing():
    rep = audit(pre_post(), id_col="仮名ID")
    assert not any(f.category == "採血時点" and f.severity in (ERROR, WARN)
                   for f in rep.findings)


def test_unknown_timing_on_a_timing_sensitive_item_is_warned():
    df = base()
    df["BUN"] = 60.0
    df["K"] = 4.5
    rep = audit(df, id_col="仮名ID")
    w = [f for f in rep.warnings if f.category == "採血時点"]
    assert w and "spKt/V" in w[0].action


def test_only_one_side_present_is_recorded():
    df = base(30)
    df["透析前BUN"] = 60.0
    rep = audit(df, id_col="仮名ID")
    assert any(f.category == "採血時点" and "だけがあり" in f.message for f in rep.findings)


# ---------------------------------------------------------------- 分画
def differential(n=40, total=100.0):
    rng = np.random.default_rng(11)
    neut = rng.normal(60, 5, n).round(1)
    eo, ba, mo = np.full(n, 2.0), np.full(n, 0.5), np.full(n, 5.0)
    lym = total - neut - eo - ba - mo
    return pd.DataFrame({
        "仮名ID": [f"P{i:03d}" for i in range(n)],
        "好中球": neut, "好酸球": eo, "好塩基球": ba, "単球": mo, "リンパ球": lym,
    })


def test_white_cell_differential_summing_to_100_is_accepted():
    rep = audit(differential(), id_col="仮名ID")
    assert not cats(rep, "検算"), msgs(rep)


def test_white_cell_differential_not_summing_to_100_is_reported():
    df = differential(total=140.0)
    rep = audit(df, id_col="仮名ID")
    assert any("白血球分画" in f.message for f in cats(rep, "検算"))


def test_band_and_seg_are_not_added_into_the_100_percent_check():
    """★好中球 = 桿状核球 + 分葉核球 なので、7 項目を足すと約 200% になる。★

    2026-09-18 に「検算には好中球を使う」と確定した。
    桿状核球・分葉核球を合計に入れてしまえば、正常なデータが全例異常になる。
    """
    df = differential()
    df["桿状核球"] = (df["好中球"] * 0.1).round(1)
    df["分葉核球"] = (df["好中球"] * 0.9).round(1)
    rep = audit(df, id_col="仮名ID")
    assert not any("白血球分画" in f.message for f in cats(rep, "検算")), msgs(rep)


def test_band_plus_seg_must_match_neutrophils():
    df = differential()
    df["桿状核球"] = 5.0
    df["分葉核球"] = 5.0                      # 合計 10% は好中球 60% と合わない
    rep = audit(df, id_col="仮名ID")
    assert any("好中球の内訳" in f.message for f in cats(rep, "検算"))


# ---------------------------------------------------------------- 項目間
def test_iron_above_tibc_is_an_error():
    """TIBC = Fe + UIBC。定義上 Fe を下回れない。TSAT が 100% を超える。"""
    df = base(30)
    df["血清鉄(Fe)"] = [200.0] * 30
    df["総鉄結合能(TIBC)"] = [150.0] * 30
    rep = audit(df, id_col="仮名ID")
    assert any("TSAT" in f.message for f in rep.errors), msgs(rep)


def test_systolic_not_above_diastolic_is_an_error():
    df = base(30)
    df["収縮期血圧"] = [80.0] * 30
    df["拡張期血圧"] = [120.0] * 30
    rep = audit(df, id_col="仮名ID")
    assert any("収縮期血圧" in f.message for f in rep.errors)


def test_corrected_calcium_inconsistent_with_payne_is_warned():
    df = base(30)
    df["カルシウム(Ca)"] = 8.0
    df["アルブミン(Alb)"] = 3.0
    df["補正Ca"] = 8.0                        # Payne 式なら 9.0 になるはず
    rep = audit(df, id_col="仮名ID")
    assert any("補正Ca" in f.message for f in rep.warnings)


def test_impossible_bmi_from_height_and_weight():
    df = base(30)
    df["身長"] = 165.0
    df["体重"] = 1.65                         # m と kg の取り違え
    rep = audit(df, id_col="仮名ID")
    assert any("BMI" in f.message for f in rep.warnings)


# ---------------------------------------------------------------- 日付
def test_ambiguous_date_order_is_an_error_even_when_parsing_failed():
    """★日と月の順序が決まらない列は、役割が datetime にならない。★

    役割だけを見ていると検査から漏れる。列名でも拾うようにしてある。
    """
    df = base(30)
    df["検査日"] = ["3/4/2013", "5/6/2014", "7/8/2015"] * 10
    rep = audit(df, id_col="仮名ID")
    assert any("並びが確定できない" in f.message for f in rep.errors), msgs(rep)


def test_future_dates_are_an_error():
    df = base(30)
    future = (pd.Timestamp.today() + pd.Timedelta(days=400)).strftime("%Y/%m/%d")
    df["採血日"] = ["2019/5/1"] * 29 + [future]
    rep = audit(df, id_col="仮名ID")
    assert any("未来の日付" in f.message for f in rep.errors)


def test_unparseable_values_in_a_date_column_are_warned():
    df = base(30)
    df["採血日"] = ["2019/5/1"] * 28 + ["平成拾年", "???"]
    rep = audit(df, id_col="仮名ID")
    assert any("解釈できない値" in f.message for f in rep.warnings)


# ---------------------------------------------------------------- 生存時間
def survival_frame():
    return pd.DataFrame({
        "仮名ID": [f"P{i}" for i in range(6)],
        "観察開始年月日": ["2013/1/1", "2013/1/1", "2013/1/1", "2013/1/1", "", "2015/1/1"],
        "死亡年月日":   ["2015/1/1", "",         "2016/1/1", "",         "2016/1/1", "2014/1/1"],
        "打切年月日":   ["2014/1/1", "2018/1/1", "",         "",         "",         ""],
    })


def test_survival_contradictions_are_all_reported():
    df = survival_frame()
    sch = Schema.infer(df, id_col="仮名ID",
                       survival_dates=("観察開始年月日", "死亡年月日", "打切年月日"))
    rep = audit(df, sch, id_col="仮名ID")
    text = msgs(rep, ERROR)
    assert "両方が入っている" in text          # P0
    assert "どちらも" in text or "も入っていない" in text   # P3
    assert "観察開始日が無い" in text           # P4
    assert "前になっている" in text             # P5


def test_blank_event_date_is_not_reported_as_missing():
    """★形式C では『イベント発生日が空欄』＝イベント無し。★

    これを欠測率として報告すると、意味のある空欄を補完させる誘導になる。
    """
    df = survival_frame()
    sch = Schema.infer(df, id_col="仮名ID",
                       survival_dates=("観察開始年月日", "死亡年月日", "打切年月日"))
    rep = audit(df, sch, id_col="仮名ID")
    assert not any(f.category == "欠測" and "死亡年月日" in f.message for f in rep.findings)
    assert any(f.category == "欠測" and f.severity == INFO for f in rep.findings)


def test_too_few_events_warns_about_epv():
    df = survival_frame()
    sch = Schema.infer(df, id_col="仮名ID",
                       survival_dates=("観察開始年月日", "死亡年月日", "打切年月日"))
    rep = audit(df, sch, id_col="仮名ID")
    assert any("イベント数" in f.message for f in rep.warnings)


# ---------------------------------------------------------------- 目的変数
def test_single_class_outcome_is_an_error():
    df = base(40)
    df["転帰"] = 1
    rep = audit(df, id_col="仮名ID", outcome="転帰", task="classification")
    assert any("種類しかない" in f.message for f in rep.errors)


def test_tiny_minority_class_is_an_error():
    df = base(60)
    df["転帰"] = [1] * 3 + [0] * 57
    rep = audit(df, id_col="仮名ID", outcome="転帰", task="classification")
    assert any("最小クラス" in f.message for f in rep.errors)


def test_class_imbalance_is_warned():
    df = base(200)
    df["転帰"] = [1] * 15 + [0] * 185
    rep = audit(df, id_col="仮名ID", outcome="転帰", task="classification")
    assert any("クラス不均衡" in f.message for f in rep.warnings)


def test_leakage_is_an_error():
    """『転帰』から導かれた列が説明変数に残ると、性能は偽物になる。"""
    df = base(80)
    df["転帰"] = [1, 0] * 40
    df["退院時状態"] = df["転帰"].map({1: "死亡", 0: "生存"})
    rep = audit(df, id_col="仮名ID", outcome="転帰", task="classification")
    assert any(f.category == "リーク" for f in rep.errors), msgs(rep)


def test_missing_outcome_is_not_to_be_imputed():
    df = base(60)
    df["転帰"] = [1, 0] * 30
    df.loc[:5, "転帰"] = np.nan
    rep = audit(df, id_col="仮名ID", outcome="転帰", task="classification")
    f = [x for x in rep.findings if x.category == "目的変数"][0]
    assert "補完してはならない" in f.action


# ---------------------------------------------------------------- 群・欠測
def test_small_group_is_warned():
    df = base(100)
    df.loc[:2, "施設"] = "Z院"                  # n=3
    rep = audit(df, id_col="仮名ID", group="施設")
    assert any(f.category == "群" for f in rep.warnings)


def test_missingness_tied_to_the_group_is_warned():
    """欠測が施設に偏っていれば、全体の中央値で埋めると群間差が人工的に作られる。"""
    n = 200
    df = base(n)
    df["施設"] = ["A院"] * 100 + ["B院"] * 100
    df["アルブミン(Alb)"] = np.random.default_rng(1).normal(3.6, 0.4, n).round(1)
    df.loc[df["施設"] == "B院", "アルブミン(Alb)"] = np.nan
    rep = audit(df, id_col="仮名ID", group="施設")
    assert any("偏っている" in f.message for f in rep.warnings)


def test_high_missing_rate_column_is_warned():
    df = base(60)
    df["尿素窒素"] = np.nan
    df.loc[:20, "尿素窒素"] = np.random.default_rng(2).normal(60, 8, 21).round(1)
    rep = audit(df, id_col="仮名ID")
    assert any(f.category == "欠測" and "欠測率" in f.message for f in rep.warnings)


# ---------------------------------------------------------------- 表記ゆれ
def test_spelling_variants_that_normalize_to_the_same_value():
    """『男』と『男 』が別の水準として数えられれば、群が 2 つに割れる。"""
    df = base(60)
    df.loc[:10, "性別"] = "男 "
    df.loc[11:20, "性別"] = "Ｍ"
    df.loc[21:30, "性別"] = "M"
    rep = audit(df, id_col="仮名ID")
    assert any(f.category == "表記ゆれ" for f in rep.warnings), msgs(rep)


# ---------------------------------------------------------------- 測定法変更
def alp_frame(n=200, step=True):
    days = pd.date_range("2016-01-01", periods=n, freq="20D")
    rng = np.random.default_rng(5)
    after = days >= pd.Timestamp("2020-04-01")
    val = np.where(after & step, rng.normal(85, 20, n), rng.normal(250, 60, n))
    return pd.DataFrame({
        "仮名ID": [f"P{i:03d}" for i in range(n)],
        "検体採取日": [d.strftime("%Y/%m/%d") for d in days],
        "アルカリフォスファターゼ(ALP)": val.round(0),
    })


def test_alp_method_change_step_is_detected():
    """★2020年4月の JSCC→IFCC で値がおよそ 1/3 になる。★

    これを病態の変化として解析すると「2020年以降の症例は骨代謝が良い」という
    存在しない所見が出る。
    """
    fs = method_change_steps(alp_frame(), "検体採取日")
    alp = [f for f in fs if "ALP" in " ".join(f.columns)]
    assert alp and alp[0].severity == ERROR
    assert "段差" in alp[0].message
    assert "層別" in alp[0].action


def test_no_step_is_recorded_not_flagged():
    fs = method_change_steps(alp_frame(step=False), "検体採取日")
    alp = [f for f in fs if "ALP" in " ".join(f.columns)]
    assert alp and alp[0].severity == INFO


def test_data_entirely_on_one_side_of_the_change_is_not_judged():
    df = alp_frame()
    df = df[pd.to_datetime(df["検体採取日"]) < "2020-04-01"]
    fs = method_change_steps(df, "検体採取日")
    alp = [f for f in fs if "ALP" in " ".join(f.columns)]
    assert alp and "判定できない" in alp[0].message


def test_change_without_a_fixed_date_is_still_reported():
    """Alb の BCG→BCP は施設ごとに時期が違う。日付が無くても存在は伝える。"""
    df = alp_frame()
    df["アルブミン(Alb)"] = 3.6
    fs = method_change_steps(df, "検体採取日")
    assert any("特定できない" in f.message for f in fs)


def test_audit_says_so_when_it_could_not_check_the_method_change():
    rep = audit(alp_frame(), id_col="仮名ID")       # date_col を渡さない
    assert any("測定法変更" in name for name, _why in rep.skipped)


def test_audit_runs_the_step_check_when_given_a_date_column():
    rep = audit(alp_frame(), id_col="仮名ID", date_col="検体採取日")
    assert any(f.category == "測定法" for f in rep.errors)
    assert not rep.skipped


# ---------------------------------------------------------------- 報告
def test_report_and_frame_are_usable():
    df = pre_post(swap=True)
    rep = audit(df, id_col="仮名ID")
    text = rep.report()
    assert "データ品質監査" in text and "致命的" in text
    f = rep.to_frame()
    assert list(f.columns) == ["重大度", "区分", "列", "件数", "所見", "対応"]
    assert len(f) == len(rep.findings)


def test_finding_str_includes_the_action():
    f = Finding(WARN, "試験", "所見", "こうすること", columns=["列A"], n=3)
    s = str(f)
    assert "所見" in s and "こうすること" in s and "3 件" in s and "列A" in s
