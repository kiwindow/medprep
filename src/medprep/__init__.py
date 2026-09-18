"""medprep — 医学研究データの前処理・記述統計・生存時間解析を自動化する。

日本腎・血液浄化AI学会（JAINBP）演習講座「鹿鳴館」の教材として開発している。

設計の原則
----------
1. **全自動は「決定の自動化」ではなく「決定の明示化」**
   `autoprep()` は 1 行で最後まで走るが、そこで下した判断（この列は ID、
   この値は欠損コード、この症例は除外）はすべて `schema.yaml` に理由つきで
   書き出される。人がそれを直して再実行できる。

2. **リークを構造的に不可能にする**
   学習を伴う変換（補完の統計量・スケーラ・エンコーダのカテゴリ集合・
   Winsorize の閾値）はすべて sklearn の ColumnTransformer に閉じ込め、
   `fit` は train にしか呼べない構造にする。

3. **推測して計算しない**
   採血時点（透析前/後）が不詳の値から透析指標を計算しない。
   日と月の順序が確定できない日付列を勝手に解釈しない。
   欠測を 0 で埋めない。異常値を 0 に丸めない。
   計算できないときは NaN を返し、理由を報告する。

4. **外れ値と入力ミスを区別する**
   Hb 0 g/dL は外れ値ではなく入力ミスであり、winsorize してはならない。
   この区別を医学領域辞書（`dict/ranges_ja.yaml`）が担う。

使い方
------
    import medprep as mp

    # ---- 層1：全自動 1 行（判断はすべて schema.yaml と rep.warnings に残る）
    rep = mp.autoprep("cohort.xlsx", outcome="eGFR_12m", task="regression",
                      group="施設", id_col="仮名ID", save=True)
    rep.show()                      # 段ごとの成否と「人の確認が要る事項」
    model.fit(rep.X_train, rep.y_train)

    # ---- 層2：1 段ずつ（第3回の演習はこちらを順に実行する）
    # 列の役割を推定し、判断を理由つきで書き出す
    sch = mp.Schema.infer(df, id_col="仮名ID", group="施設")
    sch.to_yaml("schema.yaml")      # 人が直して再実行できる

    # このまま解析してよいかを問う（audit は直さない。報告する）
    rep = mp.audit(df, sch, id_col="仮名ID", group="施設", date_col="検体採取日")
    rep.show()

    # Table 1（検定は自動で選び、選んだ理由と効果量を必ず添える）
    t1 = mp.table_one(df, sch, groupby="施設")
    print(t1.report())

    # 生存時間データ（ID / 観察開始日 / イベント発生日 / 打ち切り日 の4列）
    sf = mp.build_survival(df, id_col="仮名ID", start_date="観察開始年月日",
                           event_date="event発生年月日", censor_date="観察打ち切り年月日",
                           unit="years")
    print(sf.report())

    # 生存時間解析（KM・log-rank・Cox・比例ハザードの検定）
    s = mp.Survival.from_survival_frame(sf)
    s.km(by="施設", save="km.png")
    print(s.logrank(by="施設").report())
    cox = s.cox(covariates="auto")      # 欠測で何例落ちたかを必ず報告する
    print(cox.report())

    # 分割 → 前処理（★fit は train にしか呼べない★）
    sp = mp.split(df, sch, test_size=0.2)
    p  = mp.prepare(sp, sch)            # p.X_train / p.X_test は列名付き DataFrame
    print(mp.leak_check(p.preprocessor, sp.train, sp.test))

    # すべてを HTML 1 枚にまとめる（図は base64 で埋め込む）
    figs = mp.viz.overview(df, sch)
    mp.build_report(df, sch, audit=rep, figures=figs, path="prep_report.html")

    # 辞書駆動の掃除
    clean, rep = mp.clean_numeric(df)
    rep.show()

    # 透析指標
    r = mp.percent_cgr(sex="男性", age=60, bun_pre=60, bun_post=20,
                       cr_pre=12, cr_post=4, bw_pre=63, bw_post=60, td_hours=4)
"""

from __future__ import annotations

__version__ = "0.4.0"
__author__ = "Kazuhiro Iwadoh"
__license__ = "MIT"

from . import (
    auto,
    clean,
    dates,
    demo,
    describe,
    hd,
    loading,
    missing,
    outliers,
    paths,
    pipeline,
    quality,
    report,
    schema,
    splitting,
    survival,
    survival_input,
    tac,
    targets,
    textfmt,
    timing,
    viz,
)
from .auto import PrepResult, autoprep, quicklook
from .clean import CleanReport, build_alias_map, clean_numeric, derive, load_dict
from .dates import DateParseResult, parse_date_frame, parse_date_series
from .describe import (
    Comparison,
    ComparisonResult,
    Normality,
    TableOneResult,
    compare_groups,
    normality,
    table_one,
    target_achievement,
    target_summary,
)
from .hd import (
    bmi,
    clear_space_ratio,
    clear_space_ratio_estimated,
    clear_space_ratio_from_weight,
    corrected_ca,
    gnri,
    ideal_body_weight,
    npcr,
    percent_cgr,
    removed_mass_from_effluent,
    salt_intake,
    select_weight_for_gnri,
    sp_ktv,
    tsat,
    urr,
)
from .loading import read_any
from .missing import MissingReport, drop_missing_outcome, mcar_signals
from .missing import analyze as analyze_missing
from .outliers import NanSafeWinsorizer, OutlierReport
from .outliers import detect as detect_outliers
from .paths import RunLog, RunPaths, new_run
from .pipeline import (
    LeakageError,
    Prepared,
    Preprocessor,
    build_preprocessor,
    leak_check,
    prepare,
)
from .quality import AuditReport, Finding, audit, method_change_steps
from .report import Report, build_report
from .schema import ColumnSpec, Schema, infer_column
from .splitting import SplitResult, cv_splitter, fold_summary, mark_as, split
from .survival import (
    CoxResult,
    KMResult,
    LogRankResult,
    Survival,
    logrank_trend,
)
from .survival_input import SurvivalFrame, build_survival
from .tac import (
    TACResult,
    bun_to_urea_mg_dl,
    bun_to_urea_mmol_l,
    interdialytic_hours,
    tac_bun_simple,
    tac_linear,
    tac_trapezoid,
    urea_to_bun_mg_dl,
)
from .targets import achievement, describe_target, in_target
from .textfmt import frame_text
from .timing import POST, PRE, UNKNOWN, TimingSchema, check_requirements, detect_timing

__all__ = [
    "__version__",
    # 層1（全自動 1 行）
    "autoprep", "quicklook", "PrepResult",
    # 読み込みと保存先
    "read_any", "new_run", "RunPaths", "RunLog",
    # 表を文字で出す（日本語の幅で桁を揃える）
    "frame_text",
    # モジュール
    "auto", "clean", "dates", "demo", "describe", "hd", "loading", "missing",
    "outliers",
    "paths", "pipeline", "quality", "report", "schema", "splitting", "survival",
    "survival_input", "tac", "targets", "textfmt", "timing", "viz",
    # 列の役割とデータ品質監査
    "Schema", "ColumnSpec", "infer_column",
    "audit", "AuditReport", "Finding", "method_change_steps",
    # 日付
    "parse_date_series", "parse_date_frame", "DateParseResult",
    # 生存時間の入力
    "build_survival", "SurvivalFrame",
    # 生存時間解析
    "Survival", "KMResult", "LogRankResult", "CoxResult", "logrank_trend",
    # 欠損・外れ値
    "analyze_missing", "drop_missing_outcome", "mcar_signals", "MissingReport",
    "detect_outliers", "OutlierReport", "NanSafeWinsorizer",
    # 分割と前処理（リーク防止）
    "split", "cv_splitter", "fold_summary", "mark_as", "SplitResult",
    "Preprocessor", "prepare", "Prepared", "build_preprocessor", "leak_check",
    "LeakageError",
    # 図とレポート
    "build_report", "Report",
    # 掃除
    "clean_numeric", "derive", "load_dict", "build_alias_map", "CleanReport",
    # 採血時点
    "TimingSchema", "detect_timing", "check_requirements", "PRE", "POST", "UNKNOWN",
    # 透析指標
    "urr", "sp_ktv", "npcr", "percent_cgr",
    "clear_space_ratio", "clear_space_ratio_estimated", "clear_space_ratio_from_weight",
    "removed_mass_from_effluent", "salt_intake",
    "bmi", "ideal_body_weight", "gnri", "select_weight_for_gnri", "corrected_ca", "tsat",
    # TAC
    "tac_bun_simple", "tac_trapezoid", "tac_linear", "interdialytic_hours",
    "bun_to_urea_mg_dl", "bun_to_urea_mmol_l", "urea_to_bun_mg_dl", "TACResult",
    # 記述統計・Table 1・群間比較
    "table_one", "TableOneResult", "compare_groups", "ComparisonResult", "Comparison",
    "normality", "Normality",
    # 管理目標
    "in_target", "describe_target", "achievement", "target_achievement", "target_summary",
]
