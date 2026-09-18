"""層1 — 全自動 1 行（`mp.autoprep()`）。

    rep = mp.autoprep('cohort.xlsx', outcome='eGFR_12m', task='regression',
                      group='施設', save=True)

**ここには固有のロジックを書かない。** 層2（`schema` / `quality` / `clean` /
`missing` / `outliers` / `describe` / `splitting` / `pipeline` / `viz` / `report`）を
決まった順に呼び、結果を 1 つの入れ物に集めるだけである。

そうする理由は 2 つある。

1. 受講者が層1で見た結果を、層2で 1 段ずつ分解して追体験できる。
   `autoprep` にしか無い処理があると、その追体験が途中で途切れる。
2. **全自動は「決定の自動化」ではなく「決定の明示化」である。**
   ここで下した判断はすべて `schema.yaml` と `rep.warnings` に残り、
   人が直して再実行できる。

止まるところと、止まらないところ
--------------------------------
背骨（読む・役割の推定・監査・分割・前処理）で失敗したら **例外を投げて止まる。**
付随するもの（管理目標・生存時間・図）で失敗したら、**理由を残して飛ばす。**
黙って飛ばすことはしない（`rep.steps` に全段の成否が残る）。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd

from . import paths as _paths
from . import viz
from .clean import clean_numeric, derive, load_dict
from .describe import table_one, target_achievement
from .loading import read_any
from .missing import analyze as analyze_missing
from .missing import drop_missing_outcome, mcar_signals
from .outliers import detect as detect_outliers
from .pipeline import leak_check, prepare
from .quality import audit
from .report import build_report
from .schema import DATETIME, Schema
from .splitting import split
from .survival_input import build_survival
from .textfmt import frame_text

__version_note__ = "Phase 11"


# ================================================================== 結果
@dataclass
class PrepResult:
    """`autoprep()` が返すもの。**「処理済みデータ」ではなく「判断の記録」である。**"""

    df_raw: pd.DataFrame | None = None
    df_clean: pd.DataFrame | None = None
    schema: Schema | None = None
    schema_raw: Schema | None = None
    audit: object = None
    clean_report: object = None
    missing: object = None
    mcar: pd.DataFrame | None = None
    outliers: object = None
    table1: object = None
    achievement: pd.DataFrame | None = None
    survival: object = None                # SurvivalFrame（解析は層2でする）
    split: object = None
    prepared: object = None
    figures: viz.FigureSet | None = None
    html: object = None                    # Report
    run: _paths.RunPaths | None = None
    log: _paths.RunLog | None = None
    removed: pd.DataFrame | None = None     # 何を、いくつ、なぜ減らしたか
    dropped_outcome: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    steps: list = field(default_factory=list)      # (段, 成否, 一言)
    saved: list = field(default_factory=list)
    seconds: float = 0.0

    # ------------------------------------------------------------ 近道
    @property
    def pipeline(self):
        """fit 済みの前処理（sklearn 互換）。"""
        return self.prepared.preprocessor if self.prepared else None

    @property
    def X_train(self):
        return self.prepared.X_train if self.prepared else None

    @property
    def X_test(self):
        return self.prepared.X_test if self.prepared else None

    @property
    def y_train(self):
        return self.prepared.y_train if self.prepared else None

    @property
    def y_test(self):
        return self.prepared.y_test if self.prepared else None

    # ------------------------------------------------------------ 報告
    def report(self) -> str:
        lines = ["autoprep"]
        if self.df_raw is not None:
            lines[0] += f"  {len(self.df_raw)} 行 × {self.df_raw.shape[1]} 列"
        lines[0] += f"（{self.seconds:.1f} 秒）"

        lines.append("\n段:")
        for name, ok, detail in self.steps:
            mark = "・" if ok else "—"
            lines.append(f"  {mark} {name}" + (f"  {detail}" if detail else ""))

        if self.removed is not None and len(self.removed):
            col = self.removed[self.removed["種類"] == "列"]
            row = self.removed[self.removed["種類"] == "行"]
            val = self.removed[self.removed["種類"] == "値"]
            lines.append(
                f"\n減らしたもの: 列 {len(col)} 本 / 行 {int(row['件数'].sum())} 例 / "
                f"値 {int(val['件数'].sum())} 個（NaN 化）")
            lines.append(frame_text(self.removed))
            lines.append("（行の削除は段によって効く範囲が違う。"
                         "『生存時間の形にする』は生存時間解析にだけ効き、"
                         "モデルに渡す行列には影響しない）")

        if self.warnings:
            lines.append(f"\n★人の確認が要る事項 {len(self.warnings)} 件★"
                         "（空でないのがふつうである）:")
            for w in self.warnings:
                lines.append(f"  ✗ {w}")
        else:
            lines.append("\n人の確認が要る事項: なし")

        if self.run:
            lines.append("\n" + self.run.report_text())
        if self.saved:
            lines.append("\n保存したもの:")
            for p in self.saved:
                lines.append(f"  {p}")
        return "\n".join(lines)

    def show(self):
        print(self.report())

    def to_html(self, path=None, *, show_values: bool = False, **kwargs):
        """HTML 1 枚のレポートを作り直す（`show_values=True` で症例レベルの値も出す）。"""
        rep = _build_html(self, show_values=show_values, **kwargs)
        if path:
            rep.to_html(path)
        self.html = rep
        return rep

    def to_excel(self, path):
        rep = self.html or self.to_html()
        return rep.to_excel(path)


# ================================================================== 本体
def autoprep(
    data,
    *,
    outcome: str | None = None,
    task: str | None = None,
    group: str | None = None,
    id_col: str | None = None,
    survival: tuple | None = None,
    survival_dates: tuple | None = None,
    date_col: str | None = None,
    columns: list | None = None,
    table1_columns: list | None = None,
    test_size: float = 0.2,
    seed: int = 0,
    clean: bool = True,
    do_split: bool | str = "auto",
    do_table1: bool = True,
    do_figures: bool = True,
    save: bool = False,
    save_data: bool = True,
    method: str = "Preprocessing",
    project_folder: str = "",
    out_dir: str | None = None,
    title: str | None = None,
    show_values: bool = False,
    policy: dict | None = None,
    dic: dict | None = None,
    verbose: bool = True,
) -> PrepResult:
    """読み込みから前処理済み行列とレポートまでを 1 行で通す。

    Parameters
    ----------
    data : ファイルのパスか DataFrame
    outcome, task : 目的変数と課題（'regression' / 'classification' / 'survival'）
    group : 群間比較・グループ分割に使う列（施設など）
    survival_dates : (観察開始日, イベント発生日, 打ち切り日) の 3 列
    do_split : "auto" なら `outcome` があるときだけ分割して前処理まで行う
    save : True なら `~/lab_output/{method}/run{N}/` 以下に全出力を保存する
    save_data : `save=True` のとき、掃除済みデータと前処理済み行列も書き出す。
        ★これは症例レベルのデータである。★ 置き場所には `.gitignore` を必ず置くが、
        受け渡しには注意すること。要らなければ False。
    """
    t0 = time.time()
    res = PrepResult()
    dic = dic or load_dict()

    def step(name, ok=True, detail=""):
        res.steps.append((name, ok, detail))
        if verbose:
            print(("・" if ok else "—") + f" {name}" + (f"  {detail}" if detail else ""))

    def warn(text):
        if text not in res.warnings:
            res.warnings.append(text)

    def optional(name, fn):
        """付随する段。失敗したら理由を残して飛ばす（背骨では使わない）。"""
        try:
            return fn()
        except Exception as e:                                       # noqa: BLE001
            step(name, False, f"飛ばした（{type(e).__name__}: {e}）")
            return None

    # -------------------------------------------------------- 1) 読む
    if isinstance(data, pd.DataFrame):
        res.df_raw = data.copy()
        source = "（渡された DataFrame）"
        step("読む", True, f"{len(data)} 行 × {data.shape[1]} 列")
    else:
        res.df_raw = read_any(data)
        source = res.df_raw.attrs.get("medprep_source", str(data))
        for n in res.df_raw.attrs.get("medprep_read", []):
            step("読む", True, n)
            if "★" in n:
                warn(n)
    df = res.df_raw

    # -------------------------------------------------------- 2) 役割の推定
    res.schema_raw = Schema.infer(
        df, outcome=outcome, task=task, group=group, id_col=id_col,
        survival=survival, survival_dates=survival_dates, dic=dic)
    step("列の役割を推定する", True,
         f"{len(res.schema_raw.columns)} 列（うち解析から外す列 "
         f"{len(res.schema_raw.dropped())} 本）")

    # -------------------------------------------------------- 3) 監査
    res.audit = audit(df, res.schema_raw, id_col=id_col, outcome=outcome, task=task,
                      group=group, survival=survival, survival_dates=survival_dates,
                      date_col=date_col, dic=dic)
    step("品質を監査する", True,
         f"致命的 {len(res.audit.errors)} / 要確認 {len(res.audit.warnings)} 件")
    for f in res.audit.errors:
        warn(f"{f.category}  {f.message}  → {f.action}")

    # -------------------------------------------------------- 4) 掃除
    if clean:
        res.df_clean, res.clean_report = clean_numeric(df, dic=dic)
        res.df_clean, notes = derive(res.df_clean, dic=dic)
        res.notes += notes
        step("辞書で掃除する", True,
             f"{len(res.clean_report.to_frame())} 列を辞書と照合"
             + (f"、派生 {len(notes)} 件" if notes else ""))
    else:
        res.df_clean = df.copy()
        step("辞書で掃除する", False, "clean=False が指定された")
    dfc = res.df_clean

    # -------------------------------------------------------- 5) 役割の再推定
    #   掃除で派生列（補正Ca など）が増えるので、以降はこちらの schema を使う。
    res.schema = Schema.infer(
        dfc, outcome=outcome, task=task, group=group, id_col=id_col,
        survival=survival, survival_dates=survival_dates, dic=dic)

    # -------------------------------------------------------- 6) 生存時間の形
    if survival_dates:
        def _sf():
            start, ev, cens = survival_dates
            # ★共変量を必ず持たせる。★
            #   持たせないと `.data` は ID と (duration, event) だけになり、
            #   受け取った側が Cox に掛けられない（共変量を手で結合し直すはめになる）。
            keep = [c for c in res.schema.features()
                    if c in dfc.columns and c not in survival_dates
                    and res.schema.columns[c].role != DATETIME]   # 日付は共変量ではない
            sf = build_survival(dfc, id_col=id_col, start_date=start,
                                event_date=ev, censor_date=cens,
                                covariates=keep, unit="years")
            step("生存時間の形にする", True,
                 f"{len(sf.data)} 例（除外 {len(sf.excluded)} 例）")
            if len(sf.excluded):
                warn(f"生存時間に変換できず {len(sf.excluded)} 例を除外した"
                     "（rep.survival.excluded を見ること）")
            res.notes.append("生存時間の解析（KM・log-rank・Cox）は層2で行う: "
                             "mp.Survival(rep.survival.data, unit='years')")
            return sf
        res.survival = optional("生存時間の形にする", _sf)

    # -------------------------------------------------------- 7) 欠損
    res.missing = analyze_missing(dfc, res.schema, group=group)
    step("欠損を見る", True, f"完全症例 {res.missing.complete_rate:.1%}")
    res.mcar = optional("欠損の偏りを調べる",
                        lambda: mcar_signals(dfc, group=group) if group else
                        mcar_signals(dfc))
    if res.mcar is not None and len(res.mcar):
        step("欠損の偏りを調べる", True, f"{len(res.mcar)} 組で偏りの兆候")

    # -------------------------------------------------------- 8) 外れ値
    res.outliers = optional(
        "外れ値を見る", lambda: detect_outliers(dfc, res.schema, id_col=id_col))
    if res.outliers is not None:
        step("外れ値を見る", True, "検出のみ（★ここでは直さない★）")

    # -------------------------------------------------------- 9) Table 1
    if do_table1:
        # ★`columns` を Table 1 に回してはならない。★
        #   `columns` は「モデルに入れる特徴量」であって「表に並べる項目」ではない。
        #   両方に渡すと、日付や自由記載まで Table 1 に並び、数百行の表になる。
        res.table1 = optional(
            "Table 1 を作る",
            lambda: table_one(dfc, res.schema, groupby=group,
                              columns=table1_columns, dic=dic))
        if res.table1 is not None:
            step("Table 1 を作る", True,
                 f"{len(res.table1.to_frame())} 行" + (f"（{group} 別）" if group else ""))

    # -------------------------------------------------------- 10) 管理目標
    res.achievement = optional(
        "管理目標の達成率", lambda: target_achievement(dfc, by=group, dic=dic))
    if res.achievement is not None and len(res.achievement):
        step("管理目標の達成率", True, f"{res.achievement['項目'].nunique()} 項目")

    # -------------------------------------------------------- 11) 分割と前処理
    want_split = (outcome is not None) if do_split == "auto" else bool(do_split)
    if want_split and outcome:
        dfs = dfc
        if outcome in dfs.columns and dfs[outcome].isna().any():
            dfs, info = drop_missing_outcome(dfs, outcome)
            res.dropped_outcome = info
            step("目的変数の欠測を除く", True,
                 f"{info['残り']} 例（判定できない {info['除外']} 例を除外）")
            warn(f"目的変数 '{outcome}' の欠測 {info['除外']} 例を除外した"
                 "（★目的変数は補完してはならない★）")
            res.schema = Schema.infer(dfs, outcome=outcome, task=task, group=group,
                                      id_col=id_col, survival=survival,
                                      survival_dates=survival_dates, dic=dic)
        res.split = split(dfs, res.schema, test_size=test_size, seed=seed)
        step("train / test に分ける", True, res.split.strategy
             + f"  train {res.split.n_train} / test {res.split.n_test}")
        for w in res.split.warnings:
            warn(w)

        res.prepared = prepare(res.split, res.schema, policy, columns=columns)
        step("前処理を train だけで fit する", True,
             f"{res.prepared.X_train.shape[1]} 特徴量"
             "（★test には transform しか当てていない★）")
        for n in res.prepared.notes:
            if "★" in n:
                warn(n)
        lc = leak_check(res.prepared.preprocessor, res.split.train, res.split.test)
        bad = lc[lc["結果"] != "OK"]
        step("リークを検査する", True,
             f"{len(lc)} 項目すべて OK" if not len(bad) else f"★{len(bad)} 項目で問題★")
        for _, r in bad.iterrows():
            warn(f"リーク検査で問題: {r['検査']}  {r['内容']}")
    elif want_split:
        step("train / test に分ける", False, "outcome が指定されていない")

    # -------------------------------------------------------- 11.5) 減らしたものの記録
    #   ★何を捨てたかを言わない自動化は、信用してはならない。★
    res.removed = _removed_table(res, dfc, outcome)
    n_col = int((res.removed["種類"] == "列").sum()) if len(res.removed) else 0
    n_row = int(res.removed.loc[res.removed["種類"] == "行", "件数"].sum()) \
        if len(res.removed) else 0
    step("減らしたものを数える", True,
         f"列 {n_col} 本 / 行 {n_row} 例（rep.removed に一覧）")

    # -------------------------------------------------------- 12) 図
    if do_figures:
        res.figures = viz.overview(
            dfc, res.schema, achievement=res.achievement,
            balance=res.split.balance if res.split is not None else None)
        step("図を描く", True, f"{len(res.figures)} 枚"
             + (f"（描けなかったもの: {[t for t, _ in res.figures.skipped]}）"
                if res.figures.skipped else ""))

    # -------------------------------------------------------- 13) レポート
    res.html = _build_html(res, title=title, show_values=show_values, source=source)
    step("HTML 1 枚にまとめる", True, f"{len(res.html.sections)} 節")

    # -------------------------------------------------------- 14) 保存
    if save:
        _save_all(res, method=method, project_folder=project_folder,
                  out_dir=out_dir, source=source, step=step, t0=t0,
                  save_data=save_data)
    res.seconds = time.time() - t0

    if verbose:
        print()
        if res.warnings:
            print(f"★人の確認が要る事項が {len(res.warnings)} 件ある。"
                  "rep.show() で読むこと。空でないのがふつうである。★")
        else:
            print("人の確認が要る事項: なし")
    return res


def quicklook(data, **kwargs) -> PrepResult:
    """まず全体を見るだけ。分割も前処理もしない（第3回演習の 1 つ目の演習）。"""
    kwargs.setdefault("do_split", False)
    kwargs.setdefault("do_table1", True)
    kwargs.setdefault("save", False)
    return autoprep(data, **kwargs)


# ================================================================== 補助
def _build_html(res: PrepResult, *, title=None, show_values=False, source="", **kwargs):
    sub = kwargs.pop("subtitle", "")
    if source and not sub:
        sub = f"入力: {os.path.basename(source)}" if os.path.sep in str(source) else ""
    return build_report(
        res.df_clean if res.df_clean is not None else res.df_raw,
        res.schema,
        title=title or "前処理レポート",
        subtitle=sub,
        audit=res.audit, removed=res.removed, missing=res.missing,
        outliers=res.outliers,
        table1=res.table1,
        comparison=getattr(res.table1, "comparison", None),
        achievement=res.achievement,
        split=res.split,
        preprocessor=res.prepared.preprocessor if res.prepared else None,
        figures=res.figures, show_values=show_values, **kwargs)


def _save_all(res: PrepResult, *, method, project_folder, out_dir, source, step, t0,
              save_data=True):
    """`run{N}/` 以下に全部書き出す（構想 §7 の保存規約）。"""
    p = _paths.new_run(method, project_folder=project_folder, output_dir=out_dir)
    res.run = p
    df = res.df_clean if res.df_clean is not None else res.df_raw

    res.log = _paths.RunLog.start(
        p,
        入力ファイル名=os.path.basename(str(source)),
        使用コード=f"medprep.autoprep ({_pkg_version()})",
        実行方法="medprep.autoprep",
        行数=len(df) if df is not None else "",
        目的変数=(res.schema.target or {}).get("name", "") if res.schema else "",
    )

    def put(path, fn):
        try:
            fn(path)
            res.saved.append(path)
        except Exception as e:                                       # noqa: BLE001
            step(f"保存: {os.path.basename(path)}", False,
                 f"書けなかった（{type(e).__name__}: {e}）")

    # --- model/ ： ★再現性の中核★
    if res.schema is not None:
        put(p.file("model", "schema.yaml"), res.schema.to_yaml)
    if res.prepared is not None:
        put(p.file("model", "pipeline.pkl"), res.prepared.preprocessor.save)
    put(p.file("model", "medprep_version.txt"), _write_version)

    # --- table/
    if res.html is not None:
        put(p.file("table", "prep_tables.xlsx"), res.html.to_excel)
    if res.table1 is not None:
        put(p.file("table", "table1.xlsx"), res.table1.to_excel)
    if res.removed is not None:
        put(p.file("table", "除外の記録.xlsx"),
            lambda q: res.removed.to_excel(q, index=False))

    # --- data/ ： ★症例レベルのデータ★
    if save_data:
        _save_data(res, p, put)

    _paths.write_run_info(p, {"medprep_version": _pkg_version()})
    res.saved.append(p.file("table", "run_info.json"))

    # --- figure/
    if res.figures is not None and len(res.figures):
        try:
            res.saved += res.figures.save_all(p.figure)
        except Exception as e:                                       # noqa: BLE001
            step("保存: 図", False, f"書けなかった（{type(e).__name__}: {e}）")

    # --- report/
    if res.html is not None:
        put(p.file("report", "prep_report.html"), res.html.to_html)

    # ★キー名は既存ノートブックの LOG_CSV_COLUMNS と同じにする。★
    #   違う名前で書くと history.csv の列が 1 本増えるだけで、同じ列に並ばない。
    res.log.finish(**{
        "所要時間(秒)": round(time.time() - t0, 1),
        "欠損セル数": int(df.isna().sum().sum()) if df is not None else "",
        "説明変数の数": (res.prepared.X_train.shape[1] if res.prepared else
                    (len(res.schema.features()) if res.schema else "")),
        "入力ファイル(フルパス)": str(source),
        "確認事項": len(res.warnings),
    })
    res.saved.append(res.log.path)
    step("run フォルダに保存する", True, f"run{p.runnumber}  {p.run}")


def _save_data(res: PrepResult, p, put) -> None:
    """掃除済みデータと前処理済み行列を書き出す。

    **2 つは別ものである。**
      掃除済み  … 辞書で掃除し派生列を足した、**人が読める**データ
      前処理済み … スケール・符号化まで済ませた、**モデルに渡す**行列

    ★どちらも症例レベルのデータである。★ `run` の根には `.gitignore` を必ず置いているが、
    人に渡すときは中身を確かめること。
    """
    import os
    d = os.path.join(p.run, "data")
    os.makedirs(d, exist_ok=True)

    if res.df_clean is not None:
        put(os.path.join(d, "掃除済みデータ.xlsx"),
            lambda q: res.df_clean.to_excel(q, index=False))

    if res.prepared is None:
        return
    name = (res.schema.target or {}).get("name") if res.schema else None

    def _with_y(x, y):
        return x if (y is None or name is None) else x.assign(**{name: y})

    put(os.path.join(d, "前処理済み_train.xlsx"),
        lambda q: _with_y(res.X_train, res.y_train).to_excel(q, index=False))
    put(os.path.join(d, "前処理済み_test.xlsx"),
        lambda q: _with_y(res.X_test, res.y_test).to_excel(q, index=False))


def _removed_table(res: PrepResult, dfc, outcome) -> pd.DataFrame:
    """**何を、いくつ、なぜ減らしたか**を 1 枚の表にする。

    全自動の値打ちは「何をしたか」と同じくらい「何を捨てたか」で決まる。
    捨てたものを言わない自動化は、信用してはならない。
    """
    rows = []

    # --- 列：役割の推定で解析から外したもの
    if res.schema is not None:
        for c in res.schema.dropped():
            spec = res.schema.columns[c]
            rows.append({"種類": "列", "対象": str(c), "件数": 1,
                         "理由": spec.reason, "段": f"列の役割の推定（{spec.role}）"})

        # --- 列：残したが前処理に投入しなかったもの（日付・目的変数・生存時間の 2 列）
        if res.prepared is not None and res.prepared.preprocessor is not None:
            used = set(res.prepared.preprocessor.input_columns())
            for c in res.schema.kept():
                if c in used or c == outcome:
                    continue
                spec = res.schema.columns[c]
                rows.append({"種類": "列", "対象": str(c), "件数": 1,
                             "理由": f"役割 {spec.role} は特徴量にしない"
                                   f"（日付・生存時間の列はそのままでは説明変数にならない）",
                             "段": "前処理"})

    # --- 行：目的変数の欠測
    if res.dropped_outcome:
        info = res.dropped_outcome
        if info.get("除外"):
            rows.append({"種類": "行", "対象": f"目的変数 '{outcome}' が欠測",
                         "件数": int(info["除外"]),
                         "理由": str(info.get("理由", "")).splitlines()[0],
                         "段": "分割の前"})

    # --- 行：生存時間に変換できなかったもの
    sf = res.survival
    if sf is not None and len(getattr(sf, "excluded", [])):
        for reason, n in sf.excluded["理由"].value_counts().items():
            rows.append({"種類": "行", "対象": "生存時間に変換できない",
                         "件数": int(n), "理由": str(reason),
                         "段": "生存時間の形にする"})

    # --- 値：辞書で掃除して NaN にしたもの（行は消えない。値だけが消える）
    if res.clean_report is not None:
        for col, kind, n, detail in res.clean_report.actions:
            if "NaN" in str(kind) and n:
                rows.append({"種類": "値", "対象": str(col), "件数": int(n),
                             "理由": f"{kind}（{detail}）", "段": "辞書で掃除"})

    cols = ["種類", "対象", "件数", "理由", "段"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    order = {"列": 0, "行": 1, "値": 2}
    return (out.assign(_o=out["種類"].map(order))
               .sort_values(["_o", "件数"], ascending=[True, False])
               .drop(columns="_o").reset_index(drop=True))


def _write_version(path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_pkg_version() + "\n")


def _pkg_version() -> str:
    from . import __version__
    return f"medprep {__version__}"
