"""保存先の解決 — 既存教材（`SimpleRegressionHoldOutVer12_1` 等）の `run{N}` 規約。

```
ローカル: ~/lab_output/{手法名}/run{N}/{table,figure,model,report}/
Colab  : {COLAB_BASE}/lab_output/{手法名}/run{N}/...
ログ    : ~/lab_work/log  ／  {COLAB_BASE}/lab_work/log
```

★この規約を作り直してはならない。★
受講者の PC には既に run1, run2, ... が溜まっており、D&D アプリ（papermill
ランチャ）は `table/run_info.json` を読んで実行済みノートブックの置き場所を決める。
採番の仕組みを少しでも変えると、**過去の計算結果を上書きする。**

そのため `_existing_max_runnumber()` / `determine_runnumber()` / `create_directory()`
は既存ノートブックの実装を **1 文字も変えずに** 取り込んである
（docstring とコメントも当時のまま）。
`medprep` 側の都合で書き換えたくなったら、まずノートブックを直すこと。

既存ノートブックと違うのは 1 点だけで、設定値の渡し方である。
ノートブックは `globals().get('OUTPUT_DIR', '')` を読むが、パッケージには
そのノートブックのグローバルが無い。同じ名前のモジュール変数を置き、
`medprep.paths.OUTPUT_DIR = '...'` と書けば同じように効くようにした。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

# ================================================================== 設定
# 既存ノートブックの「環境変数セル」と同じ名前・同じ既定値。
PROJECT_DIR = ""        # '' → 使用中の .venv の位置から自動判定
WORK_DIR = ""           # '' → {PROJECT_DIR}_work
OUTPUT_DIR = ""         # '' → {PROJECT_DIR}_output
COLAB_BASE = "/content/drive/MyDrive/AI"

IN_COLAB: bool | None = None       # None → 自動判定。テストからは真偽値で固定する

#: run 番号の採番に使うフォルダ。★report を入れてはならない★
#: 既存ノートブックは ['table', 'model', 'figure'] で「使用済みか」を判定する。
#: ここに report を足すと、medprep が report だけ書いた run を
#: ノートブック側が「空き」とみなして上書きする。
RUN_FOLDERS = ["table", "model", "figure"]

#: 実際に作るフォルダ。report は D&D アプリが実行済み NB と HTML を置く場所。
ALL_FOLDERS = ["table", "model", "figure", "report"]

#: history.csv の列。★既存ノートブックの定義をそのまま使う。★
#: history.csv は log/*.json から毎回作り直す仕様なので、ここを狭めると
#: **過去のノートブック実行の列（最良モデル・R²）を消してしまう。**
LOG_CSV_COLUMNS = [
    "開始日時", "手法", "run", "状態", "入力ファイル名", "行数", "説明変数の数", "目的変数",
    "欠損セル数", "使用コード", "実行方法", "最良モデル(R2)", "最良R2", "所要時間(秒)",
    "出力フォルダ", "入力ファイル(フルパス)",
]

_GITIGNORE = "# 計算結果（個票データを含みうる。バージョン管理に載せない）\n*\n"

# 同じセッション内で続けて走らせたときに run を使い回さないための覚え書き。
# （ノートブックの `runnumber += 1` に相当する）
_LAST: dict[str, int] = {}


def in_colab() -> bool:
    """Colab で動いているか。"""
    if IN_COLAB is not None:
        return bool(IN_COLAB)
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ================================================================== 3 つの根
def resolve_project_dir(explicit: str | None = None) -> str:
    """uv の環境とプログラムのフォルダ（`~/lab`）。空なら .venv の位置から自動判定する。"""
    explicit = explicit if explicit is not None else PROJECT_DIR
    if explicit:
        return os.path.expanduser(str(explicit))
    if in_colab():
        return "/content"                           # Colab に ~/lab は存在しない
    if os.path.basename(sys.prefix) == ".venv":
        return os.path.dirname(sys.prefix)          # .../lab/.venv → .../lab
    return os.path.expanduser("~/lab")


def resolve_work_dir(explicit: str | None = None) -> str:
    """作業フォルダ（`~/lab_work`）。code / data / log がこの下にある。"""
    explicit = explicit if explicit is not None else WORK_DIR
    if explicit:
        return os.path.expanduser(str(explicit))
    if in_colab():
        return os.path.join(COLAB_BASE, "lab_work")
    p = resolve_project_dir()
    return os.path.join(os.path.dirname(p), os.path.basename(p) + "_work")


def resolve_output_dir(explicit: str | None = None) -> str:
    """計算結果の保存先（`~/lab_output`）。"""
    explicit = explicit if explicit is not None else OUTPUT_DIR
    if explicit:
        return os.path.expanduser(str(explicit))
    if in_colab():
        return os.path.join(COLAB_BASE, "lab_output")
    p = resolve_project_dir()
    return os.path.join(os.path.dirname(p), os.path.basename(p) + "_output")


def resolve_log_dir(explicit: str | None = None) -> str:
    """実行の記録を残す場所（`~/lab_work/log`）。"""
    return os.path.join(resolve_work_dir(explicit), "log")


# ================================================================== 採番
# ------------------------------------------------------------------
# ★ここから 3 つの関数は既存ノートブック（Ver12.1）からの写しである。★
#   引数の順序・名前・既定値を変えないこと。D&D アプリが壊れる。
# ------------------------------------------------------------------
def _existing_max_runnumber(project_directory):
    """保存先にある run1, run2, ... のうち最大の番号を返す（1つも無ければ 0）。

    Colab では実行のたびにランタイムが作り直され runnumber が 1 に戻るため、
    「番号を 1 から順に空きを探す」やり方だと、途中に中身の空の run が残っていた場合に
    そこへ戻って上書きしてしまう。ドライブに実際にある run 番号の最大値を下限にすることで、
    run1・run2・run3 があれば必ず run4 以降から始まるようにする。
    """
    try:
        names = os.listdir(project_directory) if os.path.isdir(project_directory) else []
    except OSError:
        return 0
    nums = [
        int(name[3:]) for name in names
        if name.startswith('run') and name[3:].isdigit()
        and os.path.isdir(os.path.join(project_directory, name))
    ]
    return max(nums) if nums else 0


def determine_runnumber(project_directory, folder_names, runnumber):
    """次に使う run 番号を決定する。

    ・保存先に既にある run 番号の最大値（と、同一セッションで使った runnumber）を下限にする。
    ・その番号の run が「未使用」（フォルダが無い、または table / model / figure がすべて空）
      なら、その番号をそのまま使う。空の run を無駄に増やさないため。
    ・使用済みなら 1 つずつ繰り上げる。
    """
    def _used(num):
        path_run = os.path.join(project_directory, f'run{num}')
        if not os.path.isdir(path_run):
            return False
        try:
            return any(
                os.path.isdir(os.path.join(path_run, str(f)))
                and os.listdir(os.path.join(path_run, str(f)))
                for f in folder_names
            )
        except OSError:
            return True          # 読めないときは「使用済み」とみなし、上書きを避ける

    num = max(int(runnumber) if runnumber else 1, _existing_max_runnumber(project_directory), 1)
    while _used(num):
        num += 1
    return num


def create_directory(path_run, folder_name):
    path_new = os.path.join(path_run, str(folder_name))
    os.makedirs(path_new, exist_ok=True)
    return path_new
# ------------------------------------------------------------------
# ★写しはここまで。★
# ------------------------------------------------------------------


# ================================================================== run
@dataclass
class RunPaths:
    """1 回の実行の保存先一式。"""

    runnumber: int
    project_name: str
    project_folder: str
    base: str
    project_directory: str
    run: str
    table: str
    model: str
    figure: str
    report: str
    log_dir: str
    notes: list = field(default_factory=list)

    def file(self, kind: str, name: str) -> str:
        """`paths.file('table', 'table1.xlsx')` → そのフルパス。"""
        if kind not in ALL_FOLDERS:
            raise ValueError(f"kind は {ALL_FOLDERS} のいずれか（渡されたのは {kind!r}）")
        return os.path.join(getattr(self, kind), name)

    def report_text(self) -> str:
        return (
            f"◇ Run number: run{self.runnumber}\n"
            f"◇ Directory for this run: {self.run}\n"
            f"◇   Tables:  {self.table}\n"
            f"◇   Models:  {self.model}\n"
            f"◇   Figures: {self.figure}\n"
            f"◇   Report:  {self.report}\n"
            f"◇   Log:     {self.log_dir}"
        )

    def show(self):
        print(self.report_text())


def ensure_gitignore(base: str) -> None:
    """結果フォルダの根に「全部無視」の .gitignore を必ず置く。

    run フォルダには個票由来の出力が集まる。誤って git リポジトリに載せる事故を
    防ぐため、実行のたびに確認する（受講者がフォルダを作り直しても復活する）。
    """
    try:
        os.makedirs(base, exist_ok=True)
        gi = os.path.join(base, ".gitignore")
        if not os.path.exists(gi):
            with open(gi, "w", encoding="utf-8") as f:
                f.write(_GITIGNORE)
    except OSError:
        pass                     # 読み取り専用の環境でも解析は止めない


def new_run(
    method: str = "Preprocessing",
    *,
    project_folder: str = "",
    output_dir: str | None = None,
    work_dir: str | None = None,
    create: bool = True,
) -> RunPaths:
    """次の `run{N}` を決めてフォルダを作る。

        p = mp.paths.new_run("Preprocessing")
        fig.savefig(p.file("figure", "km.png"))

    `create=False` なら場所を計算するだけで、何も作らない。
    """
    base = resolve_output_dir(output_dir)
    project_directory = os.path.join(base, str(project_folder), str(method))

    if create:
        ensure_gitignore(base)

    key = os.path.abspath(project_directory)
    num = determine_runnumber(project_directory, RUN_FOLDERS, _LAST.get(key, 0) + 1)
    _LAST[key] = num

    path_run = os.path.join(project_directory, f"run{num}")
    folders = {}
    for name in ALL_FOLDERS:
        folders[name] = (create_directory(path_run, name) if create
                         else os.path.join(path_run, name))

    log_dir = resolve_log_dir(work_dir)
    if create:
        os.makedirs(log_dir, exist_ok=True)

    return RunPaths(
        runnumber=num, project_name=str(method), project_folder=str(project_folder),
        base=base, project_directory=project_directory, run=path_run,
        log_dir=log_dir, **folders)


def write_run_info(paths: RunPaths, extra: dict | None = None) -> str:
    """`table/run_info.json` を書く。★D&D アプリ（papermill ランチャ）が読む。★

    キーの名前はランチャ側と合わせてある。減らしてはならない。
    """
    info = {
        "runnumber": paths.runnumber,
        "project_folder": paths.project_folder,
        "project_name": paths.project_name,
        "project_directory": paths.project_directory,
        "path_run": paths.run,
        "path_table": paths.table,
        "path_model": paths.model,
        "path_figure": paths.figure,
    }
    info.update(extra or {})
    p = os.path.join(paths.table, "run_info.json")
    os.makedirs(paths.table, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return p


# ================================================================== ログ
def rebuild_history_csv(log_dir: str) -> str:
    """log フォルダの全 JSON から history.csv を作り直す。

    ★列は `LOG_CSV_COLUMNS` ＋「JSON にあって表に無い列」の順にする。★
    既存ノートブックは固定の列だけを書き出すが、それだと medprep が書いた項目が
    落ちる。逆に medprep の列だけにすると、ノートブックの「最良R2」が消える。
    **どちらの実行も残るように、和をとって末尾に足す。**
    """
    rows, extra = [], []
    for p in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        for k in d:
            if k not in LOG_CSV_COLUMNS and k not in extra:
                extra.append(k)
        rows.append(d)

    columns = LOG_CSV_COLUMNS + extra
    table = [[d.get(c, "") for c in columns] for d in rows]
    table.sort(key=lambda r: str(r[0]))
    out = os.path.join(log_dir, "history.csv")
    # utf-8-sig にすると Excel で文字化けせずに開ける
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(table)
    return out


@dataclass
class RunLog:
    """1 回の実行の記録。`{作業フォルダ}/log/{手法}_run{N}_{日時}.json`。

    「どのデータを、どのコードで、どの設定で解析して、どこに出力したか」を残す。
    開始時に「実行中」として書き、終了時に「完了」で上書きする。
    """

    log_dir: str
    run_id: str

    @classmethod
    def start(cls, paths: RunPaths, **fields) -> RunLog:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = cls(log_dir=paths.log_dir,
                  run_id=f"{paths.project_name}_run{paths.runnumber}_{stamp}")
        log.write({
            "開始日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "手法": paths.project_name,
            "run": paths.runnumber,
            "状態": "実行中",
            "出力フォルダ": paths.run,
            **fields,
        })
        return log

    @property
    def path(self) -> str:
        return os.path.join(self.log_dir, self.run_id + ".json")

    def write(self, update: dict) -> str:
        """この実行のログ JSON に項目を追記し、history.csv を作り直す。"""
        os.makedirs(self.log_dir, exist_ok=True)
        d = {}
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                d = {}
        d.update(update)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, default=str)
        rebuild_history_csv(self.log_dir)
        return self.path

    def finish(self, **fields) -> str:
        return self.write({"状態": "完了", **fields})
