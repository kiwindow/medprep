"""第3回演習ノートブックを組み立てる。

★既存の .ipynb を上書きしない。★ VERSION を上げて別名で保存する。
"""
import json
import pathlib

C = []      # cells


def md(text):
    C.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")})


def code(text):
    C.append({"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": text.strip("\n").split("\n")})


# 改行を保つ（nbformat は各行末に \n が要る）
def _fix(cells):
    for c in cells:
        src = c["source"]
        c["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
    return cells


# ================================================================= 表紙
md("# ■ データの前処理（medprep）― 第3回 演習")

md("""
## □ 日本腎・血液浄化AI学会  学術委員会作成&nbsp;&nbsp;&nbsp;  made on September 18th, 2026

### ◇ MIT License&nbsp;&nbsp;&nbsp;&nbsp;Version 1.0 &nbsp;&nbsp;(Colab / Windows / Mac 共通版)

第1回・第2回で作った環境の上で、**解析を始める前の仕事**を扱う。
医学研究のデータは、そのままでは解析にかけられない。日付が 8 通りの書き方で混ざり、
欠損が `999` や「未測定」と書かれ、単位が途中で変わり、測定法が変わって値が段差になる。
**この回で覚えるのは、それらを見つけて記録に残す手順**である。

### ◇ この演習の一番大事な考え方

> **全自動とは「決定の自動化」ではなく「決定の明示化」である。**

`medprep` は 1 行で最後まで走る。しかし**そこで下した判断はすべて外に出る**。
「この列は ID とみなした」「この値は欠損コードとみなした」「この症例は除外した」——
その理由が `schema.yaml` と「人の確認が要る事項」に残り、**人が直して再実行できる**。

**確認事項が空で返ってきたら、まずそちらを疑うこと。** 医学データで所見ゼロはまず無い。

### ◇ このコードの構成

第１部  環境設定: 最初に**一回実行**。新たなデータ分析時は再実行<br>
   　1) Google Drive にマウント: **Colaboratory** なら必ず実行<br>
   　2) 環境変数の一括設定: 最初に**一回実行**<br>
   　3) Google Colaboratory への Library のインストール: **Colaboratory** なら必ず実行<br>
   　4) データの入力: 新規のデータのたびに実行<br>

第２部  演習（上から順に実行する）<br>
   　演習① まず全自動で走らせる（`mp.autoprep` 1 行）<br>
   　演習② 機械の判断を検分する（`schema` と監査）<br>
   　演習③ 欠損と外れ値 ―― **消してよい場合と消してはいけない場合**<br>
   　演習④ 透析前後の 2 点 ―― 採血時点の対応づけと透析量の指標<br>
   　演習⑤ Table 1 / Table 2 ―― そのまま論文に貼れる形（日英・Excel・Word）<br>
   　演習⑥ 生存時間分析 ―― KM → log-rank → Cox → 比例ハザードの確認<br>
   　演習⑦ `schema.yaml` を直して再実行し、モデルに渡す<br>

第３部  発展事項<br>

### ◇ 分析するデータの準備

1. Excel（または CSV）に、**1列が1つの変数、1行が1症例**となるように入力する。
2. **目的変数の位置は自由**。名前で指定する（環境変数セルの `OUTCOME`）。
3. **記号や単位が混ざっていてもよい。** `<0.1`、`999`、「未測定」、全角数字はそのまま入れてよい。
   何をどう解釈したかは必ず記録に残る。
4. 不明な値は**空欄**にする。**空欄を 0 で埋めてはならない。**
5. 日付は書き方が混ざっていてよい（和暦・Excel シリアル値・全角・時刻付き）。

### ◇ 使い方

◆ **Colaboratory**: このコードを Google Drive にアップロードして開き、上から順に実行する。<br>
◆ **Windows / Mac**: `jl` で JupyterLab を立ち上げ、**code** にこのコードを置いて開く。
分析する Excel / CSV は **data** に入れる。

計算結果は Colab なら `MyDrive/AI/lab_output/Preprocessing`、PC なら `~/lab_output/Preprocessing`
の中に `run1`, `run2`, ... という新しいフォルダが作られて保存される。
**過去の結果が上書きされることはない。** 実行の記録は **log** フォルダの `history.csv` に貯まる。

### ◇ 自分のデータが無い場合

「1. 環境構築 → 4) データの入力」で **［演習用の合成データを使う］** を押せば、
**合成透析コホート**（600 例・実データは一切含まない）で最後まで通せる。
上に挙げた汚れがすべて意図的に仕込んである。

自分のデータを使うときは **［ファイルを選ぶ］**（Colab では **［PC からアップロード］** も）
を押す。どちらを選んでも、その先の手順は同じである。
""")

# ================================================================= 第1部
md("# ◆ 1. 環境構築 (最初に一回だけ実行する)")

md("## 1) Google Colaboratoryを使う場合このNotebookをGoogle Driveにマウントする<br>"
   "（ローカルPCでは実行してもエラーにはならない）")

code("""
# このセルは Google Colaboratory で Google Drive をマウントする。
# ローカル（Windows / Mac）では IN_COLAB が False になり、マウントはスキップされる。
import sys

# 実行環境の判定（このセルを単独で実行できるよう、ここでも判定しておく。第1部2)でも再判定する）
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
else:
    print('◇ ローカル環境のため Google Drive のマウントはスキップした')

print(f"Python version: {sys.version.split()[0]}")
""")

md("## 2) 環境変数の一括設定&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ここを編集することで解析の対象と設定を変えられる<br>"
   "（環境変数を書き換えたらこのセルを再実行）")

code("""
# ============================================================================
# ■ 環境変数の一括設定
#   ここを書き換えて再実行するだけで、以降のすべてに反映される。
# ============================================================================
import os

# 実行環境の判定（Colab / ローカル：Windows・Mac 共通）
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ----- 保存先（第1回・第2回で作ったフォルダの規約をそのまま使う） -----
#   ~/lab          … uv の環境とプログラム        PROJECT_DIR
#   ~/lab_work     … code / data / log            WORK_DIR
#   ~/lab_output   … 計算結果 run1, run2, ...     OUTPUT_DIR
#   どれも '' のままなら自動で決まるので、触る必要はない。
PROJECT_DIR = ''
WORK_DIR    = ''
OUTPUT_DIR  = ''
COLAB_BASE  = '/content/drive/MyDrive/AI'

# ----- このノートブックの名前（実行の記録に残す） -----
NOTEBOOK_NAME = 'Preprocessing_Ver1_0'
RUN_BY        = '手動（JupyterLab / Colab）'
PROJECT_NAME  = 'Preprocessing'      # lab_output/{ここ}/run{N}/ に保存される

# ----- 読み込むデータ -----
#   ★どのデータを使うかは「4) データの入力」の**ボタン**で選ぶ。ここでは決めない。★
#   LOCAL_DATA_PATH はパスを書いたときだけ使う（自動実行・D&Dアプリ用の逃げ道）。
#   ふだんは空のままでよい。
LOCAL_DATA_PATH = ''

# ----- 列の指定（自分のデータを使うときはここを書き換える） -----
#   ★ここで指定した役割は、機械の推定より強い。★
#   ★空にしておくと、4) でデータを読んだあとに**列名の一覧から選ぶ画面**が出る。★
#   ここに書けばその画面は出ない（書いたほうが強い）。
ID_COL  = '仮名ID'        # 患者を識別する列。分割のときに同じ患者を両側に入れないために使う
GROUP   = ''              # 群間比較・グループ分割に使う列。'' なら 4) で選ぶ
DATE_COL = '検体採取日'    # 測定法変更の段差を調べるための日付列（無ければ None）

# 生存時間：観察開始日 / イベント発生日 / 打ち切り日 の 3 列。() なら 4) で選ぶ
SURVIVAL_DATES = ()

# 目的変数（回帰・分類をするとき）。None なら 4) で選ぶ（skip もできる）
OUTCOME = None
TASK    = None            # 'regression' / 'classification' / 'survival'

# ----- 分割 -----
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ----- レポート -----
#   ★既定では症例レベルの値（仮名 ID など）をレポートに出さない。★
#   レポートは HTML 1 枚でメールに乗る。必要なときだけ True にすること。
SHOW_VALUES = False

print('◇ 環境変数を設定した')
print(f'   IN_COLAB = {IN_COLAB}  （True=Colab / False=ローカルPC）')
print(f'   PROJECT_NAME = {PROJECT_NAME}')
""")

md("## 3) Google Colaboratoryへのライブラリのインストール<br>（ローカルPCで実行してもスキップされる）")

code("""
# ライブラリのインストール
#   Colab        : 実行のたびに環境が初期化されるため、毎回 pip で導入する。
#   ローカル(uv) : 事前に uv で導入済みのため、このセルはスキップされる。
#
# ★このノートブックが必要とする medprep の版★
#   古い medprep が入っていると、あとのセルが AttributeError で止まる。
#   ここで版を確かめて、足りなければ**理由を言って止める**。
REQUIRED_MEDPREP = (0, 9, 0)

if IN_COLAB:
    # -U（更新）と --no-cache-dir を付ける。付けないと、同じ版番号のまま
    # 中身だけ更新された medprep を pip が「導入済み」とみなして取りに行かない。
    !pip install -q -U --no-cache-dir "medprep @ git+https://github.com/kiwindow/medprep"
    !pip install -q ipyfilechooser
    print('◇ Colab: medprep を導入した')
else:
    print('◇ ローカル（uv）環境: pip install はスキップした')
    print('   ※ 未導入・古い場合は、ターミナルで次を実行すること:')
    print('      uv add --upgrade "medprep @ git+https://github.com/kiwindow/medprep"')
    print('      uv add ipywidgets ipyfilechooser        # 4) のボタンとファイル選択に使う')

print('')

import warnings

import matplotlib_fontja  # noqa: F401   日本語フォント

import medprep as mp

# 警告表示の抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 環境変数セルで決めた保存先を medprep に渡す
mp.paths.PROJECT_DIR = PROJECT_DIR
mp.paths.WORK_DIR    = WORK_DIR
mp.paths.OUTPUT_DIR  = OUTPUT_DIR
mp.paths.COLAB_BASE  = COLAB_BASE

def _ver(text):
    return tuple(int(x) for x in str(text).split('.')[:3])


if _ver(mp.__version__) < REQUIRED_MEDPREP:
    need = '.'.join(map(str, REQUIRED_MEDPREP))
    for line in [
        f'★medprep が古い（入っているのは {mp.__version__}、必要なのは {need} 以上）★',
        '',
        '  Colab   : このセルをもう一度実行したうえで、',
        '            「ランタイム → セッションを再起動する」を実行し、上から流し直すこと。',
        '            ★一度読み込まれた古い medprep は、再起動しないと入れ替わらない。★',
        '',
        '  ローカル : uv add --upgrade "medprep @ git+https://github.com/kiwindow/medprep"',
    ]:
        print(line)
    raise RuntimeError(f'medprep が古い（{mp.__version__} < {need}）。上の指示に従うこと')

print(f'◇ medprep {mp.__version__}')
print(f'   結果の保存先 : {mp.paths.resolve_output_dir()}')
print(f'   実行の記録   : {mp.paths.resolve_log_dir()}')
""")

md("""
## 4) **データの入力** (.xlsx, .xls, .csv) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;新たなデータを使う時にも実行

**このセルを実行すると、ボタンが 3 つ出る。どれかを押すとデータが読み込まれる。**

| ボタン | 何をするか |
|---|---|
| **演習用の合成データを使う** | 汚れを仕込んだ合成透析コホート（600 例・実データは含まない）を作って読む |
| **ファイルを選ぶ** | 手元のフォルダ（Colab では Google ドライブ）から選んで読む |
| **PC からアップロード**（Colab のみ） | PC のファイルを Colab に送ってから読む |

- **記号・単位・全角・和暦が混ざっていてよい。** どう解釈したかは必ず記録に残る。
- ボタンを押すまで `df`（データ）は空のままで、**先のセルは止まる**。
""")

code("""
import contextlib

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display

DATA_DIR = os.path.join(mp.paths.resolve_work_dir(), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

df = None          # ★ここに読み込んだデータが入る。ボタンを押すまでは空。★
data_path = None
_out = widgets.Output()


# ---------------------------------------------------------------- 列を選ぶ
#   ★環境変数に書いてあればそれが最優先。★ 空のときだけここが出る。
#   列名をすべて並べ、クリックで選ぶ。skip もできる。
_NONE = '（指定しない / skip）'


def _guess(cols, *words):
    # 列名に手がかりの語を含むものを探す。★見つからなければ選ばない★
    for w in words:
        for c in cols:
            if w in str(c):
                return c
    return _NONE


def _suggest_task(s):
    # 目的変数の見た目から回帰か分類かを提案する。★決めつけない。提案である★
    if s is None:
        return _NONE
    v = s.dropna()
    if v.empty:
        return _NONE
    if not pd.api.types.is_numeric_dtype(v) or v.nunique() <= 10:
        return 'classification'
    return 'regression'


def _ask_settings(d, auto=False):
    global GROUP, OUTCOME, TASK, SURVIVAL_DATES
    cols = list(d.columns)
    need_g = not GROUP
    need_o = not OUTCOME
    need_s = not SURVIVAL_DATES
    if not (need_g or need_o or need_s):
        print('◇ 群分け・目的変数・生存時間の列は環境変数で指定済み。選択画面は出さない。')
        return

    print('')
    print('─' * 70)
    print('◇ この表の列（この中から選ぶ）')
    for i in range(0, len(cols), 3):
        print('   ' + '  '.join(f'{j + 1:2d}. {cols[j]:<24s}' for j in range(i, min(i + 3, len(cols)))))
    print('─' * 70)

    opts = [_NONE] + cols
    w_g = widgets.Dropdown(options=opts, value=_guess(cols, '施設', '群', '病院', 'group'),
                           description='GROUP', layout=widgets.Layout(width='420px'))
    w_o = widgets.Dropdown(options=opts, value=_NONE,
                           description='OUTCOME', layout=widgets.Layout(width='420px'))
    w_t = widgets.Dropdown(options=[_NONE, 'regression', 'classification', 'survival'],
                           value=_NONE, description='TASK',
                           layout=widgets.Layout(width='420px'))
    w_s1 = widgets.Dropdown(options=opts, value=_guess(cols, '観察開始', '開始年月日', '登録日'),
                            description='観察開始日', layout=widgets.Layout(width='420px'))
    w_s2 = widgets.Dropdown(options=opts, value=_guess(cols, 'event発生', 'イベント', '発生年月日'),
                            description='イベント日', layout=widgets.Layout(width='420px'))
    w_s3 = widgets.Dropdown(options=opts, value=_guess(cols, '打ち切り', '打切', '最終観察'),
                            description='打ち切り日', layout=widgets.Layout(width='420px'))

    def _on_outcome(change):
        if w_t.value == _NONE and change['new'] != _NONE:
            w_t.value = _suggest_task(d[change['new']])
    w_o.observe(_on_outcome, names='value')

    box, out2 = [], widgets.Output()

    def _apply(_=None, skip=False, quiet=False):
        global GROUP, OUTCOME, TASK, SURVIVAL_DATES
        with (contextlib.nullcontext() if quiet else out2):
            if not quiet:
                clear_output()
            if not skip:
                if need_g and w_g.value != _NONE:
                    GROUP = w_g.value
                if need_o and w_o.value != _NONE:
                    OUTCOME = w_o.value
                if need_o and w_t.value != _NONE:
                    TASK = w_t.value
                trio = (w_s1.value, w_s2.value, w_s3.value)
                if need_s and all(x != _NONE for x in trio):
                    SURVIVAL_DATES = trio
            print(f'◇ GROUP          = {GROUP or "（指定なし）"}'
                  '   ← 指定があれば Table 2（群間比較）を作る')
            print(f'◇ OUTCOME / TASK = {OUTCOME or "（指定なし）"} / {TASK or "（指定なし）"}')
            print(f'◇ SURVIVAL_DATES = {SURVIVAL_DATES or "（指定なし）"}')
            print('')
            print('★ここで選ばなくても先へ進める。★ 演習①は群分け無しで走り、'
                  'Table 1 だけが作られる。')

    if need_g:
        box.append(w_g)
    if need_o:
        box += [w_o, w_t]
    if need_s:
        box += [w_s1, w_s2, w_s3]

    if auto:
        # 自動実行（LOCAL_DATA_PATH を書いたとき・D&Dアプリ）は待たずに当てはめる
        _apply(quiet=True)
        return
    b_ok = widgets.Button(description='この指定で進む', button_style='primary',
                          icon='check', layout=widgets.Layout(width='200px', height='38px'))
    b_no = widgets.Button(description='指定しない（skip）', icon='forward',
                          layout=widgets.Layout(width='200px', height='38px'))
    b_ok.on_click(_apply)
    b_no.on_click(lambda _: _apply(skip=True))
    display(widgets.VBox(box), widgets.HBox([b_ok, b_no]), out2)


def _load(path):
    # ★読み込みで推測したこと（文字コード・見出し行）は必ず記録に残る★
    global df, data_path
    if not path:
        print('★ファイルが選ばれていない★')
        return
    data_path = path
    df = mp.read_any(path)
    for note in df.attrs.get('medprep_read', []):
        print('  ・', note)
    print('')
    print(f'◇ {df.shape[0]} 行 × {df.shape[1]} 列 を読み込んだ')
    print(f'   {path}')
    display(df.head())
    _ask_settings(df, auto=bool(LOCAL_DATA_PATH))


def _use_demo(_=None):
    with _out:
        clear_output()
        # ★押すたびに作り直す。★
        #   以前は「ファイルが無ければ作る」にしていたが、medprep の版が上がって
        #   列が増えても**古いファイルが残ったまま**になり、透析前後の列が無い、
        #   という事故が起きた。合成データは seed 固定なので、作り直しても中身は同じ。
        p = os.path.join(DATA_DIR, 'synthetic_dialysis_cohort.xlsx')
        mp.demo.save(p)
        print(f'◇ 演習用の合成データを作り直した（medprep {mp.__version__}）: {p}')
        print('◇ 演習用の合成透析コホート（600 例。実データは一切含まない）')
        print('   透析前後の 2 点（BUN・Cr・K・体重）、身長、透析時間を含む')
        print('   欠損コード・検出限界・単位混在・日付の表記ゆれ・測定法の段差を仕込んである')
        _load(p)


def _choose_file(_=None):
    with _out:
        clear_output()
        from ipyfilechooser import FileChooser
        drive = '/content/drive/MyDrive'
        start = drive if (IN_COLAB and os.path.isdir(drive)) else DATA_DIR
        fc = FileChooser(start, filter_pattern=['*.xlsx', '*.xls', '*.csv'])
        fc.title = '<b>ファイルを選ぶと読み込む</b>'
        fc.register_callback(lambda chooser: _load(chooser.selected))
        display(fc)
        if IN_COLAB:
            print('※ PC にあるファイルは、左の【ファイル】にドラッグして置くか、')
            print('   Google ドライブに入れてから選ぶこと')


def _upload(_=None):
    with _out:
        clear_output()
        from google.colab import files
        up = files.upload()
        if not up:
            print('★ファイルが選ばれなかった★')
            return
        name = next(iter(up))
        p = os.path.join(DATA_DIR, name)
        with open(p, 'wb') as f:
            f.write(up[name])
        print(f'◇ {name} を {DATA_DIR} に置いた')
        _load(p)


_size = widgets.Layout(width='250px', height='42px')
_b_demo = widgets.Button(description='演習用の合成データを使う', button_style='primary',
                         icon='flask', layout=_size)
_b_file = widgets.Button(description='ファイルを選ぶ', icon='folder-open', layout=_size)
_b_demo.on_click(_use_demo)
_b_file.on_click(_choose_file)
_buttons = [_b_demo, _b_file]

if IN_COLAB:
    _b_up = widgets.Button(description='PC からアップロード', icon='upload', layout=_size)
    _b_up.on_click(_upload)
    _buttons.append(_b_up)

if LOCAL_DATA_PATH:
    # パスが書いてあるときはボタンを出さずに読む（自動実行・D&Dアプリ用）
    _load(os.path.expanduser(LOCAL_DATA_PATH))
else:
    print('◇ どれかのボタンを押すこと。押すまで先のセルは止まる。')
    display(widgets.HBox(_buttons), _out)
""")

# ================================================================= 演習1
md("""
# ◆ 2. 演習① まず全自動で走らせる

読み込みから前処理済みの行列とレポートまでを 1 行で通す。

**ここで見るのは結果ではなく、「機械が何を判断したか」である。**
段ごとの成否と、人の確認が要る事項が最後にまとめて出る。
""")

code("""
if df is None:
    raise RuntimeError(
        '★データが選ばれていない★  '
        '「1. 環境構築 → 4) データの入力」に戻り、ボタンを押してから、このセルを実行すること')

rep = mp.autoprep(
    df,
    group=GROUP or None, id_col=ID_COL, date_col=DATE_COL,
    survival_dates=SURVIVAL_DATES or None,
    outcome=OUTCOME or None, task=TASK or None,
    test_size=TEST_SIZE, seed=RANDOM_STATE,
    show_values=SHOW_VALUES,
    save=True, method=PROJECT_NAME,      # run{N}/ に全出力を保存する
)
""")

code("""
# 段ごとの成否と、人の確認が要る事項
rep.show()
""")

md("""
## 何を減らしたか

**全自動の値打ちは「何をしたか」と同じくらい「何を捨てたか」で決まる。**
捨てたものを言わない自動化は、信用してはならない。

| 種類 | 何が減るか |
|---|---|
| **列** | ID・重複列・自由記載は解析から外す。日付は特徴量にしない |
| **行** | ★1 行も削除しない。★ 外すのが望ましい症例に **`除外推奨` の印**を付けるだけ |
| **値** | 欠損コード（999）・あり得ない値を NaN にする（★行は消えない★） |

> **なぜ行を削除しないのか**
>
> 削除すると症例数と並びが変わり、**元のデータと症例ごとに `axis=1` で結合し直せなくなる**。
> 外すかどうかは医学的な判断であって、自動化が黙って決めることではない。
> `除外推奨` の列を見て、人が決める。
>
> ```python
> # 印の付いた症例を外して解析するなら
> use = rep.df_clean[rep.df_clean['除外推奨'] == 0]
> ```
>
> ただし **`rep.X_train` / `X_test` は目的変数が欠測の症例を含まない**（含めると学習できない）。
> こちらは必然的に部分集合なので、書き出したファイルに `元の行` と ID を付けてある。
""")

code("""
print(f'元のデータ {len(df)} 行 → 掃除済み {len(rep.df_clean)} 行（★変わらない★）')
print(f'外すのが望ましい行: {rep.n_excluded} 例（印だけ）')
print()
rep.df_clean.loc[rep.df_clean['除外推奨'] == 1,
                 ['仮名ID', '除外推奨', '除外推奨_理由']].head(10)
""")

code("""
rep.removed
""")

md("""
## 二値の列を 0/1 に直した ―― **1 がどちらかを列名で示す**

性別のように水準が 2 つの列は **0/1 の 1 本**にまとめる。
問題は「列が何本になるか」ではなく、**どちらの水準が 1 になるか**である。

機械の都合（辞書順）で決めさせると、`男/女` で記録した施設は `性別=男`、
`M/F` の施設は `性別=M` と、**同じ意味の列が別名になる。**
施設をまたいで結合した瞬間に破綻する。だから **意味で** 決める。

| 元の列 | 元の値 | 作る列 | |
|---|---|---|---|
| 性別 | 男 / 女、男性 / 女性、M / F、Male / Female | **男性** | 1=男性・0=女性 |
| 糖尿病 | あり / なし | 糖尿病 | 1=あり・0=なし |
| 転帰 | 生存 / 死亡 | **死亡** | 1=死亡・0=生存 |

`糖尿病` の列名を変えないのは、1 側の水準名が「あり」だからである。
列名を「あり」にしたら**何の列か分からなくなる。**

★どのファイルに効くか★
`解析用データ.xlsx` と `前処理済み_*.xlsx` には効く。
**`掃除済みデータ.xlsx` には効かない** ── あれは<u>人が元の記録と突き合わせる</u>
ためのものなので、`男/女` のまま残してある。
""")

code("""
rep.encoded
""")

code("""
# 解析用データはこうなっている（性別 → 男性）
print(mp.frame_text(rep.df_use[['施設', '男性', '年齢', '糖尿病']].head(8)))
""")

md("""
## 書き出されたもの ―― **3 つはどう違うのか**

同じデータを **3 段階**で書き出してある。**施した処置が 1 段ずつ違う。**
どれを使うかで結果が変わるので、ここを取り違えないこと。

| ファイル | 行 | 列 | 何のため |
|---|---|---|---|
| `data/掃除済みデータ.xlsx` | **元と同じ** | **全列** | **人が読む。** 元の記録と症例ごとに突き合わせる（ID が残っている） |
| `data/解析用データ.xlsx` | **元と同じ** | 落とす列を除く | **自分で解析する。** 行が減っていないので元データと `axis=1` で結合できる |
| `data/前処理済み_train.xlsx` / `_test.xlsx` | ★部分集合★ | 特徴量のみ | **モデルに渡す。** 元の行番号と ID が付いている |

### 段ごとに、何が足されるか

```
掃除済み   ① 欠損コード（999 等）と「未測定」を NaN に
           ② 検出限界（<0.1）を数値化      ③ 単位の混在を換算
           ④ 生理学的にあり得ない値を NaN に ⑤ 派生指標を追加（補正Ca 等）
           ⑥ 日付を解釈（和暦・全角・シリアル値）
           ⑦ 外すのが望ましい行に印（★行は削除しない★）
   ↓
解析用     ＋ ⑧ 解析に使わない列を削除（ID・重複列・自由記載）
           ＋ ⑧b 二値の列を 0/1 に（性別 → **男性**：1=男性・0=女性）
   ↓
前処理済み  ＋ ⑨ 目的変数が欠測の症例を除く   ⑩ train / test に分割
           ⑪ 欠損を補完  ⑫ スケーリング  ⑬ カテゴリをダミー化
           （★⑪〜⑬ は train だけで fit する★）
           ※二値（男/女・男性/女性・M/F）は 0/1 の 1 本にまとめる。
             列名は **男性**（1=男性・0=女性）。書き方が違っても同じ列名になる。
```

**⑨ でだけ行が減る。** だから前処理済みにだけ `元の行` と ID の列を付けてある。

そのほかに `table/書き出したデータの説明.xlsx`、`model/schema.yaml`（★再現性の中核★）、
`model/pipeline.pkl`、`report/prep_report.html` が保存される。
HTML レポートの「**1c. 書き出したデータ**」にも同じ表が入っている。

**`data/` は症例レベルのデータである。** 置き場所の根には毎回 `.gitignore` を入れているが、
人に渡すときは中身を確かめること。要らなければ `save_data=False` を渡す。
""")

code("""
rep.outputs
""")

code("""
for f in rep.saved:
    print(' ', f)
""")

md("""
## レポートを開く

`run{N}/report/prep_report.html` を開くと、監査・列の役割・欠損・外れ値・Table 1・
図が **HTML 1 枚**にまとまっている。図は埋め込みなので、そのままメールに乗る。

**冒頭に致命的な所見が並んでいる。** 10 枚目にある警告は、読まれないのと同じだからである。

> **ノートブックの中に埋め込んでは見ない。** 結果の置き場所（`lab_output`）は
> JupyterLab の表示範囲の外にあり、Colab では Google ドライブの中にある。
> どちらも**出力枠からは開けない**。下のセルが、環境に応じて手元で開く。
""")

code("""
# レポートをブラウザで開く。Colab とローカルで開き方が違う。
#
# ★ノートブックの出力枠に貼り込むことはしない。★
#   ローカル : 結果は ~/lab_output にあり、JupyterLab の表示範囲（~/lab_work）の外。
#   Colab    : 結果は Google ドライブの中にあり、出力枠からは参照できない。
# レポートは「HTML 1 枚でメールに乗る」ものなので、ブラウザで開くのが本来の使い方である。
def open_report(path):
    print(path)
    print(f'  （{os.path.getsize(path) / 1024 / 1024:.1f} MB）')
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(path)      # 手元に落ちる。ダウンロード欄から開く
            print('◇ ダウンロードした prep_report.html をブラウザで開くこと')
            print('   （Google ドライブの上のパスにも同じものが残っている）')
        except Exception as e:
            print(f'※ 自動ダウンロードできなかった（{e}）。')
            print('   左の【ファイル】から drive/MyDrive/AI/lab_output/… をたどり、')
            print('   prep_report.html を右クリックしてダウンロードすること')
    else:
        import webbrowser
        opened = webbrowser.open('file://' + os.path.abspath(path))
        print('◇ 既定のブラウザで開いた' if opened
              else '※ 開けなかった。上のパスをブラウザにドラッグして開くこと')


open_report(rep.run.file('report', 'prep_report.html'))
""")

# ================================================================= 演習2
md("""
# ◆ 3. 演習② 機械の判断を検分する

### ここが第3回の山場である。

`autoprep` が下した判断には**必ず理由が付いている**。
その理由を読み、納得できなければ直す。これが「決定の明示化」の意味である。
""")

md("## 1) 列の役割（schema）")

code("""
print(rep.schema.report())
""")

code("""
# 表で見る。「判断の根拠」の列を必ず読むこと。
rep.schema.to_frame()
""")

md("""
**確かめること**

- `drop` になった列は、本当に捨ててよいか（`id` / `duplicate_of` / `high_cardinality`）
- `numeric` になった列に、数値でないものが混ざっていないか
- `datetime` の列が、日と月を取り違えていないか

直したいときは `schema.yaml` を編集して読み直す。
""")

code("""
yaml_path = rep.run.file('model', 'schema.yaml')
print(yaml_path)
print()
with open(yaml_path, encoding='utf-8') as f:
    print(f.read()[:1500], '...')
""")

md("## 2) 品質監査 ―― このまま解析してよいかを問う")

code("""
# ★audit はデータを直さない。報告するだけである。★
rep.audit.show()
""")

code("""
# 表にして、致命的な所見から順に見る
rep.audit.to_frame()
""")

md("""
### 測定法の変更に注目する

合成データには **2020 年 4 月の ALP 測定法変更（JSCC 法 → IFCC 法）** が仕込んである。
値がおよそ 1/3 になるが、**病態は何も変わっていない。**

この段差を見落とすと「2020 年以降、患者の ALP が下がった」という誤った結論になる。
検定でも例外でも捕まらない。**日付で切って中央値を比べて初めて見える。**
""")

code("""
steps = mp.method_change_steps(df, date_col=DATE_COL)
steps
""")

# ================================================================= 演習3
md("""
# ◆ 4. 演習③ 欠損と外れ値

### 「測っていない」ことが情報である

重症だから測った／軽症だから測らなかった、という欠損は予後と結びついている。
欠損を黙って中央値で埋めると、**その情報を捨てたうえに分布を歪める。**
""")

code("""
print(rep.missing.report())
""")

code("""
# 欠損の地図。縦の筋が揃っていれば、同じ検査がまとめて行われなかったということ。
rep.missing.plot(rep.df_clean, kind='matrix')
""")

md("""
### 欠損は偏っているか

欠損が完全にランダム（MCAR）なら、欠損フラグはどの列とも関連しないはずである。
関連があれば単純補完は分布を歪める。**「どの列の欠損がどの列と結びついているか」を名指しする。**
""")

code("""
rep.mcar
""")

md("""
## 外れ値 ―― 消してよい場合と消してはいけない場合

| | 例 | 扱い |
|---|---|---|
| **入力ミス** | Hb 0 g/dL、年齢 250 歳 | **生理学的にあり得ない。** 辞書の範囲で NaN にする |
| **本物の外れ値** | Alb 1.8 g/dL、CRP 25 mg/dL | **消してはならない。** 医学ではこれが重要な症例でありうる |

この区別は統計では付かない。**医学領域辞書（`dict/ranges_ja.yaml`）が担う。**
`medprep` は既定で外れ値を削除しない。検出して報告するだけである。
""")

code("""
print(rep.outliers.report())
""")

code("""
rep.outliers.table
""")

# ================================================================= 演習4
md("""
# ◆ 5. 演習④ 透析前後の 2 点 ―― **どちらが「前」かを推測で決めない**

演習用の合成データには、**透析前と透析後の 2 点**が入っている。

| 項目 | 透析前 | 透析後 |
|---|---|---|
| 尿素窒素 | `透析前BUN` | `透析後BUN` |
| クレアチニン | `透析前クレアチニン(Cr)` | `透析後クレアチニン(Cr)` |
| カリウム | `透析前カリウム(K)` | `透析後カリウム(K)` |
| 体重 | `透析前体重` | `透析後体重` |

そのほか `身長`・`透析時間` も入っている。
これで **URR・spKt/V・nPCR・%CGR・GNRI・BMI** が算出できる。

★この合成データには汚れが 1 つ仕込んである。★
**1 施設だけ、前後の列が入れ替わっている。**
実務でいちばんよくある形である（施設ごとにエクスポートの仕様が違う）。
透析後 BUN のほうが高いのは生理学的にあり得ず、**URR が負になる。**
気づかずに計算すると、その施設だけ透析量が異常に低く見える。
""")

code("""
# 列が実際に入っているか、自分の目で確かめる
pre_post = [c for c in rep.df_clean.columns if c.startswith(('透析前', '透析後'))]
print('透析前後の列:', pre_post)
print()
print(mp.frame_text(rep.df_clean[['施設', '透析前BUN', '透析後BUN',
                                  '透析前体重', '透析後体重']].head(8)))
""")

code("""
# 採血時点の対応づけ（★推測しない。対応が付いた列だけを使う★）
ts = mp.TimingSchema.infer(rep.df_clean)
print(ts.report())
""")

code("""
# 前後の方向を検査する。★施設別に見ると、偏っているのが分かる★
rev = (rep.df_clean['透析後BUN'] > rep.df_clean['透析前BUN'])
print(mp.frame_text(
    rep.df_clean.assign(前後が逆=rev.astype(int))
       .groupby('施設')['前後が逆'].agg(['sum', 'count', 'mean'])
       .rename(columns={'sum': '逆の例数', 'count': '例数', 'mean': '割合'})
       .round(3).reset_index()))
""")

md("""
### 透析量の指標は**掃除の段で自動的に作られる**

揃っている列の組についてだけ、次の列が作られる（★揃っていなければ作らない★）。

| 作る列 | 要る列 | 式 |
|---|---|---|
| `除水量(kg)` | 透析前体重・透析後体重 | 前 − 後 |
| `URR(%)` | 透析前BUN・透析後BUN | (前 − 後) / 前 × 100 |
| `透析時間(hr)` | 透析開始時刻・透析終了時刻 | 終了 − 開始（日またぎは +24h） |
| `spKt/V` | 前後BUN・除水量・透析後体重・透析時間 | Daugirdas 第2世代式 |
| `TSAT(%)` | Fe・TIBC | Fe / TIBC × 100 |
| `iCa(mg/dL)` | Ca・Alb | Ca + (4 − Alb) |

spKt/V = −ln(R − 0.008t) + (4 − 3.5R)·UF/W
（R = 透析後BUN/透析前BUN、t = 透析時間[hr]、**UF = 除水量[L]**、W = 透析後体重[kg]）

★UF は L で入れる。★ mL で入れると 1000 倍になり、Kt/V が 2000 になる。
水 1 L = 1 kg なので、`除水量(kg)` の数値をそのまま L として使ってよい。

前後が逆の症例では URR が負になる。**NaN にして黙って埋めたりはしない。**
値が出ないことを見せるのが正しい。
""")

code("""
d = rep.df_clean
urr = mp.urr(bun_pre=d['透析前BUN'], bun_post=d['透析後BUN'])
# ★medprep は同じ指標を掃除の段で自動的に作っている（列名に単位が付く）★
out = d[['施設', '透析時間(hr)', '除水量(kg)', 'URR(%)', 'spKt/V', 'TSAT(%)', 'iCa(mg/dL)']]
print(mp.frame_text(out.groupby('施設').mean(numeric_only=True).round(2).reset_index()))
print()
print(f"URR が負になった症例: {int((d['URR(%)'] < 0).sum())} 例"
      '  ★前後が逆の施設に偏っている★')

# 手で計算しても同じになることを確かめる
urr = mp.urr(bun_pre=d['透析前BUN'], bun_post=d['透析後BUN'])
print('自動生成の URR(%) と mp.urr() の最大差:',
      float((pd.Series(urr, index=d.index) - d['URR(%)']).abs().max()))
""")

# ================================================================= 演習5
md("""
# ◆ 6. 演習⑤ Table 1 / Table 2 ―― **そのまま論文に貼れる形**

**表は 2 枚に分ける。** 読む側の目的が違うからである。

| | 中身 | p 値 |
|---|---|---|
| **Table 1** | 全症例の背景（N と分布だけ） | ★載せない★ |
| **Table 2** | 全体 + 群ごと + 群間比較 | 右端に 1 列 |

`GROUP` の指定が無ければ **Table 1 だけ**が作られる。

### 書式

```
連続変数    平均値 ± 標準偏差 [最小値, 最大値]
離散変数    n (%)
p 値        右端の 1 列だけ。★検定手法は表の中に書かず、脚注に回す★
```

検定手法を行ごとに書くと表が横に伸びて読めなくなる。しかし
**どの検定を使ったかを書かない表は査読に通らない**ので、脚注に必ず出す。

同じ数値から**日本語版と英語版**を作ってある。
人が訳し直すと、そこで誤訳と写し間違いが入るからである。
""")

md("""
### 下の表は **そのまま選んでコピーし、Word に貼れる**

Word に貼ると**表として**入る（文字の塊ではない）。配置は中詰め。
ここに出すのは日本語版だけ。英語版は Excel と Word のファイルに入っている。
""")

code("""
from IPython.display import HTML

# ★Word にそのまま貼れる形で出す（選択してコピー → Word に貼る）★
display(HTML(rep.gt.to_html('ja')))
""")

code("""
# 文字だけで見たいとき（桁が揃う）
print(rep.gt.table1.text('ja'))
""")

code("""
if rep.gt.table2 is not None:
    print(rep.gt.table2.text('ja'))
else:
    print('GROUP の指定が無いので Table 2 は作られていない。')
    print('「1. 環境構築 → 4) データの入力」でもう一度読み込み、GROUP を選ぶこと。')
""")

md("""
### 書き出されたファイル

| ファイル | 中身 |
|---|---|
| `table/Table1_2.xlsx` | **sheet1 = 日本語版 / sheet2 = 英語版** |
| `table/Table1_2.docx` | **ページを分けて日本語版 → 英語版**（表は中詰め・三本罫線） |

Word のほうは、群が多くて横に伸びる表だけ**自動で横置きページ**になる。
""")

code("""
for f in rep.saved:
    if 'Table1_2' in f:
        print(' ', f)
""")

md("""
## 検定の中身を確かめる（詳細版）

Table 2 には p 値しか出ていない。**なぜその検定を選んだか**と効果量は、
こちらの詳細版に入っている。査読で聞かれるのはこの中身である。
""")

code("""
print(rep.table1.report())
""")

md("""
### 検定の選択理由・効果量・多重比較の補正

- 連続変数は正規性の判定に従って 平均(SD) か 中央値[Q1,Q3] を選ぶ
- 3 群以上なら事後比較（Tukey HSD / Dunn）まで出す
- **全列に p を出すので、BH の q 値を必ず併記する**
- **群間の「偏り」は p ではなく SMD で見る**（p は例数で動く）
""")

code("""
rep.table1.comparison.to_frame()[['項目', '検定', 'p', 'q(BH)', '効果量', 'SMD', '判定の根拠']]
""")

md("""
## 管理目標の達成率 ―― 開区間と閉区間を取り違えない

JSDT の管理目標には**境界値を含むものと含まないものがある**。
`<` と `<=` の取り違えは、達成率を数％動かす。**論文の結論が変わる。**
""")

code("""
rep.achievement
""")

code("""
# 境界値ちょうどの症例が何例あるか。ここが取り違えの影響を受ける。
clean = rep.df_clean
for key, col in [('P', '無機リン(P)'), ('cCa', 'iCa(mg/dL)'), ('Hb', '末梢血｜血色素量(Hb)')]:
    if col not in clean.columns:
        continue
    spec = mp.load_dict()['items'][key]
    tg = spec['target']
    v = pd.to_numeric(clean[col], errors='coerce')
    ok = v.notna()
    right = mp.in_target(v, tg)
    wrong = mp.in_target(v, {**tg, 'high_inclusive': True, 'low_inclusive': True})
    a, b = right[ok].mean(), wrong[ok].mean()
    print(f"{spec['name_ja']:<14s} 正しく開区間 {a:6.1%} / 誤って閉区間 {b:6.1%}"
          f"  → 差 {b - a:+.1%}（{int((wrong[ok] != right[ok]).sum())} 例）")
""")

# ================================================================= 演習5
md("""
# ◆ 7. 演習⑥ 生存時間分析

4 列の日付（ID / 観察開始日 / イベント発生日 / 打ち切り日）から `(duration, event)` を作る。

**矛盾した症例は勝手に解釈しない。除外して、理由を表に残す。**
""")

code("""
sf = rep.survival
print(sf.report())
print()
print('除外された症例:')
sf.excluded
""")

code("""
import numpy as np

d = sf.data.rename(columns={
    'アルブミン(Alb)': 'Alb', '末梢血｜血色素量(Hb)': 'Hb',
    'C反応性蛋白(CRP)定量': 'CRP', '無機リン(P)': 'P', '透析歴_月': 'vintage'})
d['logCRP'] = np.log(pd.to_numeric(d['CRP'], errors='coerce') + 0.1)
d['低Alb'] = np.where(pd.to_numeric(d['Alb'], errors='coerce') < 3.5, 'Alb<3.5', 'Alb≧3.5')

surv = mp.Survival(d, unit='years')
km = surv.km(by='低Alb', title='アルブミン値による生存曲線',
             save=rep.run.file('figure', 'km_alb.png'))
print(km.report())
""")

code("""
print(surv.logrank(by='低Alb').report())
""")

md("""
## Cox 比例ハザード回帰 ―― **黙って減る n を見張る**

Cox は共変量に欠測がある症例を**黙って落とす**。
600 例で始めたはずが 400 例で推定されている、ということが起こる。
`medprep` は**何例落ちたかを必ず報告する。**

あわせて **EPV（イベント数 ÷ 共変量の数）** を見る。10 を下回れば推定が不安定になる。
""")

code("""
cov = [c for c in ['年齢', 'Alb', 'Hb', 'logCRP', '糖尿病', 'vintage'] if c in d.columns]
cox = surv.cox(covariates=cov)
print(cox.report())
""")

code("""
surv.forest(cox, save=rep.run.file('figure', 'cox_forest.png'))
""")

md("""
### 比例ハザード仮定の確認

Cox は「ハザード比が時間によらず一定」と仮定している。
Schoenfeld 残差の検定が有意なら**その仮定は成り立っていない**。
そのときは層別化するか、比例ハザードを仮定しない指標（RMST）を使う。
""")

code("""
surv.schoenfeld_plot(cox, save=rep.run.file('figure', 'cox_schoenfeld.png'))
print('PH 違反:', cox.ph_violations or 'なし')
""")

md("""
### 合成データの「真の係数」と照合する

演習用の合成データは、**分かっている係数から作ってある**。
推定した 95% 信頼区間が真値を含むかを確かめると、手順が正しいことを自分で検証できる。
（自分のデータでは真値は分からない。だからこそ合成データで手順を確かめておく。）
""")

code("""
truth = mp.demo.TRUE_COEFFICIENTS
m = cox.model
chk = pd.DataFrame({
    '真の係数': pd.Series(truth),
    '推定係数': m.params_,
    '95%CI下限': m.confidence_intervals_.iloc[:, 0],
    '95%CI上限': m.confidence_intervals_.iloc[:, 1]}).dropna()
chk['CIに真値を含む'] = ((chk['95%CI下限'] <= chk['真の係数'])
                    & (chk['真の係数'] <= chk['95%CI上限']))
print(f"{int(chk['CIに真値を含む'].sum())}/{len(chk)} の変数で 95%CI が真値を含む")
chk.round(3)
""")

# ================================================================= 演習6
md("""
# ◆ 8. 演習⑦ 目的変数を作り、モデルに渡す

### リークを構造的に不可能にする

「前処理をしてから分割する」と、テストデータの情報が前処理に混ざる（**リーク**）。
中央値も、スケーラも、カテゴリの一覧も、**train だけで決めなければならない。**

`medprep` では `fit` を test に呼ぶと**例外で止まる**。気をつける話ではなく、**できない構造にする。**
""")

code("""
# 目的変数：1 年以内のイベント発生
#   ★1 年経たずに打ち切られた症例は「1 年以内に起きたか」を判定できない。★
#     その症例は除く。除いた数は必ず報告される。目的変数を補完してはならない。
d1 = d.copy()
d1['1年以内イベント'] = np.where(d1['event'] == 1,
                             (d1['duration'] <= 1.0).astype(float), 0.0)
d1.loc[(d1['event'] == 0) & (d1['duration'] < 1.0), '1年以内イベント'] = np.nan

rep2 = mp.autoprep(
    d1,
    outcome='1年以内イベント', task='classification',
    group=GROUP, id_col=ID_COL,
    test_size=TEST_SIZE, seed=RANDOM_STATE,
    columns=[c for c in d1.columns if c not in ('duration', 'event', '低Alb', 'CRP')],
    save=True, method=PROJECT_NAME,
)
""")

code("""
rep2.show()
""")

code("""
# リークの検査。★通ったことの証拠として残せる表である。★
mp.leak_check(rep2.pipeline, rep2.split.train, rep2.split.test)
""")

code("""
# test に fit しようとすると止まる
try:
    mp.Preprocessor(rep2.schema).fit(rep2.split.test)
except mp.LeakageError as e:
    print('✗', str(e).split('。')[0], '。')
""")

md("""
## カテゴリはどう数値になったか ―― **1 がどちらかを列名で示す**

二値の列（男/女、男性/女性、M/F、Male/Female）は **0/1 の 1 本**にまとめる。
one-hot の結果は同じだが、**どちらを 1 にするかを辞書順ではなく意味で決める。**

| 元の列 | 元の値 | 作る列 | 値 |
|---|---|---|---|
| 性別 | 男 / 女 | **男性** | 男性=1・女性=0 |
| 性別 | M / F | **男性** | 同上（★書き方が違っても同じ列名になる★） |
| 糖尿病 | あり / なし | 糖尿病 | あり=1・なし=0 |

`性別=男` という列名にしないのは、**`M/F` で記録された施設のデータと結合したときに
`性別=M` という別の列になってしまうから**である。列名は意味で決める。

`糖尿病` の列名を変えないのは、1 側の水準名が「あり」だからである。
列名を「あり」にしたら**何の列か分からなくなる。**

水準が 3 つ以上の列（施設 A院/B院/C院）はこれまでどおり one-hot にし、
**基準にした水準（列を作らなかった水準）** を下に出す。基準が分からなければ係数は読めない。
""")

code("""
# 出てきた列。二値は 1 本にまとまっている
print('前処理後の列:', list(rep2.X_train.columns))

# 0 とした水準（＝基準）
print()
for src, base in rep2.pipeline.reference_levels().items():
    print(f'  {src}: 基準 = {base}')
""")

code("""
# 元の値と突き合わせて、向きを自分の目で確かめる
chk = pd.DataFrame({
    '元の性別': rep2.split.train.loc[rep2.X_train.index, '性別'],
    '男性': rep2.X_train['男性'],
})
print(mp.frame_text(chk.value_counts().reset_index(name='人数')))
""")

md("""
## 前処理済みの行列をそのままモデルに渡す

`rep2.X_train` は **列名の付いた DataFrame** である。
`get_dummies` で作った列の名前も残っているので、**係数を読める。**
""")

code("""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model = LogisticRegression(max_iter=2000).fit(rep2.X_train, rep2.y_train)
auc_tr = roc_auc_score(rep2.y_train, model.predict_proba(rep2.X_train)[:, 1])
auc_te = roc_auc_score(rep2.y_test, model.predict_proba(rep2.X_test)[:, 1])
print(f'ロジスティック回帰  train AUC = {auc_tr:.3f} / test AUC = {auc_te:.3f}')

coef = (pd.Series(model.coef_[0], index=rep2.X_train.columns)
        .sort_values(key=abs, ascending=False).head(10))
coef.round(3)
""")

md("""
### 再現に必要なもの

| ファイル | 中身 |
|---|---|
| `model/schema.yaml` | **列の役割と、そう判断した理由。これ 1 枚で前処理を再現できる** |
| `model/pipeline.pkl` | fit 済みの前処理（sklearn 互換） |
| `table/run_info.json` | run 番号と保存先（D&D アプリが読む） |
| `log/history.csv` | どのデータを、どのコードで、どう解析したかの一覧 |
""")

code("""
print(rep2.run.report_text())
print()
for f in rep2.saved:
    print(' ', f)
""")

md("""
ここで初めて **`data/前処理済み_train.xlsx` / `_test.xlsx`** が出る。
目的変数を決めて分割したときだけ、モデルに渡す行列が確定するからである。

この xlsx は `rep2.X_train` と同じもの（末尾に目的変数の列が付く）。
別のノートブックに渡すならこれを読む。**ただし再現に要るのは `schema.yaml` と
`pipeline.pkl` のほうである。** 前処理済みの xlsx だけを回すと、
数か月後に「この列は何だったか」が分からなくなる。
""")

code("""
rep2.removed
""")

# ================================================================= 第3部
md("""
# ◆ 9. 発展事項

## 全自動にしてはいけないもの

このパッケージの値打ちは「何を自動化したか」と同じくらい、**「何を自動化しなかったか」**で決まる。
以下は**警告を出して人に返す**。自動で処理しない。

| 項目 | 理由 |
|---|---|
| **外れ値の自動削除** | 医学では外れ値こそが重要な症例（劇症型、稀な合併症）でありうる |
| **目的変数の補完** | 補完した目的変数で学習した結果は解釈できない。除外して、除外数を報告する |
| **ステップワイズ変数選択** | 逐次選択は推定量に偏りを与え、信頼区間と p 値が無効になる |
| **p 値による Table 1 の解釈** | 群間の偏りは p ではなく SMD で見る。p は例数で動く |
| **多重比較の無補正** | 全列に p を出すので、BH の q 値を必ず併記する |
| **PH 仮定違反時の Cox の黙認** | Schoenfeld 検定が有意なら警告し、層別化か RMST を勧める |
| **EPV < 10 での多変量 Cox** | 警告を出す |
| **施設・時期をそのまま特徴量にすること** | リークの温床。投入するかを人に問う |
| **欠損の指示変数を黙って落とすこと** | 「測っていない」ことが予後情報である |

> 受講者が持ち帰るべき原理は、
> **「自動化できる部分と、医学的判断が必要な部分の境界を知ること」**である。

## この先（第4回以降）

- **MICE（多重代入）** ―― 欠損が MAR のときの正しい扱い。補完の不確実性を推定に反映させる
- **target encoding** ―― 水準の多いカテゴリ。**fold の中で fit しないとリークする**
- **競合リスク** ―― 死亡と移植のように、片方が起きるともう片方が観察できない場合
- **時系列** ―― 横持ちの検査値を時系列として扱う（第4回）

## 自分のデータで試す

環境変数セルに戻り、

1. `USE_DEMO_DATA = False` にする
2. `ID_COL` / `GROUP` / `DATE_COL` / `SURVIVAL_DATES` を自分の列名に書き換える
3. このノートブックを上から実行する

**同じコードが自分のデータでも動く。** そこが第3回の山場である。
""")

nb = {
    "cells": _fix(C),
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
VERSION = "Ver1_8"
out = str(pathlib.Path(__file__).resolve().parent / f"Preprocessing_{VERSION}.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"{len(C)} cells ->", out)
