# 実装記録：medprep Phase 0（リポジトリ整備）2026-09-18

構想 ver1.3 の Phase 0。プロトタイプを配布できる Python パッケージに仕立て、
CI を組み、GitHub に公開した。**完了。**

  https://github.com/kiwindow/medprep

CI は Ubuntu / macOS / Windows × Python 3.11 / 3.12 / 3.13 の 9 通り＋lint＋build が
すべて緑（run #4、1 分 15 秒、警告ゼロ）。

---

## 1. 置き場所

```
~/Projects/medprep          ← リポジトリ本体（初回コミット済み、working tree clean）
~/Projects/medprep/PUSH_手順.md   ← GitHub 作成と push の手順（未追跡。push 後に削除可）
~/Projects/medprep_0.1.0_initial.tar.gz   ← 転送に使った控え（削除可）
```

**Dropbox の外に置いた。** Dropbox 配下に git リポジトリを置くと、同期のタイミングで
`.git` が壊れることがあるためである。`~/Projects` は既存のフォルダを使った。

---

## 2. 公開と配布

公開済み：https://github.com/kiwindow/medprep （Public、MIT）

受講者への配布：

```bash
# uv（講座の環境）
uv add "medprep @ git+https://github.com/kiwindow/medprep"

# Colab
!pip install git+https://github.com/kiwindow/medprep
```

---

## 3. リポジトリの構成

```
medprep/
├── .github/workflows/ci.yml     9 セルのテスト行列 + lint + パッケージ組み立て
├── pyproject.toml               hatchling / src レイアウト / 依存宣言
├── README.md                    設計の原則と使い方
├── CHANGELOG.md                 Keep a Changelog 形式
├── LICENSE                      MIT（既存スイートと同じ）
├── docs/DECISIONS.md            選ばなかった道とその理由
├── src/medprep/
│   ├── __init__.py              __version__ と公開 API 50 個
│   ├── dates.py                 日付表記の正規化
│   ├── survival_input.py        生存時間 3 形式 → (duration, event)
│   ├── clean.py                 辞書駆動の掃除
│   ├── timing.py                採血時点（透析前/後/不詳）
│   ├── hd.py                    透析指標
│   ├── tac.py                   TAC-BUN
│   ├── targets.py               管理目標の達成判定
│   └── dict/ranges_ja.yaml      医学領域辞書 86 項目
├── tests/                       9 ファイル・439 テスト
└── examples/                    合成データ生成 と 通し検証
```

追跡ファイル 27、リポジトリ 1.1 MB。

**src レイアウトにした。** これだとカレントディレクトリからの誤 import が起きず、
「wheel に入れ忘れたファイル」が CI で必ず露見する。辞書 YAML がその典型で、
同梱漏れがあるとインストール後に何も動かない。

---

## 4. CI

| ジョブ | 内容 |
|---|---|
| test | **Ubuntu / macOS / Windows × Python 3.11 / 3.12 / 3.13 の 9 通り**。uv で依存を解決し pytest |
| test（続き） | 日本語フォントが実際に読めているかを確認（`font.family` が sans-serif のままなら失敗） |
| lint | `ruff check` |
| build | `uv build` で wheel を作り、**wheel だけを入れた素の環境で辞書を読めるか**を確認 |

日本語フォントの確認を CI に入れたのは、受講者が日本語のグラフを描くためである。
フォントが入らない環境では例外が出ず、黙って豆腐（□）になる。気づけない壊れ方は
機械に見張らせる。

`ruff format` は使わない。全角文字を幅 1 として数えるため、日本語の多いこの
コードベースでは「95 文字」と判定された行が実際には 190 桁で表示される。
整形すると短い行が結合されて、かえって読めなくなる（`docs/DECISIONS.md` に記録）。

---

## 5. テスト 439 個

| ファイル | 主な内容 |
|---|---|
| `test_dates.py` | 20 表記が同じ日付になる／曖昧な列は**推測せず拒否する**／年月のみは既定で拒否 |
| `test_survival_input.py` | 3 形式／矛盾は黙って解決しない／除外表に元の入力値が残る／単位／両端入れ |
| `test_clean.py` | 検出限界／**実測の正常値を誤って換算しない**／あり得ない値は winsorize せず NaN |
| `test_timing.py` | 列名からの時点判定／**不詳なら算出不可**／TAC は同一回の前後では出せない |
| `test_hd.py` | **%CGR の 12 中間値を計算例に固定**／A/V が 70.0%／異常入力は 0 に丸めず NaN |
| `test_tac.py` | 簡便式 40 mg/dL／換算／**取り違えの検出**／Ti の 168 時間検算 |
| `test_targets.py` | **境界値 20 通り**（P 5.5、補正Ca 9.5、Hb 12.0、iPTH 240、Kt/V 1.4、β2MG 30） |
| `test_dict.py` | 範囲の向き／基準範囲が plausible の内側／**目標に出典がある**／**別名の衝突** |
| `test_package.py` | 版番号／辞書の同梱／公開 API の実在／**スカラー入力→スカラー出力** |

計算例との一致を**回帰テストとして固定した**のが要点である。
今後 nPCR の式を差し替えるようなことがあれば、%CGR のテストが必ず落ちる。

---

## 6. CI が実際に見つけた欠陥

### 辞書の別名衝突（実害あり）

`HT` を**身長**の別名にしていた。`Ht` はヘマトクリットの標準的な略号である。
`build_alias_map` は先に定義された項目を優先するため、身長（先に定義）が `Ht` を取り、
**ヘマトクリット 35〜50% が「あり得ない身長（許容 50〜220 cm）」として黙って
NaN 化される**状態だった。

`test_dict.py::test_alias_map_has_no_collisions_between_items` が検出。
身長の別名から `HT` を外し（`BH` に置換）、辞書のヘッダに
「別名は全項目を通じて一意であること」を明記した。

修正後の確認：

```
辞書一致: {'Ht': 'Ht', '身長': 'height'}
  Ht    身長
42.0 165.0    ← ヘマトクリットとして正しく残る
```

### スカラーを渡すと `float()` できなかった

`percent_cgr` に 1 症例ぶんのスカラーを渡すと、戻り値の形が不揃いだった
（`delta_bw` は 0 次元、`percent_CGR` は形 (1,)）。numpy 2 では要素数 1 の配列を
`float()` に渡すとエラーになるため、

```python
float(r.percent_CGR)   # TypeError
```

となっていた。1 症例を確かめたいだけのときに `float(np.atleast_1d(...)[0])` を
書かされるのは教材として摩擦が大きい。**入力がすべてスカラーなら Python の float を
返す**ように `hd.py` と `tac.py` の全関数を揃え、`test_package.py` で 16 関数について
検査するようにした。

---

## 7. 依存の宣言

core に解析まで含めた。講座の環境は 1 回の導入で解析まで通せるほうがよい。
重い学習ライブラリ（torch / xgboost / lightgbm）はノートブック側の責務なので入れない。

```
pandas, numpy, scipy, pyyaml, scikit-learn, statsmodels,
matplotlib, seaborn, tableone, lifelines, missingno,
openpyxl, jinja2, matplotlib-fontja
```

任意の追加：`[report]`（ydata-profiling + setuptools<81）／`[impute]`／`[dimred]`／
`[spss]`／`[dev]`（pytest, ruff）。

ydata-profiling を core から外した理由は `docs/DECISIONS.md` に記録した。

---

## 8. 実機で確認したこと

| 項目 | 結果 |
|---|---|
| Python 3.11 / 3.12 / 3.13 で依存解決とインストール | 3 つとも成功 |
| 439 テスト | 3 つとも全通過 |
| `ruff check` | All checks passed |
| `uv build` | sdist と wheel を生成 |
| wheel だけを入れた素の環境 | 辞書 86 項目を読み、%CGR = 116.4464608869、TAC = 40.0、A/V = 70.0% |
| `examples/run_e2e.py` | 掃除 → 生存時間 → Table 1 → KM → Cox まで通過 |
| Mac へ転送後 | Python 19 ファイル構文エラー 0、辞書 86 項目、git 履歴 1 コミット、working tree clean |

なお、`medprep` は PyPI で未使用の名前だった（404 を確認）。将来 PyPI に出す場合も
名前の衝突はない。

---

## 9. 作業中に起きたこと（記録）

転送後の健全性確認で `python3 -c` を走らせた際、git が `.git/index.lock` を
作ったまま残した。接続フォルダでは削除が既定で無効なため git 自身が消せず、
そのままだと手元で `git add` / `git commit` が失敗する状態だった。
削除許可をいただいて `index.lock` と `__pycache__` を除去し、
`git add -n` が通ることを確認済み。現在の working tree は clean。

---

## 10. 公開後の CI 修正（run #1 〜 #4）

初回の run #1 は緑だったが、警告が 2 件出ていた。その対応で 1 度失敗を挟んでいる。

| run | 内容 | 結果 |
|---|---|---|
| #1 `83727af` | 初回公開 | 成功 54 秒。ただし警告 2 件 |
| #2 `923d1ff` | action の版上げ＋キャッシュ修正 | **失敗 14 秒** |
| #3 `2541c2c` | setup-uv を実在するタグに固定 | 成功 56 秒。元の警告 2 件は解消 |
| #4 `2c4de6d` | キャッシュ鍵をジョブごとに分離 | 成功 1 分 15 秒。**警告ゼロ** |

### run #1 の警告 2 件

1. **キャッシュが一度も無効化されない**
   `No file matched to [**/uv.lock,**/requirements*.txt]. The cache will never get invalidated.`
   ライブラリなので `uv.lock` を追跡しておらず、setup-uv がキャッシュ鍵を作れずに
   鍵が固定されていた。放っておくと古い wheel を使い続け、
   「新しい依存で動くか」を確かめるという CI の目的が失われる。
   `cache-dependency-glob: "pyproject.toml"` を指定して解消。
   あわせて**週次実行**（毎週月曜 09:00 JST）を追加した。ロックファイルを持たない
   以上、上流の新版による破壊は push の無い週にも起こるためである。

2. **Node.js 20 の廃止予告**
   `actions/checkout@v4` と `astral-sh/setup-uv@v5` が対象。v7 と v10 系に更新。

### run #2 の失敗（私の確認不足）

全 11 ジョブが起動前に落ちた。

```
Unable to resolve action `astral-sh/setup-uv@v10`, unable to find version `v10`
```

`astral-sh/setup-uv` はリリースが immutable で、**`v10` のような浮動の major タグを
作っていない**（タグ一覧は `v10.1.0` / `v10.0.1` / `v10.0.0` / `v9.0.0` …）。
以前使っていた `v5` は、この方針になる前のタグだった。
リリース名だけを確認して major タグの実在を確かめなかったのが原因である。

実在するタグ `v10.1.0` に厳密固定し、次に版を上げる人が同じ罠を踏まないよう
ワークフローに理由を書き添えた。確認は推測ではなく
`raw.githubusercontent.com/<repo>/<tag>/action.yml` を実際に取得して行う
（`v10` は 404、`v10.1.0` と `checkout@v7` は 200）。

**コードには影響が無く、壊れたのはワークフローの参照だけだった。**

### run #3 で新たに出た警告

```
Failed to save: Unable to reserve cache with key
setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-24.04-3.12.3-<hash>,
another job may be creating this cache.
```

setup-uv の既定のキャッシュ鍵は「OS + ランナーの system python + 依存ファイル」で
作られ、**matrix の Python 版が入らない**。鍵にある `3.12.3` はランナー側の
system python であって、その回で試している版ではない。そのため同一 OS の 3 セルが
同じ鍵を同時に取り合い、1 つを除いて保存に失敗していた。

実害は無い（どれか 1 つは保存され復元も効く）が、毎回 8 件の警告が出ると
注釈そのものを読まなくなる。気づけない壊れ方を機械に見張らせるために CI を
置いている以上、警告は静かにしておきたい。`cache-suffix` を
test は `py${{ matrix.python }}`、lint と build はそれぞれの名前に分けた。
cp311 と cp313 では必要な wheel も違うので、セルごとに持つほうが正しい。

---

## 11. 次

Phase 6 以降（`schema.py` / `audit.py` → `describe.py` → `survival.py` → …）。
残り 13.5 日相当。以降はこのリポジトリに直接積んでいける。
