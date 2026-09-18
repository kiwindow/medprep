# 設計上の判断の記録

実装の途中で選んだ道と、選ばなかった道の理由。あとから「なぜこうなっているのか」を
追えるようにしておく。数字は実機で測ったもの。

---

## ydata-profiling を必須の依存にしない

**やめた理由（実機で確認）**

1. `ydata_profiling` は内部で `pkg_resources` を import する。setuptools 81 以降は
   これを同梱しないため、`ModuleNotFoundError: No module named 'pkg_resources'` で
   import 自体が失敗する（setuptools 84.0.0 で再現。`setuptools<81` を入れて解消）。
2. 依存の上限固定が厳しい: `matplotlib<=3.10` `numpy<2.4` `scipy<1.17` `PyYAML<6.1`。
   Colab の既定環境はこれらより新しいことが多く、`pip install` が既存パッケージの
   ダウングレードとランタイム再起動を誘発する。第2回で Colab を教えた直後に
   これを踏ませるのは教材として悪い。
3. `numba` を引き込むためインストールが重い。

**代わりに** レポートは Jinja2 の自前テンプレートで作る。
詳細プロファイルが要る人だけ `medprep[report]` を入れる。

---

## feature-engine を使わない

当初は `Winsorizer` / `RareLabelEncoder` / `OneHotEncoder` を使う設計だったが、
2 つの実害を確認して純 scikit-learn に切り替えた。

1. `Winsorizer` は NaN を含む列で `ValueError` を投げて停止する。
   医学データで「補完前に外れ値を処理する」という正しい順序が組めない。
2. `feature_engine.encoding.OneHotEncoder` を `ColumnTransformer` の中に入れると
   `get_feature_names_out()` が
   `ValueError: input_features is not equal to feature_names_in_` で失敗する。
   列名が取れないと、下流の係数プロット・特徴量重要度・SHAP がすべて `x0, x1, …` になる。

純 sklearn 版（`min_frequency` + `set_output(transform="pandas")` +
`verbose_feature_names_out=False`）は列名付き DataFrame を返し、joblib 保存・再読込も通る。
NaN を保ったまま上下限を学習する Winsorizer は自前で持つ。

---

## pingouin を使わない

0.6.1 で出力の列名が `p-val` → `p_val` に変わっている（実機で確認）。
教材の寿命（数年）に対してこの種の破壊的変更は負債になる。
検定は `scipy.stats` と `statsmodels` を直接呼ぶ。

---

## R の gtsummary を rpy2 経由で使わない

出力の見た目は gtsummary が最良だが、Windows での R 導入・`R_HOME` 設定・文字コードが
受講者にとって高い壁になる。第1回・第2回で「Python 環境だけで完結する」ことを
教えた直後に R を要求するのは教材の一貫性を損なう。
`tableone` は同じ論文（Pollard et al., JOSS 2018）由来の実装で、Table 1 の用途には十分。

---

## japanize-matplotlib ではなく matplotlib-fontja

`japanize_matplotlib` は内部で `distutils` を import する。distutils は Python 3.12 で
標準ライブラリから削除された。講座の環境は `uv init --python ">=3.12,<3.13"` なので
本来ならここで失敗するが、環境構築コードが `setuptools` を入れており、
その互換シムが distutils を肩代わりしているため現在は動いている。
**壊れているのではなく、将来なくなるシムに依存している。**

setuptools 81 以降はこのシムに削除予告の警告を出す。外れた時点で受講者の手元で
日本語のグラフだけが豆腐（□）になり、原因が `uv add` の一行に埋もれて追いにくい。
後継の `matplotlib-fontja` に移行し、古い環境のために japanize も順に試す。

**あわせて**: IPAexGothic には `≥` `≤` の字形が無い。グラフのラベルには
「以上」「以下」または `>=` `<=` を使う。

---

## ruff format を使わない

ruff の整形は全角文字を幅 1 として数える。日本語のコメントと文字列が多いこの
コードベースでは、「95 文字」と判定された行が実際には 190 桁で表示される。
整形すると短い行が結合されて、かえって読めなくなる。

欠陥を捕まえる `ruff check`（未使用の import、未使用変数、実際のバグ）は CI で回す。

---

## 単位混在は「あり得ない範囲の外の値」だけ換算する

最初は「基準範囲の中心に最も近づく倍率を行ごとに当てる」実装にした。**これは誤りだった。**
実機で CRP 329 件（正常に近い実測値を含む）が誤って 1/10 に換算され、データが壊れた。
CRP の基準範囲上限は 0.14 mg/dL なので、実測 0.3 mg/dL が「桁が違う」と判定される。

修正後の原則:

> plausible 範囲の**外**にあり、かつ換算係数を掛けると範囲の**中**に収まる値だけを換算する。

Hb 108（g/L 記録）は範囲外 → ×0.1 で 10.8 となり範囲内 → 換算する。
CRP 0.3 は範囲内 → 触らない。

Plt の `×10³/µL` と `×10⁴/µL` のように、どちらの単位でも plausible に収まる場合は
自動判別できない。**換算せず「桁の異なる二峰性がある」と報告するだけ**にした。

---

## 辞書の別名は全項目を通じて一意でなければならない

CI で見つかった実害: `HT` を身長の別名にしていたため、`Ht`（ヘマトクリット）の列が
身長として同定され、**Ht 35〜50% が「あり得ない身長」として黙って NaN 化される**
状態だった。`tests/test_dict.py::test_alias_map_has_no_collisions_between_items` が
毎回これを検査する。

---

## スカラーを渡したらスカラーを返す

numpy 2 では要素数 1 の配列を `float()` に渡すとエラーになる。
1 症例を確かめたいだけのときに `float(np.atleast_1d(...)[0])` を書かされるのは、
教材として摩擦が大きい。入力がすべてスカラーなら Python の float を返す。
`tests/test_package.py::test_scalar_in_scalar_out` が全関数について検査する。

---

## Table 1 を tableone で包まず、自前で作る

`tableone` は良い実装で、依存にも残してある（受講者が使いたければそのまま使える）。
それでも `medprep.describe.table_one()` を自前で持つのは、
**医学領域の文脈を表に載せるため**である。

| 必要なもの | tableone | medprep |
|---|---|---|
| 項目名に単位と**採血時点**を出す | — | `BUN [mg/dL]（透析前）` |
| **なぜその検定を選んだか**を列に残す | — | 「判定の根拠」列 |
| 大標本で Shapiro-Wilk が棄却しても平均(SD) を使う判断 | 検定のみ | 歪度・尖度と併用（下記） |
| 効果量（Hedges' g / Cliff's δ / η² / ε² / Cramér's V） | — | 必ず併記 |
| 列数ぶんの多重比較（BH の q 値） | — | 併記 |
| 3群以上の事後比較（Tukey / Dunn） | — | 併記 |

透析データでは、**透析前 BUN と透析後 BUN が同じ表に並ぶ**。
時点を書かない表は、読み手が黙って読み違える。

---

## 正規性は「検定の棄却」だけでは決めない

Shapiro-Wilk を n=600 に当てると、実用上どうでもよい歪みでも p<0.05 になる。
検定だけで決めると Table 1 はほぼ全項目が 中央値[Q1,Q3] になり、読みにくくなるだけで
何も改善しない。

ここで決めたいのは「厳密に正規分布か」ではなく
**「平均と中央値のどちらがこの分布を忠実に代表するか」**である。
そこで medprep は、**検定が棄却し、かつ歪度 |skew| ≥ 0.5 または過剰尖度 ≥ 1.0**
のときにだけ非正規とする。n < 20 では検定に検出力が無いので形だけで判断する。
どちらの基準で決めたかは必ず「判定の根拠」に書き出す。

---

## 効果量に負の値を出さない（ε²）

Kruskal-Wallis の ε² = (H − k + 1)/(n − k) は、群間差が偶然より小さいと負になる。
効果量の負値は「小さい」ではなく「逆向き」と読まれるので 0 に丸める。

---

## `cramers_v(...) or float("nan")` と書かない

関連がまったく無いとき Cramér's V は **0.0** を返す。Python では 0.0 が偽なので、
`or` で既定値を与えると「算出できなかった」ことにされてしまう。
`None` かどうかで判定する。`tests/test_describe.py` が回帰テストとして固定している。
