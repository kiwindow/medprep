# medprep

医学研究データの前処理・記述統計・生存時間解析を自動化する Python パッケージ。
日本腎・血液浄化AI学会（JAINBP）演習講座「鹿鳴館」の教材として開発しています。

[![CI](https://github.com/kiwindow/medprep/actions/workflows/ci.yml/badge.svg)](https://github.com/kiwindow/medprep/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## これは何か

カルテや透析管理システムから出てきたままの表を、**そのまま回帰・分類・生存時間解析に
かけられる形に直す**ための道具です。医学データに特有の汚れ——欠損コード `999`、
検出限界 `<0.1`、単位混在、全角、和暦、生理学的にあり得ない値、日付の逆転——を
想定して作ってあります。

## 設計の原則

### 1. 全自動は「決定の自動化」ではなく「決定の明示化」

1 行で最後まで走りますが、そこで下した判断（この列は ID、この値は欠損コード、
この症例は除外）は**すべて理由つきで書き出され**、人がそれを直して再実行できます。

医学研究では「なぜこの症例が除かれたのか」を後から説明できることが成果物の価値を
決めます。自動化と説明責任を両立させる方法がこれだと考えています。

### 2. 推測して計算しない

- 採血時点（透析前／透析後）が**不詳の値から透析指標を計算しません**。
  spKt/V も %CGR も前後の組を要求する式であり、どちらか分からない値を当てはめれば
  結果は数字としては出ますが意味を失います。
- 日と月の順序が列内から確定できない日付列（`03/04/2013` ばかりの列）を
  **勝手に解釈しません**。誤れば観察期間が最大 11 か月ずれ、誰も気づきません。
- 欠測を 0 で埋めません。異常値を 0 に丸めません。
- 計算できないときは `NaN` を返し、**何が足りなかったかを報告します**。

### 3. 外れ値と入力ミスを区別する

Hb 0 g/dL は外れ値ではなく入力ミスです。winsorize してはいけません（NaN 化が正しい）。
逆に、劇症型や稀な合併症の極端値は**外れ値として消してはいけない**症例です。
この区別を医学領域辞書（`dict/ranges_ja.yaml`、86 項目）が担います。

### 4. リークを構造的に不可能にする

学習を伴う変換（補完の統計量・スケーラ・エンコーダのカテゴリ集合・Winsorize の閾値）
はすべて scikit-learn の `ColumnTransformer` に閉じ込め、`fit` は train にしか
呼べない構造にします。（`pipeline.py` は実装中）

---

## 導入

```bash
# uv（推奨。講座の環境）
uv add "medprep @ git+https://github.com/kiwindow/medprep"

# pip
pip install git+https://github.com/kiwindow/medprep

# Google Colab
!pip install git+https://github.com/kiwindow/medprep
```

Python 3.11 / 3.12 / 3.13 に対応（Windows・macOS・Linux で CI を回しています）。

---

## 使い方

### 列の役割を決め、判断を書き出す

```python
import medprep as mp

sch = mp.Schema.infer(df, id_col="仮名ID", group="施設",
                      survival_dates=("観察開始年月日", "event発生年月日", "観察打ち切り年月日"))
print(sch.report())
sch.to_yaml("schema.yaml")     # 人が直して再実行できる
```

```
               列名               役割   扱い      辞書  水準数    欠損率                     判断の根拠
             仮名ID               id drop       —   600 0.0000     人が id_col として指定した
            施設コード     duplicate_of drop       —     4 0.0000 '施設' と Cramér's V = 1.0000
               備考 high_cardinality drop       —   600 0.0000 ユニーク率 1.00 > 0.90（…）
            検体採取日         datetime keep       —   557 0.0000 100% の値を日付として解釈できた
```

**`reason` の無い判断を作りません。** 推定できなかった列は `unknown` にして人に返します。
`schema.yaml` を直して再実行し、`sch.diff(edited)` で変更点を表にできます。

### このまま解析してよいかを問う

```python
rep = mp.audit(df, sch, id_col="仮名ID", group="施設", date_col="検体採取日")
rep.show()
```

```
[✗] 採血時点: 'BUN' が透析後に下がるはずなのに、100% の症例で逆になっている（40 件）
      対象列: 透析前BUN、透析後BUN
      → '透析前BUN' と '透析後BUN' の取り違えを強く疑う。
        取り違えたまま spKt/V を計算すると、透析量が過大に評価される

[✗] 測定法: 'ALP' は 2020-04-01（JSCC 法 → IFCC 法）の前後で中央値が
    246 → 91（0.37 倍、p=4.2e-33、前 539 例 / 後 61 例）と段差になっている
      → 期間を共変量に入れるか、期間で層別すること
```

**audit は直しません。報告します。** どう直すかは人が決めることだからです。
所見には必ず「どうすればよいか」が付きます。

見ているのは、**どの列も単独では正常に見える壊れ方**です。
同じ患者が 2 行ある／透析前後の列が入れ替わっている／白血球分画の合計が 140%／
血清鉄が TIBC を超えている／欠測が施設に偏っている／目的変数から導かれた列が
説明変数に残っている（リーク）。いずれも例外を出さずに通ってしまいます。

### Table 1 を出す（検定の選択理由と効果量つき）

```python
t1 = mp.table_one(df, sch, groupby="低Alb")
print(t1.report())
t1.to_excel("table1.xlsx")
```

```
                       項目                全体           Alb<3.5           Alb≥3.5                検定      p  q(BH)    SMD 欠測
                        n               578               215               363
          年齢 [歳], 平均 (SD)       67.8 (12.0)       67.9 (11.9)       67.8 (12.1)      Welch の t 検定  0.908  0.908  0.010  3
CRP [mg/dL], 中央値 [Q1, Q3]    0.2 [0.1, 0.7]    0.3 [0.1, 0.7]    0.2 [0.1, 0.6] Mann-Whitney U 検定  0.753  0.908 -0.084  0
    補正Ca [mg/dL], 平均 (SD)         9.3 (0.8)         9.7 (0.7)         9.1 (0.7)      Welch の t 検定 <0.001 <0.001  0.874  0
```

検定は自動で選びます。**なぜその検定なのかは「判定の根拠」列に残ります。**

```
CRP [mg/dL]  Mann-Whitney U 検定  Cliff's δ = 0.016
    → 2群・非正規。Shapiro-Wilk p=8.74e-40 < 0.05 で棄却され、歪度 7.63（|skew| ≥ 0.5）も大きい
年齢 [歳]      Welch の t 検定      Hedges' g = 0.010
    → 2群・正規。Shapiro-Wilk p=0.089 ≥ 0.05。歪度 -0.19。分散の等質性を仮定しない Welch を既定にする
```

3 群以上なら事後比較（Tukey HSD / Dunn）まで出します。
項目名に**単位と採血時点**が付くのは、透析前 BUN と透析後 BUN が同じ表に並ぶからです。

### 管理目標の達成率 — 境界値を必ず見せる

```python
mp.target_achievement(df, by="施設")
```

```
    無機リン        全体           3.5 以上、5.5 未満   600  277  46.2%   CKD-MBD 2025年改訂版 Statement 3.1.2
    無機リン  【上限境界 5.5 ちょうど】        含まない       21    0   0.0%   境界値の扱いを確認すること
```

「5.5 未満」を `<= 5.5` と書くと達成率が動きます。合成データ 600 例での実測：

```
    無機リン    正しく開区間 46.2% / 誤って閉区間 49.7%  → 差 +3.5%（21 例）
    補正Ca     正しく開区間 44.2% / 誤って閉区間 50.5%  → 差 +6.3%（38 例）
    血色素量    正しく開区間 59.0% / 誤って閉区間 61.8%  → 差 +2.8%（17 例）
```

### 生存時間データを日付だけから作る

観察期間を人が計算する必要はありません。**日付を入れれば済むようにしてあります。**

```python
import pandas as pd
import medprep as mp

sf = mp.build_survival(
    df,
    id_col      = "仮名ID",             # 解析には使わない。除外症例の報告にのみ使う
    start_date  = "観察開始年月日",
    event_date  = "event発生年月日",     # イベントが起きなかった症例は空欄
    censor_date = "観察打ち切り年月日",   # イベントが起きた症例は空欄
    covariates  = ["年齢", "性別", "Alb", "Hb"],
    unit        = "years",
)
print(sf.report())
sf.data        # lifelines にそのまま渡せる DataFrame
sf.excluded    # 除外された症例（ID・理由・入力されていた元の値）
```

受け付ける日付表記:

| 入力 | 結果 |
|---|---|
| `2013/3/3` `2013-3-3` `2013.3.3` `2013,3,3` `{2013, 3, 3}` | 2013-03-03 |
| `2013-3-3 14:32:11` `2013年3月3日 14時32分` | 2013-03-03（**時刻は自動削除**） |
| `H25.3.3` `平成25年3月3日` `R5.3.3` `令和元年5月1日` | 和暦 |
| `41336`（Excel シリアル値）、`20130303`、`２０１３／３／３`（全角） | 2013-03-03 |

イベント日と打ち切り日の**両方が入っている矛盾**、どちらも空欄、日付の逆転、
未来日付、観察期間 0 を検出し、症例ごとに理由を付けて除外します。

### 生存時間解析 — 黙って減る n を見張る

```python
s = mp.Survival.from_survival_frame(sf)
s.km(by="施設", save="km.png")
print(s.logrank(by="施設").report())
cox = s.cox(covariates=["年齢", "Alb", "Hb", "logCRP", "糖尿病", "vintage"])
print(cox.report())
s.forest(cox, save="forest.png")
```

```
Cox 比例ハザード回帰  n = 508、イベント = 203、共変量 = 6、EPV = 33.8
欠測により除外: 70 例（うちイベント 24 件）

[警告] 共変量の欠測により 70 例（イベント 24 件）が解析から落ちた（578 → 508）。
       欠測の多い列: Alb: 49、vintage: 15、年齢: 3。
       Cox は完全ケースだけで推定するので、**共変量を足すと n が黙って減る**。

[警告] 比例ハザード仮定に違反している変数がある（['logCRP']）。
       (1) その変数で層別する cox(strata='...')、(2) 時間依存項を入れる、
       (3) RMST に切り替える rmst(by='...')、のいずれかを選ぶこと
```

**「p < 0.05 でした」で終わらせません。** 何をすればよいかまで書きます。

生存期間中央値に到達しない群は NaN ではなく **NR** と表示します
（「生存が良い」ではなく「追跡期間が足りない」という事実です）。
順序のある群には log-rank trend test も出します。

### 辞書駆動でデータを掃除する

```python
clean, rep = mp.clean_numeric(df)
rep.show()
```

```
                列           処理  件数                           詳細
               年齢   あり得ない値→NaN   3                   許容 0–120 歳
     末梢血｜血色素量(Hb)      単位混在を換算  15         許容範囲外の値に倍率 [0.1] を適用
     末梢血｜血色素量(Hb)   あり得ない値→NaN   3               許容 1.5–25 g/dL
    C反応性蛋白(CRP)定量   検出限界表記を数値化  38 < は 1/2 LOD を代入（policy=half）
   インタクトPTH(iPTH)    欠損コード→NaN  35                        値 999
β2マイクログロブリン(β2MG) テキスト欠損表記→NaN  25                     未測定/不明 等
```

### 透析指標

```python
r = mp.percent_cgr(sex="男性", age=60, bun_pre=60, bun_post=20,
                   cr_pre=12, cr_post=4, bw_pre=63, bw_post=60, td_hours=4)
r.percent_CGR       # 116.4464608869
r.to_frame()        # 11 ステップの中間値をすべて保持している
r.provenance        # どの式を使ったか（nPCR の算出法が変われば %CGR も変わる）
r.invalid           # 妥当性検査に引っかかった件数と理由
```

実装済み: `urr` `sp_ktv` `npcr` `percent_cgr` `clear_space_ratio`（実測法・推算法）
`salt_intake` `gnri` `corrected_ca` `tsat` `bmi`、および TAC-BUN（簡便式・台形則・時間加重近似）。

### 管理目標の達成判定

「5.5 **未満**」に 5.5 は含まれません。この 1 件の取り違えが達成率を数％動かすため、
辞書側で開区間・閉区間を区別し、**境界値ちょうどの症例数を必ず報告します**。

```python
d = mp.load_dict()
mp.achievement(df, d, {"P": "無機リン(P)", "cCa": "補正Ca"}, by="施設")
```

```
  項目               群            目標  評価可能例数  達成    達成率
無機リン              全体 3.5 以上、5.5 未満     200 104  52.0%
無機リン 【上限境界 5.5 ちょうど】          含まない       5   0   0.0%
```

---

## 医学領域辞書（`dict/ranges_ja.yaml`）

86 項目。透析・腎臓領域に特化した資産です。

```yaml
Hb:
  aliases: ["末梢血｜血色素量(Hb)", 血色素量, Hb, HGB, ヘモグロビン, Hemoglobin]
  unit: g/dL
  unit_variants: {g/L: 0.1}
  plausible: [1.5, 25]          # これを外れたら入力ミス → NaN。winsorize しない
  reference_male:   [13.7, 16.8]
  reference_female: [11.6, 14.8]
  timing_required: true
  target:
    low: 10.0
    low_inclusive: true
    high: 12.0
    high_inclusive: false       # 12.0 は含まない
    measurement_point: 週初めの採血値
    source: "腎性貧血治療ガイドライン2015年版 第2章"
```

`plausible`（生理学的にあり得ない範囲）と `reference`（基準範囲）を分けているのが要点です。
透析患者では β2MG・Cr・iPTH が基準範囲を大きく超えるのが正常なので、
**基準範囲での外れ値判定は原理的に誤り**です。

**測定法の変更**も記録しています。ALP は 2020 年 4 月の JSCC→IFCC 移行で値が
およそ 1/3 になります。長期コホートでこれを共変量に入れると、段差が「患者の変化」
として推定に混入します。

---

## 開発

```bash
git clone https://github.com/kiwindow/medprep
cd medprep
uv sync --extra dev
uv run pytest          # テスト
uv run ruff check .    # lint
uv run python examples/make_synthetic.py   # 演習用の合成データを作る
uv run python examples/run_e2e.py          # schema → audit → 掃除 → 生存時間 → Table 1 → 管理目標 → KM → Cox（全 11 段）
```

`examples/make_synthetic.py` は**実データを一切含まない**合成透析コホート（600 例）を
生成します。日付 7 表記の混在、イベント日と打ち切り日の矛盾、欠損コード、検出限界、
単位混在、あり得ない値、ALP の測定法変更による段差を意図的に仕込んであります。

---

## 状態

土台の 11 モジュールが動き、以下は実装中です。

- `missing.py` / `outliers.py` / `pipeline.py` / `split.py`
- `viz.py` / `report.py` — 単一ファイルの HTML レポート
- `autoprep()` — 全自動 1 行の結線

進捗は [CHANGELOG.md](CHANGELOG.md) を参照してください。

---

## 注意

- 本パッケージは**ローカルで完結**します。データを外部に送信しません。
- 辞書の `reference`（基準範囲）は代表値です。**自院基準での確認が必要**です。
- `target`（管理目標）は 2026-09-18 時点の JSDT 公開版に基づきます。出典を項目ごとに
  記録してありますが、版の更新にご注意ください。
- 本パッケージは研究・教育のための道具であり、診療上の判断を代替するものではありません。

## ライセンス

MIT License — [LICENSE](LICENSE) を参照。
