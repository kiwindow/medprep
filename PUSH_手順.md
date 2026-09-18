# GitHub へ公開する手順（初回のみ）

リポジトリはここに用意してあります。初回コミット済み、working tree は clean です。

    ~/Projects/medprep

Dropbox の外に置いてあります。Dropbox 配下に git リポジトリを置くと、
同期のタイミングによって `.git` が壊れることがあるためです。

---

## 1. GitHub で空のリポジトリを作る

https://github.com/new を開き、次のとおりに設定します。

| 項目 | 値 |
|---|---|
| Owner | `kiwindow` |
| Repository name | `medprep` |
| Description | 医学研究データの前処理・記述統計・生存時間解析を自動化する（JAINBP 鹿鳴館 教材） |
| 公開設定 | **Public** |
| Add a README file | **チェックしない** |
| Add .gitignore | **None** |
| Choose a license | **None** |

★下の3つは必ず「追加しない」でお願いします。★
追加すると GitHub 側に別のコミットができてしまい、手元の履歴と衝突します。

## 2. 手元から push する

ターミナルで、そのまま貼り付けてください。

    cd ~/Projects/medprep
    git remote add origin https://github.com/kiwindow/medprep.git
    git push -u origin main

認証を求められたら、GitHub のパスワードではなく
**Personal Access Token**（Settings → Developer settings → Personal access tokens）
を使います。`repo` の権限があれば足ります。

## 3. 確認

push が終わると、GitHub Actions が自動で走ります。

    https://github.com/kiwindow/medprep/actions

Ubuntu / macOS / Windows × Python 3.11 / 3.12 / 3.13 の 9 通りでテストが走り、
別に lint と「wheel を組み立てて辞書が同梱されているか」の確認が走ります。
全部緑になれば完了です。

## 4. 受講者への配布

    # uv（講座の環境）
    uv add "medprep @ git+https://github.com/kiwindow/medprep"

    # Colab
    !pip install git+https://github.com/kiwindow/medprep

---

## 手元で動かす

    cd ~/Projects/medprep
    uv sync --extra dev
    uv run pytest                              # 439 個のテスト
    uv run ruff check .                        # lint
    uv run python examples/make_synthetic.py   # 演習用の合成データを作る
    uv run python examples/run_e2e.py          # 掃除 → 生存時間 → Table 1 → KM → Cox

---

## このファイルについて

`PUSH_手順.md` は git の管理下に入れていません（`.gitignore` の対象外ですが
初回コミットには含めていません）。push が済んだら削除して構いません。

同じ階層にある `~/Projects/medprep_0.1.0_initial.tar.gz` は転送に使った控えです。
これも削除して構いません。
