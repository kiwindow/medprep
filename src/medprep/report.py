"""medprep.report — すべての結果を **HTML 1 枚** にまとめる。

なぜ 1 枚なのか
--------------
図を別ファイルにすると、メールに添付した時点でばらける。
base64 で埋め込んで単一ファイルにすれば、そのまま送れて、そのまま開ける。

何を先頭に置くか
----------------
**警告を先頭に置く。** 監査の致命的な所見が 10 枚目にあるレポートは、
読まれないのと同じである。冒頭に件数と内訳を出し、そこから本文へ降りる。

症例レベルの値を既定で出さない
------------------------------
このレポートはメールで回る。受講者が自分のデータで作ったレポートには、
**仮名 ID であっても症例を特定しうる情報**が載る。
したがって既定では、監査の「該当例」欄を件数だけにする。
必要なら `show_values=True` を**明示的に**渡す。
その場合はレポート自身が「症例レベルの値を含む」と表示する。

図には必ず表を添える
--------------------
図は PNG なので、拡大もホバーもできない。色だけに頼らせないためにも、
**同じ内容の表を隣に置く**。色覚特性のある読者にとってはこちらが本体である。
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import re
from dataclasses import dataclass, field

import pandas as pd

from . import __version__
from .viz import PALETTE, FigureSet, to_base64

ERROR, WARN, INFO = "error", "warn", "info"
_ICON = {ERROR: "✗", WARN: "!", INFO: "·"}
_LABEL = {ERROR: "致命的", WARN: "要確認", INFO: "記録"}
_STATUS = {ERROR: PALETTE.critical, WARN: PALETTE.warning, INFO: PALETTE.muted}

#: 「人の確認が要る事項」の名前。★HTML・Excel のシート・txt の 3 か所で同じ名前を使う。★
#: 表に書く文字と探す文字を二か所に書くと、片方だけ直したときに黙って食い違う。
ATTENTION = "人の確認が要る事項"
ATTENTION_TXT = f"{ATTENTION}.txt"

#: 症例の値を例として添えた部分（「（例: ['2021/13/45', …]）」）。
#: レポートは既定で症例レベルの値を出さないので、ここだけ伏せる。
#: ★リストの形のものだけを伏せる。★ 辞書の説明（「（例: 240 U/L → 80 U/L）」）は
#: 症例の値ではなく、伏せると何の段差かが読めなくなる。
_EXAMPLE_RE = re.compile(r"（例: \[[^\]]*\]）")


# ================================================================== 部品
@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)
    anchor: str = ""

    def text(self, s: str, *, kind: str = "p"):
        self.blocks.append((kind, s))
        return self

    def table(self, df: pd.DataFrame, *, caption: str = "", max_rows: int = 200):
        self.blocks.append(("table", (df, caption, max_rows)))
        return self

    def figure(self, fig, *, caption: str = "", alt: str = ""):
        self.blocks.append(("figure", (fig, caption, alt)))
        return self

    def findings(self, items: list):
        self.blocks.append(("findings", items))
        return self

    def pre(self, s: str):
        self.blocks.append(("pre", s))
        return self


@dataclass
class Report:
    title: str = "前処理レポート"
    subtitle: str = ""
    sections: list = field(default_factory=list)
    show_values: bool = False
    counts: dict = field(default_factory=lambda: {ERROR: 0, WARN: 0, INFO: 0})
    headline: list = field(default_factory=list)      # 冒頭に出す所見
    #: 人の確認が要る事項（`rep.warnings`）。None なら「まず読むこと」に載せない
    #: （autoprep を通さずに build_report を呼んだとき）。空のリストなら「なし」と書く。
    attention: list | None = None

    def section(self, title: str) -> Section:
        s = Section(title=title, anchor=_slug(title, len(self.sections)))
        self.sections.append(s)
        return s

    # ------------------------------------------------------------ 出力
    def to_html(self, path=None) -> str:
        from jinja2 import Environment

        env = Environment(autoescape=True)
        body = []
        for s in self.sections:
            body.append(f'<section id="{_html.escape(s.anchor)}">'
                        f'<h2>{_html.escape(s.title)}</h2>')
            for kind, payload in s.blocks:
                body.append(_render_block(kind, payload, self.show_values))
            body.append("</section>")

        tpl = env.from_string(_TEMPLATE)
        html = tpl.render(
            title=self.title, subtitle=self.subtitle,
            css=_CSS, body="".join(body),
            toc=[(s.anchor, s.title) for s in self.sections],
            counts=self.counts, headline=self.headline,
            attention=self.attention, attention_title=ATTENTION,
            icon=_ICON, label=_LABEL, status=_STATUS,
            show_values=self.show_values,
            version=__version__,
            generated=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        # 本文はすでに組み立て済みの HTML なので、最後に差し込む
        html = html.replace("<!--BODY-->", "".join(body))
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def to_excel(self, path, tables: dict | None = None):
        """表だけを Excel にも出す。シートごとに分ける。"""
        sheets = dict(tables or {})
        if not sheets:
            # ★1 枚目に置く。★ HTML の「まず読むこと」と同じ一覧を、同じ順で。
            if self.attention is not None:
                sheets[ATTENTION] = self.attention_frame()
            for s in self.sections:
                for kind, payload in s.blocks:
                    if kind == "table":
                        df, caption, _ = payload
                        name = _sheet_name(caption or s.title, sheets)
                        sheets[name] = df
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name[:31], index=False)
                if name == ATTENTION:
                    _fit_attention_sheet(w.sheets[name[:31]])
        return path

    def attention_frame(self) -> pd.DataFrame:
        """「人の確認が要る事項」を表にしたもの（Excel の 1 枚目と同じ）。"""
        items = list(self.attention or [])
        if not items:
            return pd.DataFrame({"番号": [""], "内容": ["なし"]})
        return pd.DataFrame({"番号": range(1, len(items) + 1), "内容": items})

    def attention_text(self, *, run: str = "", source: str = "") -> str:
        """`report/人の確認が要る事項.txt` の中身。HTML と同じ一覧を、同じ順で。"""
        items = list(self.attention or [])
        lines = [f"■ {ATTENTION}",
                 f"medprep {__version__}　{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                 + (f"　{run}" if run else "")]
        if source:
            lines.append(f"入力: {source}")
        lines.append("")
        if not items:
            lines.append("なし")
        else:
            lines.append(f"全 {len(items)} 件。空でないのがふつうである。"
                         "1 件ずつ確かめ、必要なら schema.yaml や引数を直して再実行すること。")
            lines.append("")
            width = len(str(len(items)))
            for i, t in enumerate(items, 1):
                lines.append(f"{i:>{width}}. {t}")
        if not self.show_values:
            lines += ["", "※ 症例の値の例は伏せてある（show_values=True で作り直すと出る）。"
                      "伏せていない全文は rep.warnings にある。"]
        return "\n".join(lines) + "\n"

    def to_attention_txt(self, path, **kwargs):
        """txt に書く。★UTF-8（BOM 付き）・CRLF。★ Windows のメモ帳でも Mac でも文字化けしない。"""
        with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(self.attention_text(**kwargs))
        return path


# ================================================================== 組み立て
def build_report(
    df: pd.DataFrame,
    schema=None,
    *,
    title: str = "前処理レポート",
    subtitle: str = "",
    audit=None,
    removed=None,
    outputs=None,
    encoded=None,
    run=None,
    missing=None,
    outliers=None,
    table1=None,
    gt=None,
    comparison=None,
    achievement=None,
    survival_summary=None,
    logrank=None,
    cox=None,
    split=None,
    preprocessor=None,
    figures: FigureSet | None = None,
    show_values: bool = False,
    attention: list | None = None,
    survival_frame=None,
    no_outcome=None,
    horizon=None,
    path=None,
) -> Report:
    """手元にあるものを全部 1 枚にまとめる。**無いものは黙って飛ばす。**"""
    rep = Report(title=title, subtitle=subtitle or _default_subtitle(df, schema),
                 show_values=show_values)

    # --- 冒頭：人の確認が要る事項（★「まず読むこと」の一番上に置く★）
    #   autoprep の `rep.warnings` をそのまま、同じ順で。HTML・Excel・txt で同じ一覧。
    if attention is not None:
        rep.attention = [str(t) if show_values else _EXAMPLE_RE.sub(
            "（例は show_values=True で表示）", str(t)) for t in attention]

    # --- 冒頭：警告の要約（★ここを最初に置くことが目的である★）
    if audit is not None:
        rep.counts = {ERROR: len(audit.errors), WARN: len(audit.warnings),
                      INFO: len(audit.by_severity(INFO))}
        rep.headline = [_finding_dict(f, show_values) for f in audit.errors[:8]]

    s = rep.section("このレポートの読み方")
    s.text("上から順に、<b>このデータで解析してよいか</b>（1. 監査）→ "
           "<b>列の役割</b>（2.）→ 行と列の処理（3.）→ 書き出したデータ（4.）→ "
           "欠損と外れ値（5.・6.）→ 記述統計（7.）→ 生存時間（8.）→ "
           "分割と前処理（9.）→ 図（10.）→ 再現性（11.）、と降りていく。", kind="html")
    s.text("<b>番号はノートブック（Preprocessing）の「第3部 結果の確認」（1.〜7.）と"
           "「第4部 生存時間分析と機械学習」（8.〜11.）と同じである。</b>"
           "1.〜7. は autoprep が自動で行った処理、8.〜11. は autoprep が作ったデータを使って"
           "人が決めて行う解析（Kaplan-Meier・Cox・予測モデル）の準備である。"
           "ここで気になった所は、ノートブックの同じ番号の節で、表を全部表示したり"
           "計算をやり直したりして確かめられる。"
           "ノートブックにだけある節（測定法の変更・透析前後・Kaplan-Meier と Cox など）は、"
           "同じ番号の下の小節になっている。", kind="html")
    s.text("図はすべて同じ色体系で描いてある。"
           "<b>相関は発散配色（中点の灰色が「関連なし」）、関連の強さは順次配色（濃いほど強い）</b>。"
           "色覚特性のある読者のために、隣り合う色が区別できることを検証した配色を使い、"
           "散布図は 3 色までに抑えてある。図には必ず同じ内容の表を添えている。",
           kind="html")
    if not show_values:
        s.text("症例レベルの値（仮名 ID など）は<b>含めていない</b>。"
               "必要なら <code>show_values=True</code> を明示して作り直すこと。", kind="html")
    else:
        s.text("★このレポートは<b>症例レベルの値を含む</b>。取り扱いに注意すること。★",
               kind="html")

    # --- 監査
    if audit is not None:
        s = rep.section("1. データ品質監査")
        s.text(f"{audit.n_rows} 行 × {audit.n_cols} 列。"
               f"致命的 {len(audit.errors)} 件 / 要確認 {len(audit.warnings)} 件 / "
               f"記録 {len(audit.by_severity(INFO))} 件（検査 {len(audit.checked)} 種）。")
        s.findings([_finding_dict(f, show_values) for f in audit.findings])
        if audit.skipped:
            s.table(pd.DataFrame(audit.skipped, columns=["実施できなかった検査", "理由"]),
                    caption="実施できなかった検査")

    # --- schema
    if schema is not None:
        s = rep.section("2. 列の役割（schema）")
        s.text("<b>判断には必ず理由を付けてある。</b>"
               "直したい場合は <code>schema.yaml</code> を編集して再実行する。", kind="html")
        s.table(schema.to_frame(), caption="列の役割", max_rows=300)
        if schema.unknown():
            s.text(f"★役割を推定できなかった列: {schema.unknown()}")

    # --- 減らしたもの（★何を捨てたかを言わない自動化は信用してはならない★）
    if (removed is not None and len(removed)) or (encoded is not None and len(encoded)):
        s = rep.section("3. 行と列の処理（減らしたもの）")
    if removed is not None and len(removed):
        rrow = removed[removed["種類"] == "行"]
        rcol = removed[removed["種類"] == "列"]
        rval = removed[removed["種類"] == "値"]
        s.text(f"<b>行 {int(rrow['件数'].sum())} 例に印を付け、列 {len(rcol)} 本を解析から外し、"
               f"値 {int(rval['件数'].sum())} 個を NaN にした。</b>それぞれ理由を付けてある。",
               kind="html")
        s.text("3.1 行の処理", kind="h3")
        s.text("<b>★行は 1 つも削除していない。★</b> 削除すると症例数と並びが変わり、"
               "元のデータと症例ごとに axis=1 で結合し直せなくなる。"
               "外すかどうかは <code>除外推奨</code> 列を見て人が決めること"
               "（全セルが空欄の行は <code>DROP_EMPTY_ROWS = True</code> のときだけ削除する）。",
               kind="html")
        if len(rrow):
            s.table(rrow, caption="印を付けた行・削除した行")
        else:
            s.text("印を付けた行は無い。", kind="note")
        s.text("3.2 列の処理", kind="h3")
        s.text("解析から外した列（2_解析用データから消える）と、"
               "残すが特徴量にしなかった列（ID・日付など）がある。「処置」の列で区別できる。",
               kind="html")
        if len(rcol):
            s.table(rcol, caption="解析から外した列・特徴量にしなかった列", max_rows=300)
        s.text("3.3 値の処理", kind="h3")
        s.text("欠損コード（999 など）と、生理学的にあり得ない値を NaN にした。"
               "行は残る。", kind="html")
        if len(rval):
            s.table(rval, caption="NaN にした値（列ごとの件数）", max_rows=300)
        else:
            s.text("NaN にした値は無い。", kind="note")

    # --- 二値の列
    if encoded is not None and len(encoded):
        s.text("3.4 二値の列を 0/1 に直した ―― 1 がどちらかを列名で示す", kind="h3")
        s.text("性別のように水準が 2 つの列は <b>0/1 の 1 本</b>にまとめる。"
               "問題は「列が何本になるか」ではなく <b>どちらの水準が 1 になるか</b>で、"
               "それを機械の都合（辞書順）で決めると、"
               "<code>男/女</code> で記録した施設は <code>性別=男</code>、"
               "<code>M/F</code> の施設は <code>性別=M</code> と"
               "<b>同じ意味の列が別名になる。</b>だから <u>意味で</u>決める。", kind="html")
        s.table(encoded, caption="0/1 に直した列（解析用データと前処理済みの両方に効く）")
        s.text("<b>掃除済みデータには掛けていない。</b>あれは"
               "<u>人が元の記録と突き合わせる</u>ためのものなので、"
               "<code>男/女</code> のまま残してある。", kind="html")

    # --- 書き出したデータ（★0〜6 の違いを最初に言う★）
    if outputs is not None and len(outputs):
        s = rep.section("4. 書き出したデータ ―― 0〜6 の番号の付いたファイル")
        s.text("同じデータを、<b>加工の度合いの順に 0〜6 の番号</b>を付けて "
               "<code>data/</code> に書き出してある。どれを使うかで結果が変わるので、"
               "下の表で確かめること。", kind="html")
        s.text("<b>0 元データ</b>は読み込んだそのまま。"
               "<b>1 掃除済み</b>は <u>人が元の記録と照合するため</u>のもので、ID も自由記載も残す。"
               "<b>2 解析用</b>は <u>自分で解析するため</u>のもので、使わない列を除くが ID は残し、"
               "<b>行は 1 つも減らさない</b>（元データと <code>axis=1</code> で結合できる）。<br>"
               "<b>3 機械学習用</b>は <u>回帰・分類のアプリに渡す</u>もの（目的変数がある行だけ。"
               "分割・標準化・補完はしていない）。<b>4 生存時間用</b>は KM・log-rank・Cox に使う。"
               "<b>5・6</b> はこのノートブックの中だけで使う training / test（補完・標準化済み）。",
               kind="html")
        s.table(outputs, caption="書き出したデータと、そこまでに施した処置",
                max_rows=20)
        if run is not None:
            s.text(f"保存先: {run.run}", kind="note")
        if no_outcome is not None and len(no_outcome):
            s.text("4.1 目的変数がないため 3〜6 から除いた行", kind="h3")
            s.text("これらの行は 1・2 には残っている。1 行ずつの一覧は "
                   "<code>data/目的変数がないため削除した行.xlsx</code>。", kind="html")
            cnt = (no_outcome.groupby(["理由", "除いたファイル"]).size()
                   .reset_index(name="件数")
                   if {"理由", "除いたファイル"} <= set(no_outcome.columns)
                   else no_outcome["理由"].value_counts().rename_axis("理由").reset_index(name="件数"))
            s.table(cnt, caption="理由ごとの件数")
        if horizon is not None:
            s.text("4.2 目的変数を自動で作った場合 ―― τ の決め方", kind="h3")
            s.text(f"目的変数の指定が無いので <b>{_html.escape(str(horizon.name))}</b>"
                   f"（τ 以内にイベントが起きたか）を作った。1 = {horizon.n1} 例、"
                   f"0 = {horizon.n0} 例、判定できない（τ より前に打ち切り）= "
                   f"{horizon.n_undetermined} 例。", kind="html")
            if getattr(horizon, "reason", ""):
                s.text(f"選んだ理由: {horizon.reason}", kind="note")
            s.table(horizon.table, caption="τ の候補ごとの例数")

    # --- 欠損
    if missing is not None:
        s = rep.section("5. 欠損")
        s.text(f"欠損がまったく無い行 {missing.n_complete} / {missing.n_rows}"
               f"（{missing.complete_rate:.1%}）。")
        s.table(missing.columns, caption="列ごとの欠損")
        if len(missing.patterns):
            s.table(missing.patterns, caption="欠損パターン", max_rows=25)
        if len(missing.signals):
            s.text("5.1 欠損と他の列の関連", kind="h3")
            s.text("★欠損が他の列と関連している（MCAR ではない）。"
                   "全体の中央値で埋めると群間差が人工的に作られる。")
            s.table(missing.signals, caption="欠損の偏り")
        for n in missing.notes:
            s.text(f"注記: {n}", kind="note")

    # --- 外れ値
    if outliers is not None:
        s = rep.section("6. 外れ値")
        s.text("<b>ここに出るのは「あり得るが極端」な値である。</b>"
               "生理学的にあり得ない値（Hb 0 など）は掃除の段で既に NaN にしてある。"
               "医学では外れ値こそが重要な症例でありうるので、既定では削除しない。",
               kind="html")
        t = outliers.table.copy()
        if not show_values and "該当例" in t.columns:
            t = t.drop(columns=["該当例"])
        s.table(t, caption="列ごとの外れ値")
        # ★入力ミスの可能性がある値（0.13.0〜）。IQR の印とは分けて出す。★
        sus = getattr(outliers, "suspects", None)
        if sus is not None and len(sus):
            s.text("<b>★入力ミスの可能性がある値★</b> ほかの値から桁違いに離れている"
                   "（ロバスト z が大きく、しかも次に極端な値から大きく飛び離れている）。"
                   "元の記録と照らして、誤りなら <code>schema.yaml</code> の列に "
                   "<code>plausible: [下限, 上限]</code> を書いて再実行する。"
                   "範囲の外は NaN になり、補完へ回る（winsorize で丸めない）。", kind="html")
            st = sus.copy()
            if not show_values:
                st = st.drop(columns=[c for c in ("値", "ID", "次の値", "中央値")
                                      if c in st.columns])
            s.table(st, caption="入力ミスの可能性がある値"
                    + ("" if show_values else "（値は show_values=True で表示）"))
        for n in outliers.notes:
            s.text(f"注記: {n}", kind="note")

    # --- 記述統計
    if gt is not None and gt.tables():
        s = rep.section("7. Table 1 / Table 2 ―― 論文にそのまま載る形")
        s.text("<b>Table 1 は全症例の背景、Table 2 は群間比較。</b>"
               "Table 1 に p 値は載せない（背景を述べる表で検定はしない）。"
               "群分けの指定が無ければ Table 2 は作らない。<br>"
               "連続変数は <b>平均値 ± 標準偏差 [最小値, 最大値]</b>、"
               "離散変数は <b>n (%)</b>。"
               "<b>検定手法は表の中ではなく脚注に書く</b>"
               "（行ごとに書くと表が横に伸びて読めなくなる）。<br>"
               "同じ数値から<b>日本語版と英語版</b>を作ってある。"
               "Excel（sheet1=日本語 / sheet2=英語）と Word（ページを分けて日英）に"
               "書き出してあるので、<u>そのまま原稿に貼れる</u>。", kind="html")
        for t in gt.tables():
            s.text(t.to_html("ja"), kind="html")
        for n in gt.notes:
            s.text(f"注記: {n}", kind="note")
    has7 = gt is not None and bool(gt.tables())
    if not has7 and (table1 is not None or comparison is not None
                     or (achievement is not None and len(achievement))):
        s = rep.section("7. Table 1 / Table 2 ―― 論文にそのまま載る形")
    if table1 is not None:
        s.text("7.1 Table 1（詳細版）", kind="h3")
        s.table(table1.table, caption="Table 1", max_rows=300)
        for n in table1.notes:
            s.text(f"注記: {n}", kind="note")
    if comparison is not None:
        s.text("7.2 群間比較の詳細", kind="h3")
        s.text("<b>どの検定をなぜ選んだかを「判定の根拠」列に残してある。</b>", kind="html")
        s.table(comparison.to_frame(), caption="群間比較", max_rows=200)
        ph = comparison.posthoc_frame()
        if len(ph):
            s.table(ph, caption="事後比較", max_rows=200)
    if achievement is not None and len(achievement):
        s.text("7.3 管理目標の達成率", kind="h3")
        s.text("<b>境界値ちょうどの行に注意。</b>"
               "「5.5 未満」を <code>&lt;= 5.5</code> と書くと達成率が数ポイント動く。", kind="html")
        s.table(achievement, caption="管理目標の達成率", max_rows=200)

    # --- 生存時間
    if survival_frame is not None and survival_summary is None:
        survival_summary = survival_frame.summary()
    if survival_summary is not None or logrank is not None or cox is not None:
        s = rep.section("8. 生存時間")
        s.text("観察開始日・イベント発生日・打ち切り日から、観察期間（<code>duration</code>）と"
               "イベント（<code>event</code>）を作った。"
               "<b>Kaplan-Meier・log-rank・Cox はノートブックの 8. で行う</b>"
               "（群の分け方と共変量は人が決めるため）。", kind="html")
        if survival_summary is not None:
            s.table(survival_summary, caption="生存時間の要約")
        if logrank is not None:
            s.pre(logrank.report())
        if cox is not None:
            s.pre(cox.report())
        if survival_frame is not None and len(survival_frame.excluded):
            s.table(survival_frame.excluded["理由"].value_counts()
                    .rename_axis("理由").reset_index(name="件数"),
                    caption="生存時間に変換できなかった症例（理由別。行は削除せず印を付けた）")

    # --- 分割と前処理
    if split is not None or preprocessor is not None:
        s = rep.section("9. 分割と前処理 ―― 機械学習の準備（モデルに渡す行列）")
    if split is not None:
        s.text("9.1 分割", kind="h3")
        s.text(f"{split.strategy}：train {split.n_train} 例 / test {split.n_test} 例。")
        if len(split.balance):
            s.table(split.balance, caption="train と test の比較（SMD）")
        for w in split.warnings:
            s.text(f"警告: {w}", kind="warn")
        for n in split.notes:
            s.text(f"注記: {n}", kind="note")
    if preprocessor is not None:
        s.text("9.2 前処理（training だけで fit した変換）", kind="h3")
        s.pre(preprocessor.report())
        refs = preprocessor.reference_levels()
        if refs:
            names = (getattr(preprocessor, "design", {}) or {}).get("binary_names", {})
            s.table(pd.DataFrame([{"元の列": k, "基準（0 とした水準）": v,
                                   "作った列": names.get(k, "")} for k, v in refs.items()]),
                    caption="基準水準（係数はこの水準との比になる）。"
                            "二値列は 0/1 の 1 本にまとめ、1 が何かが分かる列名にしてある"
                            "（性別 → <b>男性</b>：1=男性・0=女性）")

    # --- 図
    if figures is not None and len(figures):
        s = rep.section("10. 図")
        # ★小節番号はノートブックの 10. と同じ順に振る。★
        for k, (ttl, caption, fig) in enumerate(figures.figures, 1):
            s.text(f"10.{k} {ttl}", kind="h3")
            s.figure(fig, caption=caption, alt=ttl)
        if figures.skipped:
            s.table(pd.DataFrame(figures.skipped, columns=["描けなかった図", "理由"]),
                    caption="描けなかった図")

    # --- 再現性
    s = rep.section("11. 再現性")
    s.text(f"medprep {__version__} で作成。"
           "<b>元データ ＋ schema.yaml ＋ この版番号</b>があれば前処理を完全に再現できる。"
           "論文の Methods には「前処理は medprep、設定は補足資料の schema.yaml のとおり」"
           "と書ける状態を保つこと。", kind="html")

    if path:
        rep.to_html(path)
    return rep


# ================================================================== 描画
def _render_block(kind, payload, show_values) -> str:
    if kind == "table":
        df, caption, max_rows = payload
        return _table_html(df, caption, max_rows)
    if kind == "figure":
        fig, caption, alt = payload
        b64 = to_base64(fig)
        cap = f'<figcaption>{_html.escape(caption)}</figcaption>' if caption else ""
        return (f'<figure><img alt="{_html.escape(alt)}" '
                f'src="data:image/png;base64,{b64}">{cap}</figure>')
    if kind == "findings":
        return _findings_html(payload)
    if kind == "pre":
        return f"<pre>{_html.escape(str(payload))}</pre>"
    if kind == "html":
        return f"<p>{payload}</p>"
    if kind == "h3":
        return f"<h3>{_html.escape(str(payload))}</h3>"
    if kind == "note":
        return f'<p class="note">{_html.escape(str(payload))}</p>'
    if kind == "warn":
        return f'<p class="warn">{_html.escape(str(payload))}</p>'
    return f"<p>{_html.escape(str(payload))}</p>"


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _table_html(df: pd.DataFrame, caption: str, max_rows: int) -> str:
    if df is None or not len(df):
        return ""
    shown = df.head(max_rows)
    more = (f'<p class="note">全 {len(df)} 行のうち {max_rows} 行を表示</p>'
            if len(df) > max_rows else "")
    cap = f"<caption>{_html.escape(caption)}</caption>" if caption else ""
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in shown.columns)
    rows = []
    for _, r in shown.iterrows():
        cells = []
        for v in r:
            s = "" if _isna(v) else str(v)
            # 短いセル（数値・水準名）は折り返さず、長い説明文だけ折り返す
            cls = ' class="t-wrap"' if len(s) > 28 else ' class="t-num"'
            cells.append(f"<td{cls}>{_html.escape(s)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<div class="tw"><table>{cap}<thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>{more}")


def _findings_html(items: list) -> str:
    if not items:
        return '<p class="note">所見なし。</p>'
    out = []
    for f in items:
        sev = f["severity"]
        out.append(
            f'<div class="finding {sev}">'
            f'<div class="fhead"><span class="ficon">{_ICON[sev]}</span>'
            f'<span class="fsev">{_LABEL[sev]}</span>'
            f'<span class="fcat">{_html.escape(f["category"])}</span>'
            f'<span class="fmsg">{_html.escape(f["message"])}</span></div>'
            + (f'<div class="fcols">対象列: {_html.escape(f["columns"])}</div>'
               if f["columns"] else "")
            + f'<div class="faction">→ {_html.escape(f["action"])}</div>'
            + (f'<div class="fex">該当例: {_html.escape(f["examples"])}</div>'
               if f["examples"] else "")
            + "</div>")
    return "".join(out)


def _finding_dict(f, show_values: bool) -> dict:
    n = f"（{f.n} 件）" if f.n is not None else ""
    ex = ""
    if f.examples:
        ex = ("、".join(map(str, f.examples[:10]))
              + ("…" if len(f.examples) > 10 else "")) if show_values \
            else f"{len(f.examples)} 例（値は show_values=True で表示）"
    return {"severity": f.severity, "category": f.category,
            "message": f.message + n,
            "columns": "、".join(map(str, f.columns)) if f.columns else "",
            "action": f.action, "examples": ex}


def _default_subtitle(df, schema) -> str:
    s = f"{len(df)} 行 × {df.shape[1]} 列"
    if schema is not None and schema.target:
        s += f"／目的変数 {schema.target['name']}"
    return s


def _slug(title: str, i: int) -> str:
    base = re.sub(r"[^0-9A-Za-z]+", "-", title).strip("-").lower()
    return f"sec{i}-{base}" if base else f"sec{i}"


def _fit_attention_sheet(ws) -> None:
    """「人の確認が要る事項」のシートを読める幅にする。★文は折り返す。★

    1 件が 100 文字を超えることがある。列幅を見出し（2 文字）で決めると、
    画面の右へ流れて、印刷すると切れる。
    """
    from openpyxl.styles import Alignment

    ws.column_dimensions["A"].width = 6
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows(min_row=2):
        row[0].alignment = Alignment(vertical="top", horizontal="right")
        row[1].alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"


def _sheet_name(name: str, existing: dict) -> str:
    base = re.sub(r"[\\/*?:\[\]]", "", str(name))[:28] or "表"
    n, out = 1, base
    while out in existing:
        n += 1
        out = f"{base}{n}"
    return out


# ================================================================== 見た目
_CSS = f"""
:root {{
  --surface: {PALETTE.surface};
  --plane: #f9f9f7;
  --ink: {PALETTE.ink};
  --ink2: {PALETTE.ink_secondary};
  --muted: {PALETTE.muted};
  --grid: {PALETTE.grid};
  --axis: {PALETTE.axis};
  --error: {PALETTE.critical};
  --warn: {PALETTE.warning};
  --info: {PALETTE.muted};
  --accent: {PALETTE.categorical[0]};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: system-ui, -apple-system, "Hiragino Sans", "Noto Sans JP", sans-serif;
  font-size: 15px; line-height: 1.7;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 80px; }}
header {{ border-bottom: 2px solid var(--axis); padding-bottom: 14px; margin-bottom: 22px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 19px; margin: 40px 0 10px; padding-bottom: 6px;
      border-bottom: 1px solid var(--grid); }}
h3 {{ font-size: 16px; margin: 26px 0 6px; color: var(--ink2); }}
.sub {{ color: var(--ink2); font-size: 14px; }}
.banner {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 18px 0 8px; }}
.count {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
          padding: 10px 16px; min-width: 150px; }}
.count .n {{ font-size: 26px; font-weight: 650; }}
.count .l {{ font-size: 13px; color: var(--ink2); }}
.count.error .n {{ color: var(--error); }}
.count.warn .n {{ color: #9a6b00; }}
nav {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
       padding: 10px 18px; margin: 18px 0 8px; }}
nav a {{ color: var(--accent); text-decoration: none; margin-right: 16px;
         font-size: 14px; white-space: nowrap; }}
nav a:hover {{ text-decoration: underline; }}
section {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
           padding: 4px 20px 18px; margin-bottom: 18px; }}
.finding {{ border-left: 4px solid var(--info); background: var(--plane);
            padding: 10px 14px; margin: 10px 0; border-radius: 0 6px 6px 0; }}
.finding.error {{ border-left-color: var(--error); }}
.finding.warn {{ border-left-color: var(--warn); }}
.fhead {{ display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }}
.ficon {{ font-weight: 700; }}
.finding.error .ficon, .finding.error .fsev {{ color: var(--error); }}
.finding.warn .ficon, .finding.warn .fsev {{ color: #9a6b00; }}
.fsev {{ font-size: 12px; font-weight: 650; }}
.fcat {{ font-size: 12px; color: var(--ink2); background: var(--grid);
         padding: 1px 8px; border-radius: 10px; }}
.fmsg {{ flex: 1 1 320px; }}
.fcols, .fex {{ font-size: 13px; color: var(--ink2); margin-top: 4px; }}
.faction {{ font-size: 13.5px; color: var(--ink2); margin-top: 4px; }}
.tw {{ overflow-x: auto; margin: 10px 0; }}
/* ★横に溢れさせない。★ 溢れた表は横スクロールに隠れ、紙に刷ると『判断の根拠』の右端が消える */
table {{ border-collapse: collapse; max-width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }}
caption {{ text-align: left; font-size: 13px; color: var(--ink2); padding-bottom: 6px; }}
th, td {{ border-bottom: 1px solid var(--grid); padding: 5px 10px; text-align: left;
          vertical-align: top; }}
/* 数値は短いので折り返らない。長い説明文（判断の根拠・対応）は折り返す。
   nowrap にすると画面でも印刷でも右端で切れ、読み手には欠けて見える。 */
th {{ white-space: nowrap; }}
td {{ white-space: normal; overflow-wrap: anywhere; }}
/* ★接頭辞を付ける。★ ページ外枠の .wrap（padding 80px）と名前がぶつかると
   セルにその余白が当たり、行が 4 倍の高さに膨らむ。 */
td.t-num {{ white-space: nowrap; }}
td.t-wrap {{ max-width: 52ch; min-width: 14ch; }}
thead th {{ border-bottom: 2px solid var(--axis); color: var(--ink2);
            font-weight: 600; position: sticky; top: 0; background: var(--surface); }}
tbody tr:hover {{ background: var(--plane); }}
figure {{ margin: 10px 0 22px; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid var(--grid);
              border-radius: 6px; background: var(--surface); }}
figcaption {{ font-size: 13px; color: var(--ink2); margin-top: 6px; }}
pre {{ background: var(--plane); border: 1px solid var(--grid); border-radius: 6px;
       padding: 12px 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.55;
       white-space: pre-wrap; }}
p.note {{ font-size: 13px; color: var(--ink2); }}
h3.att {{ font-size: 15px; margin: 14px 0 6px; }}
ol.att {{ margin: 6px 0 12px; padding-left: 2.2em; }}
ol.att li {{ margin: 4px 0; line-height: 1.6; overflow-wrap: anywhere; }}
p.warn {{ font-size: 13.5px; color: #9a6b00; }}
code {{ background: var(--grid); padding: 1px 5px; border-radius: 4px; font-size: 13px; }}
footer {{ color: var(--muted); font-size: 12.5px; margin-top: 28px;
          border-top: 1px solid var(--grid); padding-top: 12px; }}
@media print {{
  body {{ background: #fff; }}
  section {{ border: none; padding: 0; }}
  nav {{ display: none; }}
}}
"""

_TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>{{ css | safe }}</style>
</head><body><div class="wrap">
<header>
  <h1>{{ title }}</h1>
  <div class="sub">{{ subtitle }}</div>
</header>

{% if counts and (counts.error or counts.warn or counts.info) %}
<div class="banner">
  <div class="count error"><div class="n">{{ counts.error }}</div>
    <div class="l">✗ 致命的 — このまま解析すると誤った結論になる</div></div>
  <div class="count warn"><div class="n">{{ counts.warn }}</div>
    <div class="l">! 要確認</div></div>
  <div class="count"><div class="n">{{ counts.info }}</div>
    <div class="l">· 記録</div></div>
</div>
{% endif %}

{% if headline or attention is not none %}
<section><h2>まず読むこと</h2>
{% if attention is not none %}
<h3 class="att">{{ attention_title }}{% if attention %}（{{ attention|length }} 件）{% endif %}</h3>
{% if attention %}
<p class="note">空でないのがふつうである。1 件ずつ確かめ、必要なら schema.yaml や引数を直して再実行すること。
同じ一覧が <code>report/{{ attention_title }}.txt</code> と <code>table/prep_tables.xlsx</code> の 1 枚目にもある。</p>
<ol class="att">{% for t in attention %}<li>{{ t }}</li>{% endfor %}</ol>
{% else %}
<p>なし</p>
{% endif %}
{% if headline %}<h3 class="att">監査の致命的な所見（詳しく）</h3>{% endif %}
{% endif %}
{% for f in headline %}
  <div class="finding {{ f.severity }}">
    <div class="fhead"><span class="ficon">{{ icon[f.severity] }}</span>
      <span class="fsev">{{ label[f.severity] }}</span>
      <span class="fcat">{{ f.category }}</span>
      <span class="fmsg">{{ f.message }}</span></div>
    {% if f.columns %}<div class="fcols">対象列: {{ f.columns }}</div>{% endif %}
    <div class="faction">→ {{ f.action }}</div>
  </div>
{% endfor %}
</section>
{% endif %}

<nav>{% for a, t in toc %}<a href="#{{ a }}">{{ t }}</a>{% endfor %}</nav>

<!--BODY-->

<footer>
  medprep {{ version }} — {{ generated }} 生成。
  {% if show_values %}★このレポートは症例レベルの値を含む。★{% else %}
  症例レベルの値は含まれていない。{% endif %}
  元データ ＋ schema.yaml ＋ この版番号で、前処理は完全に再現できる。
</footer>
</div></body></html>
"""
