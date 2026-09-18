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
            for s in self.sections:
                for kind, payload in s.blocks:
                    if kind == "table":
                        df, caption, _ = payload
                        name = _sheet_name(caption or s.title, sheets)
                        sheets[name] = df
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name[:31], index=False)
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
    comparison=None,
    achievement=None,
    survival_summary=None,
    logrank=None,
    cox=None,
    split=None,
    preprocessor=None,
    figures: FigureSet | None = None,
    show_values: bool = False,
    path=None,
) -> Report:
    """手元にあるものを全部 1 枚にまとめる。**無いものは黙って飛ばす。**"""
    rep = Report(title=title, subtitle=subtitle or _default_subtitle(df, schema),
                 show_values=show_values)

    # --- 冒頭：警告の要約（★ここを最初に置くことが目的である★）
    if audit is not None:
        rep.counts = {ERROR: len(audit.errors), WARN: len(audit.warnings),
                      INFO: len(audit.by_severity(INFO))}
        rep.headline = [_finding_dict(f, show_values) for f in audit.errors[:8]]

    s = rep.section("このレポートの読み方")
    s.text("上から順に、<b>このデータで解析してよいか</b>（監査）→ "
           "<b>列の役割</b>（schema）→ 欠損と外れ値 → 記述統計 → 生存時間 → "
           "分割と前処理、と降りていく。", kind="html")
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

    # --- 減らしたもの（★何を捨てたかを言わない自動化は信用してはならない★）
    if removed is not None and len(removed):
        s = rep.section("1b. 減らしたもの")
        col = int((removed["種類"] == "列").sum())
        row = int(removed.loc[removed["種類"] == "行", "件数"].sum())
        val = int(removed.loc[removed["種類"] == "値", "件数"].sum())
        s.text(f"<b>列 {col} 本を解析から外し、行 {row} 例に印を付け、"
               f"値 {val} 個を NaN にした。</b>それぞれ理由を付けてある。", kind="html")
        s.text("<b>★行は 1 つも削除していない。★</b> 削除すると症例数と並びが変わり、"
               "元のデータと症例ごとに axis=1 で結合し直せなくなる。"
               "外すかどうかは <code>除外推奨</code> 列を見て人が決めること。", kind="html")
        s.table(removed, caption="減らしたものの一覧", max_rows=300)

    # --- 二値の列
    if encoded is not None and len(encoded):
        s = rep.section("1d. 二値の列を 0/1 に直した ―― 1 がどちらかを列名で示す")
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

    # --- 書き出したデータ（★3 つの違いを最初に言う★）
    if outputs is not None and len(outputs):
        s = rep.section("1c. 書き出したデータ ―― 3 つはどう違うのか")
        s.text("同じデータを <b>3 段階</b>で書き出してある。"
               "<b>施した処置が 1 段ずつ違う。</b>どれを使うかで結果が変わるので、"
               "下の表で確かめること。", kind="html")
        s.text("<b>掃除済み</b>は <u>人が読むため</u>のもの。元の記録と症例ごとに"
               "突き合わせられるよう、ID も落とす予定の列も残してある。<br>"
               "<b>解析用</b>は <u>自分で解析するため</u>のもの。解析に使わない列"
               "（ID・重複列・自由記載）を除いてある。"
               "<b>行は 1 つも減っていない</b>ので、元データと "
               "<code>axis=1</code> で結合できる。<br>"
               "<b>前処理済み</b>は <u>モデルに渡すため</u>のもの。"
               "目的変数が欠測の症例を含まないので<b>行数が違う</b>。"
               "元の行番号と ID を付けてあるので、あとから突き合わせられる。",
               kind="html")
        s.table(outputs, caption="書き出したデータと、そこまでに施した処置",
                max_rows=20)
        if run is not None:
            s.text(f"保存先: {run.run}", kind="note")

    # --- schema
    if schema is not None:
        s = rep.section("2. 列の役割（schema）")
        s.text("<b>判断には必ず理由を付けてある。</b>"
               "直したい場合は <code>schema.yaml</code> を編集して再実行する。", kind="html")
        s.table(schema.to_frame(), caption="列の役割", max_rows=300)
        if schema.unknown():
            s.text(f"★役割を推定できなかった列: {schema.unknown()}")

    # --- 欠損
    if missing is not None:
        s = rep.section("3. 欠損")
        s.text(f"欠損がまったく無い行 {missing.n_complete} / {missing.n_rows}"
               f"（{missing.complete_rate:.1%}）。")
        s.table(missing.columns, caption="列ごとの欠損")
        if len(missing.patterns):
            s.table(missing.patterns, caption="欠損パターン", max_rows=25)
        if len(missing.signals):
            s.text("★欠損が他の列と関連している（MCAR ではない）。"
                   "全体の中央値で埋めると群間差が人工的に作られる。")
            s.table(missing.signals, caption="欠損の偏り")
        for n in missing.notes:
            s.text(f"注記: {n}", kind="note")

    # --- 外れ値
    if outliers is not None:
        s = rep.section("4. 外れ値")
        s.text("<b>ここに出るのは「あり得るが極端」な値である。</b>"
               "生理学的にあり得ない値（Hb 0 など）は掃除の段で既に NaN にしてある。"
               "医学では外れ値こそが重要な症例でありうるので、既定では削除しない。",
               kind="html")
        t = outliers.table.copy()
        if not show_values and "該当例" in t.columns:
            t = t.drop(columns=["該当例"])
        s.table(t, caption="列ごとの外れ値")
        for n in outliers.notes:
            s.text(f"注記: {n}", kind="note")

    # --- 記述統計
    if table1 is not None:
        s = rep.section("5. Table 1")
        s.table(table1.table, caption="Table 1", max_rows=300)
        for n in table1.notes:
            s.text(f"注記: {n}", kind="note")
    if comparison is not None:
        s = rep.section("5b. 群間比較の詳細")
        s.text("<b>どの検定をなぜ選んだかを「判定の根拠」列に残してある。</b>", kind="html")
        s.table(comparison.to_frame(), caption="群間比較", max_rows=200)
        ph = comparison.posthoc_frame()
        if len(ph):
            s.table(ph, caption="事後比較", max_rows=200)
    if achievement is not None and len(achievement):
        s = rep.section("6. 管理目標の達成率")
        s.text("<b>境界値ちょうどの行に注意。</b>"
               "「5.5 未満」を <code>&lt;= 5.5</code> と書くと達成率が数ポイント動く。", kind="html")
        s.table(achievement, caption="管理目標の達成率", max_rows=200)

    # --- 生存時間
    if survival_summary is not None or logrank is not None or cox is not None:
        s = rep.section("7. 生存時間")
        if survival_summary is not None:
            s.table(survival_summary, caption="生存時間の要約")
        if logrank is not None:
            s.pre(logrank.report())
        if cox is not None:
            s.pre(cox.report())

    # --- 分割と前処理
    if split is not None:
        s = rep.section("8. 分割")
        s.text(f"{split.strategy}：train {split.n_train} 例 / test {split.n_test} 例。")
        if len(split.balance):
            s.table(split.balance, caption="train と test の比較（SMD）")
        for w in split.warnings:
            s.text(f"警告: {w}", kind="warn")
        for n in split.notes:
            s.text(f"注記: {n}", kind="note")
    if preprocessor is not None:
        s = rep.section("9. 前処理")
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
        for ttl, caption, fig in figures.figures:
            s.text(ttl, kind="h3")
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

{% if headline %}
<section><h2>まず読むこと</h2>
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
