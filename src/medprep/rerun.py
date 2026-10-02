"""schema.yaml を使った再実行と、元の結果との照合（0.12.0〜）。

    rep = mp.autoprep(df, schema="…/run1/model/schema.yaml", save=True)   # 直して再実行
    rr  = mp.reproduce("…/run1")                                          # 再現して照合
    rr.table                                                               # ファイルごとの一致

★方針★
  * `schema.yaml` の設定（`run:` 欄）と列の判断は、`autoprep` に渡した引数より**優先**する。
    再現が目的なので、ノートブックの設定を後から変えても結果が変わらない方が安全である。
    食い違いは黙って直さず、「人の確認が要る事項」に一覧で出す。
  * 列の役割の推定そのものは変えない。推定したあとで、`schema.yaml` の判断で上書きする。
    同じデータ・同じ版なら上書きは何も変えない（＝再現）。人が直した所だけが変わる。
  * 0.11.x 以前の `schema.yaml`（`run:` 欄が無い）も読む。読めない設定は引数の値を使い、
    そのことを「人の確認が要る事項」に書く。
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from .schema import (
    BINARY,
    CONSTANT,
    DATETIME,
    DUPLICATE,
    EVENT,
    GROUP,
    HIGH_CARDINALITY,
    ID,
    NOMINAL,
    NUMERIC,
    ORDINAL,
    OUTCOME,
    TEXT,
    TIME,
    TIME_OF_DAY,
    UNKNOWN,
    Schema,
)

#: schema.yaml の role に書いてよい値
VALID_ROLES = {ID, OUTCOME, TIME, EVENT, GROUP, DATETIME, NUMERIC, BINARY, ORDINAL, NOMINAL,
               HIGH_CARDINALITY, TIME_OF_DAY, TEXT, CONSTANT, DUPLICATE, UNKNOWN}
#: keep にしたとき特徴量になる役割（それ以外は keep にしても前処理の段で外れる）
FEATURE_ROLES = {NUMERIC, ORDINAL, BINARY, NOMINAL, GROUP}
#: 人が直してよい（＝上書きする）項目。欠損率・水準数などデータから決まるものは上書きしない
EDITABLE = ("role", "action", "timing", "dict_key", "unit", "order", "value_map", "duplicate_of")

#: `run:` 欄に残す autoprep の設定と、その既定値（★autoprep の既定と同じにすること★）
RUN_DEFAULTS = {
    "outcome": None, "task": None, "group": None, "id_col": None,
    "survival": None, "survival_dates": None, "date_col": None,
    "columns": None, "table1_columns": None,
    "test_size": 0.2, "seed": 0, "clean": True, "drop_empty_rows": False, "do_split": "auto",
}
_TUPLES = ("survival", "survival_dates")


# ================================================================== 設定
def _norm(v):
    """比較と YAML 書き出しのための正規化（tuple → list、空 → None、numpy → Python）。"""
    if v is None or v == "" or (isinstance(v, (list, tuple)) and len(v) == 0):
        return None
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _denorm(key, v):
    """YAML から読んだ値を autoprep の引数の形に戻す。"""
    v = _norm(v)
    if key in _TUPLES and v is not None:
        return tuple(v)
    return v


def run_block(settings: dict, version: str) -> dict:
    """`schema.yaml` の `run:` 欄。"""
    out = {"medprep": version}
    for k in RUN_DEFAULTS:
        out[k] = _norm(settings.get(k))
    return out


def load_schema(schema) -> Schema:
    """パス・Schema・dict のどれでも受けて Schema にし、書き間違いを調べる。"""
    if isinstance(schema, Schema):
        given = copy.deepcopy(schema)
    elif isinstance(schema, dict):
        import yaml
        given = Schema.from_yaml(yaml.safe_dump(schema, allow_unicode=True))
    else:
        path = os.fspath(schema)
        if not os.path.exists(path):
            raise FileNotFoundError(f"schema.yaml が見つからない: {path}")
        given = Schema.from_yaml(path)
    bad = []
    for c, s in given.columns.items():
        if s.role not in VALID_ROLES:
            bad.append(f"'{c}' の role '{s.role}'（使える値: {sorted(VALID_ROLES)}）")
        if s.action not in ("keep", "drop"):
            bad.append(f"'{c}' の action '{s.action}'（keep か drop）")
    if bad:
        raise ValueError("schema.yaml の書き方に誤りがある。直してから再実行すること:\n  "
                         + "\n  ".join(bad))
    return given


def resolve_settings(given: Schema, passed: dict, data_columns) -> tuple[dict, list]:
    """autoprep の設定を決める。**schema.yaml を優先し、食い違いを文で返す。**"""
    out = dict(passed)
    msgs: list = []
    run = given.run or {}
    if run:
        diffs = []
        for k, default in RUN_DEFAULTS.items():
            if k not in run:
                continue                       # 後の版で足した設定。引数の値を使う
            s_val, p_val = _norm(run[k]), _norm(passed.get(k))
            if p_val != s_val and p_val != _norm(default):
                diffs.append(f"{k}: 引数 {p_val!r} → schema.yaml の {s_val!r}")
            out[k] = _denorm(k, run[k])
        if diffs:
            msgs.append("★schema.yaml と、autoprep に渡した設定（ノートブックの環境変数）とで"
                        "違うものがある。schema.yaml の値を使った★  " + "；".join(diffs))
        used = [f"{k}={_norm(run[k])!r}" for k in RUN_DEFAULTS
                if k in run and _norm(run[k]) is not None]
        msgs.append("schema.yaml の設定（run 欄）で実行した: " + "、".join(used))
        return out, msgs

    # ---- 0.11.x 以前の schema.yaml（run 欄が無い）
    cols = set(map(str, data_columns))
    took = []
    tg = given.target or {}
    if out.get("outcome") is None and tg.get("name") and tg.get("inferred_role") != "auto" \
            and str(tg["name"]) in cols:
        out["outcome"] = tg["name"]
        out["task"] = out.get("task") or tg.get("task")
        took.append(f"目的変数 {tg['name']}")
    sv = given.survival or {}
    if out.get("survival_dates") is None and out.get("survival") is None and sv:
        if sv.get("form") == "C" and sv.get("start_date"):
            out["survival_dates"] = (sv["start_date"], sv["event_date"], sv["censor_date"])
            took.append("生存時間の 3 列")
        elif sv.get("time") and sv.get("event"):
            out["survival"] = (sv["time"], sv["event"])
            took.append("観察期間とイベントの列")
    for key, role, label in (("id_col", ID, "ID"), ("group", GROUP, "群")):
        if out.get(key) is None:
            c = next((c for c, s in given.columns.items()
                      if s.role == role and str(c) in cols), None)
            if c is not None:
                out[key] = c
                took.append(f"{label} {c}")
    msgs.append(
        "★この schema.yaml には autoprep の設定（run 欄）が無い（medprep 0.11.x 以前の形式）。★ "
        + (("、".join(took) + " は schema.yaml から読み、") if took else "")
        + f"seed={out.get('seed')!r}・test_size={out.get('test_size')!r}・"
          f"date_col={out.get('date_col')!r}・drop_empty_rows={out.get('drop_empty_rows')!r} は"
          "引数の値を使った。元の実行と同じ値か確かめること")
    return out, msgs


# ================================================================== 列の判断の上書き
def apply_overrides(inferred: Schema, given: Schema) -> list:
    """推定した schema を、schema.yaml の判断で上書きする。変えた所の一覧を返す。"""
    edits = []
    for c, g in given.columns.items():
        sp = inferred.columns.get(c)
        if sp is None:
            continue
        # ★観察期間・イベントの役割は、推定のあとで autoprep 自身が付け直す。★
        #   人が直したものではないので、上書きはしても「直した判断」には数えない。
        system = g.role in (TIME, EVENT) or c in ("duration", "event", "_start", "_end")
        changed = []
        for f in EDITABLE:
            gv, iv = getattr(g, f), getattr(sp, f)
            if gv is None or gv == iv:       # 書いていない項目は「意見なし」とみなす
                continue
            changed.append((f, iv, gv))
            setattr(sp, f, copy.deepcopy(gv))
        if changed and not system:
            if any(f in ("role", "action") for f, _, _ in changed) and g.reason:
                sp.reason = g.reason
            edits.append((c, changed))
    if given.policy:
        inferred.policy = copy.deepcopy(given.policy)
    return edits


def describe_edits(edits: list, final: Schema, limit: int = 20) -> list:
    """上書きした所を「人の確認が要る事項」の文にする。"""
    if not edits:
        return []
    parts = []
    for c, changed in edits[:limit]:
        parts.append(f"'{c}': " + "、".join(f"{f} {a!r}→{b!r}" for f, a, b in changed))
    more = f"（ほか {len(edits) - limit} 列）" if len(edits) > limit else ""
    msgs = ["schema.yaml で直した判断（自動の推定と違うもの）を使った: "
            + "；".join(parts) + more]
    useless = [c for c, changed in edits
               if c in final.columns and final.columns[c].action == "keep"
               and final.columns[c].role not in FEATURE_ROLES | {OUTCOME, ID, TIME, EVENT}
               and any(f == "action" for f, _, _ in changed)]
    if useless:
        msgs.append("★keep に直したが、役割のために特徴量にならない列がある★ "
                    + "、".join(f"'{c}'（role {final.columns[c].role}）" for c in useless)
                    + "。特徴量にしたいときは role も numeric・nominal などに直すこと")
    return msgs


def coverage_messages(given: Schema, final: Schema) -> list:
    """schema.yaml とデータとで、列が食い違うときの文。"""
    msgs = []
    gone = [c for c in given.columns if c not in final.columns]
    new = [c for c in final.columns if c not in given.columns]
    if gone:
        msgs.append(f"schema.yaml にあるが、今回のデータに無い列（使わなかった）: {gone[:20]}"
                    + (f" ほか {len(gone) - 20} 列" if len(gone) > 20 else ""))
    if new:
        msgs.append(f"★schema.yaml に無い列がある（役割は自動で推定した）★ {new[:20]}"
                    + (f" ほか {len(new) - 20} 列" if len(new) > 20 else "")
                    + "。元の実行とデータが違う可能性がある")
    return msgs


# ================================================================== 再現と照合
@dataclass
class Reproduction:
    """`mp.reproduce()` の結果。`table` がファイルごとの照合。"""
    original: str
    reproduced: str
    table: pd.DataFrame
    version_original: str
    version_now: str
    rep: object = None
    schema_diff: pd.DataFrame | None = None
    notes: list = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """書き出したデータと列の判断がすべて元と一致したか（版の違いは含めない）。"""
        t = self.table[self.table["対象"] != "medprep の版"]
        return bool(len(t)) and bool((t["一致"] == "一致").all())

    def report(self) -> str:
        from .textfmt import frame_text
        head = (f"元の実行   : {self.original}\n再現した実行: {self.reproduced}\n"
                f"medprep の版: 元 {self.version_original} / 今 {self.version_now}\n")
        verdict = ("◇ すべて一致した。この run は、元データ・schema.yaml・medprep の版から再現できる。"
                   if self.identical else
                   "★一致しないものがある。下の表の「違い」を確かめること★")
        return head + verdict + "\n\n" + frame_text(self.table) + (
            "\n\n" + "\n".join(f"[注記] {n}" for n in self.notes) if self.notes else "")

    def __repr__(self) -> str:
        return self.report()

    def _repr_html_(self) -> str:
        import html as _h
        return "<pre>" + _h.escape(self.report()) + "</pre>"


def _first_line(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.readline().strip()
    except OSError:
        return "（不明）"


def _same_frame(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.equals(b):
        return True
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False,
                                      rtol=1e-9, atol=1e-12)
        return True
    except AssertionError:
        return False


def _compare_book(a: str, b: str) -> tuple[str, str]:
    if not os.path.exists(b):
        return "不一致", "再現した側にファイルが無い"
    xa = pd.read_excel(a, sheet_name=None)
    xb = pd.read_excel(b, sheet_name=None)
    if list(xa) != list(xb):
        return "不一致", f"シートが違う（元 {list(xa)} / 再現 {list(xb)}）"
    bad = []
    for name in xa:
        if not _same_frame(xa[name], xb[name]):
            sa, sb = xa[name].shape, xb[name].shape
            bad.append(f"{name}（元 {sa[0]}×{sa[1]} / 再現 {sb[0]}×{sb[1]}）")
    if bad:
        return "不一致", "差のあるシート: " + "、".join(bad)
    return "一致", f"{len(xa)} 枚のシートが同じ"


def reproduce(run, data=None, *, full: bool = False, verbose: bool = False,
              **settings) -> Reproduction:
    """保存した run を、元データ・schema.yaml から再実行し、元の結果と照合する。

    run  : 元の `…/Preprocessing/run{N}` フォルダ
    data : 元データ。省略すると `run{N}/data/0_元データ.xlsx` を使う
    full : True なら図と Table 1 も作る（照合には要らないので既定では作らない）
    settings : schema.yaml に run 欄が無い（0.11.x 以前の）とき、元の設定を渡す（seed=42 など）。
        run 欄があれば、そちらが優先される

    再実行の結果は、同じ置き場所の次の `run{N+1}` に保存する（再現も 1 回の実行として残す）。
    """
    from .auto import _pkg_version, autoprep

    run = os.path.abspath(os.fspath(run))
    schema_path = os.path.join(run, "model", "schema.yaml")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.yaml が無い: {schema_path}（run フォルダを指定すること）")
    raw_path = os.path.join(run, "data", "0_元データ.xlsx")
    if data is None:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"元データの複製が無い: {raw_path}（save_data=False で実行した run は、"
                "data= に元データを渡すこと）")
        data = pd.read_excel(raw_path, sheet_name=0)

    v_orig = _first_line(os.path.join(run, "model", "medprep_version.txt"))
    v_now = _pkg_version()

    # 置き場所: …/{project_folder}/{method}/run{N}
    project_dir = os.path.dirname(run)
    method = os.path.basename(project_dir)
    project_folder = ""
    try:
        with open(os.path.join(run, "table", "run_info.json"), encoding="utf-8") as f:
            project_folder = json.load(f).get("project_folder", "") or ""
    except (OSError, ValueError):
        pass
    base = os.path.dirname(project_dir)
    if project_folder:
        base = os.path.dirname(base)

    rep = autoprep(data, schema=schema_path, save=True, method=method,
                   project_folder=project_folder, out_dir=base,
                   do_figures=full, do_table1=full, verbose=verbose,
                   title=f"{os.path.basename(run)} の再現", **settings)

    rows = [{"対象": "medprep の版", "一致": "一致" if v_orig == v_now else "不一致",
             "違い": "" if v_orig == v_now else
             f"元は {v_orig}。同じ版を入れて再実行すること"
             f"（medprep_version.txt の 2 行目にコマンドがある）"}]
    old = Schema.from_yaml(schema_path)
    d = old.diff(rep.schema)
    rows.append({"対象": "schema.yaml（列の判断）", "一致": "一致" if not len(d) else "不一致",
                 "違い": "" if not len(d) else f"{len(d)} か所（rr.schema_diff）"})
    run_old = {k: v for k, v in (old.run or {}).items() if k != "medprep"}
    run_new = {k: v for k, v in (rep.schema.run or {}).items() if k != "medprep"}
    if run_old:
        dk = [k for k in run_new if _norm(run_old.get(k)) != _norm(run_new.get(k))]
        rows.append({"対象": "schema.yaml（設定 run）", "一致": "一致" if not dk else "不一致",
                     "違い": "、".join(dk)})
    new_data = os.path.join(rep.run.run, "data")
    for fn in sorted(os.listdir(os.path.join(run, "data"))):
        if not fn.endswith(".xlsx") or fn.startswith("~$"):
            continue
        ok, why = _compare_book(os.path.join(run, "data", fn), os.path.join(new_data, fn))
        rows.append({"対象": f"data/{fn}", "一致": ok, "違い": "" if ok == "一致" else why})
    rr = Reproduction(original=run, reproduced=rep.run.run, table=pd.DataFrame(rows),
                      version_original=v_orig, version_now=v_now, rep=rep, schema_diff=d)
    if not run_old:
        rr.notes.append("元の schema.yaml に run 欄が無い（0.11.x 以前）。seed などは"
                        f"{settings or '引数の既定値'} で再実行したので、元と違えば一致しない。"
                        "mp.reproduce(run, seed=42) のように元の値を渡すこと")
    for w in rep.warnings:
        if "schema.yaml" in w and "★" in w:
            rr.notes.append(w)
    return rr
