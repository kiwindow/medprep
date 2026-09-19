"""medprep.clean — 辞書駆動で数値列を掃除する（プロトタイプ）。

処理の順序（この順序に意味がある）
  1. 列名を辞書の aliases と照合して項目を同定する
  2. 文字列を数値化する（全角、カンマ、単位トークン、検出限界 "<0.1" の分解）
  3. 欠損コード（999 等）を NaN にする  ※分布から浮いている場合のみ
  4. plausible 範囲外を NaN にする（= 入力ミス。外れ値処理ではない）
  5. 単位混在を検出して換算する
  --- ここまでが「掃除」。補完・外れ値処理・スケーリングは分割の後に行う ---
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .textfmt import frame_text

_DICT_PATH = Path(__file__).parent / "dict" / "ranges_ja.yaml"
_LOD_RE = re.compile(r"^\s*([<>≦≧＜＞]|以下|以上|未満)?\s*([0-9.]+)\s*(.*)$")


def load_dict(path=None) -> dict:
    return yaml.safe_load(open(path or _DICT_PATH, encoding="utf-8"))


def _norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").lower()


def build_alias_map(d: dict) -> dict:
    m = {}
    for key, v in d["items"].items():
        for a in [key, v.get("name_ja", "")] + list(v.get("aliases") or []):
            if a:
                m.setdefault(_norm(a), key)
    return m


@dataclass
class CleanReport:
    matched: dict = field(default_factory=dict)        # 列名 -> 辞書キー
    unmatched: list = field(default_factory=list)
    actions: list = field(default_factory=list)        # (列, 種別, 件数, 詳細)
    flags: dict = field(default_factory=dict)          # 追加したフラグ列

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.actions, columns=["列", "処理", "件数", "詳細"])

    def show(self):
        print(f"辞書に一致した列: {len(self.matched)} / 未一致: {len(self.unmatched)}")
        if self.unmatched:
            print("  未一致:", ", ".join(map(str, self.unmatched[:15])))
        f = self.to_frame()
        print(frame_text(f) if len(f) else "  （処理なし）")


def _to_numeric(series: pd.Series, lod_policy="half"):
    """文字列混じりの列を数値化し、検出限界フラグを返す。"""
    lo = pd.Series(False, index=series.index)
    hi = pd.Series(False, index=series.index)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for i, v in series.items():
        if pd.isna(v):
            continue
        if isinstance(v, (int, float, np.number)):
            out.at[i] = float(v)
            continue
        s = unicodedata.normalize("NFKC", str(v)).strip().replace(",", "")
        if s in ("", "-", "ー", "―", ".", "・"):
            continue
        m = _LOD_RE.match(s)
        if not m:
            continue
        op, num, _rest = m.groups()
        try:
            val = float(num)
        except ValueError:
            continue
        if op in ("<", "＜", "未満"):
            lo.at[i] = True
            out.at[i] = val / 2 if lod_policy == "half" else (val if lod_policy == "lod" else np.nan)
        elif op in (">", "＞", "以上", "≧"):
            hi.at[i] = True
            out.at[i] = val
        elif op in ("以下", "≦"):
            lo.at[i] = True
            out.at[i] = val / 2 if lod_policy == "half" else val
        else:
            out.at[i] = val
    return out, lo, hi


def _detect_unit_scale(x: pd.Series, spec: dict):
    """単位混在を検出する。

    【設計上の原則】基準範囲への近さで行ごとに倍率を当てるのは誤り。
    実測の正常値まで「補正」してしまう（CRP 0.3 mg/dL を 0.03 にする等）。
    ここでは保守的に、**plausible 範囲の外にあり、かつ変換係数を掛けると
    範囲の中に収まる値だけ** を換算する。それ以外は一切触らない。

    Returns
    -------
    (倍率の Series, 何も換算しなかったなら None)
    """
    variants = {float(f): u for u, f in (spec.get("unit_variants") or {}).items() if f}
    if not variants or x.dropna().empty:
        return None
    lo, hi = spec["plausible"]
    outside = x.notna() & ((x < lo) | (x > hi))
    if not outside.any():
        return None
    scale = pd.Series(1.0, index=x.index)
    for i in x.index[outside]:
        v = x.at[i]
        for f in sorted(variants, key=lambda f: abs(np.log10(f))):
            if lo <= v * f <= hi:
                scale.at[i] = f
                break
    return scale if (scale != 1.0).any() else None


def _report_unit_bimodality(x: pd.Series, spec: dict):
    """換算はせず、桁の異なる二峰性があれば「疑い」として報告するだけ。

    Plt の ×10³/µL と ×10⁴/µL のように、どちらの単位でも plausible の中に
    収まってしまう場合は自動判別できない。人に返すのが正しい。
    """
    v = pd.to_numeric(x, errors="coerce").dropna()
    v = v[v > 0]
    if len(v) < 30:
        return None
    lg = np.log10(v)
    gap = lg.max() - lg.min()
    if gap < 0.8:
        return None
    # 1 桁の幅でヒストグラムを取り、間に谷があるか見る
    hist, edges = np.histogram(lg, bins=max(8, int(gap * 6)))
    nz = np.flatnonzero(hist)
    if len(nz) < 2:
        return None
    # 連続する非ゼロ区間が 2 つ以上あり、間隔が 0.7 桁以上あれば二峰性を疑う
    breaks = np.flatnonzero(np.diff(nz) > 1)
    for b in breaks:
        if edges[nz[b + 1]] - edges[nz[b] + 1] >= 0.7:
            return (round(10 ** edges[nz[b] + 1], 4), round(10 ** edges[nz[b + 1]], 4))
    return None


def clean_numeric(
    df: pd.DataFrame,
    dic: dict | None = None,
    lod_policy: str = "half",
    apply_plausible: bool = True,
    apply_missing_codes: bool = True,
    fix_units: bool = True,
    add_flags: bool = True,
) -> tuple[pd.DataFrame, CleanReport]:
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    codes = set(dic.get("missing_codes", {}).get("numeric", []))
    texts = {_norm(t) for t in dic.get("missing_codes", {}).get("text", [])}
    out = df.copy()
    rep = CleanReport()

    for col in df.columns:
        key = amap.get(_norm(col))
        if key is None:
            rep.unmatched.append(col)
            continue
        rep.matched[col] = key
        spec = dic["items"][key]

        # --- テキスト欠損表記を先に NaN にする
        s = out[col]
        if s.dtype == object:
            mask_txt = s.map(lambda v: _norm(v) in texts if isinstance(v, str) else False)
            if mask_txt.any():
                s = s.mask(mask_txt)
                rep.actions.append((col, "テキスト欠損表記→NaN", int(mask_txt.sum()), "未測定/不明 等"))

        # --- 数値化 + 検出限界
        num, lo, hi = _to_numeric(s, lod_policy)
        if lo.any() or hi.any():
            rep.actions.append((col, "検出限界表記を数値化", int(lo.sum() + hi.sum()),
                                f"< は 1/2 LOD を代入（policy={lod_policy}）"))
            if add_flags:
                if lo.any():
                    # ★印の列は、値の列と見間違えない名前にする。★
                    #   `CRP__censored_low` は Excel で列名が切れると
                    #   「CRP」に見え、**ほぼ全部 0 なので「CRP が全部 0」
                    #   と読まれる**。先頭に日本語で用途を書いておく。
                    out[f"検出限界未満_{col}"] = lo.astype(int)
                    rep.flags[f"検出限界未満_{col}"] = col
                if hi.any():
                    out[f"検出限界超_{col}"] = hi.astype(int)
                    rep.flags[f"検出限界超_{col}"] = col

        # --- 欠損コード（分布から浮いている値のみ）
        if apply_missing_codes:
            plo, phi = spec["plausible"]
            cand = sorted({c for c in codes if (num == c).sum() > 0})
            hit = [c for c in cand if c > phi or (num == c).sum() >= 3 and c >= 999 and
                   (num[(num != c)].quantile(0.99) if num[(num != c)].notna().any() else 0) < c]
            for c in hit:
                n = int((num == c).sum())
                num = num.mask(num == c)
                rep.actions.append((col, "欠損コード→NaN", n, f"値 {c}"))

        # --- 単位混在（あり得ない範囲の外にある値だけを換算する）
        if fix_units:
            scale = _detect_unit_scale(num, spec)
            if scale is not None:
                n = int((scale != 1.0).sum())
                num = num * scale.reindex(num.index).fillna(1.0)
                rep.actions.append((col, "単位混在を換算", n,
                                    f"許容範囲外の値に倍率 {sorted(set(scale[scale != 1.0]))} を適用"))
            bim = _report_unit_bimodality(num, spec)
            if bim:
                rep.actions.append((col, "【要確認】桁の異なる二峰性", 0,
                                    f"{bim[0]} と {bim[1]} の間に谷。単位混在の可能性（自動換算はしない）"))

        # --- 生理学的にあり得ない値（入力ミス）
        if apply_plausible:
            plo, phi = spec["plausible"]
            bad = num.notna() & ((num < plo) | (num > phi))
            if bad.any():
                rep.actions.append((col, "あり得ない値→NaN", int(bad.sum()),
                                    f"許容 {plo}–{phi} {spec.get('unit','')}"))
                num = num.mask(bad)

        out[col] = num

    return out, rep


# ======================================================== 透析関連の派生指標
_TIME_RE = re.compile(r"^\s*(\d{1,2})\s*[:：時]\s*(\d{1,2})\s*分?\s*$")

#: 派生指標の列名。★単位を列名に書く。★ 単位の無い数値は必ずどこかで誤読される。
UF_COL, URR_COL, TD_COL = "除水量(kg)", "URR(%)", "透析時間(hr)"
KTV_COL, TSAT_COL, ICA_COL = "spKt/V", "TSAT(%)", "iCa(mg/dL)"
ICAP_COL, VINTAGE_COL = "iCa×P", "透析年数"


def to_hours(v) -> float:
    """時刻を「0 時からの時間」に直す。読めなければ NaN。

    実務の時刻列は `9:30`・`09:30`・`9時30分`・全角コロン・Excel のシリアル小数
    （0.395833… = 9:30）が平気で混ざる。**読めないものを 0 にはしない。**
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    if isinstance(v, (pd.Timestamp, datetime.datetime)):
        return v.hour + v.minute / 60 + v.second / 3600
    if isinstance(v, datetime.time):
        return v.hour + v.minute / 60 + v.second / 3600
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        if 0.0 <= x < 1.0:                      # Excel のシリアル小数
            return x * 24
        return x if 0 <= x < 24 else float("nan")
    m = _TIME_RE.match(unicodedata.normalize("NFKC", str(v)))
    if not m:
        return float("nan")
    h, mi = int(m.group(1)), int(m.group(2))
    return h + mi / 60 if (0 <= h < 24 and 0 <= mi < 60) else float("nan")


def session_hours(start, end) -> pd.Series:
    """開始時刻と終了時刻から透析時間（hr）を作る。

    ★日をまたぐ夜間透析がある。★ 終了が開始より早いときは翌日とみなして 24 を足す。
    24 時間を超える値は入力ミスなので NaN にする（**勝手に丸めない**）。
    """
    a = pd.Series([to_hours(v) for v in start], index=getattr(start, "index", None))
    b = pd.Series([to_hours(v) for v in end], index=getattr(end, "index", None))
    h = b - a
    h = h.mask(h < 0, h + 24)
    return h.mask((h <= 0) | (h > 12))


def _timing_of(col: str) -> str:
    from .timing import detect_timing
    return detect_timing(col)[0]


def _pick(df, amap, key, timing=None):
    """辞書キー `key` に一致する列を選ぶ。`timing` を指定すればその時点のものだけ。

    時点つきが見つからないときは**時点不詳の列で代用し、注記を残す**。
    """
    from .timing import detect_timing
    exact, plain = None, None
    for c in df.columns:
        t, base = detect_timing(c)
        k = amap.get(_norm(base)) or amap.get(_norm(c))
        if k != key:
            continue
        if timing is not None and t == timing and exact is None:
            exact = c
        elif t == "unknown" and plain is None:
            plain = c
    if timing is None:
        return plain or exact, False
    return (exact, False) if exact else (plain, plain is not None)


def _time_col(df, *words):
    for c in df.columns:
        nc = _norm(c)
        if all(w in nc for w in words):
            return c
    return None


def _date_like(df, *words, exclude=("時刻",)):
    for c in df.columns:
        nc = _norm(c)
        if all(w in nc for w in words) and not any(x in nc for x in exclude):
            return c
    return None


def derive_vintage(df: pd.DataFrame, *, start_col: str | None = None,
                   end_col: str | None = None) -> tuple[pd.DataFrame, list]:
    """透析開始年月日から **透析年数（切り捨て）** を作る。

    ★日付を解釈したあとに呼ぶこと。★ 文字列のままでは引き算ができない。

    終了日は「その検査を計算している日」である。検体採取日があればそれを使い、
    無ければ観察開始年月日で代用する（**どちらを使ったかは必ず注記に出す**）。
    どちらも無ければ作らない。**今日の日付では代用しない** ――
    走らせた日によって値が変わる列を、黙って混ぜてはならない。

    0 年は「導入後 1 年未満」を意味する。導入日が不明な症例は NaN。
    """
    from .dates import parse_date_series

    out, notes = df.copy(), []
    if VINTAGE_COL in out.columns:
        return out, notes
    sc = start_col or _date_like(out, "透析開始") or _date_like(out, "導入", "日")
    ec = end_col or _date_like(out, "検体採取") or _date_like(out, "採血", "日") \
        or _date_like(out, "測定", "日") or _date_like(out, "観察開始")
    if not sc or not ec or sc == ec:
        return out, notes

    def _dt(col):
        v = out[col]
        if pd.api.types.is_datetime64_any_dtype(v):
            return v
        return parse_date_series(v).values

    a, b = _dt(sc), _dt(ec)
    years = (b - a).dt.days / 365.25
    out[VINTAGE_COL] = np.floor(years).where(years.notna() & (years >= 0))
    n_unknown = int(out[VINTAGE_COL].isna().sum())
    n_neg = int((years < 0).sum())
    notes.append(
        f"{VINTAGE_COL} を '{sc}' から '{ec}' までの年数（★切り捨て★）として算出した"
        + (f"。算出できない症例が {n_unknown} 例（導入日が空欄・解釈不能）" if n_unknown else "")
        + (f"。開始日が終了日より後の症例が {n_neg} 例あり NaN にした" if n_neg else ""))
    return out, notes


def derive_dialysis(df: pd.DataFrame, dic: dict | None = None) -> tuple[pd.DataFrame, list]:
    """透析の指標を、**入力が揃っている列の組についてだけ**作る。

    | 作る列 | 要る列 | 式 |
    |---|---|---|
    | `除水量(kg)` | 透析前体重・透析後体重 | 前 − 後 |
    | `URR(%)` | 透析前BUN・透析後BUN | (前 − 後) / 前 × 100 |
    | `透析時間(hr)` | 透析開始時刻・透析終了時刻 | 終了 − 開始（日またぎは +24h） |
    | `spKt/V` | 前後BUN・除水量・透析後体重・透析時間 | Daugirdas 第2世代式 |
    | `TSAT(%)` | Fe・TIBC | Fe / TIBC × 100 |
    | `iCa(mg/dL)` | Ca・Alb | Payne 式（Alb < 4.0 のとき Ca + (4 − Alb)） |

    ★揃っていない組は作らない。★ 片方だけで推測した値を入れると、
    それが計算値なのか実測値なのか、あとから誰にも分からなくなる。
    既に同じ名前の列があるときも作らない（人が入れた値を上書きしない）。
    """
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    out, notes = df.copy(), []

    def borrowed(name, col, flag):
        if flag:
            notes.append(f"{name}: 採血時点の書かれていない列 '{col}' を使った。"
                         "★時点が違えば値の意味も違う。列名に透析前/透析後を書くこと★")

    # ---- 1) 透析時間（★他の指標がこれに依存するので最初に作る★）
    c_start = _time_col(out, "開始", "時") or _time_col(out, "start", "time")
    c_end = _time_col(out, "終了", "時") or _time_col(out, "end", "time")
    if c_start and c_end and TD_COL not in out.columns:
        h = session_hours(out[c_start], out[c_end])
        out[TD_COL] = h.round(2)
        bad = int(h.isna().sum() - (out[c_start].isna() | out[c_end].isna()).sum())
        notes.append(f"{TD_COL} を '{c_start}' と '{c_end}' から算出した"
                     + (f"（読めない/あり得ない時刻が {bad} 件あり NaN にした）" if bad > 0 else ""))

    # ---- 2) 除水量
    w_pre, f1 = _pick(out, amap, "weight", "pre")
    w_post, f2 = _pick(out, amap, "weight", "post")
    if w_pre and w_post and w_pre != w_post and UF_COL not in out.columns:
        uf = pd.to_numeric(out[w_pre], errors="coerce") - pd.to_numeric(out[w_post], errors="coerce")
        out[UF_COL] = uf.round(2)
        borrowed(UF_COL, w_pre, f1)
        borrowed(UF_COL, w_post, f2)
        neg = int((uf < 0).sum())
        notes.append(f"{UF_COL} = 透析前体重 − 透析後体重 を算出した"
                     + (f"（負が {neg} 件 — 前後が入れ替わっている疑い）" if neg else ""))

    # ---- 3) URR
    b_pre, f1 = _pick(out, amap, "BUN", "pre")
    b_post, f2 = _pick(out, amap, "BUN", "post")
    if b_pre and b_post and b_pre != b_post and URR_COL not in out.columns:
        pre = pd.to_numeric(out[b_pre], errors="coerce")
        post = pd.to_numeric(out[b_post], errors="coerce")
        urr_ = (pre - post) / pre.replace(0, np.nan) * 100
        out[URR_COL] = urr_.round(2)
        borrowed(URR_COL, b_pre, f1)
        borrowed(URR_COL, b_post, f2)
        neg = int((urr_ < 0).sum())
        notes.append(f"{URR_COL} = (透析前BUN − 透析後BUN)/透析前BUN×100 を算出した"
                     + (f"（負が {neg} 件 — 透析後のほうが高い。前後が逆の疑い）" if neg else ""))

    # ---- 4) spKt/V（Daugirdas 第2世代式）
    td_col = TD_COL if TD_COL in out.columns else (_pick(out, amap, "Td")[0])
    if (b_pre and b_post and UF_COL in out.columns and w_post and td_col
            and KTV_COL not in out.columns):
        pre = pd.to_numeric(out[b_pre], errors="coerce")
        post = pd.to_numeric(out[b_post], errors="coerce")
        t = pd.to_numeric(out[td_col], errors="coerce")
        uf = pd.to_numeric(out[UF_COL], errors="coerce")          # ★L（= kg）★
        w = pd.to_numeric(out[w_post], errors="coerce")
        r = (post / pre.replace(0, np.nan))
        inner = r - 0.008 * t
        # ★log の中が 0 以下なら計算できない。★ 黙って埋めない。NaN にする。
        ktv = -np.log(inner.where(inner > 0)) + (4 - 3.5 * r) * uf / w.replace(0, np.nan)
        out[KTV_COL] = ktv.round(3)
        n_bad = int(ktv.isna().sum() - (pre.isna() | post.isna() | t.isna()
                                        | uf.isna() | w.isna()).sum())
        notes.append(
            "spKt/V を Daugirdas 第2世代式で算出した "
            "（spKt/V = −ln(R − 0.008t) + (4 − 3.5R)·UF/W、R=透析後BUN/透析前BUN、"
            "t=透析時間[hr]、★UF=除水量[L]★、W=透析後体重[kg]）"
            + (f"。算出できなかった例が {n_bad} 件（R − 0.008t ≦ 0）" if n_bad > 0 else ""))

    # ---- 5) TSAT
    fe, _ = _pick(out, amap, "Fe")
    tibc, _ = _pick(out, amap, "TIBC")
    have_tsat = (_pick(out, amap, "TSAT")[0] is not None) or (TSAT_COL in out.columns)
    if fe and tibc and not have_tsat:
        v = (pd.to_numeric(out[fe], errors="coerce")
             / pd.to_numeric(out[tibc], errors="coerce").replace(0, np.nan) * 100)
        out[TSAT_COL] = v.round(2)
        bad = int((v > 100).sum())
        notes.append(f"{TSAT_COL} = Fe/TIBC×100 を算出した"
                     + (f"（100% 超が {bad} 件 — Fe>TIBC は入力ミス）" if bad else ""))

    # ---- 6) iCa（補正カルシウム）
    ca, f1 = _pick(out, amap, "Ca", "pre")
    alb, f2 = _pick(out, amap, "Alb", "pre")
    if ca and alb and ICA_COL not in out.columns:
        c = pd.to_numeric(out[ca], errors="coerce")
        a = pd.to_numeric(out[alb], errors="coerce")
        # ★Payne 式：補正するのは Alb < 4.0 のときだけ。★
        #   Alb ≥ 4.0 では補正せず実測 Ca をそのまま使う（日本の運用に合わせる）。
        #   ここを無条件にすると、Alb 4.5 の症例で iCa が実測より 0.5 低くなる。
        #
        # ★Alb が欠測なら算出不能。★ Ca をそのまま入れると、補正されていない値が
        #   iCa として黙って混ざり、管理目標の達成率も回帰係数も静かにずれる。
        #   `np.where(np.nan < 4.0, ...)` は False に落ちるので、必ず mask する。
        ica = np.where(a < 4.0, c + (4.0 - a), c)
        out[ICA_COL] = pd.Series(ica, index=out.index).mask(a.isna() | c.isna()).round(2)
        borrowed(ICA_COL, ca, f1)
        borrowed(ICA_COL, alb, f2)
        n_unknown = int((a.isna() & c.notna()).sum())
        notes.append(f"{ICA_COL} を Payne 式で算出した"
                     "（Alb < 4.0 のとき Ca + (4 − Alb)、Alb ≥ 4.0 のとき実測 Ca）"
                     + (f"。Alb が欠測の {n_unknown} 例は算出不能として NaN にした"
                        if n_unknown else ""))
    ph, _ = _pick(out, amap, "P")
    if ICA_COL in out.columns and ph and ICAP_COL not in out.columns:
        out[ICAP_COL] = (out[ICA_COL] * pd.to_numeric(out[ph], errors="coerce")).round(2)
        notes.append(f"{ICAP_COL} = {ICA_COL} × P を算出した")
    return out, notes


def derive(df: pd.DataFrame, dic: dict | None = None) -> tuple[pd.DataFrame, list]:
    """辞書に formula を持つ派生指標のうち、入力が揃っているものを計算する。"""
    dic = dic or load_dict()
    amap = build_alias_map(dic)
    col_of = {}
    for c in df.columns:
        k = amap.get(_norm(c))
        if k and k not in col_of:
            col_of[k] = c
    out, notes = df.copy(), []

    def has(*keys):
        return all(k in col_of for k in keys)

    if has("BMI") is False and has("height", "weight") and "BMI" not in col_of:
        h = pd.to_numeric(out[col_of["height"]], errors="coerce") / 100
        w = pd.to_numeric(out[col_of["weight"]], errors="coerce")
        out["BMI"] = (w / (h ** 2)).round(2)
        notes.append("BMI = 体重 / 身長(m)² を算出した")

    # ★透析まわりの派生指標はここに一本化してある。★
    #   補正Ca（iCa）・TSAT もここで作る。式と単位は derive_dialysis の表を見ること。
    out, more = derive_dialysis(out, dic)
    notes += more
    return out, notes
