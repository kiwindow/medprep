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
                    out[f"{col}__censored_low"] = lo.astype(int)
                    rep.flags[f"{col}__censored_low"] = col
                if hi.any():
                    out[f"{col}__censored_high"] = hi.astype(int)
                    rep.flags[f"{col}__censored_high"] = col

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

    if has("Ca", "Alb") and "cCa" not in col_of:
        ca, alb = out[col_of["Ca"]], out[col_of["Alb"]]
        # ★Alb が欠測なら補正Ca は「算出不能」であって Ca ではない。★
        #   np.where(np.nan < 4.0, ...) は False に落ちるので、そのまま書くと
        #   **補正されていない Ca が補正Ca として黙って混ざる**。
        #   達成率も回帰係数も、その分だけ静かにずれる。
        cca = np.where(alb < 4.0, ca + (4.0 - alb), ca)
        n_unknown = int(alb.isna().sum())
        out["補正Ca"] = pd.Series(cca, index=out.index).mask(alb.isna() | ca.isna())
        notes.append("補正Ca を Payne 式で算出した（Alb<4.0 のとき Ca+(4.0-Alb)）"
                     + (f"。Alb が欠測の {n_unknown} 例は算出不能として NaN にした"
                        if n_unknown else ""))
        col_of["cCa"] = "補正Ca"
    if has("cCa", "P"):
        out["補正Ca×P"] = out[col_of["cCa"]] * out[col_of["P"]]
        notes.append("補正Ca×P を算出した")
    if has("Fe", "TIBC"):
        out["TSAT"] = out[col_of["Fe"]] / out[col_of["TIBC"]] * 100
        bad = (out["TSAT"] > 100).sum()
        notes.append("TSAT = Fe/TIBC×100 を算出した"
                     + (f"（100%超が {bad} 件 — Fe>TIBC は入力ミス）" if bad else ""))
    return out, notes
