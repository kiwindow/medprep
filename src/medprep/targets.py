"""medprep.targets — 管理目標の達成判定。開区間と閉区間を正しく区別する。

「5.5 未満」を `<= 5.5` と実装すると、5.5 ちょうどの症例が達成側に入る。
検査値は 1 桁の小数で報告されるため、境界値ちょうどの症例は実際に多く出る。
この 1 件の取り違えが達成率を数％動かすので、辞書側で
`high_inclusive: false` を持ち、ここで厳密に評価する。
"""

from __future__ import annotations

import pandas as pd


def describe_target(t: dict) -> str:
    """管理目標を日本語の文字列にする。"""
    lo, hi = t.get("low"), t.get("high")
    li = t.get("low_inclusive", True)
    hi_inc = t.get("high_inclusive", True)
    parts = []
    if lo is not None:
        parts.append(f"{lo} {'以上' if li else 'を超える'}")
    if hi is not None:
        parts.append(f"{hi} {'以下' if hi_inc else '未満'}")
    s = "、".join(parts) if parts else "範囲指定なし"
    if t.get("measurement_point"):
        s += f"（{t['measurement_point']}）"
    return s


def in_target(x, t: dict) -> pd.Series:
    """目標範囲内なら True。欠損は NA のまま返す（False にしない）。"""
    v = pd.to_numeric(pd.Series(x), errors="coerce")
    ok = pd.Series(True, index=v.index)
    lo, hi = t.get("low"), t.get("high")
    if lo is not None:
        ok &= (v >= lo) if t.get("low_inclusive", True) else (v > lo)
    if hi is not None:
        ok &= (v <= hi) if t.get("high_inclusive", True) else (v < hi)
    return ok.where(v.notna())


def achievement(df: pd.DataFrame, dic: dict, colmap: dict,
                by: str | None = None) -> pd.DataFrame:
    """管理目標の達成率をまとめる。

    Parameters
    ----------
    colmap : {辞書キー: 列名}   例 {"P": "無機リン(P)", "cCa": "補正Ca"}
    by     : 群分け列（施設など）。指定すると群別に出す。
    """
    rows = []
    for key, col in colmap.items():
        spec = dic["items"].get(key)
        if not spec or "target" not in spec or col not in df.columns:
            continue
        t = spec["target"]
        ok = in_target(df[col], t)
        groups = [("全体", slice(None))] if by is None else \
                 [("全体", slice(None))] + [(g, df[by] == g) for g in sorted(df[by].dropna().unique())]
        for gname, m in groups:
            sub = ok if m is slice(None) else ok[m]
            n = int(sub.notna().sum())
            k = int(sub.sum()) if n else 0
            rows.append({
                "項目": spec["name_ja"], "群": gname, "目標": describe_target(t),
                "評価可能例数": n, "達成": k,
                "達成率": f"{k / n:.1%}" if n else "—",
                "出典": t.get("source", ""),
            })
        # 境界値ちょうどの症例数を必ず出す（開区間／閉区間の取り違えの影響を可視化する）
        for bound, incl, label in ((t.get("low"), t.get("low_inclusive", True), "下限"),
                                   (t.get("high"), t.get("high_inclusive", True), "上限")):
            if bound is None:
                continue
            n_edge = int((pd.to_numeric(df[col], errors="coerce") == bound).sum())
            if n_edge:
                rows.append({
                    "項目": spec["name_ja"], "群": f"【{label}境界 {bound} ちょうど】",
                    "目標": "含む" if incl else "含まない",
                    "評価可能例数": n_edge, "達成": n_edge if incl else 0,
                    "達成率": "100.0%" if incl else "0.0%",
                    "出典": "境界値の扱いを確認すること",
                })
    return pd.DataFrame(rows)


def conditional_target(spec: dict, condition_mask=None) -> list:
    """条件付き目標（iPTH の「活性型ビタミンD製剤のみで管理する場合」等）を返す。"""
    t = spec.get("target", {})
    return t.get("conditional", [])
