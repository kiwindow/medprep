"""表を文字で出すときの体裁。

**`DataFrame.to_string()` をそのまま使ってはならない。**
既定の pandas は桁を**文字数**で揃える。日本語は等幅フォントで 2 桁ぶんの幅を取るので、
「アルブミン(Alb)」（9 文字・幅 14）と「年齢」（2 文字・幅 4）が同じ桁数として扱われ、
**表がずれる。** 医学データの列名はほぼ日本語なので、ずれない表のほうが珍しい。

pandas には表示幅で揃える設定があるので、報告を出すあいだだけそれを使う。
`pd.set_option` で全体を書き換えると利用者の設定を壊すため、`option_context` で囲う。
"""
from __future__ import annotations

import pandas as pd


def frame_text(df: pd.DataFrame, *, index: bool = False,
               max_colwidth: int | None = None, width: int = 1000) -> str:
    """表を、日本語の幅で桁の揃った文字列にする。

        print(mp.frame_text(rep.schema.to_frame()))

    `max_colwidth` は既定で **None（切り詰めない）**。
    この package の表の値は「判断の根拠」「対応」のように、**切り詰めたら意味を失う**
    ものが多い。狭くしたいときだけ数値を渡すこと。
    （`display.max_colwidth` は `to_string()` には効かない。引数で渡す必要がある。）
    """
    with pd.option_context(
        "display.unicode.east_asian_width", True,   # ★これが本体★
        "display.max_columns", None,
        "display.width", width,
        "display.max_rows", 2000,
    ):
        return df.to_string(index=index, max_colwidth=max_colwidth)
