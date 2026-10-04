"""0.12.2 — pandas 3 の str 型の列でも、文字列の列として扱う。

pandas 3 では文字列だけの列の型が object ではなく str（StringDtype）になる。
`dtype == object` で判定していた 2 か所が、文字列の列を見落としていた:
- 全セルが空欄の行: '' や全角空白だけの行を「空」と数えなかった
- 辞書で掃除: 「未測定」などの欠損表記を NaN にした記録が残らなかった
ここでは pandas 2 でも同じことが起きるよう、列を明示的に str 型にして確かめる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import medprep as mp
from medprep.auto import _blank_mask

STR = pd.StringDtype()


def test_blank_strings_in_a_str_column_count_as_blank():
    df = pd.DataFrame({
        "ID": pd.Series(["a", "", "　", None], dtype=STR),
        "x": [1.0, np.nan, np.nan, np.nan],
    })
    m = _blank_mask(df)
    assert m.all(axis=1).tolist() == [False, True, True, True]


def test_missing_words_in_a_str_column_are_recorded():
    df = pd.DataFrame({"アルブミン(Alb)": pd.Series(["3.5", "未測定", "3.8", "4.0"], dtype=STR)})
    clean, rep = mp.clean_numeric(df)
    assert np.isnan(clean["アルブミン(Alb)"].iloc[1])
    assert clean["アルブミン(Alb)"].iloc[0] == 3.5
    assert any(a[1] == "テキスト欠損表記→NaN" and a[2] == 1 for a in rep.actions)
