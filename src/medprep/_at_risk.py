"""Kaplan-Meier 曲線の下に付ける at-risk 表（lifelines 0.30.3 の写し）。

なぜ自前で持つか
----------------
lifelines 0.30.2 以前の ``lifelines.plotting.add_at_risk_counts`` は、
長さ 1 の配列を ``int()`` に渡している。NumPy 2.4 以降ではこれが
``TypeError: only 0-dimensional arrays can be converted to Python scalars`` になり、
**KM 曲線の図そのものが作れずに止まる**（SetupLab の ~/lab は lifelines 0.30.0・
NumPy 2.4.6 を固定しているので必ず起きる）。lifelines 0.30.3 で直ったが、
受講者の環境の lifelines の版は medprep からは決められない。

そこで、0.30.3 の関数を（ほぼそのまま）ここに写し、lifelines の版に関係なく
Colab でもローカルでも同じ at-risk 表が出るようにした。
0.30.0 との違いは「数を ``np.asarray(c).item()`` で取り出す」2 か所だけである。

出典とライセンス
----------------
lifelines (https://github.com/CamDavidsonPilon/lifelines) 0.30.3,
``lifelines/plotting.py`` の ``add_at_risk_counts`` と補助関数。

    MIT License
    Copyright (c) 2017 Cameron Davidson-Pilon

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from __future__ import annotations

import numpy as np

ROWS = ("At risk", "Censored", "Events")


def _scalar(c) -> int:
    """長さ 1 の配列でもスカラーでも int にする（NumPy 2.4 以降でも動く）。"""
    return int(np.asarray(c).item())


def add_at_risk_counts(*fitters, labels=None, rows_to_show=None, ypos=-0.6,
                       xticks=None, ax=None,
                       at_risk_count_from_start_of_period=False, **kwargs):
    """lifelines 0.30.3 の ``add_at_risk_counts`` と同じ。引数も同じ。"""
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = ax.figure
    if labels is None:
        labels = [f._label for f in fitters]
    elif labels is False:
        labels = [None] * len(fitters)
    if rows_to_show is None:
        rows_to_show = list(ROWS)
    else:
        bad = [r for r in rows_to_show if r not in ROWS]
        if bad:
            raise ValueError(f"at_risk_rows は {list(ROWS)} から選ぶ（{bad} は使えない）")
    n_rows = len(rows_to_show)

    # 表を描く軸を、元の軸の下にもう 1 本作る
    ax2 = ax.twiny()
    ax_height = (ax.get_position().y1 - ax.get_position().y0) * fig.get_figheight()
    ax2.spines["bottom"].set_position(("axes", ypos / ax_height))
    for side in ("top", "right", "bottom", "left"):
        ax2.spines[side].set_visible(False)
    ax2.xaxis.tick_bottom()
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    if xticks is None:
        xticks = [t for t in ax.get_xticks() if min_time <= t <= max_time]
    ax2.set_xticks(xticks)
    ax2.xaxis.set_ticks_position("none")
    ax2.yaxis.set_ticks_position("none")

    usetex = mpl.rcParams["text.usetex"]
    ticks = ax2.get_xticks()
    ticklabels = []
    for tick in ticks:
        lbl = ""
        counts = []
        for f in fitters:
            if at_risk_count_from_start_of_period:
                et = f.event_table.assign(at_risk=lambda x: x.at_risk)
            else:
                et = f.event_table.assign(at_risk=lambda x: x.at_risk - x.removed)
            if not et.loc[:tick].empty:
                et = (
                    et.loc[:tick, ["at_risk", "censored", "observed"]]
                    .agg({"at_risk": lambda x: x.tail(1).values.item(),
                          "censored": "sum", "observed": "sum"})
                    .rename({"at_risk": "At risk", "censored": "Censored",
                             "observed": "Events"})
                    .fillna(0)
                )
                counts.extend([_scalar(c) for c in et.loc[rows_to_show]])
            else:
                counts.extend([0 for _ in range(n_rows)])

        if n_rows > 1:
            if tick == ticks[0]:
                max_length = len(str(max(counts)))
                for i, c in enumerate(counts):
                    if i % n_rows == 0:
                        name = labels[int(i / n_rows)]
                        head = rf"\textbf{{{name}}}" if usetex else f"{name}"
                        lbl += ("\n" if i > 0 else "") + head + "\n"
                    row = rows_to_show[i % n_rows]
                    lbl += (row.rjust(10, " ") + " " * (max_length - len(str(c)) + 3)
                            + f"{c:>{max_length}d}\n")
            else:
                for i, c in enumerate(counts):
                    if i % n_rows == 0 and i > 0:
                        lbl += "\n\n"
                    lbl += f"\n{c}"
        else:
            # 1 行だけ出すときは詰めた形
            if tick == ticks[0]:
                max_length = len(str(max(counts)))
                lbl += rows_to_show[0] + "\n"
                for i, c in enumerate(counts):
                    lbl += (labels[i].rjust(10, " ") + " " * (max_length - len(str(c)) + 3)
                            + f"{c:>{max_length}d}\n")
            else:
                for c in counts:
                    lbl += f"\n{c}"
        ticklabels.append(lbl)
    # 数を比べやすいように右揃え
    ax2.set_xticklabels(ticklabels, ha="right", **kwargs)
    return ax
