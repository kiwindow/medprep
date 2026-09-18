"""medprep.splitting — 学習用と検証用への分割。**リークはここで始まる。**

分割そのものは一行で書けるが、医学データでは次の 3 つを外すと
**あとの工程を何一つ正しくできなくなる。**

1. **同じ患者を train と test の両方に入れない。**
   縦持ちのデータ（1 人が複数行）を行単位で分けると、同じ患者の別の測定が
   両側に入る。モデルは「その患者を覚える」だけで高い精度を出し、
   **新しい患者ではまったく動かない。** `quality.audit` が「同一 ID の複数行」を
   検出するのはこのためであり、ここでは `group=` で患者単位に分ける。

2. **層化を忘れない。**
   イベント率 5% のデータを層化せずに分けると、test のイベントが 0 件になることがある。
   分類なら目的変数で、生存時間ならイベントで層化する。

3. **時間で分けるべき場面がある。**
   「過去のデータで学習し、未来の症例で検証する」のが臨床予測モデルの本来の検証である。
   無作為分割はこれより必ず楽観的な値を出す。`time_order=` で切り替える。

分割した結果には印（`df.attrs`）を付ける。`pipeline.Preprocessor` は
**test に fit しようとしたらそこで止まる**。

モジュール名について
--------------------
公開する関数の名前は `split()` である。モジュール名を `split.py` にすると
`import medprep.split` がパッケージ属性 `medprep.split`（関数）を上書きし、
`mp.split(df)` が突然呼べなくなる（`quality.py` と同じ理由）。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from .describe import smd_categorical, smd_continuous
from .schema import BINARY, EVENT, GROUP, ID, NOMINAL, NUMERIC, ORDINAL, OUTCOME, Schema
from .textfmt import frame_text

TRAIN, TEST = "train", "test"
ATTR = "medprep_split"


# ================================================================== 結果
@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str = "random"
    stratify_by: str | None = None
    group_by: str | None = None
    seed: int = 0
    balance: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def report(self) -> str:
        lines = [f"分割（{self.strategy}）  train {self.n_train} 例 / test {self.n_test} 例"
                 + (f"、層化: {self.stratify_by}" if self.stratify_by else "")
                 + (f"、グループ: {self.group_by}" if self.group_by else "")
                 + f"、seed={self.seed}"]
        if len(self.balance):
            lines += ["\ntrain と test の比較:", frame_text(self.balance)]
        for w in self.warnings:
            lines.append(f"[警告] {w}")
        for n in self.notes:
            lines.append(f"[注記] {n}")
        return "\n".join(lines)

    def show(self):
        print(self.report())


# ================================================================== 分割
def split(
    df: pd.DataFrame,
    schema: Schema | None = None,
    *,
    test_size: float = 0.2,
    outcome: str | None = None,
    task: str | None = None,
    stratify: str | bool = "auto",
    group: str | bool = "auto",
    event: str | None = None,
    time_order: str | None = None,
    seed: int = 0,
    balance_columns: list | None = None,
) -> SplitResult:
    """train / test に分ける。

    Parameters
    ----------
    stratify : "auto" なら task と schema から決める。列名を渡せばその列で層化。
               False で層化しない。
    group    : "auto" なら schema の id 列に重複があれば自動でその列を使う
               （**同じ患者が両側に入るのを防ぐ**）。列名／False も可。
    time_order : 日付列を渡すと、無作為ではなく**時間順**に分ける
                 （過去で学習し未来で検証する）。
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size は 0 と 1 の間（渡されたのは {test_size}）")
    notes, warnings = [], []

    outcome = outcome or (schema.target["name"] if (schema and schema.target) else None)
    task = task or (schema.target.get("task") if (schema and schema.target) else None)
    event = event or (schema.survival.get("event") if (schema and schema.survival) else None)

    group_col = _resolve_group(df, schema, group, notes, warnings)
    strat_col, strat_values, strat_note = _resolve_stratify(
        df, schema, stratify, outcome, task, event, group_col)
    if strat_note:
        notes.append(strat_note)

    if time_order:
        tr_idx, te_idx, strategy = _time_split(df, time_order, test_size, notes, warnings)
        strat_col = None
    elif group_col:
        tr_idx, te_idx, strategy = _group_split(
            df, group_col, strat_values, test_size, seed, notes)
    else:
        strategy = "無作為（層化あり）" if strat_values is not None else "無作為"
        tr_idx, te_idx = train_test_split(
            df.index, test_size=test_size, random_state=seed,
            stratify=strat_values.to_numpy() if strat_values is not None else None)

    train, test = df.loc[tr_idx].copy(), df.loc[te_idx].copy()
    _mark(train, TRAIN)
    _mark(test, TEST)

    res = SplitResult(train=train, test=test, strategy=strategy, stratify_by=strat_col,
                      group_by=group_col, seed=seed, notes=notes, warnings=warnings)
    res.balance = balance_table(train, test, schema=schema, columns=balance_columns)
    _check(res, outcome, event, task)
    return res


def _resolve_group(df, schema, group, notes, warnings) -> str | None:
    if group is False:
        return None
    if isinstance(group, str) and group != "auto":
        if group not in df.columns:
            raise ValueError(f"グループ列 '{group}' がデータに無い")
        return group
    # auto: schema の id 列に重複があれば、それは 1 人 1 行ではない
    if schema is None:
        return None
    ids = [c for c, s in schema.columns.items() if s.role == ID and c in df.columns]
    for c in ids:
        dup = int(df[c].duplicated().sum())
        if dup:
            warnings.append(
                f"'{c}' に重複がある（{dup} 行）。1 人が複数行ある縦持ちのデータなので、"
                f"**行単位で分けると同じ患者が train と test の両方に入る**。"
                f"'{c}' を単位として分割した。行単位にしたい場合は group=False を渡すこと")
            return c
    return None


def _resolve_stratify(df, schema, stratify, outcome, task, event, group_col):
    if stratify is False:
        return None, None, None
    if isinstance(stratify, str) and stratify != "auto":
        if stratify not in df.columns:
            raise ValueError(f"層化に使う列 '{stratify}' がデータに無い")
        return stratify, df[stratify], f"'{stratify}' で層化した"
    # auto
    if event and event in df.columns:
        return event, df[event], (
            f"生存時間なので**イベント '{event}' で層化**した。"
            f"層化しないと test のイベント数が偶然 0 になることがある")
    if outcome and outcome in df.columns:
        s = df[outcome]
        nu = s.nunique(dropna=True)
        is_class = (task or "").startswith("class") or (nu <= 10 and not task)
        if is_class:
            if s.isna().any():
                return None, None, (
                    f"目的変数 '{outcome}' に欠測があるため層化しなかった。"
                    f"missing.drop_missing_outcome() で先に除くこと")
            return outcome, s, f"分類なので目的変数 '{outcome}' で層化した"
        return None, None, (
            f"回帰なので層化していない。分布を揃えたい場合は "
            f"stratify=pd.qcut(df['{outcome}'], 4) 相当の列を作って渡すこと")
    return None, None, None


def _group_split(df, group_col, strat_values, test_size, seed, notes):
    groups = df[group_col].astype(str)
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise ValueError(
            f"'{group_col}' の値が {n_groups} 種類しかないためグループ単位で分割できない")
    # ★グループ数より多くは分けられない。★
    #   施設（3 施設）を単位にして test_size=0.25 と書くと 4 分割が要求され、
    #   sklearn は分かりにくいエラーで止まる。ここで実現可能な分割数に丸める。
    want = max(2, int(round(1 / test_size)))
    n_splits = min(want, n_groups)
    notes.append(f"'{group_col}' を単位に分割した（同じ値の行は必ず同じ側に入る）。"
                 f"test の割合は {1 / n_splits:.0%} 前後になる（グループ単位なので厳密には揃わない）")
    if n_splits < want:
        notes.append(
            f"'{group_col}' は {n_groups} 種類しかないため、要求された test_size="
            f"{test_size} は実現できない。{n_splits} 分割にした"
            f"（test はおよそ {1 / n_splits:.0%}）")
    if n_groups < 10:
        notes.append(
            f"グループが {n_groups} 個しかない。**施設単位の分割は外部妥当性の検証**"
            f"（別の施設でも動くか）であり、無作為分割より厳しい評価になる。"
            f"意図した検証かどうかを確かめること")
    if strat_values is not None and strat_values.notna().all():
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tr, te = next(sp.split(df, strat_values, groups))
        return df.index[tr], df.index[te], "グループ単位（層化あり）"
    sp = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(sp.split(df, groups=groups))
    return df.index[tr], df.index[te], "グループ単位"


def _time_split(df, time_order, test_size, notes, warnings):
    from .dates import parse_date_series
    if time_order not in df.columns:
        raise ValueError(f"時間順に使う列 '{time_order}' がデータに無い")
    t = df[time_order]
    if not pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_datetime64_any_dtype(t):
        r = parse_date_series(t)
        if r.order.startswith("ambiguous"):
            raise ValueError(f"'{time_order}' の日と月の並びが確定できないため時間順に分けられない")
        t = r.values
    order = t.sort_values(kind="mergesort").index
    n_test = max(1, int(round(len(df) * test_size)))
    tr_idx, te_idx = order[:-n_test], order[-n_test:]
    n_na = int(pd.Series(t).isna().sum())
    if n_na:
        warnings.append(f"'{time_order}' が欠測の {n_na} 行がある。"
                        f"並べ替えでは末尾に来るため test 側に寄る")
    notes.append(
        f"**時間順に分割した**（'{time_order}' の古い順に train、新しい順に test）。"
        f"無作為分割は同じ時期のデータが両側に入るため、必ずこれより楽観的な値を出す。"
        f"臨床予測モデルの検証はこちらが本来の姿である")
    return tr_idx, te_idx, "時間順"


# ================================================================== 検査
def balance_table(train: pd.DataFrame, test: pd.DataFrame,
                  schema: Schema | None = None,
                  columns: list | None = None) -> pd.DataFrame:
    """train と test の分布を比べる。**|SMD| < 0.1 が目安。**"""
    if columns is None:
        if schema is not None:
            columns = [c for c in schema.kept()
                       if schema.columns[c].role in
                       (NUMERIC, BINARY, ORDINAL, NOMINAL, GROUP, OUTCOME, EVENT)]
        else:
            columns = [c for c in train.columns
                       if pd.api.types.is_numeric_dtype(train[c])
                       or train[c].nunique(dropna=True) <= 20]
    rows = []
    for c in columns:
        if c not in train.columns or c not in test.columns:
            continue
        a, b = train[c], test[c]
        if pd.api.types.is_numeric_dtype(a) and a.nunique(dropna=True) > 2:
            v = smd_continuous(a, b)
            rows.append({"列": c, "train": f"{pd.to_numeric(a, errors='coerce').mean():.3g}",
                         "test": f"{pd.to_numeric(b, errors='coerce').mean():.3g}",
                         "要約": "平均", "SMD": round(v, 3) if pd.notna(v) else None})
        else:
            if a.nunique(dropna=True) > 20:
                continue
            v = smd_categorical(a, b)
            top = str(a.dropna().mode().iloc[0]) if a.notna().any() else "—"
            rows.append({
                "列": c, "train": f"{(a.astype(str) == top).mean():.1%}",
                "test": f"{(b.astype(str) == top).mean():.1%}",
                "要約": f"'{top}' の割合", "SMD": round(v, 3) if pd.notna(v) else None})
    out = pd.DataFrame(rows)
    if len(out):
        out["判定"] = ["—" if pd.isna(v) else ("偏り" if abs(v) >= 0.2 else "可")
                       for v in out["SMD"]]
    return out


def _check(res: SplitResult, outcome, event, task):
    if len(res.balance):
        bad = res.balance[res.balance["判定"] == "偏り"]
        if len(bad):
            res.warnings.append(
                f"train と test で分布が偏っている列がある（|SMD| ≥ 0.2: {list(bad['列'])}）。"
                f"seed を変えるか、その列で層化すること。"
                f"偏ったまま評価すると、性能の差が『モデルの差』ではなく『分割の運』になる")
    if event and event in res.test.columns:
        n = int(pd.to_numeric(res.test[event], errors="coerce").fillna(0).sum())
        if n < 10:
            res.warnings.append(
                f"test のイベントが {n} 件しかない。C-index も生存曲線も不安定になる。"
                f"test_size を上げるか、交差検証に切り替えること")
    if outcome and outcome in res.test.columns and (task or "").startswith("class"):
        vc = res.test[outcome].value_counts()
        if len(vc) and int(vc.min()) < 5:
            res.warnings.append(
                f"test の最小クラスが {int(vc.min())} 例しかない（'{vc.idxmin()}'）。"
                f"評価指標が 1 例で大きく動く")
    if res.n_test < 30:
        res.warnings.append(f"test が {res.n_test} 例しかない。"
                            f"1 回の分割では評価が不安定なので交差検証を使うこと")
    res.notes.append("分割した train / test には印を付けた。"
                     "pipeline.Preprocessor は test に fit しようとすると止まる")


# ================================================================== 交差検証
def cv_splitter(
    df: pd.DataFrame,
    schema: Schema | None = None,
    *,
    n_splits: int = 5,
    outcome: str | None = None,
    task: str | None = None,
    event: str | None = None,
    group: str | bool = "auto",
    seed: int = 0,
) -> tuple:
    """交差検証の分割器を作る。`(splitter, y, groups, notes)` を返す。

        sp, y, g, notes = mp.cv_splitter(df, sch, n_splits=5)
        for tr, te in sp.split(df, y, g):
            ...
    """
    notes, warnings = [], []
    outcome = outcome or (schema.target["name"] if (schema and schema.target) else None)
    task = task or (schema.target.get("task") if (schema and schema.target) else None)
    event = event or (schema.survival.get("event") if (schema and schema.survival) else None)
    group_col = _resolve_group(df, schema, group, notes, warnings)
    notes += warnings

    strat_col, strat_values, note = _resolve_stratify(
        df, schema, "auto", outcome, task, event, group_col)
    if note:
        notes.append(note)
    groups = df[group_col].astype(str) if group_col else None

    if group_col and strat_values is not None and strat_values.notna().all():
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        notes.append(f"StratifiedGroupKFold（{n_splits} 分割、'{group_col}' 単位、"
                     f"'{strat_col}' で層化）")
    elif group_col:
        sp = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        notes.append(f"GroupKFold（{n_splits} 分割、'{group_col}' 単位）")
    elif strat_values is not None and strat_values.notna().all():
        sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        notes.append(f"StratifiedKFold（{n_splits} 分割、'{strat_col}' で層化）")
    else:
        sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        notes.append(f"KFold（{n_splits} 分割）")

    notes.append("★前処理は fold ごとに train で fit し直すこと。★ "
                 "分割の外で 1 回だけ fit すると、fold の test が fit に混ざる")
    return sp, strat_values, groups, notes


def fold_summary(df: pd.DataFrame, splitter, y=None, groups=None,
                 outcome: str | None = None) -> pd.DataFrame:
    """各 fold の例数と目的変数の分布を表にする。**回す前に確かめる。**"""
    rows = []
    for i, (tr, te) in enumerate(splitter.split(df, y, groups), start=1):
        row = {"fold": i, "train": len(tr), "test": len(te)}
        if outcome and outcome in df.columns:
            s = df[outcome].iloc[te]
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 10:
                row["test の陽性率"] = f"{pd.to_numeric(s, errors='coerce').mean():.1%}"
            elif s.nunique(dropna=True) <= 10:
                row["test の内訳"] = "、".join(
                    f"{k}={v}" for k, v in s.value_counts().items())
            else:
                row["test の平均"] = f"{pd.to_numeric(s, errors='coerce').mean():.3g}"
        if groups is not None:
            row["testのグループ数"] = int(pd.Series(groups).iloc[te].nunique())
            overlap = set(pd.Series(groups).iloc[tr]) & set(pd.Series(groups).iloc[te])
            row["両側に出るグループ"] = len(overlap)
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================== 印
def _mark(df: pd.DataFrame, kind: str) -> None:
    """分割の由来を DataFrame に書き込む。pipeline がこれを見て fit を拒む。"""
    with contextlib.suppress(Exception):
        df.attrs[ATTR] = kind


def split_kind(df: pd.DataFrame) -> str | None:
    """`split()` が付けた印を読む。付いていなければ None。"""
    v = None
    with contextlib.suppress(Exception):
        v = df.attrs.get(ATTR)
    return v if v in (TRAIN, TEST) else None


def mark_as(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """人が明示的に印を付ける（`split()` を通さずに用意した train/test 用）。"""
    if kind not in (TRAIN, TEST):
        raise ValueError("kind は 'train' か 'test'")
    out = df.copy()
    _mark(out, kind)
    return out


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, group: str) -> list:
    """train と test の両方に現れるグループ（＝リーク）を返す。"""
    if group not in train.columns or group not in test.columns:
        return []
    return sorted(set(train[group].dropna().astype(str))
                  & set(test[group].dropna().astype(str)))


def levels_only_in_test(train: pd.DataFrame, test: pd.DataFrame,
                        columns: list | None = None) -> pd.DataFrame:
    """test にしか出てこないカテゴリ水準を探す。

    one-hot のときに `handle_unknown` を誤ると、ここで黙って落ちるか例外になる。
    どちらにせよ**先に知っておくべき**である。
    """
    cols = columns or [c for c in train.columns
                       if c in test.columns
                       and not pd.api.types.is_numeric_dtype(train[c])
                       and train[c].nunique(dropna=True) <= 50]
    rows = []
    for c in cols:
        a = set(train[c].dropna().astype(str))
        b = set(test[c].dropna().astype(str))
        only = sorted(b - a)
        if only:
            rows.append({"列": c, "test にしか無い水準": "、".join(only),
                         "該当例数": int(test[c].astype(str).isin(only).sum())})
    return pd.DataFrame(rows)
